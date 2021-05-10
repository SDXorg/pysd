"""
These functions have no direct analog in the standard python data analytics
stack, or require information about the internal state of the system beyond
what is present in the function call. We provide them in a structure that
makes it easy for the model elements to call.
"""

import inspect
import os
import sys
import re
import random
import warnings
import pathlib
from importlib.machinery import SourceFileLoader
from ast import literal_eval

import numpy as np
import pandas as pd
import xarray as xr
from funcsigs import signature

from . import utils
from .external import External, Excels

from pysd._version import __version__

try:
    import scipy.stats as stats

    def bounded_normal(minimum, maximum, mean, std, seed):
        """ Implements vensim's BOUNDED NORMAL function """
        # np.random.seed(seed)  # we could bring this back later, but for now, ignore
        return stats.truncnorm.rvs(minimum, maximum, loc=mean, scale=std)

except ImportError:
    warnings.warn("Scipy required for functions:"
                  "- Bounded Normal (falling back to unbounded normal)")

    def bounded_normal(minimum, maximum, mean, std, seed):
        """ Warning: using unbounded normal due to no scipy """
        return np.random.normal(mean, std)

small_vensim = 1e-6  # What is considered zero according to Vensim Help


class Stateful(object):
    # the integrator needs to be able to 'get' the current state of the object,
    # and get the derivative. It calculates the new state, and updates it.
    # The state can be any object which is subject to basic (element-wise)
    # algebraic operations
    def __init__(self):
        self._state = None

    def __call__(self, *args, **kwargs):
        return self.state

    def initialize(self):
        raise NotImplementedError

    @property
    def state(self):
        if self._state is None:
            raise AttributeError('Attempt to call stateful element before it is initialized.')
        return self._state

    @state.setter
    def state(self, new_value):
        self._state = new_value

    def update(self, state):
        self.state = state


class Integ(Stateful):
    def __init__(self, ddt, initial_value):
        """

        Parameters
        ----------
        ddt: function
            This will become an attribute of the object
        initial_value
        """
        super().__init__()
        self.init_func = initial_value
        self.ddt = ddt
        self.shape_info = None

    def initialize(self):
        self.state = self.init_func()
        if isinstance(self.state, xr.DataArray):
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_value):
        if self.shape_info is not None:
            self._state = xr.DataArray(data=new_value, **self.shape_info)
        else:
            self._state = new_value


class Delay(Stateful):
    # note that we could have put the `delay_input` argument as a parameter to
    # the `__call__` function, and more closely mirrored the vensim syntax.
    # However, people may get confused this way in thinking that they need
    # only one delay object and can call it with various arguments to delay
    # whatever is convenient. This method forces them to acknowledge that
    # additional structure is being created in the delay object.

    def __init__(self, delay_input, delay_time, initial_value, order):
        """

        Parameters
        ----------
        delay_input: function
        delay_time: function
        initial_value: function
        order: function
        coords: dictionary (optional)
        dims: list (optional)
        """
        super().__init__()
        self.init_func = initial_value
        self.delay_time_func = delay_time
        self.input_func = delay_input
        self.order_func = order
        self.order = None
        self.shape_info = None

    def initialize(self):
        order = self.order_func()

        if order != int(order):
            warnings.warn('Casting delay order from %f to %i' % (
                order, int(order)))

        self.order = int(order)  # The order can only be set once

        init_state_value = self.init_func() * self.delay_time_func()\
                           / self.order

        if isinstance(init_state_value, xr.DataArray):
            # broadcast self.state
            self.state = init_state_value.expand_dims({
                'delay': np.arange(self.order)}, axis=0)
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}
        else:
            self.state = np.array([init_state_value] * self.order)

    def __call__(self):
        if self.shape_info:
            return self.state[-1].reset_coords('delay', drop=True)\
                   / (self.delay_time_func() / self.order)
        else:
            return self.state[-1] / (self.delay_time_func() / self.order)

    def ddt(self):
        outflows = self.state / (self.delay_time_func() / self.order)
        inflows = np.roll(outflows, 1, axis=0)
        if self.shape_info:
            inflows[0] = self.input_func().values
        else:
            inflows[0] = self.input_func()
        return inflows - outflows

class SampleIfTrue(Stateful):
    def __init__(self, condition, actual_value, initial_value):
        """

        Parameters
        ----------
        condition: function
        actual_value: function
        initial_value: function
        """
        super().__init__()
        self.condition = condition
        self.actual_value = actual_value
        self.initial_value = initial_value
    
    def initialize(self):
        self.state = self.initial_value()
        if isinstance(self.state, xr.DataArray):
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}
    
    def __call__(self):
        self.state = sample_if_true(self.condition(), self.actual_value(), self.state)
        return self.state
    
    def ddt(self):
        return 0

class Smooth(Stateful):
    def __init__(self, smooth_input, smooth_time, initial_value, order):
        super().__init__()
        self.init_func = initial_value
        self.smooth_time_func = smooth_time
        self.input_func = smooth_input
        self.order_func = order
        self.order = None

    def initialize(self):
        self.order = self.order_func()  # The order can only be set once
        self.state = np.array([self.init_func()] * self.order)

    def __call__(self):
        return self.state[-1]

    def ddt(self):
        targets = np.roll(self.state, 1)
        targets[0] = self.input_func()
        return (targets - self.state) * self.order / self.smooth_time_func()


class Trend(Stateful):
    def __init__(self, trend_input, average_time, initial_trend):
        super().__init__()
        self.init_func = initial_trend
        self.average_time_function = average_time
        self.input_func = trend_input

    def initialize(self):
        self.state = self.input_func()\
            / (1 + self.init_func() * self.average_time_function())

    def __call__(self):
        return zidz(self.input_func() - self.state,
                    self.average_time_function() * abs(self.state))

    def ddt(self):
        return (self.input_func() - self.state) / self.average_time_function()


class Initial(Stateful):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def initialize(self):
        self.state = self.func()

    def ddt(self):
        return 0

    def update(self, state):
        # this doesn't change once it's set up.
        pass


class Macro(Stateful):
    """
    The Model class implements a stateful representation of the system,
    and contains the majority of methods for accessing and modifying model
    components.

    When the instance in question also serves as the root model object
    (as opposed to a macro or submodel within another model) it will have
    added methods to facilitate execution.
    """

    def __init__(self, py_model_file, params=None, return_func=None,
                 time=None, time_initialization=None):
        """
        The model object will be created with components drawn from a
        translated python model file.

        Parameters
        ----------
        py_model_file : <string>
            Filename of a model which has already been converted into a
            python format.
        get_time:
            needs to be a function that returns a time object
        params
        return_func
        """
        super().__init__()
        self.time = time
        self.time_initialization = time_initialization

        # need a unique identifier for the imported module.
        module_name = os.path.splitext(py_model_file)[0]\
                      + str(random.randint(0, 1000000))
        try:
            self.components = SourceFileLoader(module_name,
                                               py_model_file).load_module()
        except TypeError:
            raise ImportError(
                "\n\nNot able to import the model. "
                + "This may be because the model was compiled with an "
                + "earlier version of PySD, you can check on the top of "
                + " the model file you are trying to load."
                + "\nThe current version of PySd is :"
                + "\n\tPySD " + __version__ + "\n\n"
                + "Please translate again the model with the function"
                + " read_vensim or read_xmile.")

        if __version__.split(".")[0]\
           != self.get_pysd_compiler_version().split(".")[0]:
            raise ImportError(
                "\n\nNot able to import the model. "
                + "The model was compiled with a "
                + "not compatible version of PySD:"
                + "\n\tPySD " + self.get_pysd_compiler_version()
                + "\n\nThe current version of PySd is:"
                + "\n\tPySD " + __version__ + "\n\n"
                + "Please translate again the model with the function"
                + " read_vensim or read_xmile.")

        if params is not None:
            self.set_components(params)

        # Get the collections of stateful elements and external elements
        self._stateful_elements = [
            getattr(self.components, name) for name in dir(self.components)
            if isinstance(getattr(self.components, name), Stateful)
        ]
        self._external_elements = [
            getattr(self.components, name) for name in dir(self.components)
            if isinstance(getattr(self.components, name), External)
        ]

        if return_func is not None:
            self.return_func = getattr(self.components, return_func)
        else:
            self.return_func = lambda: 0

        self.py_model_file = py_model_file

    def __call__(self):
        return self.return_func()

    def get_pysd_compiler_version(self):
        """
        Returns the version of pysd complier that used for generating
        this model
        """
        return self.components.__pysd_version__

    def initialize(self, initialization_order=None):
        """
        This function tries to initialize the stateful objects.

        In the case where an initialization function for `Stock A` depends on
        the value of `Stock B`, if we try to initialize `Stock A` before
        `Stock B` then we will get an error, as the value will not yet exist.

        In this case, just skip initializing `Stock A` for now, and
        go on to the other state initializations. Then come back to it and
        try again.
        """

        # Initialize time
        if self.time is None:
            if self.time_initialization is None:
                self.time = Time()
            else:
                self.time = self.time_initialization()

        self.components.cache.clean()
        self.components.cache.time = self.time()

        self.components._init_outer_references({
            'scope': self,
            'time': self.time
        })

        # Initialize external elements
        for element in self._external_elements:
            element.initialize()

        Excels.clean()

        # Initialize stateful elements
        remaining = set(self._stateful_elements)
        while remaining:
            progress = set()
            for element in remaining:
                try:
                    element.initialize()
                    progress.add(element)
                except (KeyError, TypeError, AttributeError):
                    pass

            if progress:
                remaining.difference_update(progress)
            else:
                raise KeyError('Unresolvable Reference: Probable circular initialization' +
                               '\n'.join([repr(e) for e in remaining]))

    def ddt(self):
        return np.array([component.ddt() for component in self._stateful_elements], dtype=object)

    @property
    def state(self):
        return np.array([component.state for component in self._stateful_elements], dtype=object)

    @state.setter
    def state(self, new_value):
        [component.update(val) for component, val in zip(self._stateful_elements, new_value)]

    def get_coords(self, param):
        """
        Returns the the coordinates and dims of model element if it has,
        otherwise returns None
        >>> model.set_components('birth_rate')
        >>> model.set_components('Birth Rate')
        """
        func_name = utils.get_value_by_insensitive_key_or_value(param,
            self.components._namespace) or param

        try:
            # TODO: make this less fragile, may crash if the component
            # takes arguments and is a xarray
            value = getattr(self.components, func_name)()
            dims = list(value.dims)
            coords = {coord: list(value.coords[coord].values)
                      for coord in value.coords}
            return coords, dims
        except Exception:
            return None

    def set_components(self, params):
        """ Set the value of exogenous model elements.
        Element values can be passed as keyword=value pairs in the function call.
        Values can be numeric type or pandas Series.
        Series will be interpolated by integrator.

        Examples
        --------

        >>> model.set_components({'birth_rate': 10})
        >>> model.set_components({'Birth Rate': 10})

        >>> br = pandas.Series(index=range(30), values=np.sin(range(30))
        >>> model.set_components({'birth_rate': br})


        """
        # It might make sense to allow the params argument to take a pandas series, where
        # the indices of the series are variable names. This would make it easier to
        # do a Pandas apply on a DataFrame of parameter values. However, this may conflict
        # with a pandas series being passed in as a dictionary element.

        for key, value in params.items():

            func_name = utils.get_value_by_insensitive_key_or_value(key,
                self.components._namespace)
            try:
                # TODO: make this less fragile, may crash if the component
                # takes arguments and is an xarray
                # can use the __doc__ but will not be backward compatible:
                # dims = literal_eval(re.findall("Subs: .+",
                # getattr(self.components, func_name).__doc__)[0][6:])
                dims = getattr(self.components, func_name)().dims
            except Exception:
                dims = None

            if isinstance(value, np.ndarray) or isinstance(value, list):
                raise ValueError('When setting ' + key +'\n'
                                 + 'Setting subscripted must be done'
                                 + 'using a xarray.DataArray with the '
                                 + 'correct dimensions or a constant value '
                                 + '(https://pysd.readthedocs.io/en/master/basic_usage.html)')

            if func_name is None:
                raise NameError('%s is not recognized as a model component'
                                % key)

            if isinstance(value, pd.Series):
                new_function = self._timeseries_component(value, dims)
            elif callable(value):
                new_function = value
            else:
                new_function = self._constant_component(value, dims)

            # this won't handle other statefuls...
            if '_integ_' + func_name in dir(self.components):
                warnings.warn("Replacing the equation of stock"
                              + "{} with params".format(key),
                              stacklevel=2)

            setattr(self.components, func_name, new_function)

    def _timeseries_component(self, series, dims):
        """ Internal function for creating a timeseries model element """
        # this is only called if the set_component function recognizes a
        # pandas series
        # TODO: raise a warning if extrapolating from the end of the series.
        if isinstance(series.values[0], xr.DataArray):
            return lambda: utils.rearrange(xr.concat(series.values,
                series.index).interp(concat_dim=self.time()).reset_coords(
                'concat_dim', drop=True),dims, self.components._subscript_dict)

        else:
            if dims:
                return lambda: utils.rearrange(np.interp(self.time(),
                    series.index, series.values),
                    dims, self.components._subscript_dict)
            return lambda: np.interp(self.time(), series.index, series.values)

    def _constant_component(self, value, dims):
        """ Internal function for creating a constant model element """
        if dims:
            return lambda: utils.rearrange(value, dims,
                                           self.components._subscript_dict)
        return lambda: value

    def set_state(self, t, state):
        """ Set the system state.

        Parameters
        ----------
        t : numeric
            The system time

        state : dict
            A (possibly partial) dictionary of the system state.
            The keys to this dictionary may be either pysafe names or
            original model file names
        """
        self.time.update(t)
        self.components.cache.reset(t)

        for key, value in state.items():
            # TODO Implement map with reference between component and stateful element?
            component_name = utils.get_value_by_insensitive_key_or_value(key,
                self.components._namespace)
            if component_name is not None:
                stateful_name = '_integ_%s' % component_name
            else:
                component_name = key
                stateful_name = key

            try:
                # TODO make this less fragile (avoid using the __doc__)
                if component_name[:7] == '_integ_':
                    # we need to check the original expression to retrieve
                    # the dimensions
                    dims = literal_eval(re.findall("Subs: .+",
                        getattr(self.components,
                                component_name[7:]).__doc__)[0][6:])
                else:
                    dims = literal_eval(re.findall("Subs: .+",
                        getattr(self.components,
                                component_name).__doc__)[0][6:])
            except Exception:
                dims = None

            if isinstance(value, np.ndarray) or isinstance(value, list):
                raise ValueError('When setting ' + key +'\n'
                                 + 'Setting subscripted must be done'
                                 + 'using a xarray.DataArray with the '
                                 + 'correct dimensions or a constant value '
                                 + '(https://pysd.readthedocs.io/en/master/basic_usage.html)')

            # Try to update stateful component
            if hasattr(self.components, stateful_name):
                try:
                    element = getattr(self.components, stateful_name)
                    if dims:
                        value = utils.rearrange(value, dims,
                            self.components._subscript_dict)
                    element.update(value)
                except AttributeError:
                    print("'%s' has no state elements, assignment failed")
                    raise
            else:
                # Try to override component
                try:
                    setattr(self.components, component_name,
                            self._constant_component(value, dims))
                except AttributeError:
                    print("'%s' has no component, assignment failed")
                    raise

    def doc(self):
        """
        Formats a table of documentation strings to help users remember
        variable names, and understand how they are translated into
        python safe names.

        Returns
        -------
        docs_df: pandas dataframe
            Dataframe with columns for the model components:
                - Real names
                - Python safe identifiers (as used in model.components)
                - Units string
                - Documentation strings from the original model file
        """
        collector = []
        for name, varname in self.components._namespace.items():
            try:
                # TODO correct this when Original Eqn is in several lines
                docstring = getattr(self.components, varname).__doc__
                lines = docstring.split('\n')

                for unit_line in range(3, 9):
                    # this loop detects where Units: starts as
                    # sometimes eqn could be split in several lines
                    if re.findall('Units:', lines[unit_line]):
                        break
                if unit_line == 3:
                    eqn = lines[2].replace("Original Eqn:", "").strip()
                else:
                    eqn = '; '.join([l.strip() for l in lines[3:unit_line]])

                collector.append(
                    {'Real Name': name,
                     'Py Name': varname,
                     'Eqn': eqn,
                     'Unit': lines[unit_line].replace("Units:", "").strip(),
                     'Lims': lines[unit_line+1].replace("Limits:", "").strip(),
                     'Type': lines[unit_line+2].replace("Type:", "").strip(),
                     'Subs': lines[unit_line+3].replace("Subs:", "").strip(),
                     'Comment': '\n'.join(lines[(unit_line+4):]).strip()})
            except Exception:
                pass

        docs_df = pd.DataFrame(collector)
        docs_df.fillna('None', inplace=True)

        order = ['Real Name', 'Py Name', 'Unit', 'Lims',
                 'Type', 'Subs', 'Eqn', 'Comment']
        return docs_df[order].sort_values(by='Real Name').reset_index(drop=True)

    def __str__(self):
        """ Return model source files """

        # JT: Might be helpful to return not only the source file, but
        # also how the instance differs from that source file. This
        # would give a more accurate view of the current model.
        string = 'Translated Model File: ' + self.py_model_file
        if hasattr(self, 'mdl_file'):
            string += '\n Original Model File: ' + self.mdl_file

        return string


class Time(object):
    def __init__(self, t=None, dt=None):
        self._t = t
        self._step = dt
        self.stage = None

    def __call__(self):
        return self._t

    def step(self):
        return self._step

    def update(self, value):
        if self._t is not None:
            self._step = value - self._t

        self._t = value


class Model(Macro):
    def __init__(self, py_model_file, initialize, missing_values):
        """ Sets up the python objects """
        super().__init__(py_model_file, None, None, Time())
        self.time.stage = 'Load'
        self.missing_values = missing_values
        if initialize:
            self.initialize()

    def initialize(self):
        """ Initializes the simulation model """
        self.time.update(self.components.initial_time())
        self.time.stage = 'Initialization'
        External.missing = self.missing_values
        super().initialize()

    def _build_euler_timeseries(self, return_timestamps=None):
        """
        - The integration steps need to include the return values.
        - There is no point running the model past the last return value.
        - The last timestep will be the last in that requested for return
        - Spacing should be at maximum what is specified by the integration time step.
        - The initial time should be the one specified by the model file, OR
          it should be the initial condition.
        - This function needs to be called AFTER the model is set in its initial state
        Parameters
        ----------
        return_timestamps: numpy array
          Must be specified by user or built from model file before this function is called.

        Returns
        -------
        ts: numpy array
            The times that the integrator will use to compute time history
        """
        t_0 = self.time()
        t_f = return_timestamps[-1]
        dt = self.components.time_step()
        ts = np.arange(t_0, t_f, dt, dtype=np.float64)

        # Add the returned time series into the integration array. Best we can do for now.
        # This does change the integration ever so slightly, but for well-specified
        # models there shouldn't be sensitivity to a finer integration time step.
        ts = np.sort(np.unique(np.append(ts, return_timestamps)))
        return ts

    def _format_return_timestamps(self, return_timestamps=None):
        """
        Format the passed in return timestamps value as a numpy array.
        If no value is passed, build up array of timestamps based upon
        model start and end times, and the 'saveper' value.
        """
        if return_timestamps is None:
            # Build based upon model file Start, Stop times and Saveper
            # Vensim's standard is to expect that the data set includes the `final time`,
            # so we have to add an extra period to make sure we get that value in what
            # numpy's `arange` gives us.
            return_timestamps_array = np.arange(
                self.components.initial_time(),
                self.components.final_time() + self.components.saveper(),
                self.components.saveper(), dtype=np.float64
            )
        elif inspect.isclass(range) and isinstance(return_timestamps, range):
            return_timestamps_array = np.array(return_timestamps, ndmin=1)
        elif isinstance(return_timestamps, (list, int, float, np.ndarray)):
            return_timestamps_array = np.array(return_timestamps, ndmin=1)
        elif isinstance(return_timestamps, pd.Series):
            return_timestamps_array = return_timestamps.as_matrix()
        else:
            raise TypeError('`return_timestamps` expects a list, array, pandas Series, '
                            'or numeric value')
        return return_timestamps_array

    def run(self, params=None, return_columns=None, return_timestamps=None,
            initial_condition='original', reload=False, progress=False):
        """ Simulate the model's behavior over time.
        Return a pandas dataframe with timestamps as rows,
        model elements as columns.

        Parameters
        ----------
        params : dictionary
            Keys are strings of model component names.
            Values are numeric or pandas Series.
            Numeric values represent constants over the model integration.
            Timeseries will be interpolated to give time-varying input.

        return_timestamps : list, numeric, numpy array(1-D)
            Timestamps in model execution at which to return state information.
            Defaults to model-file specified timesteps.

        return_columns : list of string model component names
            Returned dataframe will have corresponding columns.
            Defaults to model stock values.

        initial_condition : 'original'/'o', 'current'/'c', (t, {state})
            The starting time, and the state of the system (the values of all the stocks)
            at that starting time.

            * 'original' (default) uses model-file specified initial condition
            * 'current' uses the state of the model after the previous execution
            * (t, {state}) lets the user specify a starting time and (possibly partial)
              list of stock values.

        reload : bool
            If true, reloads the model from the translated model file before making changes

        progress : bool
            If true, a progressbar will be shown during integration

        Examples
        --------

        >>> model.run(params={'exogenous_constant': 42})
        >>> model.run(params={'exogenous_variable': timeseries_input})
        >>> model.run(return_timestamps=[1, 2, 3.1415, 4, 10])
        >>> model.run(return_timestamps=10)
        >>> model.run(return_timestamps=np.linspace(1, 10, 20))

        See Also
        --------
        pysd.set_components : handles setting model parameters
        pysd.set_initial_condition : handles setting initial conditions

        """
        if reload:
            self.reload()

        self.progress = progress

        if params:
            self.set_components(params)
        self.set_initial_condition(initial_condition)

        return_timestamps = self._format_return_timestamps(return_timestamps)

        t_series = self._build_euler_timeseries(return_timestamps)

        if return_columns is None:
            return_columns = self._default_return_columns()

        self.time.stage = 'Run'
        self.components.cache.clean()

        capture_elements, return_addresses = utils.get_return_elements(
            return_columns, self.components._namespace, self.components._subscript_dict)

        res = self._integrate(t_series, capture_elements, return_timestamps)

        return_df = utils.make_flat_df(res, return_addresses)
        return_df.index = return_timestamps

        return return_df

    def reload(self):
        """Reloads the model from the translated model file, so that all the
        parameters are back to their original value.
        """
        self.__init__(self.py_model_file, initialize=True, missing_values=self.missing_values)

    def _default_return_columns(self):
        """
        Return a list of the model elements that does not include lookup functions
        or other functions that take parameters.
        """
        return_columns = []
        parsed_expr = []

        for key, value in self.components._namespace.items():
            if hasattr(self.components, value):
                sig = signature(getattr(self.components, value))
                # The `*args` reference handles the py2.7 decorator.
                if len(set(sig.parameters) - {'args'}) == 0:
                    expr = self.components._namespace[key]
                    if expr not in parsed_expr:
                        return_columns.append(key)
                        parsed_expr.append(expr)

        return return_columns

    def set_initial_condition(self, initial_condition):
        """ Set the initial conditions of the integration.

        Parameters
        ----------
        initial_condition : <string> or <tuple>
            Takes on one of the following sets of values:

            * 'original'/'o' : Reset to the model-file specified initial condition.
            * 'current'/'c' : Use the current state of the system to start
              the next simulation. This includes the simulation time, so this
              initial condition must be paired with new return timestamps
            * (t, {state}) : Lets the user specify a starting time and list of stock values.

        >>> model.set_initial_condition('original')
        >>> model.set_initial_condition('current')
        >>> model.set_initial_condition((10, {'teacup_temperature': 50}))

        See Also
        --------
        PySD.set_state()

        """

        if isinstance(initial_condition, tuple):
            # Todo: check the values more than just seeing if they are a tuple.
            self.set_state(*initial_condition)
        elif isinstance(initial_condition, str):
            if initial_condition.lower() in ['original', 'o']:
                self.initialize()
            elif initial_condition.lower() in ['current', 'c']:
                pass
            else:
                raise ValueError('Valid initial condition strings include:  \n' +
                                 '    "original"/"o",                       \n' +
                                 '    "current"/"c"')
        else:
            raise TypeError('Check documentation for valid entries')

    def _euler_step(self, dt):
        """ Performs a single step in the euler integration,
        updating stateful components

        Parameters
        ----------
        dt : float
            This is the amount to increase time by this step
        """
        self.state = self.state + self.ddt() * dt

    def _integrate(self, time_steps, capture_elements, return_timestamps):
        """
        Performs euler integration

        Parameters
        ----------
        time_steps: iterable
            the time steps that the integrator progresses over
        capture_elements: list
            which model elements to capture - uses pysafe names
        return_timestamps:
            which subset of 'timesteps' should be values be returned?

        Returns
        -------
        outputs: list of dictionaries

        """
        # Todo: consider adding the timestamp to the return elements, and using that as the index
        outputs = []

        if self.progress:
            # initialize progress bar
            progressbar = utils.ProgressBar(len(time_steps)-1)
        else:
            # when None is used the update will do nothing
            progressbar = utils.ProgressBar(None)

        for t2 in time_steps[1:]:
            if self.time() in return_timestamps:
                outputs.append({key: getattr(self.components, key)() for key in capture_elements})
            self._euler_step(t2 - self.time())
            self.time.update(t2)  # this will clear the stepwise caches
            self.components.cache.reset(t2)
            progressbar.update()

        # need to add one more time step, because we run only the state updates in the previous
        # loop and thus may be one short.
        if self.time() in return_timestamps:
            outputs.append({key: getattr(self.components, key)() for key in capture_elements})

        progressbar.finish()

        return outputs


def ramp(time, slope, start, finish=0):
    """
    Implements vensim's and xmile's RAMP function

    Parameters
    ----------
    time: function
        The current time of modelling
    slope: float
        The slope of the ramp starting at zero at time start
    start: float
        Time at which the ramp begins
    finish: float
        Optional. Time at which the ramp ends

    Returns
    -------
    response: float
        If prior to ramp start, returns zero
        If after ramp ends, returns top of ramp
    Examples
    --------

    """

    t = time()
    if t < start:
        return 0
    else:
        if finish <= 0:
            return slope * (t - start)
        elif t > finish:
            return slope * (finish - start)
        else:
            return slope * (t - start)


def step(time, value, tstep):
    """"
    Implements vensim's STEP function

    Parameters
    ----------
    value: float
        The height of the step
    tstep: float
        The time at and after which `result` equals `value`

    Returns
    -------
    - In range [-inf, tstep) returns 0
    - In range [tstep, +inf] returns `value`
    """
    return value if time() >= tstep else 0


def pulse(time, start, duration):
    """ Implements vensim's PULSE function

    In range [-inf, start) returns 0
    In range [start, start + duration) returns 1
    In range [start + duration, +inf] returns 0
    """
    t = time()
    return 1 if start <= t < start + duration else 0


def pulse_train(time, start, duration, repeat_time, end):
    """ Implements vensim's PULSE TRAIN function

    In range [-inf, start) returns 0
    In range [start + n * repeat_time, start + n * repeat_time + duration) return 1
    In range [start + n * repeat_time + duration, start + (n+1) * repeat_time) return 0
    """
    t = time()
    if start <= t < end:
        return 1 if (t - start) % repeat_time < duration else 0
    else:
        return 0


def pulse_magnitude(time, magnitude, start, repeat_time=0):
    """ Implements xmile's PULSE function

    PULSE:             Generate a one-DT wide pulse at the given time
       Parameters:     2 or 3:  (magnitude, first time[, interval])
                       Without interval or when interval = 0, the PULSE is generated only once
       Example:        PULSE(20, 12, 5) generates a pulse value of 20/DT at time 12, 17, 22, etc.

    In rage [-inf, start) returns 0
    In range [start + n * repeat_time, start + n * repeat_time + dt) return magnitude/dt
    In rage [start + n * repeat_time + dt, start + (n + 1) * repeat_time) return 0

    """
    t = time()
    if repeat_time <= small_vensim:
        if abs(t - start) < time.step():
            return magnitude * time.step()
        else:
            return 0
    else:
        if abs((t - start) % repeat_time) < time.step():
            return magnitude * time.step()
        else:
            return 0


def lookup(x, xs, ys):
    """
    Intermediate values are calculated with linear interpolation between
    the intermediate points. Out-of-range values are the same as the
    closest endpoint (i.e, no extrapolation is performed).
    """
    return np.interp(x, xs, ys)


def lookup_extrapolation(x, xs, ys):
    """
    Intermediate values are calculated with linear interpolation between
    the intermediate points. Out-of-range values are calculated with linear
    extrapolation from the last two values at either end.
    """
    if x < xs[0]:
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        k = dy / dx
        return ys[0] + (x - xs[0]) * k
    if x > xs[-1]:
        dx = xs[-1] - xs[-2]
        dy = ys[-1] - ys[-2]
        k = dy / dx
        return ys[-1] + (x - xs[-1]) * k
    return np.interp(x, xs, ys)


def lookup_discrete(x, xs, ys):
    """
    Intermediate values take on the value associated with the next lower
    x-coordinate (also called a step-wise function). The last two points
    of a discrete graphical function must have the same y value.
    Out-of-range values are the same as the closest endpoint
    (i.e, no extrapolation is performed).
    """
    for index in range(0, len(xs)):
        if x < xs[index]:
            return ys[index - 1] if index > 0 else ys[index]
    return ys[-1]


def if_then_else(condition, val_if_true, val_if_false):
    """
    Implements Vensim's IF THEN ELSE function.

    Parameters
    ----------
    condition: bool or xarray.DataArray of bools
    val_if_true: function
        Value to evaluate and return when condition is true.
    val_if_false: function
        Value to evaluate and return when condition is false.

    Returns
    -------
    The value depending on the condition.

    """
    if isinstance(condition, xr.DataArray):
        if condition.all():
            return val_if_true()
        elif not condition.any():
            return val_if_false()

        return xr.where(condition, val_if_true(), val_if_false())

    return val_if_true() if condition else val_if_false()


def sample_if_true(condition, actual_value, saved_value):
    """
    Implements Vensim's SAMPLE IF TRUE function.

    Parameters
    ----------
    condition: bool or xarray.DataArray of bools
    actual_value: int, float or xarray.DataArray
        Value to return when condition is true.
    saved_value: int, float or xarray.DataArray
        Value to return when condition is false.

    Returns
    -------
    The value depending on the condition.

    """
    if isinstance(condition, xr.DataArray):
        if condition.all():
            return actual_value
        elif not condition.any():
            return saved_value

        return xr.where(condition, actual_value, saved_value)

    return actual_value if condition else saved_value


def xidz(numerator, denominator, value_if_denom_is_zero):
    """
    Implements Vensim's XIDZ function.
    This function executes a division, robust to denominator being zero.
    In the case of zero denominator, the final argument is returned.

    Parameters
    ----------
    numerator: float or xarray.DataArray
    denominator: float or xarray.DataArray
        Components of the division operation
    value_if_denom_is_zero: float or xarray.DataArray
        The value to return if the denominator is zero

    Returns
    -------
    numerator / denominator if denominator > 1e-6
    otherwise, returns value_if_denom_is_zero

    """
    if isinstance(denominator, xr.DataArray):
        return xr.where(np.abs(denominator) < small_vensim,
                        value_if_denom_is_zero,
                        numerator * 1.0 / denominator)

    if abs(denominator) < small_vensim:
        return value_if_denom_is_zero
    else:
        return numerator * 1.0 / denominator


def zidz(numerator, denominator):
    """
    This function bypasses divide-by-zero errors,
    implementing Vensim's ZIDZ function

    Parameters
    ----------
    numerator: float or xarray.DataArray
        value to be divided
    denominator: float or xarray.DataArray
        value to devide by

    Returns
    -------
    result of division numerator/denominator if denominator is not zero,
    otherwise zero.

    """
    if isinstance(denominator, xr.DataArray):
        return xr.where(np.abs(denominator) < small_vensim,
                        0,
                        numerator * 1.0 / denominator)

    if abs(denominator) < small_vensim:
        return 0
    else:
        return numerator * 1.0 / denominator


def active_initial(time, expr, init_val):
    """
    Implements vensim's ACTIVE INITIAL function
    Parameters
    ----------
    time: function
        The current time function
    expr
    init_val

    Returns
    -------

    """
    if time.stage == 'Initialization':
        return init_val
    else:
        return expr()

def random_0_1():
    """
    Implements Vensim's RANDOM 0 1 function.

    Returns
    -------
    A random number from the uniform distribution between 0 and 1.
    """
    return np.random.uniform(0,1)

def random_uniform(m, x, s):
    """
    Implements Vensim's RANDOM UNIFORM function.

    Parameters
    ----------
    m: int
        Minimum value that the function will return.
    x: int
        Maximun value that the function will return.
    s: int
        A stream ID for the distribution to use. In most cases should be 0.
    
    Returns
    -------
    A random number from the uniform distribution between m and x (exclusive of the endpoints).
    """
    if(s!=0):
        warnings.warn("Random uniform with a nonzero seed value, may not give the same result as vensim", RuntimeWarning)

    return np.random.uniform(m, x)


def incomplete(*args):
    warnings.warn(
        'Call to undefined function, calling dependencies and returning NaN',
        RuntimeWarning, stacklevel=2)

    return np.nan


def not_implemented_function(*args):
    raise NotImplementedError(
        'Not implemented function {}'.format(args[0]))


def log(x, base):
    """
    Implements Vensim's LOG function with change of base

    Parameters
    ----------
    x: input value
    base: base of the logarithm

    Returns
    -------
    float
      the log of 'x' in base 'base'
    """
    return np.log(x) / np.log(base)


def sum(x, dim=None):
    """
    Implements Vensim's SUM function

    Parameters
    ----------
    x: xarray.DataArray
      Input value
    dim: list of strs (optional)
      Dimensions to apply the function over.
      If not given the function will be applied over all dimensions

    Returns
    -------
    xarray.DataArray or float
      The result of the sum operation in the given dimensions

    """
    # float returned if the function is applied over all the dimensions
    if dim is None or set(x.dims) == set(dim):
        return float(x.sum())

    return x.sum(dim=dim)


def prod(x, dim=None):
    """
    Implements Vensim's PROD function

    Parameters
    ----------
    x: xarray.DataArray
      Input value
    dim: list of strs (optional)
      Dimensions to apply the function over.
      If not given the function will be applied over all dimensions

    Returns
    -------
    xarray.DataArray or float
      The result of the product operation in the given dimensions

    """
    # float returned if the function is applied over all the dimensions
    if dim is None or set(x.dims) == set(dim):
        return float(x.prod())

    return x.prod(dim=dim)


def vmin(x, dim=None):
    """
    Implements Vensim's Vmin function

    Parameters
    ----------
    x: xarray.DataArray
      Input value
    dim: list of strs (optional)
      Dimensions to apply the function over.
      If not given the function will be applied over all dimensions

    Returns
    -------
    xarray.DataArray or float
      The result of the minimum value over the given dimensions

    """
    # float returned if the function is applied over all the dimensions
    if dim is None or set(x.dims) == set(dim):
        return float(x.min())

    return x.min(dim=dim)


def vmax(x, dim=None):
    """
    Implements Vensim's VMAX function

    Parameters
    ----------
    x: xarray.DataArray
      Input value
    dim: list of strs (optional)
      Dimensions to apply the function over.
      If not given the function will be applied over all dimensions

    Returns
    -------
    xarray.DataArray or float
      The result of the maximum value over the dimensions

    """
    # float returned if the function is applied over all the dimensions
    if dim is None or set(x.dims) == set(dim):
        return float(x.max())

    return x.max(dim=dim)

