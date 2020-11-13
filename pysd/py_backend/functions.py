"""
functions.py

These are supports for functions that are included in modeling software but have no
straightforward equivalent in python.

"""

from __future__ import division, absolute_import

import imp
import inspect
import os
import sys
import random
import warnings
from functools import wraps

import numpy as np
import pandas as _pd
import pandas as pd
import xarray as xr
from funcsigs import signature

from . import utils
from .external import External

import traceback

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


def cache(horizon):
    """
    Put a wrapper around a model function

    Decorators with parameters are tricky, you have to
    essentially create a decorator that returns a decorator,
    which itself then returns the function wrapper.

    Parameters
    ----------
    horizon: string
        - 'step' means cache just until the next timestep
        - 'run' means cache until the next initialization of the model

    Returns
    -------
    new_func: decorated function
        function wrapping the original function, handling caching

    """

    def cache_step(func):
        """ Decorator for caching at a step level"""

        @wraps(func)
        def cached(*args):
            """Step wise cache function"""
            try:  # fails if cache is out of date or not instantiated
                data = func.__globals__['__data']
                assert cached.cache_t == data['time']()
                assert hasattr(cached, 'cache_val')
                assert cached.cache_val is not None
            except (AssertionError, AttributeError):
                cached.cache_val = func(*args)
                data = func.__globals__['__data']
                cached.cache_t = data['time']()
            return cached.cache_val

        return cached

    def cache_run(func):
        """ Decorator for caching at  the run level"""

        @wraps(func)
        def cached(*args):
            """Run wise cache function"""
            try:  # fails if cache is not instantiated
                return cached.cache_val
            except AttributeError:
                cached.cache_val = func(*args)
                return cached.cache_val

        return cached

    if horizon == 'step':
        return cache_step

    elif horizon == 'run':
        return cache_run

    else:
        raise (AttributeError('Bad horizon for cache decorator'))


class Stateful(object):
    # the integrator needs to be able to 'get' the current state of the object,
    # and get the derivative. It calculates the new state, and updates it. The state
    # can be any object which is subject to basic (element-wise) algebraic operations
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
        super(Integ, self).__init__()
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
    # However, people may get confused this way in thinking that they need only one
    # delay object and can call it with various arguments to delay whatever is convenient.
    # This method forces them to acknowledge that additional structure is being created
    # in the delay object.

    def __init__(self, delay_input, delay_time, initial_value, order, coords=None, dims=None):
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
        super(Delay, self).__init__()
        self.init_func = initial_value
        self.delay_time_func = delay_time
        self.input_func = delay_input
        self.order_func = order
        self.order = None
        self.broadcast = 0
        self.coords = coords
        self.dims = dims

    def initialize(self):
        order = self.order_func()
        if isinstance(order, xr.DataArray):
            order = order.values
        while isinstance(order, np.ndarray): 
            order = order[0]
        if order != int(order):
            warnings.warn('Casting delay order from %f to %i' % (order, int(order)))
        self.order = int(order)  # The order can only be set once

        init_state_value = self.init_func() * self.delay_time_func() / self.order

        if self.coords:
            # subscripted delay
            # brodcast init_state_value with the dimensions
            coords = self.coords.copy()
            coords['delay'] = np.arange(self.order)

            # broadcast self.state
            broadcast  = xr.DataArray(data=0,\
                dims=['delay'] + self.dims, coords=coords)
            self.state = broadcast + init_state_value

            # for broadcasting in the future
            self.broadcast  = xr.DataArray(data=0,\
                dims=self.dims, coords=self.coords)

        else:
            self.state = np.array([init_state_value] * self.order)

    def __call__(self):
        if self.coords:
            return self.state[-1].reset_coords('delay', drop=True)\
                   / (self.delay_time_func() / self.order)
        else:
            return self.state[-1] / (self.delay_time_func() / self.order)

    def ddt(self):
        outflows = self.state / (self.delay_time_func() / self.order)
        inflows = np.roll(outflows, 1)
        if self.coords:
            inflows[0] = (self.broadcast + self.input_func()).values
        else:
            inflows[0] = self.input_func()
        return inflows - outflows

class Smooth(Stateful):
    def __init__(self, smooth_input, smooth_time, initial_value, order):
        super(Smooth, self).__init__()
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
        super(Trend, self).__init__()
        self.init_func = initial_trend
        self.average_time_function = average_time
        self.input_func = trend_input

    def initialize(self):
        self.state = self.input_func() / (1 + self.init_func() * self.average_time_function())

    def __call__(self):
        return zidz(self.input_func() - self.state, self.average_time_function() * abs(self.state))

    def ddt(self):
        return (self.input_func() - self.state) / self.average_time_function()


class Initial(Stateful):
    def __init__(self, func):
        super(Initial, self).__init__()
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
    and contains the majority of methods for accessing and modifying model components.

    When the instance in question also serves as the root model object (as opposed to a
    macro or submodel within another model) it will have added methods to facilitate
    execution.
    """

    def __init__(self, py_model_file, params=None, return_func=None, time=None, time_initialization=None):
        """
        The model object will be created with components drawn from a translated python
        model file.

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
        super(Macro, self).__init__()
        self.time = time
        self.time_initialization = time_initialization

        # need a unique identifier for the imported module.
        module_name = os.path.splitext(py_model_file)[0] + str(random.randint(0, 1000000))
        self.components = imp.load_source(module_name,
                                          py_model_file)

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
        """ Returns the version of pysd complier that used for generating this model """
        return self.components.__pysd_version__

    def initialize(self, initialization_order=None):
        """
        This function tries to initialize the stateful objects.

        In the case where an initialization function for `Stock A` depends on
        the value of `Stock B`, if we try to initialize `Stock A` before `Stock B`
        then we will get an error, as the value will not yet exist.

        In this case, just skip initializing `Stock A` for now, and
        go on to the other state initializations. Then come back to it and try again.
        """

        # Initialize time
        if self.time is None:
            if self.time_initialization is None:
                self.time = Time()
            else:
                self.time = self.time_initialization()

        self.components._init_outer_references({
            'scope': self,
            'time': self.time
        })

        # Initialize external elements
        for element in self._external_elements:
            element.initialize()
        
        self.components.external.Excels.clean()

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
            if isinstance(value, pd.Series):
                new_function = self._timeseries_component(value)
            elif callable(value):
                new_function = value
            else:
                new_function = self._constant_component(value)

            func_name = utils.get_value_by_insensitive_key_or_value(key, self.components._namespace)

            if func_name is None:
                raise NameError('%s is not recognized as a model component' % key)

            if '_integ_' + func_name in dir(self.components):  # this won't handle other statefuls...
                warnings.warn("Replacing the equation of stock {} with params".format(key),
                              stacklevel=2)

            setattr(self.components, func_name, new_function)

    def _timeseries_component(self, series):
        """ Internal function for creating a timeseries model element """
        # this is only called if the set_component function recognizes a pandas series
        # Todo: raise a warning if extrapolating from the end of the series.
        return lambda: np.interp(self.time(), series.index, series.values)

    def _constant_component(self, value):
        """ Internal function for creating a constant model element """
        return lambda: value

    def set_state(self, t, state):
        """ Set the system state.

        Parameters
        ----------
        t : numeric
            The system time

        state : dict
            A (possibly partial) dictionary of the system state.
            The keys to this dictionary may be either pysafe names or original model file names
        """
        self.time.update(t)

        for key, value in state.items():
            # TODO Implement map with reference between component and stateful element?
            component_name = utils.get_value_by_insensitive_key_or_value(key, self.components._namespace)
            if component_name is not None:
                stateful_name = '_integ_%s' % component_name
            else:
                component_name = key
                stateful_name = key

            # Try to update stateful component
            if hasattr(self.components, stateful_name):
                try:
                    element = getattr(self.components, stateful_name)
                    element.update(value)
                except AttributeError:
                    print("'%s' has no state elements, assignment failed")
                    raise
            else:
                # Try to override component
                try:
                    setattr(self.components, component_name, self._constant_component(value))
                except AttributeError:
                    print("'%s' has no component, assignment failed")
                    raise

    def clear_caches(self):
        """ Clears the Caches for all model elements """
        for element_name in dir(self.components):
            element = getattr(self.components, element_name)
            if hasattr(element, 'cache_val'):
                delattr(element, 'cache_val')

    def doc(self):
        """
        Formats a table of documentation strings to help users remember variable names, and
        understand how they are translated into python safe names.

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
                docstring = getattr(self.components, varname).__doc__
                lines = docstring.split('\n')
                collector.append({'Real Name': name,
                                  'Py Name': varname,
                                  'Eqn': lines[2].replace("Original Eqn:", "").strip(),
                                  'Unit': lines[3].replace("Units:", "").strip(),
                                  'Lims': lines[4].replace("Limits:", "").strip(),
                                  'Type': lines[5].replace("Type:", "").strip(),
                                  'Comment': '\n'.join(lines[7:]).strip()})
            except:
                pass

        docs_df = _pd.DataFrame(collector)
        docs_df.fillna('None', inplace=True)

        order = ['Real Name', 'Py Name', 'Unit', 'Lims', 'Type', 'Eqn', 'Comment']
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
    def __init__(self, py_model_file):
        """ Sets up the python objects """
        super(Model, self).__init__(py_model_file, None, None, Time())
        self.time.stage = 'Load'
        self.initialize()

    def initialize(self):
        """ Initializes the simulation model """
        self.time.update(self.components.initial_time())
        self.time.stage = 'Initialization'
        super(Model, self).initialize()

    def reset_state(self):
        warnings.warn('reset_state is deprecated. use `initialize` instead',
                      DeprecationWarning, stacklevel=2)
        self.initialize()

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
        elif isinstance(return_timestamps, _pd.Series):
            return_timestamps_array = return_timestamps.as_matrix()
        else:
            raise TypeError('`return_timestamps` expects a list, array, pandas Series, '
                            'or numeric value')
        return return_timestamps_array

    def run(self, params=None, return_columns=None, return_timestamps=None,
            initial_condition='original', reload=False):
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

        if params:
            self.set_components(params)
        self.clear_caches()
        self.set_initial_condition(initial_condition)

        return_timestamps = self._format_return_timestamps(return_timestamps)

        t_series = self._build_euler_timeseries(return_timestamps)

        if return_columns is None:
            return_columns = self._default_return_columns()

        self.time.stage = 'Run'
        self.clear_caches()

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
        self.__init__(self.py_model_file)

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
                    if not expr in parsed_expr:
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

        for t2 in time_steps[1:]:
            if self.time() in return_timestamps:
                outputs.append({key: getattr(self.components, key)() for key in capture_elements})
            self._euler_step(t2 - self.time())
            self.time.update(t2)  # this will clear the stepwise caches

        # need to add one more time step, because we run only the state updates in the previous
        # loop and thus may be one short.
        if self.time() in return_timestamps:
            outputs.append({key: getattr(self.components, key)() for key in capture_elements})

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
    small = 1e-6  # What is considered zero according to Vensim Help
    if repeat_time <= small:
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
    Intermediate values are calculated with linear interpolation between the intermediate points.
    Out-of-range values are the same as the closest endpoint (i.e, no extrapolation is performed).
    """
    return np.interp(x, xs, ys)


def lookup_extrapolation(x, xs, ys):
    """
    Intermediate values are calculated with linear interpolation between the intermediate points.
    Out-of-range values are calculated with linear extrapolation from the last two values at either end.
    """
    length = len(xs)
    if x < xs[0]:
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        k = dy / dx
        return ys[0] + (x - xs[0]) * k
    if x > xs[-1]:
        dx = xs[length - 1] - xs[length - 2]
        dy = ys[length - 1] - ys[length - 2]
        k = dy / dx
        return ys[length - 1] + (x - xs[length - 1]) * k
    return np.interp(x, xs, ys)


def lookup_discrete(x, xs, ys):
    """
    Intermediate values take on the value associated with the next lower x-coordinate (also called a step-wise function). The last two points of a discrete graphical function must have the same y value.
    Out-of-range values are the same as the closest endpoint (i.e, no extrapolation is performed).
    """
    for index in range(0, len(xs)):
        if x < xs[index]:
            return ys[index - 1] if index > 0 else ys[index]
    return ys[len(ys) - 1]


def if_then_else(condition, val_if_true, val_if_false):
    
    if isinstance(condition, xr.DataArray):
        if condition.values.size == 1:
            return val_if_true if condition.values[0] else val_if_false
        else:
            return xr.where(condition, val_if_true, val_if_false)
    else:
        return val_if_true if condition else val_if_false

def xidz(numerator, denominator, value_if_denom_is_zero):
    """
    Implements Vensim's XIDZ function.
    This function executes a division, robust to denominator being zero.
    In the case of zero denominator, the final argument is returned.

    Parameters
    ----------
    numerator: float
    denominator: float
        Components of the division operation
    value_if_denom_is_zero: float
        The value to return if the denominator is zero

    Returns
    -------
    numerator / denominator if denominator > 1e-6
    otherwise, returns value_if_denom_is_zero
    """
    small = 1e-6  # What is considered zero according to Vensim Help
    if abs(denominator) < small:
        return value_if_denom_is_zero
    else:
        return numerator * 1.0 / denominator


def zidz(numerator, denominator):
    """
    This function bypasses divide-by-zero errors,
    implementing Vensim's ZIDZ function

    Parameters
    ----------
    numerator: float
        value to be divided
    denominator: float
        value to devide by

    Returns
    -------
    result of division numerator/denominator if denominator is not zero,
    otherwise zero.
    """
    # Todo: make this work for arrays
    small = 1e-6  # What is considered zero according to Vensim Help
    if abs(denominator) < small:
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


def random_uniform(m, x, s):
    return np.random.uniform(m, x)


def incomplete(*args):
    warnings.warn('Call to undefined function, calling dependencies and returning NaN',
                  RuntimeWarning, stacklevel=2)

    return np.nan


def log(x, base):
    """
    Implements vensim's LOG function with change of base
    Parameters
    ----------
    x: input value
    base: base of the logarithm
    """
    return np.log(x) / np.log(base)

#TODO document this new functions...
def and_(x, y):
    #TODO check logical operations between matrix and vectors
    return xr.ufuncs.logical_and(x, y)


def or_(x, y):
    return xr.ufuncs.logical_or(x, y)


def sum(x, dim=None):

    # Return float if x is DataArray and dim is None
    if dim is None:
        return float(x.sum())

    return x.sum(dim=dim)


def prod(x, dim=None):

    # Return float if x is DataArray and dim is None
    if dim is None:
        return float(x.prod())

    return x.prod(dim=dim)


def vmin(x, dim=None):

    # Return float if x is DataArray and dim is None
    if dim is None:
        return float(x.min())

    return x.min(dim=dim)


def vmax(x, dim=None):

    # Return float if x is DataArray and dim is None
    if dim is None:
        return float(x.max())

    return x.max(dim=dim)

