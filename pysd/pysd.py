"""
pysd.py

Contains all the code that will be directly accessed by the user in normal operation.
Also contains some private members to facilitate integration, setup, etc.

History
--------
August 15, 2014: created
June 6 2015: Major updates - version 0.2.5
Jan 2016: Rework to handle subscripts
May 2016: Updates to handle grammar refactoring
"""

import pandas as _pd
import numpy as np
import imp
import time
import utils
import functions


# Todo: add a logical way to run two or more models together, using the same integrator.
# Todo: work out an RK4 adaptive integrator


def read_vensim(mdl_file):
    """
    Construct a model from Vensim `.mdl` file.

    Parameters
    ----------
    mdl_file : <string>
        The relative path filename for a raw Vensim `.mdl` file

    Returns
    -------
    model: a PySD class object
        Elements from the python model are loaded into the PySD class and ready to run

    Examples
    --------
    >>> model = read_vensim('../tests/test-models/samples/teacup/teacup.mdl')
    """
    from vensim2py import translate_vensim
    py_model_file = translate_vensim(mdl_file)
    model = PySD(py_model_file)
    model.mdl_file = mdl_file
    return model


def load(py_model_file):
    """
    Load a python-converted model file.

    Parameters
    ----------
    py_model_file : <string>
        Filename of a model which has already been converted into a
         python format.

    Examples
    --------
    >>> model = load('../tests/test-models/samples/teacup/teacup.py')
    """
    return PySD(py_model_file)


class PySD(object):
    """
    PySD is the default class charged with running a model.
    It can be initialized by passing a string to a translated python model file.
    The import functions pull models and create this class.
    """

    def __init__(self, py_model_file):
        """
        Construct a PySD object from a translated model python file

        Parameters
        ----------
        py_model_file : <string>
            Filename of a model which has already been converted into a
            python format.

        """
        # need a unique identifier for the imported module. Use the time.
        module_name = str(time.time()).replace('.', '')
        self.components = imp.load_source(module_name,
                                          py_model_file)

        self._stateful_elements = [getattr(self.components, name) for name in dir(self.components)
                                   if isinstance(getattr(self.components, name),
                                                 functions.Stateful)]

        self.py_model_file = py_model_file
        self.reset_state()

    def __str__(self):
        """ Return model source files """

        # JT: Might be helpful to return not only the source file, but
        # also how the instance differs from that source file. This
        # would give a more accurate view of the current model.
        str = 'Translated Model File: ' + self.py_model_file
        if hasattr(self, 'mdl_file'):
            str += '\n Original Model File: ' + self.mdl_file

        return str

    def run(self, params=None, return_columns=None, return_timestamps=None,
            initial_condition='original'):
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

        if params:
            self.set_components(params)

        self.set_initial_condition(initial_condition)

        return_timestamps = self._format_return_timestamps(return_timestamps)

        t_series = self._build_euler_timeseries(return_timestamps)

        if return_columns is None:
            return_columns = self.components._namespace.keys()

        self.components._stage = 'Run'
        self.clear_caches()

        capture_elements, return_addresses = utils.get_return_elements(
            return_columns, self.components._namespace, self.components._subscript_dict)

        res = self._integrate(t_series, capture_elements, return_timestamps)

        return_df = utils.make_flat_df(res, return_addresses)
        return_df.index = return_timestamps

        return return_df

    def reset_state(self):
        """
        Sets the model state to the state described in the model file.
        Builds the state vector from scratch

        """

        # We give the state and the time parameters leading underscores so that
        # if there are variables in the model named 't' or 'state' there are no
        # conflicts

        self.components._t = self.components.initial_time()
        self.components._stage = 'Initialization'

        def initialize_state():
            """
            This function tries to initialize the state vector.

            In the case where an initialization function for `Stock A` depends on
            the value of `Stock B`, if we try to initialize `Stock A` before `Stock B`
            then we will get an error, as the value will not yet exist.

            In this case, just skip initializing `Stock A` for now, and
            go on to the other state initializations. Then call whole function again.

            Each time the function is called, we should be able to make some progress
            towards full initialization, by initializing at least one more state.
            If we don't then references are unresolvable, so we should throw an error.
            """
            retry_flag = False
            making_progress = False
            initialization_order = []
            for element in self._stateful_elements:
                try:
                    element.initialize()
                    making_progress = True
                    initialization_order.append(repr(element))
                except KeyError:
                    retry_flag = True
                except TypeError:
                    retry_flag = True
                except AttributeError:
                    retry_flag = True
            if not making_progress:
                raise KeyError('Unresolvable Reference: Probable circular initialization' +
                               '\n'.join(initialization_order))
            if retry_flag:
                initialize_state()

        if self._stateful_elements:  # if there are no stocks, don't try to initialize!
            initialize_state()

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
            if isinstance(value, _pd.Series):
                new_function = self._timeseries_component(value)
            else:
                new_function = self._constant_component(value)

            if key in self.components._namespace.keys():
                func_name = self.components._namespace[key]
            elif key in self.components._namespace.values():
                func_name = key
            else:
                raise NameError('%s is not recognized as a model component' % key)

            if isinstance(getattr(self.components, func_name), functions.Stateful)
                warnings.warn("Replacing the equation of stock {} with params".format(key),
                              stacklevel=2)


            setattr(self.components, func_name, new_function)

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
        self.components._t = t

        for key, value in state.items():
            if key in self.components._namespace.keys():
                element_name = 'integ_%s' % self.components._namespace[key]
            elif key in self.components._namespace.values():
                element_name = 'integ_%s' % key
            else:  # allow the user to specify the stateful object directly
                element_name = key

            try:
                element = getattr(self.components, element_name)
                element.update(value)
            except AttributeError:
                print("'%s' has no state elements, assignment failed")
                raise

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
                self.reset_state()
            elif initial_condition.lower() in ['current', 'c']:
                pass
            else:
                raise ValueError('Valid initial condition strings include:  \n' +
                                 '    "original"/"o",                       \n' +
                                 '    "current"/"c"')
        else:
            raise TypeError('Check documentation for valid entries')

    def clear_caches(self):
        """ Clears the Caches for all model elements """
        for element_name in dir(self.components):
            element = getattr(self.components, element_name)
            if hasattr(element, 'cache_val'):
                delattr(element, 'cache_val')

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
        t_0 = self.components.time()
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
        Format the passed in return timestamps value if it exists,
        or build up array of timestamps based upon the model saveper
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
        elif isinstance(return_timestamps, (list, int, float, long, np.ndarray)):
            return_timestamps_array = np.array(return_timestamps, ndmin=1)
        elif isinstance(return_timestamps, _pd.Series):
            return_timestamps_array = return_timestamps.as_matrix()
        elif isinstance(return_timestamps, np.ndarray):
            return_timestamps_array = return_timestamps
        else:
            raise TypeError('`return_timestamps` expects a list, array, pandas Series, '
                            'or numeric value')
        return return_timestamps_array

    def _timeseries_component(self, series):
        """ Internal function for creating a timeseries model element """
        # this is only called if the set_component function recognizes a pandas series
        # Todo: raise a warning if extrapolating from the end of the series.
        return lambda: np.interp(self.components.time(), series.index, series.values)

    def _constant_component(self, value):
        """ Internal function for creating a constant model element """
        return lambda: value

    def _euler_step(self, dt):
        """ Performs a single step in the euler integration,
        updating stateful components

        Parameters
        ----------
        dt : float
            This is the amount to increase time by this step
        """
        new_states = [component.state + component.ddt() * dt
                      for component in self._stateful_elements]
        [component.update(new_state)
         for component, new_state in zip(self._stateful_elements, new_states)]

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
            if self.components._t in return_timestamps:
                outputs.append({key: getattr(self.components, key)() for key in capture_elements})
            self._euler_step(t2 - self.components._t)
            self.components._t = t2  # this will clear the stepwise caches

        # need to add one more time step, because we run only the state updates in the previous
        # loop and thus may be one short.
        if self.components._t in return_timestamps:
            outputs.append({key: getattr(self.components, key)() for key in capture_elements})

        return outputs

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

        from parsimonious.grammar import Grammar
        from parsimonious.nodes import NodeVisitor

        varnames = filter(lambda x: not x.startswith('_') and x not in ('cache', 'functions', 'np'),
                          dir(self.components))

        docstrings = [getattr(self.components, name).__doc__ for name in varnames]

        g = """\
            sdVar = (sep? name sep "-"* sep model_name_wrap sep unit sep+ comment? " "*)?
            sep = ws "\\n" ws
            ws = " "*
            name = ~"[A-z ]+"
            model_name_wrap = '(' model_name ')'
            model_name = ~"[A-z_]+"
            unit = ~"[A-z\\, \\/\\*\\[\\]\\?0-9]*"
            comment = ~"[A-z _+-/*\\n]+"
            """

        class SDVarDoc(NodeVisitor):
            def __init__(self, grammar, text):
                self.sdVar = {}
                ast = Grammar(grammar).parse(text)
                self.visit(ast)

            def visit_name(self, n, vc):
                self.sdVar['name'] = n.text

            def visit_model_name(self, n, vc):
                self.sdVar['model_name'] = n.text

            def visit_unit(self, n, vc):
                self.sdVar['unit'] = n.text

            def visit_comment(self, n, vc):
                self.sdVar['comment'] = n.text

            def generic_visit(self, n, vc):
                pass

        docstring_list = list()
        for ds in filter(None, docstrings):
            docstring_list.append(SDVarDoc(g, ds).sdVar)

        # Convert docstring_list, a list of dictionaries, to a Pandas DataFrame,
        # for easy printing down the line.
        df_headers = ['Real Name', 'Py Name', 'Unit', 'Comment']
        docs_df = _pd.DataFrame(docstring_list, columns=df_headers)

        return docs_df
