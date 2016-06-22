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

Contributors
------------
James Houghton <james.p.houghton@gmail.com>
Mounir Yakzan
"""

import pandas as _pd
import numpy as np
import imp
import time
import utils
import tabulate

from documentation import SDVarDoc


# Todo: Add a __doc__ function that summarizes the docstrings of the whole model
# Todo: Give the __doc__ function a 'short' and 'long' option
# Todo: add a logical way to run two or more models together, using the same integrator.
# Todo: add the state dictionary to the model file, to give some functionality to it even
# without the pysd class
# Todo: seems to be some issue with multiple imports - need to create a new instance...
# Todo: work out an RK4 adaptive integrator

def read_xmile(xmile_file):
    """ Construct a model object from `.xmile` file.

    Parameters
    ----------
    xmile_file : <string>
        The relative path filename for a raw xmile file

    Examples
    --------
    >>> model = read_vensim('Teacup.xmile')
    """
    #from translators import translate_xmile
    #py_model_file = translate_xmile(xmile_file)
    #model = load(py_model_file)
    #model.__str__ = 'Import of ' + xmile_file
    #return model


def read_vensim(mdl_file):
    """ Construct a model from Vensim `.mdl` file.

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
    model = load(py_model_file)
    return model


def load(py_model_file):
    """ Load a python-converted model file.

    Parameters
    ----------
    py_model_file : <string>
        Filename of a model which has already been converted into a
         python format.

    Examples
    --------
    >>> model = load('../tests/test-models/samples/teacup/teacup.py')
    """

    # need a unique identifier for the imported module. Use the time.
    module_name = str(time.time()).replace('.', '')
    components = imp.load_source(module_name, py_model_file)

    components._stocknames = [name[2:-3] for name in dir(components)  # strip to just the name
                              if name.startswith('_d') and name.endswith('_dt')]

    # pointers to the various derivative functions for each of the stocks
    components._dfuncs = {name: getattr(components, '_d%s_dt' % name)
                          for name in components._stocknames}

    funcnames = filter(lambda x: not x.startswith('_'), dir(components))
    components._funcs = {name: getattr(components, name) for name in funcnames}

    varnames = filter(lambda x: not x.startswith('_') and not x in ('cache','functions','np'), dir(components))
    components._docstrings = [getattr(components,name).__doc__ for name in varnames]

    components.__str__ = 'This is the list of model components' + str(dir(components)) ## To have PySD.__str__ produce a string

    model = PySD(components)
    model.reset_state()
    return model


class PySD(object):
    """
        PySD is the default class charged with running a model.

        It can be initialized by passing an existing component class.

        The import functions pull models and create this class.
    """

    def __init__(self, components):
        """ Construct a PySD object built around the component class """
        self.components = components

    def __str__(self):
        """ Return model source file """
        return self.components.__str__

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

        >>> model.run(params={'exogenous_constant':42})
        >>> model.run(params={'exogenous_variable':timeseries_input})
        >>> model.run(return_timestamps=[1,2,3.1415,4,10])
        >>> model.run(return_timestamps=10)
        >>> model.run(return_timestamps=np.linspace(1,10,20))

        See Also
        --------
        pysd.set_components : handles setting model parameters
        pysd.set_initial_condition : handles setting initial conditions

        """
        # Todo: think of a better way to handle the separation between return timestamps and
        # integration tseries

        if params:
            self.set_components(params)

        self.set_initial_condition(initial_condition)

        t_series = self._build_euler_timeseries(return_timestamps)

        return_timestamps = self._format_return_timestamps(return_timestamps)

        if return_columns is None:
            return_columns = [utils.dict_find(self.components._namespace, x)
                              for x in self.components._stocknames]

        capture_elements, return_addresses = utils.get_return_elements(
            return_columns, self.components._namespace, self.components._subscript_dict)

        res = self._integrate(self.components._dfuncs, t_series,
                              capture_elements, return_timestamps)

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

        # todo: check that this isn't being called twice, unnecessarily?

        self.components._t = self.components.initial_time()  # set the initial time
        self.components._state = dict()

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
            for key in self.components._stocknames:
                try:
                    init_func = getattr(self.components, '_init_%s'%key)
                    self.components._state[key] = init_func()
                    making_progress = True
                    initialization_order.append(key)
                except KeyError:  # may also need to catch TypeError?
                    retry_flag = True
            if not making_progress:
                raise KeyError('Unresolvable Reference: Probable circular initialization'+
                               '\n'.join(initialization_order))
            if retry_flag:
                initialize_state()

        if self.components._stocknames:  # if there are no stocks, don't try to initialize!
            initialize_state()

    def set_components(self, params):
        """ Set the value of exogenous model elements.
        Element values can be passed as keyword=value pairs in the function call.
        Values can be numeric type or pandas Series.
        Series will be interpolated by integrator.

        Examples
        --------
        >>> br = pandas.Series(index=range(30), values=np.sin(range(30))
        >>> model.set_components(birth_rate=br)
        >>> model.set_components(birth_rate=10)

        """
        for key, value in params.iteritems():
            if isinstance(value, _pd.Series):
                new_function = self._timeseries_component(value)
            else:  # Todo: check here for valid value...
                new_function = self._constant_component(value)
            setattr(self.components, key, new_function)
            self.components._funcs[key] = new_function  # facilitates lookups

    def set_state(self, t, state):
        """ Set the system state.

        Parameters
        ----------
        t : numeric
            The system time

        state : dict
            Idelly a complete dictionary of system state, but a partial
            state dictionary will work if you're confident that the remaining
            state elements are correct.
        """
        self.components._t = t
        self.components._state.update(state)

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
        >>> model.set_initial_condition( (10,{'teacup_temperature':50}) )

        See Also
        --------
        pysd.set_state()
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
                raise ValueError('Valid initial condition strings include:  \n'+
                                 '    "original"/"o",                       \n'+
                                 '    "current"/"c"')
        else:
            raise TypeError('Check documentation for valid entries')

    def _build_euler_timeseries(self, return_timestamps=None):
        # Todo: Add the returned timeseries into the integration array. Best we can do for now.

        return np.arange(self.components.initial_time(),
                         self.components.final_time() + self.components.time_step(),
                         self.components.time_step(), dtype=np.float64)

    def _format_return_timestamps(self, return_timestamps=None):
        """
        Format the passed in return timestamps value if it exists,
        or build up array of timestamps based upon the model saveper
        """
        if return_timestamps is None:
            # Vensim's standard is to expect that the data set includes the `final time`,
            # so we have to add an extra period to make sure we get that value in what
            # numpy's `arange` gives us.
            return_timestamps_array = np.arange(self.components.initial_time(),
                                self.components.final_time() + self.components.saveper(),
                                self.components.saveper(), dtype=np.float64)
        elif isinstance(return_timestamps, (list, int, float, long, np.ndarray)):
            return_timestamps_array = np.array(return_timestamps, ndmin=1)
        else:
            raise TypeError('`return_timestamps` expects a list, array, or numeric value')
        return return_timestamps_array

    def _timeseries_component(self, series):
        """ Internal function for creating a timeseries model element """
        return lambda: np.interp(self.components._t, series.index, series.values)

    def _constant_component(self, value):
        """ Internal function for creating a constant model element """
        return lambda: value

    def _euler_step(self, ddt, state, dt):
        """ Performs a single step in the euler integration

        Parameters
        ----------
        ddt : dictionary
            list of the names of the derivative functions
        state : dictionary
            This is the state dictionary, where stock names are keys and values are
            the number or array values at the current timestep
        dt : float
            This is the amount to increase

        Returns
        -------
        new_state : dictionary
             a dictionary with keys corresponding to the input 'state' dictionary,
             after those values have been updated with one euler step
        """
        # Todo: instead of a list of dfuncs, just use locals() http://stackoverflow.com/a/834451/6361632

        return {key: dfunc()*dt + state[key] for key, dfunc in ddt.iteritems()}


    def _integrate(self, derivative_functions, timesteps, capture_elements, return_timestamps):
        """
        Performs euler integration

        Parameters
        ----------
        derivative_functions
        timesteps
        capture_elements

        Returns
        -------

        """
        # Todo: consider adding the timestamp to the return elements, and using that as the index
        outputs = []

        for i, t2 in enumerate(timesteps[1:]):
            if self.components._t in return_timestamps:
                outputs.append({key: self.components._funcs[key]() for key in capture_elements})
            self.components._state = self._euler_step(derivative_functions,
                                                      self.components._state,
                                                      t2 - self.components._t)
            self.components._t = t2  # this will clear the stepwise caches

        # need to add one more timestep, because we run only the state updates in the previous
        # loop and thus may be one short.
        if self.components._t in return_timestamps:
            outputs.append({key: self.components._funcs[key]() for key in capture_elements})

        return outputs

    def doc(self):
        docstringList=list()

        grammar = """\
            sdVar = (sep? name sep "-"* sep modelNameWrap sep unit sep+ comment? " "*)?
            sep = ws "\\n" ws
            ws = " "*
            name = ~"[A-z ]+"
            modelNameWrap = '(' modelName ')'
            modelName = ~"[A-z_]+"
            unit = ~"[A-z\\, \\/\\*\\[\\]\\?0-9]*"
            comment = ~"[A-z _+-/*\\n]+"
            """

        for ds in filter(None,self.components._docstrings):
            docstringList.append(SDVarDoc(grammar,ds).sdVar)

        dsdf = _pd.DataFrame(docstringList) ## Convert docstringlist, a list of dicitonaries, to a Pandas dataframe, for easy printing down the line.

        dsheaders=['name','modelName','unit','comment']

        dstable = tabulate.tabulate(dsdf[dsheaders],headers=dsheaders,tablefmt='orgtbl')

        return dstable