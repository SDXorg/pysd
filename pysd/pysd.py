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
from math import fmod
import time

# Todo: Add a __doc__ function that summarizes the docstrings of the whole model
# Todo: Give the __doc__ function a 'short' and 'long' option
# Todo: add a logical way to run two or more models together, using the same integrator.
# Todo: add the state dictionary to the model file, to give some functionality to it even
# without the pysd class
# Todo: seems to be some issue with multiple imports - need to create a new instance...

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
    # so that we can get values by their real names, not their python names
    components._funcs.update({name: getattr(components, components.namespace[name])
                              for name in components.namespace})

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
        self.record = []

    def __str__(self):
        """ Return model source file """
        return self.components.__str__

    def run(self, params=None, return_columns=None, return_timestamps=None,
            initial_condition='original', collect=False, flatten_subscripts=False):
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

        collect: binary (T/F)
            When running multiple simulations, collect the results in a way
            that we can access down the road.

        flatten_subscripts : binary (T/F)
            If set to `True`, will format the output dataframe in two dimensions, each
             cell of the dataframe containing a number. The number of columns of the
             dataframe will be expanded.
            If set to `False`, the dataframe cells corresponding to subscripted elements
             will take the form of numpy arrays within the cells of the dataframe. The
             columns will correspond to the model elements.


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

        if params:
            self.set_components(params)

        self.set_initial_condition(initial_condition)

        tseries = self._build_timeseries(return_timestamps)

        # the odeint expects the first timestamp in the tseries to be the initial condition,
        # so we may need to add the t0 if it is not present in the tseries array
        # Todo: with the euler integrator, this may no longer be the case. Reevaluate.
        addtflag = tseries[0] != self.components._t
        if addtflag:
            tseries = np.insert(tseries, 0, self.components._t)

        if return_columns is None:
            return_columns = self.components._stocknames

        res = self._integrate(self.components._dfuncs, tseries, return_columns)
        return_df = _pd.DataFrame(data=res, index=tseries)

        if flatten_subscripts:
            # Todo: short circuit this if there is nothing to flatten
            return_df = self._flatten_dataframe(return_df)

        if addtflag: # Todo: add a test case in which this is necessary to test_functionality
            return_df.drop(return_df.index[0], inplace=True)

        if collect:
            self.record.append(return_df)  # we could just record the state, and expand it later...

        return return_df

    # We give the state and the time parameters leading underscores so that
    # if there are variables in the model named 't' or 'state' there are no
    # conflicts

    def reset_state(self):
        """Sets the model state to the state described in the model file. """
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

    def get_record(self):
        """ Return the recorded model information.
        Returns everything as a big long dataframe.

        >>> model.get_record()
        """
        return _pd.concat(self.record)

    def clear_record(self):
        """ Reset the recorder.

        >>> model.clear_record()
        """
        self.record = []

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

    def _build_timeseries(self, return_timestamps=None):
        """ Build up array of timestamps

        """

        # Todo: rework this for the euler integrator, to be the dt series plus the return timestamps
        # Todo: maybe cache the result of this function?
        if return_timestamps is None:
            # Vensim's standard is to expect that the data set includes the `final time`,
            # so we have to add an extra period to make sure we get that value in what
            # numpy's `arange` gives us.
            tseries = np.arange(self.components.initial_time(),
                                self.components.final_time()+self.components.time_step(),
                                self.components.time_step())
        elif isinstance(return_timestamps, (list, int, float, long, np.ndarray)):
            tseries = np.array(return_timestamps, ndmin=1)
        else:
            raise TypeError('`return_timestamps` expects a list, array, or numeric value')
        return tseries

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

    def _integrate(self, ddt, timesteps, return_elements):
        """
        Performs euler integration

        Parameters
        ----------
        ddt:
        timesteps
        return_elements

        Returns
        -------

        """
        outputs = range(len(timesteps))
        for i, t2 in enumerate(timesteps):
            self.components._state = self._euler_step(ddt, self.components._state, t2-self.components._t)
            self.components._t = t2
            outputs[i] = {key: self.components._funcs[key]() for key in return_elements}

        return outputs


    def _flatten_dataframe(self, dataframe):
        """
        Formats model output for easy comparison or storage in a 2d spreadsheet.

        Parameters
        ----------
        dataframe : pandas dataframe
            The output of a model simulation, with variable names as column names
             and timeseries as the indices. In this dataframe may be some columns
             representing variables with subscripts, whose values are held within
             numpy arrays within each cell of the dataframe.

        Returns
        -------
        flat_dataframe : pandas dataframe
            Dataframe containing all of the information of the output, but flattened such
             that each cell of the dataframe contains only a number, not a full array.
             Extra columns will be added to represent each of the elements of the arrays
             in question.
        """
        def sort(dictionary):
            return [dictionary.keys()[dictionary.values().index(x)] for x in sorted(dictionary.values())]

        def pandasnamearray(varname):
            stocklen = 1
            stockmod = []

            for i in sort(varname.dimension_dir):
                stocklen *= len(self.components._subscript_dict[i])
                stockmod.append(len(self.components._subscript_dict[i]))

            for i in range(len(stockmod)):
                stockmod[i]=np.prod(stockmod[i+1:])

            interstock = np.ndarray(stocklen, object)
            interstock[:] = ""

            for i, j in enumerate(sort(varname.dimension_dir)):
                for k in range(stocklen):
                    interstock[k] += "__"+sort(self.components._subscript_dict[j])[int(fmod(k/stockmod[i], len(self.components._subscript_dict[j])))]

            return [interstock[i].strip("__") for i in range(len(interstock))]  # take off the ends

        def dataframeexpand(pddf):
            result = []
            for pos, name in enumerate(pddf.columns):
                # todo: don't try and flatten if alreay a single number
                # except if its a single-element subscript, we may want to do this?
                if (isinstance(pddf[name].loc[0], np.ndarray) and
                        np.max(pddf[name].loc[0].size) > 1): #in the case that a single number makes its way into an array?
                    result.append(pddf[name].apply(lambda x: _pd.Series(x.flatten())))
                    result[pos].columns = ([name+'__'+pandasnamearray(getattr(self.components, name))[x] for x in range(len(pandasnamearray(getattr(self.components,name))))])
                else:
                    result.append(_pd.DataFrame(pddf[name]))
                    result[pos].columns = [name]
            pddf = _pd.concat([result[x] for x in range(len(result))], axis=1)

            return pddf
        return dataframeexpand(dataframe)
