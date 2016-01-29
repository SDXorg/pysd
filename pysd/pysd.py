"""
pysd.py

Contains all the code that will be directly accessed by the user in normal operation.
Also contains some private members to facilitate integration, setup, etc.

History
--------
August 15, 2014 : created
June 6 2015 : Major updates - version 0.2.5
Jan 2016 : Rework to handle subscripts

Contributors
------------
James Houghton <james.p.houghton@gmail.com>
Mounir Yakzan
"""

import pandas as _pd
import numpy as np
import imp
from math import fmod

# Todo: add a logical way to run two or more models together, using the same integrator.
# Todo: add model element caching
# Todo: add a way to flatten the result of a subscripted simulation, for comparison with vensim
# Todo: add the state dictionary to the model file, to give some functionality to it even without the pysd class
# Todo: initialize the model on load by calling reset state
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
    from translators import translate_xmile
    py_model_file = translate_xmile(xmile_file)
    model = load(py_model_file)
    model.__str__ = 'Import of ' + xmile_file
    return model


def read_vensim(mdl_file):
    """ Construct a model from Vensim `.mdl` file.

    Parameters
    ----------
    mdl_file : <string>
        The relative path filename for a raw Vensim `.mdl` file

    Examples
    --------
    >>> model = read_vensim('Teacup.mdl')
    """
    from translators import translate_vensim
    py_model_file = translate_vensim(mdl_file)
    model = load(py_model_file)
    model.__str__ = 'Import of ' + mdl_file
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
    >>> model = load('Teacup.py')
    """
    # Todo: This whole function is a mess. Consider including parts in the `init` of the PySD class
    components = imp.load_source('modulename', py_model_file)
    components._stocknames = [name[2:-3] for name in dir(components) if name.startswith('_') and name.endswith('_dt')]
    components._dfuncs = {name: getattr(components, '_d%s_dt'%name) for name in components._stocknames}
    funcnames = filter(lambda x: not x.startswith('_'), dir(components))
    components._funcs = {name: getattr(components, name) for name in funcnames}

    # set up the caches
    # Todo: make a robust way to tell that we're only caching the right things
    # Todo: make caching optional?
    nocache = (['_t', 'time_step', 'time', 'initial_time', 'final_time', 'division', 'functions',
                'np', 'saveper', '_stocknames', '_state', '_funcs', '_dfuncs', '_subscript_dict'] +
          ['_d%s_dt'%s for s in components._dfuncs.keys()] +  #these are only called once
          ['_%s_init'%s for s in components._dfuncs.keys()] + #these are only called once
          ['%s'%s for s in components._dfuncs.keys()]) #these are pass-throughs anyways
    cache_list = filter(lambda x: not x.startswith('__') and x not in nocache, dir(components))
    [setattr(components,name, cache(getattr(components, name), components)) for name in cache_list]

    model = PySD(components)
    model.reset_state()
    model.__str__ = 'Import of ' + py_model_file
    return model


def cache(func, components):
    """
    Put a wrapper around a model function
    Parameters
    ----------
    func : function in the components module

    components : the components module

    Returns
    -------
    new_func : function wrapping the original function, handling caching
    """
    # Todo: add some error checking, that `func` is actually a function, etc.
    # Todo: find a more appropriate place for this to live.
    def new_func(*args):
        try:
            assert new_func.t == components._t  # fails if cache is out of date or not instantiated
        except:
            new_func.cache = func(*args)
            new_func.t = components._t
        return new_func.cache
    new_func.func_dict = func.func_dict  # propagate attributes (like dim_dict) to the wrap function
    return new_func


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

    def run(self, params={}, return_columns=[], return_timestamps=[],
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

        if self.components._stocknames:
            if not return_columns:
                return_columns = self.components._stocknames

            res = self._integrate(self.components._dfuncs, tseries, return_columns)
            return_df = _pd.DataFrame(data=res, index=tseries)

        else:
            outdict = {}
            for key in return_columns:
                outdict[key] = self.components._funcs[key]()
            return_df = _pd.DataFrame(index=tseries, data=outdict)

        if flatten_subscripts:
            # Todo: short circuit this if there is nothing to flatten
            return_df = self._flatten_dataframe(return_df)

        if addtflag:  # Todo: add a test case in which this is necessary to test_functionality
            return_df.drop(return_df.index[0], inplace=True)

        if collect:
            self.record.append(return_df)  # we could just record the state, and expand it later...

        return return_df

    # We give the state and the time parameters leading underscores so that
    # if there are variables in the model named 't' or 'state' there are no
    # conflicts

    def reset_state(self):
        """Sets the model state to the state described in the model file. """
        self.components._t = self.components.initial_time()  # set the initial time
        self.components._state = dict()
        retry_flag = False
        for key in self.components._stocknames:
            # We have to do a loop here because there are cases where the initialization will
            # call a function, and that function may not have its own initial conditions defined
            # just yet. There is the potential that if the model has a reference loop,
            # this will become an infinite loop.
            # Todo: make this more robust to infinite looping
            try:
                init_func = getattr(self.components, '_%s_init'%key)
                self.components._state[key] = init_func()
            except TypeError:
                retry_flag = True
        if retry_flag:
            self.reset_state()

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

    def _build_timeseries(self, return_timestamps):
        """ Build up array of timestamps """

        # Todo: rework this for the euler integrator, to be the dt series plus the return timestamps
        # Todo: maybe cache the result of this function?
        if return_timestamps == []:
            tseries = np.arange(self.components.initial_time(),
                                self.components.final_time(),
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

    def _step(self, ddt, state, dt):
        outdict = {}
        for key in ddt:
            outdict[key] = ddt[key]()*dt + state[key]
        return outdict

    def _integrate(self, ddt, timesteps, return_elements):
        """

        Parameters
        ----------
        ddt : dictionary where keys are stock names and values are functions
        timesteps
        return_elements

        Returns
        -------

        """
        # Todo: write proper docstrings.
        outputs = range(len(timesteps))
        for i, t2 in enumerate(timesteps):
            self.components._state = self._step(ddt, self.components._state, t2-self.components._t)
            self.components._t = t2
            outdict = {}
            for key in return_elements:
                outdict[key] = self.components._funcs[key]()
            outputs[i] = outdict

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

            interstock = np.ndarray(stocklen,object)
            interstock[:] = ""

            for i, j in enumerate(sort(varname.dimension_dir)):
                for k in range(stocklen):
                    interstock[k] += "__"+sort(self.components._subscript_dict[j])[int(fmod(k/stockmod[i], len(self.components._subscript_dict[j])))]

            return [interstock[i].strip("__") for i in range(len(interstock))]  # take off the ends

        def dataframeexpand(pddf):
            result = []
            for pos, name in enumerate(pddf.columns):
                # todo: don't try and flatten if alreay a single number
                if isinstance(pddf[name].loc[0], np.ndarray):
                    result.append(pddf[name].apply(lambda x: _pd.Series(x.flatten())))
                    result[pos].columns=([name+'__'+pandasnamearray(getattr(self.components, name))[x] for x in range(len(pandasnamearray(getattr(self.components,name))))])
                else:
                    result.append(_pd.DataFrame(pddf[name]))
                    result[pos].columns=[name]
            pddf=_pd.concat([result[x] for x in range(len(result))],axis=1)

            return pddf
        return dataframeexpand(dataframe)
