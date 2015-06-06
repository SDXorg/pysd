'''
created: August 15, 2014
last update: June 6 2015
version 0.2.5
James Houghton <james.p.houghton@gmail.com>
'''

#pysd specific imports
import translators as _translators
import functions

#third party imports
from scipy.integrate import odeint as _odeint
import pandas as _pd
import numpy as np


######################################################
# Todo:
# - passing optional arguments in the run command through to the integrator,
#       to give a finer level of control to those who know what to do with them. (such as `tcrit`)
# - add a logical way to run two or more models together, using the same integrator.
# - it might help with debugging if we did cross compile to an actual class or module, in an actual text file somewhere.
######################################################


######################################################
# Issues:
#
# If two model components (A, and B) depend on the same third model component (C)
# then C will get computed twice. If C itself is dependant on many upstream components (D, E, F, etc)
# then these will also be computed multiple times.
#
# As the model class depends on its internal state for calculating
# the values of each element, instead of passing that state
# through the function execution network, we can't use caching
# to prevent multiple execution, as we don't know when to update the cache
# (maybe in the calling function?)
#
# Also, this multi-calculation bears the risk that if the state is
# changed during an execution step, the resulting calculation will be
# corrupted, and we won't have any indication of this corruption.
######################################################

def read_XMILE(XMILE_file):
    """ Construct a model object from XMILE source """
    pass #to be updated when we actually have xmile files to test against...

def read_vensim(mdl_file):
    """ Construct a model from Vensim .mdl file """
    component_class = _translators.import_vensim(mdl_file)
    return pysd(component_class)


class pysd:
    def __init__(self, component_class):
        self.components = component_class()
        self.record = []
    
    def __str__(self):
        """ Return model source file """
        return self.components.__str__
    
    
    def run(self, params={}, return_columns=[], return_timestamps=[],
                  initial_condition='original', collect=False):
        
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
            'original' (default) uses model-file specified initial condition
            'current' uses the state of the model after the previous execution
            (t, {state}) lets the user specify a starting time and (possibly partial)
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
        
        if params:
            self.set_components(params)
    
        if initial_condition != 'current':
            self.set_initial_condition(initial_condition)

        tseries = self._build_timeseries(return_timestamps)

        # the odeint expects the first timestamp in the tseries to be the initial condition,
        # so we may need to add the t0 if it is not present in the tseries array
        addtflag = tseries[0] != self.components.t
        if addtflag:
            tseries = np.insert(tseries, 0, self.components.t)

        res = _odeint(func=self.components.d_dt,
                      y0=self.components.state_vector(),
                      t=tseries,
                      hmax=self.components.time_step())
        
        state_df = _pd.DataFrame(data=res,
                                 index=tseries,
                                 columns=self.components._stocknames)

        return_df = self.extend_dataframe(state_df, return_columns) if return_columns else state_df

        if addtflag:
            return_df.drop(return_df.index[0], inplace=True)

        if collect:
            self.record.append(return_df) #we could alternately just record the state, and expand it later...
        
        return return_df


    def get_record(self):
        """ Return the recorded model information. """
        return _pd.concat(self.record)
    
    def clear_record(self):
        """ Reset the recorder. """
        self.record = []


    def set_components(self, params):
        """ Set the value of exogenous model elements.
        Element values can be passed as keyword=value pairs in the function call.
        Values can be numeric type or pandas Series.
            Series will be interpolated by integrator.
            
        Examples
        --------
        >>> set_components(birth_rate=10)
            
        >>> br = pandas.Series(index=range(30), values=np.sin(range(30))
        >>> set_components(birth_rate=br)
        """
        updates_dict = {}
        for key, value in params.iteritems():
            if isinstance(value, _pd.Series):
                updates_dict[key] = self._timeseries_component(value)
            else: #could check here for valid value...
                updates_dict[key] = self._constant_component(value)
    
        self.components.__dict__.update(updates_dict)


    def extend_dataframe(self, state_df, return_columns):
        """ Calculates model values at given system states """
        #there may be a better way to use the integrator that lets us report
        #more values than just the stocks. In the meantime, we have to go
        #through the returned values again, set up the model, and measure them.
        
        def get_values(row):
            t = row.name
            state = dict(row)
            self.set_state(t,state)
        
            return_vals = {}
            for column in return_columns: #there must be a faster way to do this...
                func = getattr(self.components, column)
                return_vals[column] = func()
    
            return _pd.Series(return_vals)
        
        return state_df.apply(get_values, axis=1)
        

    def set_state(self, t, state):
        """ Set the system state 
        t : numeric, system time
        state: complete dictionary of system state
        """
        self.components.t = t
        self.components.state.update(state)


    def set_initial_condition(self, initial_condition):
        """ Set the initial conditions of the integration """
    
        if isinstance(initial_condition, tuple):
            self.set_state(*initial_condition) #we should probably check the values more than just seeing if they are a tuple.
        elif isinstance(initial_condition, str):
            if initial_condition.lower() in ['original','o']:
                self.components.reset_state()
            elif initial_condition.lower() in ['current', 'c']:
                pass
            else:
                raise ValueError('Valid initial condition strings include:\n"original"/"o",\n"current"/"c"')
        else:
            raise TypeError('Check documentation for valid entries')


    def _build_timeseries(self, return_timestamps):
        """ Build up array of timestamps """
        if return_timestamps == []:
            tseries = np.arange(self.components.initial_time(),
                                self.components.final_time(),
                                self.components.time_step())
        elif isinstance(return_timestamps, (list, int, float, long, np.ndarray)):
            tseries = np.array(return_timestamps, ndmin=1)
        else:
            raise TypeError('The `return_timestamps` parameter expects a list, array, or numeric value')
        return tseries


    #these could be better off in a model creation class
    def _timeseries_component(self, series):
        return lambda: np.interp(self.components.t, series.index, series.values)
    
    def _constant_component(self, value):
        return lambda: value



def help():
    print_supported_vensim_functions()

def print_supported_vensim_functions():
    print 'Vensim'.ljust(25) + 'Python'
    print ''.ljust(50,'-')
    for key, value in _translators.vensim2py.dictionary.iteritems():
        print key.ljust(25) + value

