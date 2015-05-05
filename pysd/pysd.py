'''
created: August 15, 2014
last update: March 28 2015
version 0.2.1
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
#
# - add a function to simplify parameter modification
# - we may think about passing optional arguments to the run command through to the integrator,
# to give a finer level of control to those who know what to do with it.
# - It would be neat to have a logical way to run two or more models together, using the same integrator.
# - it might help with debugging if we did cross compile to an actual class or module, in an actual text file somewhere.
# - Add support for cython
#
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


# need to re-implement xmile translator
def read_XMILE(XMILE_file):
    """
        This is currently broken =(
        """
    pass

def read_vensim(mdl_file):
    """
        Only a subset of vensim functions are supported.
        """
    component_class = _translators.import_vensim(mdl_file)
    return pysd(component_class)

def help():
    print_supported_vensim_functions()

def print_supported_vensim_functions():
    print 'Vensim'.ljust(25) + 'Python'
    print ''.ljust(50,'-')
    for key, value in _translators.vensim2py.dictionary.iteritems():
        print key.ljust(25) + value


class pysd:
    def __init__(self, component_class):
        self.components = component_class() #this is where we create an instance of the model subclass
        self.record = []
    
    def __str__(self):
        """
            Build this up to return a string with model equations in it
            """
        return self.components.__str__
    
    
    def run(self, params={}, return_columns=[], return_timestamps=[], initial_condition='original',
            collect=False):
        """
            Runs the model from the initial time all the way through the
            final time, returning a pandas dataframe of results.
            
            If ::return_timestamps:: is set, the dataframe will have these as
            its index. Otherwise, the index will be every timestep from the start
            of simulation to the finish.
            
            if ::return_columns:: is set, the dataframe will have these columns.
            Otherwise, it will return the value of the stocks.
            
            If ::params:: is set, modifies the model according to parameters
            before execution.
            
            format for params should be:
            params={'parameter1':value, 'parameter2':value}
            
            initial_condition can take a variety of values:
            - None or 'original' reinitializes the model at the initial conditions 
                 specified in the model file, and resets the simulation time accordingly
            - 'current' preserves the current state of the model, including the simulaion time
            - a tuple (t, dict) sets the simulation time to t, and the
            """
        
        ### Todo:
        #
        # - check that the return_timestamps is a collection - a list of timestamps, or something.
        #      if not(ie, a single value), make it one.
        
        if params:
            self.set_components(params)
        
        
        ##### Setting timestamp options
        if return_timestamps:
            tseries = return_timestamps
        else:
            tseries = np.arange(self.components.initial_time(),
                                self.components.final_time(),
                                self.components.time_step())
        
        ##### Setting initial condition options
        if isinstance(initial_condition, tuple):
            self.components.t = initial_condition[0] #this is brittle!
            self.components.state.update(initial_condition[1]) #this is also brittle. Assumes that the dictionary passed in has valid values
        elif isinstance(initial_condition, str):
            if initial_condition.lower() in ['original','o']:
                self.components.reset_state() #resets the state of the system to what the model contains (but not the parameters)
            elif initial_condition.lower() in ['current', 'c']:
                pass #placeholder - this is a valid option, but we don't modify anything
            else:
                assert False #we should throw an error if there is an unrecognized option
        else:
            assert False

        initial_values = self.components.state_vector()

        addtflag = False
        if tseries[0] != self.components.t:
            tseries = [self.components.t]+tseries
            addtflag = True

        ######Setting integrator options:
        #
        # - we may choose to use the odeint parameter ::tcrit::, which identifies singularities
        # near which the integrator should exercise additional caution.
        # we could get these values from any time-type parameters passed to a step/pulse type function
        #
        # - there may be a better way to use the integrator that lets us report additional values at
        # different timestamps. We may also want to use pydstool.
        #
        # the odeint expects the first timestamp in the tseries to be the initial condition,
        # so we may need to add the t0 if it is not present in the tseries array
        res = _odeint(self.components.d_dt, initial_values, tseries, hmax=self.components.time_step())
        
        stocknames = sorted(self.components.state.keys())
        values = _pd.DataFrame(data=res, index=tseries, columns=stocknames)


        if return_columns:
            # this is a super slow, super bad way to do this. Recode ASAP
            out = []
            for t, row in zip(values.index, values.to_dict(orient='records')):
                self.components.state.update(row)
                self.components.t = t
        
                for column in return_columns:
                    func = getattr(self.components, column)
                    row[column] = func()
        
                out.append(row)
        
            values = _pd.DataFrame(data=out, index=values.index)[return_columns] #this is ugly

        if addtflag:
            values = values.iloc[1:] #there is probably a faster way to drop the initial row

        if collect:
            self.record.append(values)
        
        return values

    def get_record(self):
        """
        returns the recorded model information
        """
        return _pd.concat(self.record)
    
    def clear_record(self):
        """
        Resets the recorder
            """
        self.record = []


    def set_components(self, params={}):
        """
           sets the equation of a model element matching the dictionary 
           key to the dictionary value:
           
           set({'delay':4})
           if given a pandas series, allows for interpolation amongst that series.
           
           
        """
        # Todo:
        # - Update the function docstring
        # - Make this able to handle states
        
        
        for key, value in params.iteritems():
            if isinstance(value, _pd.Series):
                self.components.__dict__.update({key:self._timeseries_component(value)})
            else:
                self.components.__dict__.update({key:self._constant_component(value)})
    
    
    def _timeseries_component(self, series):
        return lambda: np.interp(self.components.t, series.index, series.values)
    
    def _constant_component(self, value):
        return lambda: value

#def run_togeter(models=[], initial_conditions='original', collect=False)
#
# to make this work, we have to be comfortable reaching into a model class, running the integration ourseves, etc


