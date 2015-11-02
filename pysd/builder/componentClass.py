from __future__ import division
import inspect
from pysd import functions
from itertools import izip
from xray import DataArray
import numpy as np

class ComponentClass(object):
    """
    This is a template class to be subclassed and fleshed out by the translation tools.

    A function should be added for every flow or auxiliary variable, having the name of that
    variable, taking no parameters, which calculates the value of the variable.

    A function should be added for each stock called d<stockname>_dt() which calculates
    the net inflow to the stock as a function of the flows.

    A function should be added for each stock called <stockname>_init() which calculates
    the (potentially dynamically generated) initial value for the stock.

    This docstring will be rewritten and dynamically generated.
    """

    def __init__(self):
        self._stocknames = [name[:-5] for name in dir(self) if name[-5:] == '_init']
        self._stocknames.sort() #inplace
        self._dfuncs = [getattr(self, 'd%s_dt'%name) for name in self._stocknames]
        self.state = dict(zip(self._stocknames, [None]*len(self._stocknames)))
        self.reset_state()
        self.__doc__ = self.doc()
        self.functions = functions.Functions(self)

    def doc(self):
        """This function returns an aggregation of all of the docstrings of all of the
        elements in the model.
        """
        #docstring = self.__str__ + '\n\n'
        #this needs to have a way to make a 'short' docstring...
        docstring = ''
        for method in dir(self):
            if method not in ['__doc__', '__init__', '__module__', '__str__', 't',
                              'd_dt', 'reset_state', 'model_description', 'saveper',
                              'state', 'state_vector']:
                if method[0] not in ['_']:
                    try:
                        docstring += inspect.getdoc(getattr(self, method)) + '\n\n'
                    except:
                        pass

        return docstring

    def initial_time(self):
        """This function represents the initial time as set in the model file
        It should be overwritten during the class extension.
        """
        return 0

    def reset_state(self):
        """Sets the model state to the state described in the model file. """
        self.t = self.initial_time() #set the initial time
        retry_flag = False
        for key in self.state.keys():
            try:
                self.state[key] = eval('self.'+key+'_init()') #set the initial state
            except TypeError:
                retry_flag = True
        if retry_flag:
            #this is so that if a stock references the initial value of another stock,
            #we can try all of the stocks that don't have that type of reference, and
            #then come back to try it again
            self.reset_state() #potential for infinite loop!

#    def d_dt(self, state_vector, t):
#        """The primary purpose of this function is to interact with the integrator.
#        It takes a state vector, sets the state of the system based on that vector,
#        and returns a derivative of the state vector
#        """
#        #state = dict(zip(self._stocknames, state_vector))
#        state = dict(izip(self._stocknames, state_vector)) #izip seems to be about 5% faster
#        self.state.update(state)
#        self.t = t
#
#        #return map(lambda x: x(), self._dfuncs)
#        return [func() for func in self._dfuncs]


    def d_dt(self, state_vector, t):
        """The primary purpose of this function is to interact with the integrator.
        It takes a state vector, sets the state of the system based on that vector,
        and returns a derivative of the state vector
        """        
        self.set_state(state_vector)
        self.t = t
        
        derivative_vector = []
        for func in self._dfuncs:
            res = func()
            if isinstance(res, DataArray):
                res = res.values.flatten()
                derivative_vector += list(res)
            else:
                derivative_vector.append(res) #the way we have to handle this is really ugly...
            
        return derivative_vector

    def set_state(self, state_vector):
        i = 0
        for key in self._stocknames:
            if isinstance(self.state[key], DataArray):
                shape = self.state[key].shape
                size = self.state[key].size
                self.state[key].loc[:,:].values = np.array(state_vector[i:i+size]).reshape(shape)
                i += size
            else:
                self.state[key] = state_vector[i]
                i += 1
        
    def get_state(self):
        #if we keep this, we should make it fully a list comprehension
        state_vector = []
        for item in [self.state[key] for key in self._stocknames]:
            if isinstance(item, DataArray):
                state_vector += list(item.values.flatten())
            else:
                state_vector.append(item) #this must be a number...
        return state_vector

#    def state_vector(self):
#        """This function interacts with the integrator by setting the
#        initial values of the integrator to the state vector.
#        It returns the values of the state dictionary, sorted
#        alphabetically by key.
#        """
#        #return [self.state[key] for key in self._stocknames]
#        return map(lambda x: self.state[x], self._stocknames)
#        #should test that 'map' is actually faster, because list comprehension is clearer

    def time(self):
        """This helper function allows the model components to
        access the time component directly.
        """
        return self.t

    def file(self):
        return 'hi'