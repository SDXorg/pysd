# This file is a minimalist version of pysd
#
#



from __future__ import division
import numpy as np
from scipy.integrate import odeint
import itertools


class MinModel(object):
    ##########  boilerplate stuff from the existing pysd #########
    def __init__(self):
        self._stocknames = [name[:-5] for name in dir(self) if name[-5:] == '_init']
        self._stocknames.sort() #inplace
        self._dfuncs = [getattr(self, 'd%s_dt'%name) for name in self._stocknames]
        self.state = dict(zip(self._stocknames, [None]*len(self._stocknames)))
        self.reset_state()

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
            self.reset_state() #potential for infinite loop!

    ########### Stuff we have to modify to make subscripts work #########
    def d_dt(self, state_vector, t):
        """The primary purpose of this function is to interact with the integrator.
        It takes a state vector, sets the state of the system based on that vector,
        and returns a derivative of the state vector
        """        
        self.set_state(state_vector)
        self.t = t
        
        return [func() for func in self._dfuncs]

    def set_state(self, state_vector):
        i = 0
        for key in self._stocknames:
            self.state[key] = state_vector[i]

        
    def get_state(self):
        #if we keep this, we should make it fully a list comprehension
        return [self.state[key] for key in self._stocknames]
    
    
    ######### model specific components (that go in the model file)

    
    def stock(self):
        return self.state['stock']
    
    def stock_init(self):
        return 1
    
    def dstock_dt(self):
        return self.flow()
    
    def constant(self):
        return 3
    
    def flow(self):
        return self.constant() * self.stock()
    
    def initial_time(self):
        return 0


a = MinModel()
print odeint(a.d_dt, a.get_state(), range(10))
