
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def initial_values():
    """
    
    """

    output = np.ndarray((3,2))
    output[:,:] = np.array([[1, 2],
       [3, 4],
       [5, 6]])

    return output

def stock_a():
    return _state['stock_a']

def _stock_a_init():
    return initial_values()*np.ones((3,2))

def _dstock_a_dt():
    return inflow_a()

def inflow_a():
    """
    
    """

    output = np.ndarray((3,2))
    output[:,:] = rate_a()

    return output

def rate_a():
    """
    
    """

    output = np.ndarray((3,2))
    output[:,0] = 0.01, 0.03, 0.05 
    output[:,1] = 0.02, 0.04, 0.06

    return output

def final_time():
    """
    
    """

    output = 100
	

    return output

def initial_time():
    """
    
    """

    output = 0
	

    return output

def saveper():
    """
    
    """

    output = time_step()

    return output

def time_step():
    """
    
    """

    output = 1
	

    return output
