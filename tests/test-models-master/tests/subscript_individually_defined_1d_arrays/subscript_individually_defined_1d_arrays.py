
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def inflow_a():
    """
    
    """

    output = np.ndarray((3))
    output[:] = rate_a()

    return output

def stock_a():
    return _state['stock_a']

def _stock_a_init():
    return 0*np.ones((3))

def _dstock_a_dt():
    return inflow_a()

def rate_a():
    """
    
    """

    output = np.ndarray((3))
    output[0] = 0.01 
    output[1] = 0.2 
    output[2] = 0.3

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
