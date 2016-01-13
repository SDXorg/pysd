
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def inflow_a():
    """
    
    """

    output = np.ndarray((3,2,2))
    output[:,:,:] = rate_a()

    return output

def stock_a():
    return _state['stock_a']

def _stock_a_init():
    return *np.ones(())

def _dstock_a_dt():
    return inflow_a()

def initial_values():
    """
    
    """

    output = np.ndarray((3,2,2))
    output[:,:,0] = [[u'1, 2; 3, 4; 5, 6; ']]
    output[:,:,1] = [[u'2, 4; 5, 3; 1, 4;\n\t']]

    return output

def rate_a():
    """
    
    """

    output = np.ndarray((3,2,2))
    output[:,:,0] = [[u'0.01, 0.02; 0.03, 0.04; 0.05, 0.06; ']]
    output[:,:,1] = [[u'0.02, 0.05; 0.02, 0.04; 0.05, 0.06;\n\t']]

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
