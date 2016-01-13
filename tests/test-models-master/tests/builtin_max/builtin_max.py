
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def output():
    """
    
    """

    output = np.maximum(time(), 5)
	

    return output

def final_time():
    """
    
    """

    output = 10
	

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
