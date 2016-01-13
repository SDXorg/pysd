
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def output():
    """
    
    """

    output = functions.if_then_else(time()>5, 1, 0)
	

    return output

def final_time():
    """
    
    """

    output = 12
	

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

    output = 0.25
	

    return output
