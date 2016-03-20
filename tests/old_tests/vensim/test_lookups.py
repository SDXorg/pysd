
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

# Share the `time` function with the module for `step`, `pulse`, etc.
functions.__builtins__.update({'time':time})


def lookup_function_call():
    """
    
    """
    loc_dimension_dir = 0 
    output = lookup_function_table(accumulation())

    return output

def rate():
    """
    
    """
    loc_dimension_dir = 0 
    output = lookup_function_call()

    return output

def accumulation():
    return _state['accumulation']

def _accumulation_init():
    try:
        loc_dimension_dir = accumulation.dimension_dir
    except:
        loc_dimension_dir = 0
    return 0

def _daccumulation_dt():
    try:
        loc_dimension_dir = accumulation.dimension_dir
    except:
        loc_dimension_dir = 0
    return rate()

def final_time():
    """
    
    """
    loc_dimension_dir = 0 
    output = 100

    return output

def initial_time():
    """
    
    """
    loc_dimension_dir = 0 
    output = 0

    return output

def saveper():
    """
    
    """
    loc_dimension_dir = 0 
    output = time_step()

    return output

def time_step():
    """
    
    """
    loc_dimension_dir = 0 
    output = 1

    return output
