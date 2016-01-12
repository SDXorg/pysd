
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def lookup_function_call():
    """
    Type: Flow or Auxiliary
        
    """
    return lookup_function_table(time())

def rate():
    """
    Type: Flow or Auxiliary
        
    """
    return lookup_function_call()

def accumulation():
    return _state['accumulation']

def _accumulation_init():
    return 0

def _daccumulation_dt():
    return rate()

def lookup_function_table(x):
    return functions.lookup(x,
                            lookup_function_table.xs,
                            lookup_function_table.ys)

lookup_function_table.xs = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
lookup_function_table.ys = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0]

def final_time():
    """
    Type: Flow or Auxiliary
        
    """
    return 45

def initial_time():
    """
    Type: Flow or Auxiliary
        
    """
    return 0

def saveper():
    """
    Type: Flow or Auxiliary
        
    """
    return time_step()

def time_step():
    """
    Type: Flow or Auxiliary
        
    """
    return 0.25
