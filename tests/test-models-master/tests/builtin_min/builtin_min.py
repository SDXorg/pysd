
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def output():
    """
    Type: Flow or Auxiliary
        
    """
    return min(time(), 5)

def final_time():
    """
    Type: Flow or Auxiliary
        
    """
    return 10

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
    return 1
