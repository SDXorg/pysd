
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def associativity():
    """
    Type: Flow or Auxiliary
        
    """
    return -2**2

def output():
    """
    Type: Flow or Auxiliary
        
    """
    return time()**2

def test():
    """
    Type: Flow or Auxiliary
        
    """
    return self.functions.if_then_else(associativity()==4, 1, 0)

def final_time():
    """
    Type: Flow or Auxiliary
        
    """
    return 4

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
