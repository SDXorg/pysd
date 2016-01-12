
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def equality():
    """
    Type: Flow or Auxiliary
        
    """
    return self.functions.if_then_else(quotient()==quotient_target(), 1 , 0 )

def denominator():
    """
    Type: Flow or Auxiliary
        
    """
    return 4

def numerator():
    """
    Type: Flow or Auxiliary
        
    """
    return 3

def quotient():
    """
    Type: Flow or Auxiliary
        
    """
    return numerator()/denominator()

def quotient_target():
    """
    Type: Flow or Auxiliary
        
    """
    return 0.75

def final_time():
    """
    Type: Flow or Auxiliary
        
    """
    return 1

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
