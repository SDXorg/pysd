
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def flowa():
    """
    Type: Flow or Auxiliary
        
    """
    return 0.1

def stocka():
    return _state['stocka']

def _stocka_init():
    return -5

def _dstocka_dt():
    return flowa()

def test_integer():
    """
    Type: Flow or Auxiliary
        
    """
    return int(stocka())

def test_modulo():
    """
    Type: Flow or Auxiliary
        
    """
    return np.mod(stocka(), 3 )

def final_time():
    """
    Type: Flow or Auxiliary
        
    """
    return 100

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
