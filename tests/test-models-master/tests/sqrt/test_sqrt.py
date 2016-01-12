
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def flowa():
    """
    Type: Flow or Auxiliary
        
    """
    return stocka()

def stocka():
    return _state['stocka']

def _stocka_init():
    return 0.001

def _dstocka_dt():
    return flowa()

def test_sqrt():
    """
    Type: Flow or Auxiliary
        
    """
    return np.sqrt(stocka())

def final_time():
    """
    Type: Flow or Auxiliary
        
    """
    return 20

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
