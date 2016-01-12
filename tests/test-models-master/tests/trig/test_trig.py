
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def inflowa():
    """
    Type: Flow or Auxiliary
        
    """
    return 1

def stocka():
    return _state['stocka']

def _stocka_init():
    return -10

def _dstocka_dt():
    return inflowa()

def test_arccos():
    """
    Type: Flow or Auxiliary
        
    """
    return np.arccos(test_cos())

def test_arcsin():
    """
    Type: Flow or Auxiliary
        
    """
    return np.arcsin(test_sin())

def test_arctan():
    """
    Type: Flow or Auxiliary
        
    """
    return np.arctan(test_tan())

def test_cos():
    """
    Type: Flow or Auxiliary
        
    """
    return np.cos(stocka())

def test_sin():
    """
    Type: Flow or Auxiliary
        
    """
    return np.sin(stocka())

def test_tan():
    """
    Type: Flow or Auxiliary
        
    """
    return np.tan(stocka())

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
    return 0.125
