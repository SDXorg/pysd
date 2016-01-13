
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def inflowa():
    """
    
    """

    output = 1
	

    return output

def stocka():
    return _state['stocka']

def _stocka_init():
    return -10

def _dstocka_dt():
    return inflowa()

def test_arccos():
    """
    
    """

    output = np.arccos(test_cos())
	

    return output

def test_arcsin():
    """
    
    """

    output = np.arcsin(test_sin())
	

    return output

def test_arctan():
    """
    
    """

    output = np.arctan(test_tan())
	

    return output

def test_cos():
    """
    
    """

    output = np.cos(stocka())
	

    return output

def test_sin():
    """
    
    """

    output = np.sin(stocka())
	

    return output

def test_tan():
    """
    
    """

    output = np.tan(stocka())
	

    return output

def final_time():
    """
    
    """

    output = 20
	

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

    output = 0.125
	

    return output
