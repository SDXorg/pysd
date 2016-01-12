
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def inputa():
    """
    Type: Flow or Auxiliary
        
    """
    return test_pulse()+test_pulse_train()+test_ramp()+test_step()

def stocka():
    return _state['stocka']

def _stocka_init():
    return 0

def _dstocka_dt():
    return inputa()

def test_pulse():
    """
    Type: Flow or Auxiliary
        
    """
    return self.functions.pulse(3, 2 )

def test_pulse_train():
    """
    Type: Flow or Auxiliary
        
    """
    return self.functions.pulse_train(7 , 1 , 2 , 12)

def test_ramp():
    """
    Type: Flow or Auxiliary
        
    """
    return self.functions.ramp(1, 14 , 17 )

def test_step():
    """
    Type: Flow or Auxiliary
        
    """
    return self.functions.step(1, 1)

def final_time():
    """
    Type: Flow or Auxiliary
        
    """
    return 25

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
    return 0.0625
