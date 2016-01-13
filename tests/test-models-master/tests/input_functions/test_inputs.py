
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def inputa():
    """
    
    """

    output = test_pulse()+test_pulse_train()+test_ramp()+test_step()

    return output

def stocka():
    return _state['stocka']

def _stocka_init():
    return 0

def _dstocka_dt():
    return inputa()

def test_pulse():
    """
    
    """

    output = functions.pulse(3, 2 )
	

    return output

def test_pulse_train():
    """
    
    """

    output = functions.pulse_train(7 , 1 , 2 , 12)
	

    return output

def test_ramp():
    """
    
    """

    output = functions.ramp(1, 14 , 17 )
	

    return output

def test_step():
    """
    
    """

    output = functions.step(1, 1)
	

    return output

def final_time():
    """
    
    """

    output = 25
	

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

    output = 0.0625
	

    return output
