
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def characteristic_time():
    """
    
    """
    output = 10

    return output

def heat_loss_to_room():
    """
    
    """
    output = (teacup_temperature()-room_temperature())/characteristic_time()

    return output

def room_temperature():
    """
    
    """
    output = 70

    return output

def teacup_temperature():
    return _state['teacup_temperature']

def _teacup_temperature_init():
    return 180

def _dteacup_temperature_dt():
    return -heat_loss_to_room()

def final_time():
    """
    
    """
    output = 30

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
