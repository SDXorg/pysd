
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def characteristic_time():
    """
    
    """
    loc_dimension_dir = 0 
    output = 10

    return output

def heat_loss_to_room():
    """
    
    """
    loc_dimension_dir = 0 
    output = (teacup_temperature()-room_temperature())/characteristic_time()

    return output

def room_temperature():
    """
    
    """
    loc_dimension_dir = 0 
    output = 70

    return output

def teacup_temperature():
    return _state['teacup_temperature']

def _teacup_temperature_init():
    try:
        loc_dimension_dir = teacup_temperature.dimension_dir
    except:
        loc_dimension_dir = 0
    return 180

def _dteacup_temperature_dt():
    try:
        loc_dimension_dir = teacup_temperature.dimension_dir
    except:
        loc_dimension_dir = 0
    return -heat_loss_to_room()

def final_time():
    """
    
    """
    loc_dimension_dir = 0 
    output = 30

    return output

def initial_time():
    """
    
    """
    loc_dimension_dir = 0 
    output = 0

    return output

def saveper():
    """
    
    """
    loc_dimension_dir = 0 
    output = time_step()

    return output

def time_step():
    """
    
    """
    loc_dimension_dir = 0 
    output = 0.125

    return output
