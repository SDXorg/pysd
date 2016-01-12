
from __future__ import division
import numpy as np
from pysd import functions

def characteristic_time():
    """
    Type: Flow or Auxiliary
        
    """
    return 10

def heat_loss_to_room():
    """
    Type: Flow or Auxiliary
        
    """
    return (teacup_temperature()- room_temperature()) / characteristic_time()

def room_temperature():
    """
    Type: Flow or Auxiliary
        
    """
    return 70

def teacup_temperature():
    return _state['teacup_temperature']

def _teacup_temperature_init():
    return 180

def _dteacup_temperature_dt():
    return -heat_loss_to_room()

def final_time():
    """
    Type: Flow or Auxiliary
        
    """
    return 30

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
