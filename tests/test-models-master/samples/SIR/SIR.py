
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def contact_infectivity():
    """
    Type: Flow or Auxiliary
        
    """
    return 0.3

def duration():
    """
    Type: Flow or Auxiliary
        
    """
    return 5

def infectious():
    return _state['infectious']

def _infectious_init():
    return 5

def _dinfectious_dt():
    return succumbing()-recovering()

def recovered():
    return _state['recovered']

def _recovered_init():
    return 0

def _drecovered_dt():
    return recovering()

def recovering():
    """
    Type: Flow or Auxiliary
        
    """
    return infectious()/duration()

def succumbing():
    """
    Type: Flow or Auxiliary
        
    """
    return susceptible()*infectious()/total_population()* contact_infectivity()

def susceptible():
    return _state['susceptible']

def _susceptible_init():
    return total_population()

def _dsusceptible_dt():
    return -succumbing()

def total_population():
    """
    Type: Flow or Auxiliary
        
    """
    return 1000

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
    return 0.03125
