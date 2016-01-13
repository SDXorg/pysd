
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def contact_infectivity():
    """
    
    """

    output = 0.3
	

    return output

def duration():
    """
    
    """

    output = 5
	

    return output

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
    
    """

    output = infectious()/duration()

    return output

def succumbing():
    """
    
    """

    output = susceptible()*infectious()/total_population()* contact_infectivity()

    return output

def susceptible():
    return _state['susceptible']

def _susceptible_init():
    return total_population()

def _dsusceptible_dt():
    return -succumbing()

def total_population():
    """
    
    """

    output = 1000
	

    return output

def final_time():
    """
    
    """

    output = 100
	

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

    output = 0.03125
	

    return output
