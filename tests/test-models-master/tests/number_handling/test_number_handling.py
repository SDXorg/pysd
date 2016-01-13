
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def equality():
    """
    
    """

    output = functions.if_then_else(quotient()==quotient_target(), 1 , 0 )
	

    return output

def denominator():
    """
    
    """

    output = 4
	

    return output

def numerator():
    """
    
    """

    output = 3
	

    return output

def quotient():
    """
    
    """

    output = numerator()/denominator()

    return output

def quotient_target():
    """
    
    """

    output = 0.75
	

    return output

def final_time():
    """
    
    """

    output = 1
	

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

    output = 1
	

    return output
