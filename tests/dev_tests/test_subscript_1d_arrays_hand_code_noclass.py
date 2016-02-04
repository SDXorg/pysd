from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
#from pysd import component_module
from xray import DataArray                                     

dim_dict = {'One Dimensional Subscript': ['Entry 1', 'Entry 2', 'Entry 3']}

def rate_a():
    return rate_a.value
rate_a.value = DataArray([.01,.02,.03], dim_dict)

def dstock_a_dt():
    return inflow_a()

def stock_a_init():
    return stock_a_init.value
stock_a_init.value = DataArray([0,0,0], dim_dict)


def stock_a():
    return state["stock_a"]

def inflow_a():
    return rate_a()


                                         
def final_time():
    """Type: Flow or Auxiliary
    """
    return 30 

def initial_time():
    """Type: Flow or Auxiliary
    """
    return 0 

def saveper():
    """Type: Flow or Auxiliary
    """
    return time_step() 

def time_step():
    """Type: Flow or Auxiliary
    """
    return 0.125 
