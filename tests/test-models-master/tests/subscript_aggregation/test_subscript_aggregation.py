
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def subscript_1d_sum():
    """
    
    """

    output = np.ndarray((2))
    output[:] = np.sum(stock_a(),0)

    return output

def subscript_2d_max():
    """
    
    """

    output = vmax(stock_a())
	

    return output

def subscript_2d_min():
    """
    
    """

    output = vmin(stock_a())
	

    return output

def subscript_2d_product():
    """
    
    """

    output = prod(stock_a())
	

    return output

def subscript_2d_sum():
    """
    
    """

    output = np.sum(stock_a())
	

    return output

def subscript_secondary_max():
    """
    
    """

    output = vmax(subscript_1d_max())
	

    return output

def subscript_1d_max():
    """
    
    """

    output = np.ndarray((3))
    output[:] = vmax(stock_a(),1)

    return output

def subscript_1d_min():
    """
    
    """

    output = np.ndarray((2))
    output[:] = vmin(stock_a(),0)

    return output

def subscript_1d_product():
    """
    
    """

    output = np.ndarray((3))
    output[:] = prod(stock_a(),1)

    return output

def subscript_secondary_sum():
    """
    
    """

    output = np.sum(subscript_1d_sum())
	

    return output

def subscript_secondary_product():
    """
    
    """

    output = prod(subscript_1d_product())
	

    return output

def subscript_secondary_min():
    """
    
    """

    output = vmin(subscript_1d_min())
	

    return output

def initial_values():
    """
    
    """

    output = np.ndarray((3,2))
    output[:,:] = np.array([[1, 2],
       [3, 4],
       [5, 6]])

    return output

def stock_a():
    return _state['stock_a']

def _stock_a_init():
    return initial_values()*np.ones((3,2))

def _dstock_a_dt():
    return inflow_a()

def inflow_a():
    """
    
    """

    output = np.ndarray((3,2))
    output[:,:] = rate_a()

    return output

def rate_a():
    """
    
    """

    output = np.ndarray((3,2))
    output[:,:] = np.array([[ 0.01,  0.02],
       [ 0.03,  0.04],
       [ 0.05,  0.06]])

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

    output = 1
	

    return output
