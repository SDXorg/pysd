
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t

def normal_flow():
    """
    Type: Flow or Auxiliary
        
    """
    return import_element()

def import_element():
    """
    Type: Flow or Auxiliary
        
    """
    return 10

def aux_with_pec_characters():
    """
    Type: Flow or Auxiliary
        
    """
    return 21

def aux_with_entirely_superfuluous_parenthetical_comment():
    """
    Type: Flow or Auxiliary
        
    """
    return 90

def flow_w_division_lists_and_initialconstruction_functions():
    """
    Type: Flow or Auxiliary
        
    """
    return aux_with_pec_characters()

def flow_with_stepfunction_call():
    """
    Type: Flow or Auxiliary
        
    """
    return aux_with_entirely_superfuluous_parenthetical_comment()

def flowwith_a_fewarithmetic__characters():
    """
    Type: Flow or Auxiliary
        
    """
    return 4

def hyphenatedstockname():
    return _state['hyphenatedstockname']

def _hyphenatedstockname_init():
    return 0

def _dhyphenatedstockname_dt():
    return normal_flow()-flowwith_a_fewarithmetic__characters()

def stock_with_n_newline_character():
    return _state['stock_with_n_newline_character']

def _stock_with_n_newline_character_init():
    return 67

def _dstock_with_n_newline_character_dt():
    return flow_w_division_lists_and_initialconstruction_functions()-flow_with_stepfunction_call()

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
    return 1
