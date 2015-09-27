from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def normal_flow(self):
        """Type: Flow or Auxiliary
        """
        return self.import_element() 

    def import_element(self):
        """Type: Flow or Auxiliary
        """
        return 10 

    def aux_with_pec_characters(self):
        """Type: Flow or Auxiliary
        """
        return 21 

    def aux_with_entirely_superfuluous_parenthetical_comment(self):
        """Type: Flow or Auxiliary
        """
        return 90 

    def flow_w_division_lists_and_initialconstruction_functions(self):
        """Type: Flow or Auxiliary
        """
        return self.aux_with_pec_characters() 

    def flow_with_stepfunction_call(self):
        """Type: Flow or Auxiliary
        """
        return self.aux_with_entirely_superfuluous_parenthetical_comment() 

    def flowwith_a_fewarithmetic__characters(self):
        """Type: Flow or Auxiliary
        """
        return 4 

    def dhyphenatedstockname_dt(self):                       
        return self.normal_flow()-self.flowwith_a_fewarithmetic__characters()                           

    def hyphenatedstockname_init(self):                      
        return 0                           

    def hyphenatedstockname(self):                            
        """ Stock: hyphenatedstockname =                      
                 self.normal_flow()-self.flowwith_a_fewarithmetic__characters()                          
                                             
        Initial Value: 0                    
        Do not overwrite this function       
        """                                  
        return self.state["hyphenatedstockname"]              
                                             
    def dstock_with_n_newline_character_dt(self):                       
        return self.flow_w_division_lists_and_initialconstruction_functions()-self.flow_with_stepfunction_call()                           

    def stock_with_n_newline_character_init(self):                      
        return 67                           

    def stock_with_n_newline_character(self):                            
        """ Stock: stock_with_n_newline_character =                      
                 self.flow_w_division_lists_and_initialconstruction_functions()-self.flow_with_stepfunction_call()                          
                                             
        Initial Value: 67                    
        Do not overwrite this function       
        """                                  
        return self.state["stock_with_n_newline_character"]              
                                             
    def final_time(self):
        """Type: Flow or Auxiliary
        """
        return 100 

    def initial_time(self):
        """Type: Flow or Auxiliary
        """
        return 0 

    def saveper(self):
        """Type: Flow or Auxiliary
        """
        return self.time_step() 

    def time_step(self):
        """Type: Flow or Auxiliary
        """
        return 1 

