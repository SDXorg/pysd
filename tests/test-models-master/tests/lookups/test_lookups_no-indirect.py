from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def rate(self):
        """Type: Flow or Auxiliary
        """
        return self.lookup_function_call() 

    def lookup_function_call(self):
        """Type: Flow or Auxiliary
        """
        return self.time() 

    def daccumulation_dt(self):                       
        return self.rate()                           

    def accumulation_init(self):                      
        return 0                           

    def accumulation(self):                            
        """ Stock: accumulation =                      
                 self.rate()                          
                                             
        Initial Value: 0                    
        Do not overwrite this function       
        """                                  
        return self.state["accumulation"]              
                                             
    def initial_time(self):
        """Type: Flow or Auxiliary
        """
        return 0 

    def final_time(self):
        """Type: Flow or Auxiliary
        """
        return 45 

    def time_step(self):
        """Type: Flow or Auxiliary
        """
        return 0.25 

