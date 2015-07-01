from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def flow(self):
        """Type: Flow or Auxiliary
        """
        return self.lookup_function_call() 

    def dintegrate_lookup_dt(self):                       
        return self.flow()                           

    def integrate_lookup_init(self):                      
        return 0                           

    def integrate_lookup(self):                            
        """ Stock: integrate_lookup =                      
                 self.flow()                          
                                             
        Initial Value: 0                    
        Do not overwrite this function       
        """                                  
        return self.state["integrate_lookup"]              
                                             
    def lookup_function_call(self):
        """Type: Flow or Auxiliary
        """
        return self.lookup_function_table(self.lookup_function_parameter()) 

    def lookup_function_parameter(self):
        """Type: Flow or Auxiliary
        """
        return 0.345 

    def lookup_function_table(self, x):                                      
        return self.functions.lookup(x,                   
                                     self.lookup_function_table.xs,          
                                     self.lookup_function_table.ys)          
                                                          
    lookup_function_table.xs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]                                            
    lookup_function_table.ys = [0.0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 1.1]                                            
                                                          
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

