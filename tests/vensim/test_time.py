from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def flow(self):
        """Type: Flow or Auxiliary
        """
        return self.time() 

    def dstock_dt(self):                       
        return self.flow()                           

    def stock_init(self):                      
        return 0                           

    def stock(self):                            
        """ Stock: stock =                      
                 self.flow()                          
                                             
        Initial Value: 0                    
        Do not overwrite this function       
        """                                  
        return self.state["stock"]              
                                             
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

