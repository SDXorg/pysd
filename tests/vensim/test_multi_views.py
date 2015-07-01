from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def flow_1(self):
        """Type: Flow or Auxiliary
        """
        return 3 

    def flow_2(self):
        """Type: Flow or Auxiliary
        """
        return self.stock_1() 

    def dstock_1_dt(self):                       
        return self.flow_1()                           

    def stock_1_init(self):                      
        return 0                           

    def stock_1(self):                            
        """ Stock: stock_1 =                      
                 self.flow_1()                          
                                             
        Initial Value: 0                    
        Do not overwrite this function       
        """                                  
        return self.state["stock_1"]              
                                             
    def dstock_2_dt(self):                       
        return self.flow_2()                           

    def stock_2_init(self):                      
        return 0                           

    def stock_2(self):                            
        """ Stock: stock_2 =                      
                 self.flow_2()                          
                                             
        Initial Value: 0                    
        Do not overwrite this function       
        """                                  
        return self.state["stock_2"]              
                                             
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

