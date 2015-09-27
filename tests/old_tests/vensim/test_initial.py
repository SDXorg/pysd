from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def initial_stocka(self, inval):                  
        if not hasattr(self.initial_stocka, "value"): 
            self.initial_stocka.im_func.value = inval 
        return self.initial_stocka.value             

    def stocka_initial_measured_value(self):
        """Type: Flow or Auxiliary
        """
        return self.initial_stocka(self.stocka()) 

    def inflowa(self):
        """Type: Flow or Auxiliary
        """
        return self.stocka() 

    def initial_inflowa(self, inval):                  
        if not hasattr(self.initial_inflowa, "value"): 
            self.initial_inflowa.im_func.value = inval 
        return self.initial_inflowa.value             

    def stocka_initial_set_value(self):
        """Type: Flow or Auxiliary
        """
        return self.initial_inflowa(self.inflowa()) 

    def dstocka_dt(self):                       
        return self.inflowa()                           

    def stocka_init(self):                      
        return 10                           

    def stocka(self):                            
        """ Stock: stocka =                      
                 self.inflowa()                          
                                             
        Initial Value: 10                    
        Do not overwrite this function       
        """                                  
        return self.state["stocka"]              
                                             
    def final_time(self):
        """Type: Flow or Auxiliary
        """
        return 10 

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

