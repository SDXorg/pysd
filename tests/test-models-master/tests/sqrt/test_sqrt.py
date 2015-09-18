from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def flowa(self):
        """Type: Flow or Auxiliary
        """
        return self.stocka() 

    def dstocka_dt(self):                       
        return self.flowa()                           

    def stocka_init(self):                      
        return 0.001                           

    def stocka(self):                            
        """ Stock: stocka =                      
                 self.flowa()                          
                                             
        Initial Value: 0.001                    
        Do not overwrite this function       
        """                                  
        return self.state["stocka"]              
                                             
    def test_sqrt(self):
        """Type: Flow or Auxiliary
        """
        return np.sqrt(self.stocka()) 

    def final_time(self):
        """Type: Flow or Auxiliary
        """
        return 20 

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

