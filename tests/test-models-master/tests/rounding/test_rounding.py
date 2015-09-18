from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def flowa(self):
        """Type: Flow or Auxiliary
        """
        return 0.1 

    def dstocka_dt(self):                       
        return self.flowa()                           

    def stocka_init(self):                      
        return -5                           

    def stocka(self):                            
        """ Stock: stocka =                      
                 self.flowa()                          
                                             
        Initial Value: -5                    
        Do not overwrite this function       
        """                                  
        return self.state["stocka"]              
                                             
    def test_integer(self):
        """Type: Flow or Auxiliary
        """
        return int(self.stocka()) 

    def test_modulo(self):
        """Type: Flow or Auxiliary
        """
        return np.mod(self.stocka(), 3 ) 

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

