from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def inflowa(self):
        """Type: Flow or Auxiliary
        """
        return 1 

    def dstocka_dt(self):                       
        return self.inflowa()                           

    def stocka_init(self):                      
        return -10                           

    def stocka(self):                            
        """ Stock: stocka =                      
                 self.inflowa()                          
                                             
        Initial Value: -10                    
        Do not overwrite this function       
        """                                  
        return self.state["stocka"]              
                                             
    def test_arccos(self):
        """Type: Flow or Auxiliary
        """
        return np.arccos(self.test_cos()) 

    def test_arcsin(self):
        """Type: Flow or Auxiliary
        """
        return np.arcsin(self.test_sin()) 

    def test_arctan(self):
        """Type: Flow or Auxiliary
        """
        return np.arctan(self.test_tan()) 

    def test_cos(self):
        """Type: Flow or Auxiliary
        """
        return np.cos(self.stocka()) 

    def test_sin(self):
        """Type: Flow or Auxiliary
        """
        return np.sin(self.stocka()) 

    def test_tan(self):
        """Type: Flow or Auxiliary
        """
        return np.tan(self.stocka()) 

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
        return 0.125 

