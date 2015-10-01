from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def associativity(self):
        """Type: Flow or Auxiliary
        """
        return -2**2 

    def output(self):
        """Type: Flow or Auxiliary
        """
        return self.time()**2 

    def test(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.if_then_else(self.associativity()==4, 1, 0) 

    def final_time(self):
        """Type: Flow or Auxiliary
        """
        return 4 

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

