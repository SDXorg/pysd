from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def denominator(self):
        """Type: Flow or Auxiliary
        """
        return 4 

    def numerator(self):
        """Type: Flow or Auxiliary
        """
        return 3 

    def quotient(self):
        """Type: Flow or Auxiliary
        """
        return self.numerator()/self.denominator() 

    def quotient_target(self):
        """Type: Flow or Auxiliary
        """
        return 0.75 

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

