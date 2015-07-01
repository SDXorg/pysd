from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def fleau(self):
        """Type: Flow or Auxiliary
        """
        return self.stahk() 

    def initial_fleau(self, inval):                  
        if not hasattr(self.initial_fleau, "value"): 
            self.initial_fleau.im_func.value = inval 
        return self.initial_fleau.value             

    def test_initial_fleau(self):
        """Type: Flow or Auxiliary
        """
        return self.initial_fleau(self.fleau()) 

    def dstahk_dt(self):                       
        return self.fleau()                           

    def stahk_init(self):                      
        return 10                           

    def stahk(self):                            
        """ Stock: stahk =                      
                 self.fleau()                          
                                             
        Initial Value: 10                    
        Do not overwrite this function       
        """                                  
        return self.state["stahk"]              
                                             
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

