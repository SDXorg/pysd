from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def heat_loss_to_room(self):
        """Type: Flow or Auxiliary
        """
        return (self.teacup_temperature()-self.room_temperature())/self.characteristic_time() 

    def characteristic_time(self):
        """Type: Flow or Auxiliary
        """
        return 10 

    def room_temperature(self):
        """Type: Flow or Auxiliary
        """
        return 70 

    def dteacup_temperature_dt(self):                       
        return  - self.heat_loss_to_room()                           

    def teacup_temperature_init(self):                      
        return 180                           

    def teacup_temperature(self):                            
        """ Stock: teacup_temperature =                      
                  - self.heat_loss_to_room()                          
                                             
        Initial Value: 180                    
        Do not overwrite this function       
        """                                  
        return self.state["teacup_temperature"]              
                                             
    def initial_time(self):
        """Type: Flow or Auxiliary
        """
        return 0 

    def final_time(self):
        """Type: Flow or Auxiliary
        """
        return 30 

    def time_step(self):
        """Type: Flow or Auxiliary
        """
        return 0.125 

