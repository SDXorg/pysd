from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder
from xray import DataArray                                     
                                                                
class Components(builder.ComponentClass):                       

    dim_dict = {'suba': ['subb1', 'subb2'],
                'subb': ['suba1', 'suba2', 'suba3']}

    def characteristic_time(self):
        """Type: Flow or Auxiliary
        """
        return 10 

    def heat_loss_to_room(self):
        """Type: Flow or Auxiliary
        """
        return (self.teacup_temperature() - self.room_temperature()) / self.characteristic_time()

    def room_temperature(self):
        """Type: Flow or Auxiliary
        """
        return room_temperature.value
    room_temperature.value = DataArray([[50, 60, 70]], dim_dict[['subb']])

    def dteacup_temperature_dt(self):                       
        return -self.heat_loss_to_room()                           

    def teacup_temperature_init(self):
        return teacup_temperature_init.value
    teacup_temperature_init.value = DataArray([[180,160],[100,10],[1,190]], self.dim_dict)

    def teacup_temperature(self):                            
        """ Stock: teacup_temperature =                      
                 -self.heat_loss_to_room()                          
                                             
        Initial Value: 180                    
        Do not overwrite this function       
        """                                  
        return self.state["teacup_temperature"]              
                                             
    def final_time(self):
        """Type: Flow or Auxiliary
        """
        return 30 

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

