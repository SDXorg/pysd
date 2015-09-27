from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def particularly_efflusive_auxilliary_descriptor(self):
        """Type: Flow or Auxiliary
        """
        return 2 

    def product(self):
        """Type: Flow or Auxiliary
        """
        return self.particularly_efflusive_auxilliary_descriptor()* self.semi_infinite_component_moniker()* self.terrifically_elaborate_element_handle()* self.very_long_variable_name() 

    def semi_infinite_component_moniker(self):
        """Type: Flow or Auxiliary
        """
        return 3 

    def terrifically_elaborate_element_handle(self):
        """Type: Flow or Auxiliary
        """
        return 1 

    def very_long_variable_name(self):
        """Type: Flow or Auxiliary
        """
        return 4 

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

