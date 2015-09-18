from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def inputa(self):
        """Type: Flow or Auxiliary
        """
        return self.test_pulse()+self.test_pulse_train()+self.test_ramp()+self.test_step() 

    def dstocka_dt(self):                       
        return self.inputa()                           

    def stocka_init(self):                      
        return 0                           

    def stocka(self):                            
        """ Stock: stocka =                      
                 self.inputa()                          
                                             
        Initial Value: 0                    
        Do not overwrite this function       
        """                                  
        return self.state["stocka"]              
                                             
    def test_pulse(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.pulse(3, 2 ) 

    def test_pulse_train(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.pulse_train(7 , 1 , 2 , 12) 

    def test_ramp(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.ramp(1, 14 , 17 ) 

    def test_step(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.step(1, 1) 

    def final_time(self):
        """Type: Flow or Auxiliary
        """
        return 25 

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
        return 0.0625 

