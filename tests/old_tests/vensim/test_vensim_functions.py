from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def test_modulo(self):
        """Type: Flow or Auxiliary
        """
        return np.mod(self.test_pulse_train(), 17) 

    def flow(self):
        """Type: Flow or Auxiliary
        """
        return self.test_tan() 

    def test_pulse_train(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.pulse_train(self.test_min(), 2 , 5 , 50 ) 

    def test_cos(self):
        """Type: Flow or Auxiliary
        """
        return np.cos(self.test_pulse()) 

    def test_exp(self):
        """Type: Flow or Auxiliary
        """
        return np.exp(self.test_step()) 

    def test_if_then_else(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.if_then_else(self.test_sqrt()>0, 65.5 , -9.2 ) 

    def test_integer(self):
        """Type: Flow or Auxiliary
        """
        return int(self.test_sin()) 

    def test_ln(self):
        """Type: Flow or Auxiliary
        """
        return self.test_random_uniform()+ np.log(4) 

    def test_max(self):
        """Type: Flow or Auxiliary
        """
        return max(self.test_random_normal(), 4) 

    def test_min(self):
        """Type: Flow or Auxiliary
        """
        return min(self.test_ramp(), 71.993) 

    def test_random_uniform(self):
        """Type: Flow or Auxiliary
        """
        return np.random.rand(self.test_integer(), 2* self.test_integer(), 0 ) 

    def test_pulse(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.pulse(self.test_modulo(), 5 ) 

    def test_sqrt(self):
        """Type: Flow or Auxiliary
        """
        return np.sqrt(self.test_exp()) 

    def test_ramp(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.ramp(self.test_max(), 5 , 10 ) 

    def test_random_normal(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.bounded_normal(0 , 1000 , self.test_ln(), 5 , 0 ) 

    def test_tan(self):
        """Type: Flow or Auxiliary
        """
        return np.tan(self.test_cos()) 

    def test_sin(self):
        """Type: Flow or Auxiliary
        """
        return np.sin(self.test_if_then_else()) 

    def test_step(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.step(self.test_abs(), 10 ) 

    def ddummy_dt(self):                       
        return self.flow()                           

    def dummy_init(self):                      
        return 0                           

    def dummy(self):                            
        """ Stock: dummy =                      
                 self.flow()                          
                                             
        Initial Value: 0                    
        Do not overwrite this function       
        """                                  
        return self.state["dummy"]              
                                             
    def test_abs(self):
        """Type: Flow or Auxiliary
        """
        return abs(-5) 

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

