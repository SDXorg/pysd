from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def adjustment_time(self):
        """Type: Flow or Auxiliary
        """
        return 2 

    def input_smooth_adjustment_time_flow_1_of_3(self):
        """Type: Flow or Auxiliary
        """
        return (self.input() - self.input_smooth_adjustment_time_stock_1_of_3()) / (1.*self.adjustment_time()/3) 

    def input_smooth_adjustment_time_flow_2_of_3(self):
        """Type: Flow or Auxiliary
        """
        return (self.input_smooth_adjustment_time_stock_1_of_3() - self.input_smooth_adjustment_time_stock_2_of_3())/(1.*self.adjustment_time()/3) 

    def input_smooth_adjustment_time_flow_3_of_3(self):
        """Type: Flow or Auxiliary
        """
        return (self.input_smooth_adjustment_time_stock_2_of_3() - self.input_smooth_adjustment_time_stock_3_of_3())/(1.*self.adjustment_time()/3) 

    def dinput_smooth_adjustment_time_stock_1_of_3_dt(self):                       
        return self.input_smooth_adjustment_time_flow_1_of_3()                           

    def input_smooth_adjustment_time_stock_1_of_3_init(self):                      
        return 0                            

    def input_smooth_adjustment_time_stock_1_of_3(self):                            
        """ Stock: input_smooth_adjustment_time_stock_1_of_3 =                      
                 self.input_smooth_adjustment_time_flow_1_of_3()                          
                                             
        Initial Value: 0                     
        Do not overwrite this function       
        """                                  
        return self.state["input_smooth_adjustment_time_stock_1_of_3"]              
                                             
    def dinput_smooth_adjustment_time_stock_2_of_3_dt(self):                       
        return self.input_smooth_adjustment_time_flow_2_of_3()                           

    def input_smooth_adjustment_time_stock_2_of_3_init(self):                      
        return 0                            

    def input_smooth_adjustment_time_stock_2_of_3(self):                            
        """ Stock: input_smooth_adjustment_time_stock_2_of_3 =                      
                 self.input_smooth_adjustment_time_flow_2_of_3()                          
                                             
        Initial Value: 0                     
        Do not overwrite this function       
        """                                  
        return self.state["input_smooth_adjustment_time_stock_2_of_3"]              
                                             
    def dinput_smooth_adjustment_time_stock_3_of_3_dt(self):                       
        return self.input_smooth_adjustment_time_flow_3_of_3()                           

    def input_smooth_adjustment_time_stock_3_of_3_init(self):                      
        return 0                            

    def input_smooth_adjustment_time_stock_3_of_3(self):                            
        """ Stock: input_smooth_adjustment_time_stock_3_of_3 =                      
                 self.input_smooth_adjustment_time_flow_3_of_3()                          
                                             
        Initial Value: 0                     
        Do not overwrite this function       
        """                                  
        return self.state["input_smooth_adjustment_time_stock_3_of_3"]              
                                             
    def function_output(self):
        """Type: Flow or Auxiliary
        """
        return self.input_smooth_adjustment_time_flow_3_of_3() 

    def input(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.step(5 , 5) 

    def order_of_smoothing(self):
        """Type: Flow or Auxiliary
        """
        return 3 

    def per_stock_adjustment_time(self):
        """Type: Flow or Auxiliary
        """
        return self.adjustment_time()/self.order_of_smoothing() 

    def sm_flow_1(self):
        """Type: Flow or Auxiliary
        """
        return (self.input()- self.sm_stock_1())/self.per_stock_adjustment_time() 

    def sm_flow_2(self):
        """Type: Flow or Auxiliary
        """
        return (self.sm_stock_1()-self.sm_stock_2())/self.per_stock_adjustment_time() 

    def sm_flow_3(self):
        """Type: Flow or Auxiliary
        """
        return (self.sm_stock_2()-self.sm_stock_3())/self.per_stock_adjustment_time() 

    def dsm_stock_1_dt(self):                       
        return self.sm_flow_1()                           

    def sm_stock_1_init(self):                      
        return 0                           

    def sm_stock_1(self):                            
        """ Stock: sm_stock_1 =                      
                 self.sm_flow_1()                          
                                             
        Initial Value: 0                    
        Do not overwrite this function       
        """                                  
        return self.state["sm_stock_1"]              
                                             
    def dsm_stock_2_dt(self):                       
        return self.sm_flow_2()                           

    def sm_stock_2_init(self):                      
        return 0                           

    def sm_stock_2(self):                            
        """ Stock: sm_stock_2 =                      
                 self.sm_flow_2()                          
                                             
        Initial Value: 0                    
        Do not overwrite this function       
        """                                  
        return self.state["sm_stock_2"]              
                                             
    def dsm_stock_3_dt(self):                       
        return self.sm_flow_3()                           

    def sm_stock_3_init(self):                      
        return 0                           

    def sm_stock_3(self):                            
        """ Stock: sm_stock_3 =                      
                 self.sm_flow_3()                          
                                             
        Initial Value: 0                    
        Do not overwrite this function       
        """                                  
        return self.state["sm_stock_3"]              
                                             
    def structure_output(self):
        """Type: Flow or Auxiliary
        """
        return self.sm_stock_3() 

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
        return 0.25 

