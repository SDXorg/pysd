from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def outflow_delay1(self):
        """Type: Flow or Auxiliary
        """
        return delay1(Inflow Delay1, delay) 

    def inflow_delay1(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.step(5 , 10 ) 

    def inflow_delay1i(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.step(5 , 10 ) 

    def dstock_delay1i_dt(self):                       
        return self.inflow_delay1i()-self.outflow_delay1i()                           

    def stock_delay1i_init(self):                      
        return 5                           

    def stock_delay1i(self):                            
        """ Stock: stock_delay1i =                      
                 self.inflow_delay1i()-self.outflow_delay1i()                          
                                             
        Initial Value: 5                    
        Do not overwrite this function       
        """                                  
        return self.state["stock_delay1i"]              
                                             
    def inflow_delay3i(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.step(5 , 10 ) 

    def dstock_delay3i_dt(self):                       
        return self.inflow_delay3i()-self.outflow_delay3i()                           

    def stock_delay3i_init(self):                      
        return 5                           

    def stock_delay3i(self):                            
        """ Stock: stock_delay3i =                      
                 self.inflow_delay3i()-self.outflow_delay3i()                          
                                             
        Initial Value: 5                    
        Do not overwrite this function       
        """                                  
        return self.state["stock_delay3i"]              
                                             
    def outflow_delay1i(self):
        """Type: Flow or Auxiliary
        """
        return DELAY1I(Inflow Delay1I, delay , 2 ) 

    def outflow_delay3i(self):
        """Type: Flow or Auxiliary
        """
        return DELAY3I(Inflow Delay3I, delay , 3 ) 

    def dstock_delay1_dt(self):                       
        return self.inflow_delay1()-self.outflow_delay1()                           

    def stock_delay1_init(self):                      
        return 5                           

    def stock_delay1(self):                            
        """ Stock: stock_delay1 =                      
                 self.inflow_delay1()-self.outflow_delay1()                          
                                             
        Initial Value: 5                    
        Do not overwrite this function       
        """                                  
        return self.state["stock_delay1"]              
                                             
    def inflow_delayn(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.step(10, 5) 

    def inflow_delay3(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.step(5 , 10 ) 

    def inflow_delayn_delay_delay_flow_1_of_7(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delayn() 

    def inflow_delayn_delay_delay_flow_2_of_7(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delayn_delay_delay_stock_1_of_6()/(1.*self.delay()/6) 

    def inflow_delayn_delay_delay_flow_3_of_7(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delayn_delay_delay_stock_2_of_6()/(1.*self.delay()/6) 

    def inflow_delayn_delay_delay_flow_4_of_7(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delayn_delay_delay_stock_3_of_6()/(1.*self.delay()/6) 

    def inflow_delayn_delay_delay_flow_5_of_7(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delayn_delay_delay_stock_4_of_6()/(1.*self.delay()/6) 

    def inflow_delayn_delay_delay_flow_6_of_7(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delayn_delay_delay_stock_5_of_6()/(1.*self.delay()/6) 

    def inflow_delayn_delay_delay_flow_7_of_7(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delayn_delay_delay_stock_6_of_6()/(1.*self.delay()/6) 

    def dinflow_delayn_delay_delay_stock_1_of_6_dt(self):                       
        return self.inflow_delayn_delay_delay_flow_1_of_7() - self.inflow_delayn_delay_delay_flow_2_of_7()                           

    def inflow_delayn_delay_delay_stock_1_of_6_init(self):                      
        return 10  * (self.delay() / 6)                           

    def inflow_delayn_delay_delay_stock_1_of_6(self):                            
        """ Stock: inflow_delayn_delay_delay_stock_1_of_6 =                      
                 self.inflow_delayn_delay_delay_flow_1_of_7() - self.inflow_delayn_delay_delay_flow_2_of_7()                          
                                             
        Initial Value: 10  * (self.delay() / 6)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflow_delayn_delay_delay_stock_1_of_6"]              
                                             
    def dinflow_delayn_delay_delay_stock_2_of_6_dt(self):                       
        return self.inflow_delayn_delay_delay_flow_2_of_7() - self.inflow_delayn_delay_delay_flow_3_of_7()                           

    def inflow_delayn_delay_delay_stock_2_of_6_init(self):                      
        return 10  * (self.delay() / 6)                           

    def inflow_delayn_delay_delay_stock_2_of_6(self):                            
        """ Stock: inflow_delayn_delay_delay_stock_2_of_6 =                      
                 self.inflow_delayn_delay_delay_flow_2_of_7() - self.inflow_delayn_delay_delay_flow_3_of_7()                          
                                             
        Initial Value: 10  * (self.delay() / 6)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflow_delayn_delay_delay_stock_2_of_6"]              
                                             
    def dinflow_delayn_delay_delay_stock_3_of_6_dt(self):                       
        return self.inflow_delayn_delay_delay_flow_3_of_7() - self.inflow_delayn_delay_delay_flow_4_of_7()                           

    def inflow_delayn_delay_delay_stock_3_of_6_init(self):                      
        return 10  * (self.delay() / 6)                           

    def inflow_delayn_delay_delay_stock_3_of_6(self):                            
        """ Stock: inflow_delayn_delay_delay_stock_3_of_6 =                      
                 self.inflow_delayn_delay_delay_flow_3_of_7() - self.inflow_delayn_delay_delay_flow_4_of_7()                          
                                             
        Initial Value: 10  * (self.delay() / 6)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflow_delayn_delay_delay_stock_3_of_6"]              
                                             
    def dinflow_delayn_delay_delay_stock_4_of_6_dt(self):                       
        return self.inflow_delayn_delay_delay_flow_4_of_7() - self.inflow_delayn_delay_delay_flow_5_of_7()                           

    def inflow_delayn_delay_delay_stock_4_of_6_init(self):                      
        return 10  * (self.delay() / 6)                           

    def inflow_delayn_delay_delay_stock_4_of_6(self):                            
        """ Stock: inflow_delayn_delay_delay_stock_4_of_6 =                      
                 self.inflow_delayn_delay_delay_flow_4_of_7() - self.inflow_delayn_delay_delay_flow_5_of_7()                          
                                             
        Initial Value: 10  * (self.delay() / 6)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflow_delayn_delay_delay_stock_4_of_6"]              
                                             
    def dinflow_delayn_delay_delay_stock_5_of_6_dt(self):                       
        return self.inflow_delayn_delay_delay_flow_5_of_7() - self.inflow_delayn_delay_delay_flow_6_of_7()                           

    def inflow_delayn_delay_delay_stock_5_of_6_init(self):                      
        return 10  * (self.delay() / 6)                           

    def inflow_delayn_delay_delay_stock_5_of_6(self):                            
        """ Stock: inflow_delayn_delay_delay_stock_5_of_6 =                      
                 self.inflow_delayn_delay_delay_flow_5_of_7() - self.inflow_delayn_delay_delay_flow_6_of_7()                          
                                             
        Initial Value: 10  * (self.delay() / 6)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflow_delayn_delay_delay_stock_5_of_6"]              
                                             
    def dinflow_delayn_delay_delay_stock_6_of_6_dt(self):                       
        return self.inflow_delayn_delay_delay_flow_6_of_7() - self.inflow_delayn_delay_delay_flow_7_of_7()                           

    def inflow_delayn_delay_delay_stock_6_of_6_init(self):                      
        return 10  * (self.delay() / 6)                           

    def inflow_delayn_delay_delay_stock_6_of_6(self):                            
        """ Stock: inflow_delayn_delay_delay_stock_6_of_6 =                      
                 self.inflow_delayn_delay_delay_flow_6_of_7() - self.inflow_delayn_delay_delay_flow_7_of_7()                          
                                             
        Initial Value: 10  * (self.delay() / 6)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflow_delayn_delay_delay_stock_6_of_6"]              
                                             
    def outflow_delayn(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delayn_delay_delay_flow_7_of_7() 

    def dstock_delayn_dt(self):                       
        return self.inflow_delayn()-self.outflow_delayn()                           

    def stock_delayn_init(self):                      
        return 0                           

    def stock_delayn(self):                            
        """ Stock: stock_delayn =                      
                 self.inflow_delayn()-self.outflow_delayn()                          
                                             
        Initial Value: 0                    
        Do not overwrite this function       
        """                                  
        return self.state["stock_delayn"]              
                                             
    def delay(self):
        """Type: Flow or Auxiliary
        """
        return 10 

    def inflow_delay3_delay_delay_flow_1_of_4(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delay3() 

    def inflow_delay3_delay_delay_flow_2_of_4(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delay3_delay_delay_stock_1_of_3()/(1.*self.delay()/3) 

    def inflow_delay3_delay_delay_flow_3_of_4(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delay3_delay_delay_stock_2_of_3()/(1.*self.delay()/3) 

    def inflow_delay3_delay_delay_flow_4_of_4(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delay3_delay_delay_stock_3_of_3()/(1.*self.delay()/3) 

    def dinflow_delay3_delay_delay_stock_1_of_3_dt(self):                       
        return self.inflow_delay3_delay_delay_flow_1_of_4() - self.inflow_delay3_delay_delay_flow_2_of_4()                           

    def inflow_delay3_delay_delay_stock_1_of_3_init(self):                      
        return 0 * (self.delay() / 3)                           

    def inflow_delay3_delay_delay_stock_1_of_3(self):                            
        """ Stock: inflow_delay3_delay_delay_stock_1_of_3 =                      
                 self.inflow_delay3_delay_delay_flow_1_of_4() - self.inflow_delay3_delay_delay_flow_2_of_4()                          
                                             
        Initial Value: 0 * (self.delay() / 3)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflow_delay3_delay_delay_stock_1_of_3"]              
                                             
    def dinflow_delay3_delay_delay_stock_2_of_3_dt(self):                       
        return self.inflow_delay3_delay_delay_flow_2_of_4() - self.inflow_delay3_delay_delay_flow_3_of_4()                           

    def inflow_delay3_delay_delay_stock_2_of_3_init(self):                      
        return 0 * (self.delay() / 3)                           

    def inflow_delay3_delay_delay_stock_2_of_3(self):                            
        """ Stock: inflow_delay3_delay_delay_stock_2_of_3 =                      
                 self.inflow_delay3_delay_delay_flow_2_of_4() - self.inflow_delay3_delay_delay_flow_3_of_4()                          
                                             
        Initial Value: 0 * (self.delay() / 3)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflow_delay3_delay_delay_stock_2_of_3"]              
                                             
    def dinflow_delay3_delay_delay_stock_3_of_3_dt(self):                       
        return self.inflow_delay3_delay_delay_flow_3_of_4() - self.inflow_delay3_delay_delay_flow_4_of_4()                           

    def inflow_delay3_delay_delay_stock_3_of_3_init(self):                      
        return 0 * (self.delay() / 3)                           

    def inflow_delay3_delay_delay_stock_3_of_3(self):                            
        """ Stock: inflow_delay3_delay_delay_stock_3_of_3 =                      
                 self.inflow_delay3_delay_delay_flow_3_of_4() - self.inflow_delay3_delay_delay_flow_4_of_4()                          
                                             
        Initial Value: 0 * (self.delay() / 3)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflow_delay3_delay_delay_stock_3_of_3"]              
                                             
    def outflow_delay3(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delay3_delay_delay_flow_4_of_4() 

    def dstock_delay3_dt(self):                       
        return self.inflow_delay3()-self.outflow_delay3()                           

    def stock_delay3_init(self):                      
        return 5                           

    def stock_delay3(self):                            
        """ Stock: stock_delay3 =                      
                 self.inflow_delay3()-self.outflow_delay3()                          
                                             
        Initial Value: 5                    
        Do not overwrite this function       
        """                                  
        return self.state["stock_delay3"]              
                                             
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

