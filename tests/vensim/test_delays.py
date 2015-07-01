from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def inflowa(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.step(10, 5) 

    def inflow(self):
        """Type: Flow or Auxiliary
        """
        return self.functions.step(5 , 10 ) 

    def inflowa_delay_delay_flow_1_of_7(self):
        """Type: Flow or Auxiliary
        """
        return self.inflowa() 

    def inflowa_delay_delay_flow_2_of_7(self):
        """Type: Flow or Auxiliary
        """
        return self.inflowa_delay_delay_stock_1_of_6()/(1.*self.delay()/6) 

    def inflowa_delay_delay_flow_3_of_7(self):
        """Type: Flow or Auxiliary
        """
        return self.inflowa_delay_delay_stock_2_of_6()/(1.*self.delay()/6) 

    def inflowa_delay_delay_flow_4_of_7(self):
        """Type: Flow or Auxiliary
        """
        return self.inflowa_delay_delay_stock_3_of_6()/(1.*self.delay()/6) 

    def inflowa_delay_delay_flow_5_of_7(self):
        """Type: Flow or Auxiliary
        """
        return self.inflowa_delay_delay_stock_4_of_6()/(1.*self.delay()/6) 

    def inflowa_delay_delay_flow_6_of_7(self):
        """Type: Flow or Auxiliary
        """
        return self.inflowa_delay_delay_stock_5_of_6()/(1.*self.delay()/6) 

    def inflowa_delay_delay_flow_7_of_7(self):
        """Type: Flow or Auxiliary
        """
        return self.inflowa_delay_delay_stock_6_of_6()/(1.*self.delay()/6) 

    def dinflowa_delay_delay_stock_1_of_6_dt(self):                       
        return self.inflowa_delay_delay_flow_1_of_7() - self.inflowa_delay_delay_flow_2_of_7()                           

    def inflowa_delay_delay_stock_1_of_6_init(self):                      
        return 10  * (self.delay() / 6)                           

    def inflowa_delay_delay_stock_1_of_6(self):                            
        """ Stock: inflowa_delay_delay_stock_1_of_6 =                      
                 self.inflowa_delay_delay_flow_1_of_7() - self.inflowa_delay_delay_flow_2_of_7()                          
                                             
        Initial Value: 10  * (self.delay() / 6)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflowa_delay_delay_stock_1_of_6"]              
                                             
    def dinflowa_delay_delay_stock_2_of_6_dt(self):                       
        return self.inflowa_delay_delay_flow_2_of_7() - self.inflowa_delay_delay_flow_3_of_7()                           

    def inflowa_delay_delay_stock_2_of_6_init(self):                      
        return 10  * (self.delay() / 6)                           

    def inflowa_delay_delay_stock_2_of_6(self):                            
        """ Stock: inflowa_delay_delay_stock_2_of_6 =                      
                 self.inflowa_delay_delay_flow_2_of_7() - self.inflowa_delay_delay_flow_3_of_7()                          
                                             
        Initial Value: 10  * (self.delay() / 6)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflowa_delay_delay_stock_2_of_6"]              
                                             
    def dinflowa_delay_delay_stock_3_of_6_dt(self):                       
        return self.inflowa_delay_delay_flow_3_of_7() - self.inflowa_delay_delay_flow_4_of_7()                           

    def inflowa_delay_delay_stock_3_of_6_init(self):                      
        return 10  * (self.delay() / 6)                           

    def inflowa_delay_delay_stock_3_of_6(self):                            
        """ Stock: inflowa_delay_delay_stock_3_of_6 =                      
                 self.inflowa_delay_delay_flow_3_of_7() - self.inflowa_delay_delay_flow_4_of_7()                          
                                             
        Initial Value: 10  * (self.delay() / 6)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflowa_delay_delay_stock_3_of_6"]              
                                             
    def dinflowa_delay_delay_stock_4_of_6_dt(self):                       
        return self.inflowa_delay_delay_flow_4_of_7() - self.inflowa_delay_delay_flow_5_of_7()                           

    def inflowa_delay_delay_stock_4_of_6_init(self):                      
        return 10  * (self.delay() / 6)                           

    def inflowa_delay_delay_stock_4_of_6(self):                            
        """ Stock: inflowa_delay_delay_stock_4_of_6 =                      
                 self.inflowa_delay_delay_flow_4_of_7() - self.inflowa_delay_delay_flow_5_of_7()                          
                                             
        Initial Value: 10  * (self.delay() / 6)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflowa_delay_delay_stock_4_of_6"]              
                                             
    def dinflowa_delay_delay_stock_5_of_6_dt(self):                       
        return self.inflowa_delay_delay_flow_5_of_7() - self.inflowa_delay_delay_flow_6_of_7()                           

    def inflowa_delay_delay_stock_5_of_6_init(self):                      
        return 10  * (self.delay() / 6)                           

    def inflowa_delay_delay_stock_5_of_6(self):                            
        """ Stock: inflowa_delay_delay_stock_5_of_6 =                      
                 self.inflowa_delay_delay_flow_5_of_7() - self.inflowa_delay_delay_flow_6_of_7()                          
                                             
        Initial Value: 10  * (self.delay() / 6)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflowa_delay_delay_stock_5_of_6"]              
                                             
    def dinflowa_delay_delay_stock_6_of_6_dt(self):                       
        return self.inflowa_delay_delay_flow_6_of_7() - self.inflowa_delay_delay_flow_7_of_7()                           

    def inflowa_delay_delay_stock_6_of_6_init(self):                      
        return 10  * (self.delay() / 6)                           

    def inflowa_delay_delay_stock_6_of_6(self):                            
        """ Stock: inflowa_delay_delay_stock_6_of_6 =                      
                 self.inflowa_delay_delay_flow_6_of_7() - self.inflowa_delay_delay_flow_7_of_7()                          
                                             
        Initial Value: 10  * (self.delay() / 6)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflowa_delay_delay_stock_6_of_6"]              
                                             
    def outflowa(self):
        """Type: Flow or Auxiliary
        """
        return self.inflowa_delay_delay_flow_7_of_7() 

    def dstocka_dt(self):                       
        return self.inflowa()-self.outflowa()                           

    def stocka_init(self):                      
        return 0                           

    def stocka(self):                            
        """ Stock: stocka =                      
                 self.inflowa()-self.outflowa()                          
                                             
        Initial Value: 0                    
        Do not overwrite this function       
        """                                  
        return self.state["stocka"]              
                                             
    def delay(self):
        """Type: Flow or Auxiliary
        """
        return 10 

    def inflow_delay_delay_flow_1_of_4(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow() 

    def inflow_delay_delay_flow_2_of_4(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delay_delay_stock_1_of_3()/(1.*self.delay()/3) 

    def inflow_delay_delay_flow_3_of_4(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delay_delay_stock_2_of_3()/(1.*self.delay()/3) 

    def inflow_delay_delay_flow_4_of_4(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delay_delay_stock_3_of_3()/(1.*self.delay()/3) 

    def dinflow_delay_delay_stock_1_of_3_dt(self):                       
        return self.inflow_delay_delay_flow_1_of_4() - self.inflow_delay_delay_flow_2_of_4()                           

    def inflow_delay_delay_stock_1_of_3_init(self):                      
        return 0 * (self.delay() / 3)                           

    def inflow_delay_delay_stock_1_of_3(self):                            
        """ Stock: inflow_delay_delay_stock_1_of_3 =                      
                 self.inflow_delay_delay_flow_1_of_4() - self.inflow_delay_delay_flow_2_of_4()                          
                                             
        Initial Value: 0 * (self.delay() / 3)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflow_delay_delay_stock_1_of_3"]              
                                             
    def dinflow_delay_delay_stock_2_of_3_dt(self):                       
        return self.inflow_delay_delay_flow_2_of_4() - self.inflow_delay_delay_flow_3_of_4()                           

    def inflow_delay_delay_stock_2_of_3_init(self):                      
        return 0 * (self.delay() / 3)                           

    def inflow_delay_delay_stock_2_of_3(self):                            
        """ Stock: inflow_delay_delay_stock_2_of_3 =                      
                 self.inflow_delay_delay_flow_2_of_4() - self.inflow_delay_delay_flow_3_of_4()                          
                                             
        Initial Value: 0 * (self.delay() / 3)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflow_delay_delay_stock_2_of_3"]              
                                             
    def dinflow_delay_delay_stock_3_of_3_dt(self):                       
        return self.inflow_delay_delay_flow_3_of_4() - self.inflow_delay_delay_flow_4_of_4()                           

    def inflow_delay_delay_stock_3_of_3_init(self):                      
        return 0 * (self.delay() / 3)                           

    def inflow_delay_delay_stock_3_of_3(self):                            
        """ Stock: inflow_delay_delay_stock_3_of_3 =                      
                 self.inflow_delay_delay_flow_3_of_4() - self.inflow_delay_delay_flow_4_of_4()                          
                                             
        Initial Value: 0 * (self.delay() / 3)                    
        Do not overwrite this function       
        """                                  
        return self.state["inflow_delay_delay_stock_3_of_3"]              
                                             
    def outflow(self):
        """Type: Flow or Auxiliary
        """
        return self.inflow_delay_delay_flow_4_of_4() 

    def dstock_dt(self):                       
        return self.inflow()-self.outflow()                           

    def stock_init(self):                      
        return 5                           

    def stock(self):                            
        """ Stock: stock =                      
                 self.inflow()-self.outflow()                          
                                             
        Initial Value: 5                    
        Do not overwrite this function       
        """                                  
        return self.state["stock"]              
                                             
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

