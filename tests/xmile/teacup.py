from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def maxlen(self):
        """maxlen = 50 
        Type: Flow or Auxiliary 
         """
        return 50 

    def tdiff(self):
        """tdiff = self.kfac()*(self.rmtmp()-self.cuptemp()) 
        Type: Flow or Auxiliary 
         """
        return self.kfac()*(self.rmtmp()-self.cuptemp()) 

    def rmtmp(self):
        """rmtmp = 68 
        Type: Flow or Auxiliary 
         """
        return 68 

    def kfac(self):
        """kfac = .17 
        Type: Flow or Auxiliary 
         """
        return .17 

    def dcuptemp_dt(self):                       
        return self.tdiff()                           

    def cuptemp_init(self):                      
        return 170                           

    def cuptemp(self):                            
        """ Stock: cuptemp =                      
                 self.tdiff()                          
                                             
        Initial Value: 170                    
        Do not overwrite this function       
        """                                  
        return self.state["cuptemp"]              
                                             
    def initial_time(self):
        """initial_time = 0.0 
        Type: Flow or Auxiliary 
         """
        return 0.0 

    def final_time(self):
        """final_time = 50.0 
        Type: Flow or Auxiliary 
         """
        return 50.0 

    def time_step(self):
        """time_step = 0.25 
        Type: Flow or Auxiliary 
         """
        return 0.25 

