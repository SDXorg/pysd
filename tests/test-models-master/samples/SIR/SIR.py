from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder                                        
                                                                
class Components(builder.ComponentClass):                       
                                                                
    def succumbing(self):
        """Type: Flow or Auxiliary
        """
        return self.susceptible()*self.infectious()/self.total_population()*self.contact_infectivity() 

    def recovering(self):
        """Type: Flow or Auxiliary
        """
        return self.infectious()/self.duration() 

    def total_population(self):
        """Type: Flow or Auxiliary
        """
        return 1000 

    def duration(self):
        """Type: Flow or Auxiliary
        """
        return 5 

    def contact_infectivity(self):
        """Type: Flow or Auxiliary
        """
        return 0.3 

    def dsusceptible_dt(self):                       
        return  - self.succumbing()                           

    def susceptible_init(self):                      
        return self.total_population()                           

    def susceptible(self):                            
        """ Stock: susceptible =                      
                  - self.succumbing()                          
                                             
        Initial Value: self.total_population()                    
        Do not overwrite this function       
        """                                  
        return self.state["susceptible"]              
                                             
    def dinfectious_dt(self):                       
        return self.succumbing() - self.recovering()                           

    def infectious_init(self):                      
        return 5                           

    def infectious(self):                            
        """ Stock: infectious =                      
                 self.succumbing() - self.recovering()                          
                                             
        Initial Value: 5                    
        Do not overwrite this function       
        """                                  
        return self.state["infectious"]              
                                             
    def drecovered_dt(self):                       
        return self.recovering()                           

    def recovered_init(self):                      
        return 0                           

    def recovered(self):                            
        """ Stock: recovered =                      
                 self.recovering()                          
                                             
        Initial Value: 0                    
        Do not overwrite this function       
        """                                  
        return self.state["recovered"]              
                                             
    def initial_time(self):
        """Type: Flow or Auxiliary
        """
        return 0 

    def final_time(self):
        """Type: Flow or Auxiliary
        """
        return 100 

    def time_step(self):
        """Type: Flow or Auxiliary
        """
        return 0.03125 

