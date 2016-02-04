from __future__ import division                                 
import numpy as np                                              
from pysd import functions                                      
from pysd import builder
from xray import DataArray                                     
                                                                
class Components(builder.ComponentClass):

    dim_dict = {}

    def rate_a(self):
        return DataArray([[.01,.02],[.03,.04],[.05,.06]], self.dim_dict)

    def dstock_a_dt(self):
        return self.inflow_a()

    def stock_a_init(self):
        return DataArray([[1,2],[3,4],[5,6]], self.dim_dict)
        #note, have to expand the initial condition to the full size of the stock

    def stock_a(self):
        return self.state["stock_a"]

    def inflow_a(self):
        return self.rate_a()


    def sub1d_sum(self):
        return self.stock_a().sum('Row Subscripts')

    def sub_secondary_sum(self):
        return self.sub1d_sum().sum('Column Subscripts')


    dim_dict['Column Subscripts'] = ['Column 1', 'Column 2'],
    dim_dict['Row Subscripts'] = ['Entry 1', 'Entry 2', 'Entry 3']
                                             
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
        return 1

