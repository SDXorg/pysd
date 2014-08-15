'''
August 15, 2014
James Houghton <james.p.houghton@gmail.com>
'''

import XMILE2Py
from scipy.integrate import odeint
import pandas as pd
import numpy as np

def read_XMILE(XMILE_file):
        model, params = XMILE2Py.import_XMILE(XMILE_file)
        return PySD(model, params)


class PySD:
    def __init__(self, model, params):
        self.model = model
        self.tstart = params['tstart']
        self.tstop = params['tstop']
        self.dt = params['dt']
        self.stocknames = params['stocknames']
        self.initial_values = params['initial_values']
        
    def run(self):
        tseries = np.arange(self.tstart, self.tstop, self.dt)
        res = odeint(self.model, self.initial_values, tseries)
        return res
    
    def run_pandas(self):
        tseries = np.arange(self.tstart, self.tstop, self.dt)
        res = odeint(self.model, self.initial_values, tseries)
        return pd.DataFrame(data=res, index=tseries, columns=self.stocknames)