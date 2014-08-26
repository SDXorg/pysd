'''
August 15, 2014
James Houghton <james.p.houghton@gmail.com>
'''

import XMILE2Py as _XMILE2Py
import vensim2py as _vensim2py
from scipy.integrate import odeint as _odeint
import pandas as _pd
import networkx as _nx
import numpy as np
import functions

#version 0.0.3

def read_XMILE(XMILE_file):
    """
        This function calls the XMILE2Py module to return an execution network and model parameters
        """
    model, params = _XMILE2Py.import_XMILE(XMILE_file)
    return pysd(model, params)

def read_vensim(mdl_file):
    """
        This function calls the vensim2Py module to return an execution network and model parameters
        """
    model, params = _vensim2py.import_vensim(mdl_file)
    return pysd(model, params)


class pysd:
    def __init__(self, model, params):
        self._execution_network = model #note this fundamentally different from version 0.0.1
        self.tstart = params['tstart']
        self.tstop = params['tstop']
        self.dt = params['dt']
        self.stocknames = params['stocknames']
        self.initial_values = params['initial_values']
        self.debug=False
    
    def __str__(self):
        return 'initial values: [' + \
               ', '.join([str(key) + ':' + str(value) for key, value in zip(self.stocknames, self.initial_values)]) + \
               ']\n tstart: ' + str(self.tstart) + '\n tstop: ' + str(self.tstop) + '\n dt: ' + str(self.dt)

    def _draw_model_network(self):
        _nx.draw(self._execution_network, with_labels=True)

    def get_derivative_function_string(self, build_model='default'):
        """
            In this function we build the build_model (an execution network)
            into a valid executable python function, and return that function.
            
            We use a version of the execution network passed into the function
            instead of the class's `self.execution_network` version, because it is entirely
            likely that we are building a modified version of the xmile original,
            say if we want to change a parameter value, or fix the value of an
            intermediate auxilliary to break a loop.
            
            At some point in the future, we should remove references to self.stocknames
            so that the function is truly portable to any model. For now, referencing
            the cached versions is efficient.
            """
        
        if build_model == 'default':
            build_model = self._execution_network #dont need a copy
        
        func_string = "def dstocks_dt(stocks, t): \n\n"
        func_string += "    "+', '.join(sorted(self.stocknames))+", = stocks\n\n" #the additional comma forces it to unpack
        
        #topo sorting the graph returns the nodes in an acceptable order for execution
        for node_name in _nx.algorithms.topological_sort(build_model, reverse=True):
            if build_model.node[node_name]: #the leaves will return empty dicts - leaves are stocks
                func_string += "    "+node_name + ' = ' + str(build_model.node[node_name]['eqn'])+'\n' #dont forget to indent
        
        func_string += "\n    return ["+ ', '.join(['d'+stock+'_dt' for stock in sorted(self.stocknames)])+']'
        #caution here- this only works because the derivative names are derived from the stock names
        return func_string
    
    def _build_derivative_function(self, build_model='default'):
        """
            Separate out the string building from function building
            because it may be helpful for the user to see the function
            definition to aid in debugging
            """
        func_string = self.get_derivative_function_string(build_model)
        if self.debug:
            print func_string
        exec(func_string)
        return dstocks_dt
    
    def run(self, params={}, return_type='pandas'):
        """
        Runs the model, returning either a numpy array of stock values,
        or a pandas dataframe (with labels and timestamps).
        
        if params is set, modifies the model according to parameters
        before execution.
        
        format for params should be:
        params={'parameter1':value, 'parameter2':value}
            
        """
        #if there are modification parameters, make a new model, otherwise use the default
        run_model = self._modify_execution_network(params) if params else 'default'
        
        dstocks_dt = self._build_derivative_function(run_model)
        tseries = np.arange(self.tstart, self.tstop, self.dt)
        res = _odeint(dstocks_dt, self.initial_values, tseries)
        
        if return_type == 'numpy':
            return res
        elif return_type == 'pandas':
            return _pd.DataFrame(data=res, index=tseries, columns=self.stocknames)

    def _modify_execution_network(self, params={}):
        build_model = self._execution_network.copy()
        for key, value in params.iteritems():
            build_model.node[key]['eqn'] = value
        
        #here we may want to remove any nodes that no longer participate
        #in the calculation, which may happen if fixing a value breaks a
        #loop. This would basically be for execution speed.
        
        #check that the model is still valid
        self._validate_execution_graph(build_model)
        
        return build_model
    
    def _validate_execution_graph(self, components):
        """
            if its a well formed set of equations, it should be a directed acyclic graph
            there may be other checks that we want to run in the future
            """
        assert _nx.algorithms.is_directed_acyclic_graph(components)
    
    def get_free_parameters(self):
        """
            return the components with no dependencies, that are not stocks.
            these are parameters that can be modified without changing the model structurally.
            """
        param_name_list = [node for node, out_degree in self._execution_network.out_degree_iter() \
                           if out_degree == 0 and node not in self.stocknames]
        return dict([(nodename, self._execution_network.node[nodename]['eqn']) for nodename in param_name_list])
