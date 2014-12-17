'''
    created: August 15, 2014
    last update: September 2, 2014
    James Houghton <james.p.houghton@gmail.com>
    '''

import translators as _translators
from scipy.integrate import odeint as _odeint
import pandas as _pd
import networkx as _nx
import numpy as np
import functions

#version 0.0.4

def read_XMILE(XMILE_file):
    """
        As of pysd v0.1.0, we only support a subset of the full XMILE standard.
        Supported functions include:
        """
    
    model, params = _translators.import_XMILE(XMILE_file)
    return pysd(model, params)

def read_vensim(mdl_file):
    """
        As of pysd v0.1.0, we only support a subset of Vensim's capabilities.
        Supported functions include:
        """
    model, params = _translators.import_vensim(mdl_file)
    return pysd(model, params)


class pysd:
    def __init__(self, model, params):
        self._execution_network = model
        self.tstart = params['tstart']
        self.tstop = params['tstop']
        self.dt = params['dt']
        self.stocknames = params['stocknames'] #this should come to us sorted
        self.debug=False
        self.stock_values = 'Run model first'
        if params['initial_values'][0]:
            self.initial_values = params['initial_values']
        else:
            self.initial_values = self._get_initial_values(self.stocknames)
    
    def __str__(self):
        string  = '\n'.join(self._stringify_necessary_equations())
        string += '\n\ninitial values: [' + \
            ', '.join([str(key) + ':' + str(value) for key, value in zip(self.stocknames, self.initial_values)]) + \
            ']\ntstart: ' + str(self.tstart) + '\ntstop: ' + str(self.tstop) + '\ndt: ' + str(self.dt)
        
        return string
    
    def _get_initial_values(self, stocknames):
        init_values_func = self._build_model_function(elements=[stock+'_init' for stock in stocknames])
        return init_values_func(stocknames, 0)
    
    def _draw_model_network(self):
        _nx.draw(self._execution_network, with_labels=True)
    
    
    def _stringify_necessary_equations(self, elements=[]):
        """
            Return a sorted list of strings of equations necessary to calculate elements.
            
            if elements left as empty list, returns full network.
            
            Want to take a slice of the network containing only the elements and
            the components that the elements depend on, and then sort that slice
            """
        if elements: #if the function is called for a subset of the model, take the right subgraph
            relevant_nodes = set(elements)
            for element in elements:
                relevant_nodes = relevant_nodes.union(_nx.descendants(self._execution_network, element))
            subgraph = _nx.subgraph(self._execution_network, list(relevant_nodes)) #if this turns out to be an empty set, returns all nodes.
        else:
            subgraph = self._execution_network
        
        #topo sorting the graph returns the nodes in an acceptable order for execution
        string_list = []
        for node_name in _nx.algorithms.topological_sort(subgraph, reverse=True):
            if subgraph.node[node_name]: #the leaves will return empty dicts - leaves are stocks
                string_list.append(node_name + ' = ' + str(subgraph.node[node_name]['eqn']) )
        
        return string_list
    
    
    def _build_model_function(self, elements=[]):
        """
            We may even pre-compute elements which are not dependent on stock values
            ie. only elements that are ancestors of (ie, depend on) stocks get included, while
            the remaining values get precomputed.
            """
        num_stocks = len(self.stocknames)
        func_string = "def model_function(stocks, t): \n\n"
        #the additional comma forces it to unpack even single stocks
        #otherwise, they would remain in the list.
        func_string += "    "+', '.join(sorted(self.stocknames)) + ','
        func_string += " = stocks\n\n"
        for string in self._stringify_necessary_equations(elements):
            func_string += "    " + string + "\n"
        
        outputs = elements #['d'+stock+'_dt' for stock in sorted(self.stocknames)]
        func_string += "\n    return ["+ ', '.join(outputs)+']'
        
        if self.debug:
            print func_string
        
        exec(func_string)
        return model_function
    
    
    def run(self, params={}, return_type='pandas', return_columns=[]):
        """
            Runs the model, returning either a numpy array of stock values,
            or a pandas dataframe (with labels and timestamps).
            
            if params is set, modifies the model according to parameters
            before execution.
            
            format for params should be:
            params={'parameter1':value, 'parameter2':value}
            
            return_type can be: 'pandas' or 'numpy' or 'none'
            """
        if params:
            self.set_eqn(params)
        
        # we need our function to only return the derivatives, and in the correct order!
        elements = ['d'+stock+'_dt' for stock in sorted(self.stocknames)]
        dstocks_dt = self._build_model_function(elements)
        tseries = np.arange(self.tstart, self.tstop, self.dt)
        res = _odeint(dstocks_dt, self.initial_values, tseries, hmax=self.dt)
        
        self.stock_values = _pd.DataFrame(data=res, index=tseries, columns=self.stocknames)
        
        
        if return_type == 'numpy':
            return res
        elif return_type == 'pandas':
            return self.stock_values
        elif return_type == 'none':
            pass #we might do this if we're planning to use the 'measure' function to get most of our results
    
    def measure(self, elements, timestamps):
        """
            This function lets us come back and measure values that we didn't return during
            the run. This is necessary because scipy's odeint only lets you return derivatives
            of values to the function, and its complicated to find extra ways
            to return additional values from the derivatives calculation function.
            
            This also lets us sample at arbitrary times, which may not have been included in the original tseries
            I'm not sure if this is a good way to deal with these issues, as it requires
            recomputing values, the interpolation is inefficient, and the
            
            This still has issues with duplicate values, etc
            Also, this code is really ugly
            """
        
        #this part does a weird interpolation to estimate the stock values at the given times
        ts = _pd.DataFrame(index=list(set(timestamps)-set(self.stock_values.index)), columns=self.stock_values.columns)
        #the set arithmetic means we should only add values that arent in the dataset
        # need to work out why we have duplicate timestamps AND EXTERMINATE THEM
        lookup_stocks = _pd.concat([self.stock_values, ts]).sort_index().interpolate().loc[timestamps]
        
        
        stock_elements = [element for element in elements if element in self.stocknames]
        return_stocks = lookup_stocks[stock_elements]
        
        non_stock_elements = [element for element in elements if element not in self.stocknames]
        model_function = self._build_model_function(non_stock_elements)
        measurement_list = [] #this is awful, there has to be a more elegant solution
        for index, stocks in lookup_stocks.iterrows():
            measurement_list.append(dict(zip(non_stock_elements, model_function(stocks[self.stocknames].values, index))))
        
        return_non_stocks =_pd.DataFrame(measurement_list, index=lookup_stocks.index)

        return return_non_stocks.join(return_stocks)
    
    
    def set_eqn(self, params={}):
        """
           sets the equation of a model element matching the dictionary 
           key to the dictionary value:
           
           set({'delay':4})
           
        """
        for key, value in params.iteritems():
            self._execution_network.node[key]['eqn'] = value
        
        self._validate_execution_graph(self._execution_network)
    
    
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
