# This is the base class that gets built up into a model by addition of functions.
# A description of how this process works can be found here:
#    http://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object

# Todo:
# - the __doc__ attribute isn't something that you can call, need to rework it, along with the __str_ method
# it would be nice to dynamically generate the docstring (or some sortof 'help' string)
# so that if the model is modified, the user can see the current state of modifications

import inspect
from pysd import functions

class ComponentClass:
    """
    This is a template class to be subclassed and fleshed out by the translation tools.
    
    A function should be added for every flow or auxiliary variable, having the name of that 
    variable, taking no parameters, which calculates the value of the variable.
    
    A function should be added for each stock called d<stockname>_dt() which calculates
    the net inflow to the stock as a function of the flows.
    
    A function should be added for each stock called <stockname>_init() which calculates
    the (potentially dynamically generated) initial value for the stock.
    
    This docstring will be rewritten and dynamically generated.
    
    Example:
    --------
    >>> class Components(builder.ComponentClass):
    >>>     __str__ = 'Undefined'
    >>>     _stocknames = []
    >>>     _dfuncs = []
    
    """
    
    def __init__(self):
        self.state = dict(zip(self._stocknames, [None]*len(self._stocknames) ))
        self.reset_state()
        self.__doc__ = self.doc()
    
    
    def doc(self):
        #docstring = self.__str__ + '\n\n'
        docstring = ''
        for method in dir(self):
            if method not in ['__doc__', '__init__', '__module__', '__str__', 't',
                              'd_dt', 'reset_state', 'model_description', 'saveper',
                              'state', 'state_vector']:
                if method[0] not in ['_']:
                    try:
                        docstring +=  inspect.getdoc(getattr(self, method)) + '\n\n'
                    except:
                        pass

        return docstring
    
    def initial_time(self):
        """
            This function represents the initial time as set in the model file
            It should be overwritten during the class extension.
        """
        return 0


    def reset_state(self):
        self.t = self.initial_time() #set the initial time
        
        for key in self.state.keys():
            self.state[key] = eval('self.'+key+'_init()') #set the initial state


    def d_dt(self, state_vector, t):
        """
            The primary purpose of this function is to interact with the integrator.
            It takes a state vector, sets the state of the system based on that vector,
            and returns a derivative of the state vector
            """
        state = dict(zip(self._stocknames, state_vector))
        self.state.update(state)
        self.t = t
        
        return [func(self) for func in self._dfuncs]
    
    
    
    def state_vector(self):
        """
            This function interacts with the integrator by setting the 
            initial values of the integrator to the state vector.
            
            It returns the values of the state dictionary, sorted 
            alphabetically by key.
            """
        return [self.state[key] for key in self._stocknames]


    def time(self):
        """
            This helper function allows the model components to 
            access the time component directly.
        """
        return self.t


def add_stock(component_class, identifier, expression, initial_condition):
    #Add the identifier to the list of stocks, in order
    component_class._stocknames.append(identifier)
    component_class._stocknames.sort()
    index = component_class._stocknames.index(identifier)
    
    #create a 'derivative function' that can be
    # called by the d_dt boilerplate function and passed to the integrator
    funcstr = ('def d%s_dt(self):\n'%identifier +
               '    return %s'%expression   )
    exec funcstr in component_class.__dict__
         
    #create an 'intialization function' of the form '<stock>_init()' that 
    # can be called when the model is reset to initialize the state variable
    funcstr = ('def %s_init(self):\n'%identifier +
               '    return %s'%initial_condition)
    exec funcstr in component_class.__dict__
    
    #create a function that points to the state dictionary, to let other
    # components reference the state without explicitly having to know that
    # it is a stock. This is the function that gets an elaborated docstring
    funcstr = ('def %s(self):\n'%identifier +
               '    """    %s = %s \n'%(identifier, expression) + #include the docstring
               '        Initial Value: %s \n'%initial_condition +
               '        Type: Stock \n' +
               '        Do not overwrite this function\n' +
               '    """\n' +
               '    return self.state["%s"]'%identifier)
    exec funcstr in component_class.__dict__
    component_class._dfuncs.insert(index, getattr(component_class, 'd%s_dt'%identifier))



def add_flaux(component_class, identifier, expression):

    funcstr = ('def    %s(self):\n'%identifier +
               '    """%s = %s \n'%(identifier, expression) +
               '       Type: Flow or Auxiliary \n ' +
               '    """\n' +
               '    return %s'%expression)
    exec funcstr in component_class.__dict__

def add_to_element_docstring(component_class, identifier, string):
    
        entry = getattr(component_class, identifier)
        
        if hasattr(entry, 'im_func'): #most functions
            entry.im_func.func_doc += string
        else: #the lookups - which are represented as callable classes, instead of functions
            entry.__doc__ += string

def add_lookup(component_class, identifier, range, copair_list):
    # in the future, we may want to check in bounds for the range. for now, lazy...
    xs, ys = zip(*copair_list)
    lookup_func = functions.lookup(xs, ys)
    lookup_func.__doc__ = \
            ('%s is lookup with coordinates:\n%s'%(identifier, copair_list) +
             'Type: Flow or Auxiliary \n')
    
    component_class.__dict__.update({identifier:lookup_func})

    


