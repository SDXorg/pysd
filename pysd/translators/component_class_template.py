# This is the base class that gets built up into a model by addition of functions.
# A description of how this process works can be found here:
#    http://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object

# Todo:
# - the __doc__ attribute isn't something that you can call, need to rework it, along with the __str_ method
#
#
#
#
#
#
#
#
#
#
#
#




class component_class_template:

    def __init__(self):
        self.reset_state()
    
    def __doc__(self):
        docstring = self.__str__ + '\n\n'
        for method in dir(self):
            if method not in ['__doc__', '__init__', '__module__', '__str__', 't',
                              'd_dt', 'reset_state', 'model_description', 'saveper',
                              'state', 'state_vector']:
                if method[0] not in ['_']:
                    try:
                        docstring += method + '\n' + inspect.getdoc(getattr(self, method)) + '\n\n'
                    except:
                        pass

        return docstring

    def reset_state(self):
        self.t = self.initial_time() #set the initial time
        
        for key in self.state.keys():
            self.state[key] = eval('self._'+key+'_init()') #set the initial state
        pass
    
    def d_dt(self, state_vector, t):
        """
            The primary purpose of this function is to interact with the integrator.
            It takes a state vector, sets the state of the system based on that vector,
            and returns a derivative of the state vector
            """
        sorted_keys = sorted(self.state.keys())
        
        for key, newval in zip(sorted_keys,state_vector):
            self.state[key] = newval
        
        self.t = t
        
        return [eval('self.d'+key+'_dt()') for key in sorted_keys]
    
    
    def state_vector(self):
        """
            This function interacts with the integrator by setting the initial values of the state vector.
            It returns the values of the state dictionary, sorted alphabetically by key
            """
        return [self.state[key] for key in sorted(self.state.keys())]
