"""builder.py
Modified June 26 2015
James Houghton
james.p.houghton@gmail.com

This submodule contains everything that is needed to construct a system dynamics model
in python, using syntax that is compatible with the pysd model simulation functionality.

These functions could be used to construct a system dynamics model from scratch, in a
pinch. Due to the highly visual structure of system dynamics models, I still recommend
using an external model construction tool such as vensim or stella/iThink to build and
debug models.
"""

# Todo:
# - the __doc__ attribute isn't something that you can call, need to rework it,
# along with the __str_ method
# - make the doc function accept a 'short' argument.


import re
import keyword
from componentClass import ComponentClass

def new_model(filename):
    """Starts a new python file (saved under `filename`) where translations from
    the vensim model file or XMILE model file will be implemented as a python class.
    """
    string = ('from __future__ import division                                 \n' +
              'import numpy as np                                              \n' +
              'from pysd import functions                                      \n' +
              'from pysd import builder                                        \n' +
              '                                                                \n' +
              'class Components(builder.ComponentClass):                       \n' +
              '                                                                \n'
             )
    with open(filename, 'w') as outfile: #the 'w' setting gives us a clean slate.
        outfile.write(string)


def add_stock(filename, identifier, expression, initial_condition):
    """Adds a stock to the python model file based upon the interpreted expressions
    for the initial condition.

    identifier: <string> valid python identifier
        Our translators are responsible for translating the model identifiers into
        something that python can use as a function name.

    expression: <string>
        Note that these expressions will be added as a function body within the model class.
        They need to be written with with appropriate syntax, ie:
        `self.functioncall() * self.otherfunctioncall()`

    initial_condition: <string>
        An expression that defines the value that the stock should take on when the model
        is initialized. This may be a constant, or a call to other model elements.

    """
    #create a 'derivative function' that can be
    # called by the d_dt boilerplate function and passed to the integrator
    dfuncstr = ('    def d%s_dt(self):                       \n'%identifier +
                '        return %s                           \n\n'%expression
               )

    #create an 'intialization function' of the form '<stock>_init()' that
    # can be called when the model is reset to initialize the state variable
    ifuncstr = ('    def %s_init(self):                      \n'%identifier +
                '        return %s                           \n\n'%initial_condition
               )

    #create a function that points to the state dictionary, to let other
    # components reference the state without explicitly having to know that
    # it is a stock. This is the function that gets an elaborated docstring
    sfuncstr = ('    def %s(self):                            \n'%identifier +
                '        """ Stock: %s =                      \n'%identifier +
                '                 %s                          \n'%expression +
                '                                             \n' +
                '        Initial Value: %s                    \n'%initial_condition +
                '        Do not overwrite this function       \n' +
                '        """                                  \n' +
                '        return self.state["%s"]              \n'%identifier +
                '                                             \n'
               )

    with open(filename, 'a') as outfile:
        outfile.write(dfuncstr)
        outfile.write(ifuncstr)
        outfile.write(sfuncstr)



def add_flaux(filename, identifier, expression, doc=''):
    """Adds a flow or auxiliary element to the model.

    identifier: <string> valid python identifier
        Our translators are responsible for translating the model identifiers into
        something that python can use as a function name.

    expression: <string>
        Note that these expressions will be added as a function body within the model class.
        They need to be written with with appropriate syntax, ie:
        `self.functioncall() * self.otherfunctioncall()`
    """
    docstring = ('Type: Flow or Auxiliary\n        '+
                 '\n        '.join(doc.split('\n')))

    funcstr = ('    def %s(self):\n'%identifier +
               '        """%s"""\n'%docstring +
               '        return %s \n\n'%expression)

    with open(filename, 'a') as outfile:
        outfile.write(funcstr)

    return 'self.%s()'%identifier


def add_lookup(filename, identifier, valid_range, copair_list):
    """Constructs a function that implements a lookup.
    The function encodes the coordinate pairs as numeric values in the python file.

    identifier: <string> valid python identifier
        Our translators are responsible for translating the model identifiers into
        something that python can use as a function name.

    range: tuple
        Minimum and maximum bounds on the lookup. Currently, we don't do anything
        with this, but in the future, may use it to implement some error checking.

    copair_list: a list of tuples, eg. [(0, 1), (1, 5), (2, 15)]
        The coordinates of the lookup formatted in coordinate pairs.

    """
    # in the future, we may want to check in bounds for the range. for now, lazy...
    xs, ys = zip(*copair_list)
    xs_str = str(list(xs))
    ys_str = str(list(ys))

    #warning: this may create a class attribute of the function for the lookups, when what we want
    # is an instance attribute. Not sure...

    funcstr = ('    def %s(self, x):                                      \n'%identifier +
               '        return self.functions.lookup(x,                   \n' +
               '                                     self.%s.xs,          \n'%identifier +
               '                                     self.%s.ys)          \n'%identifier +
               '                                                          \n' +
               '    %s.xs = %s                                            \n'%(identifier, xs_str) +
               '    %s.ys = %s                                            \n'%(identifier, ys_str) +
               '                                                          \n'
              )

    with open(filename, 'a') as outfile:
        outfile.write(funcstr)


def add_initial(filename, component):
    """ Implement vensim's `INITIAL` command as a build-time function.
        component cannot be a full expression, must be a reference to
        a single external element.
    """

#        This is a bit of a weird construction. For a simpler demo, try:
#        class test:
#            def method(self, inval):
#                if not hasattr(self.method, 'value'):
#                    self.method.im_func.value = inval
#                return self.method.value
#
#        a = test()
#        print a.method('hi')
#        print a.method('there')

    naked_component = component[5:-2]
    funcstr = ('    def initial_%s(self, inval):                  \n'%naked_component +
               '        if not hasattr(self.initial_%s, "value"): \n'%naked_component +
               '            self.initial_%s.im_func.value = inval \n'%naked_component +
               '        return self.initial_%s.value             \n\n'%naked_component
              )
    with open(filename, 'a') as outfile:
        outfile.write(funcstr)

    return 'self.initial_%s(%s)'%(naked_component, component)



def add_n_delay(filename, delay_input, delay_time, initial_value, order):
    """Constructs stock and flow chains that implement the calculation of
    a delay.

    delay_input: <string>
        Reference to the model component that is the input to the delay

    delay_time: <string>
        Can be a number (in string format) or a reference to another model element
        which will calculate the delay. This is calculated throughout the simulation
        at runtime.

    initial_value: <string>
        This is used to initialize the stocks that are present in the delay. We
        initialize the stocks with equal values so that the outflow in the first
        timestep is equal to this value.

    order: int
        The number of stocks in the delay pipeline. As we construct the delays at
        build time, this must be an integer and cannot be calculated from other
        model components. Anything else will yield a ValueError.

    """
    try:
        order = int(order)
    except ValueError:
        print "Order of delay must be an int. (Can't even be a reference to an int. Sorry...)"
        raise

    #depending in these cases on input to be formatted as 'self.variable()' (or number)
    naked_input = delay_input[5:-2]
    naked_delay = delay_time[5:-2] if delay_time[:5] == 'self.' else delay_time
    delay_name = '%s_delay_%s'%(naked_input, naked_delay)


    flowlist = []
    #use 1-based indexing for stocks in the delay chain so that (n of m) makes sense.
    flowlist.append(add_flaux(filename,
                              identifier='%s_flow_1_of_%i'%(delay_name, order+1),
                              expression=delay_input))

    for i in range(2, order+2):
        flowlist.append(add_flaux(filename,
                                  identifier='%s_flow_%i_of_%i'%(delay_name, i, order+1),
                                  expression='self.%s_stock_%i_of_%i()/(1.*%s/%i)'%(
                                              delay_name, i-1, order, delay_time, order)))

    for i in range(1, order+1):
        add_stock(filename,
                  identifier='%s_stock_%i_of_%i'%(delay_name, i, order),
                  expression=flowlist[i-1]+' - '+flowlist[i],
                  initial_condition='%s * (%s / %i)'%(initial_value, delay_time, order))

    return flowlist[-1]



def add_n_smooth(filename, smooth_input, smooth_time, initial_value, order):
    """Constructs stock and flow chains that implement the calculation of
    a smoothing function.

    delay_input: <string>
        Reference to the model component that is the input to the smoothing function

    smooth_time: <string>
        Can be a number (in string format) or a reference to another model element
        which will calculate the delay. This is calculated throughout the simulation
        at runtime.

    initial_value: <string>
        This is used to initialize the stocks that are present in the delay. We
        initialize the stocks with equal values so that the outflow in the first
        timestep is equal to this value.

    order: int
        The number of stocks in the delay pipeline. As we construct the delays at
        build time, this must be an integer and cannot be calculated from other
        model components. Anything else will yield a ValueError.

    """
    try:
        order = int(order)
    except ValueError:
        print "Order of delay must be an int. (Can't even be a reference to an int. Sorry...)"
        raise

         #depending in these cases on input to be formatted as 'self.smooth_input()' (or number)
    naked_input = smooth_input[5:-2]
    naked_smooth = smooth_time[5:-2] if smooth_time[:5] == 'self.' else smooth_time
    smooth_name = '%s_smooth_%s'%(naked_input, naked_smooth)


    flowlist = []
    #use 1-based indexing for stocks in the delay chain so that (n of m) makes sense.
    prev = smooth_input
    current = 'self.%s_stock_1_of_%i()'%(smooth_name, order)
    flowlist.append(add_flaux(filename,
                              identifier='%s_flow_1_of_%i'%(smooth_name, order),
                              expression='(%s - %s) / (1.*%s/%i)'%(
                                           prev, current, smooth_time, order)))

    for i in range(2, order+1):
        prev = 'self.%s_stock_%i_of_%i()'%(smooth_name, i-1, order)
        current = 'self.%s_stock_%i_of_%i()'%(smooth_name, i, order)
        flowlist.append(add_flaux(filename,
                                  identifier='%s_flow_%i_of_%i'%(smooth_name, i, order),
                                  expression='(%s - %s)/(1.*%s/%i)'%(
                                              prev, current, smooth_time, order)))

    for i in range(1, order+1):
        add_stock(filename,
                  identifier='%s_stock_%i_of_%i'%(smooth_name, i, order),
                  expression=flowlist[i-1],
                  initial_condition=initial_value)

    return flowlist[-1]



def make_python_identifier(string):
    """Takes an arbitrary string and creates a valid Python identifier.
    
    Identifiers must follow the convention outlined here:
    https://docs.python.org/2/reference/lexical_analysis.html#identifiers
    """
    #Todo:
    # It is possible that this will create non-unique variable names that we'll
    # need to deal with

    string = string.lower()

    # remove leading and trailing whitespace
    string = string.strip()

    # Make spaces into underscores
    string = string.replace(' ', '_')

    # Remove invalid characters
    string = re.sub('[^0-9a-zA-Z_]', '', string)

    # Remove leading characters until we find a letter or underscore
    string = re.sub('^[^a-zA-Z_]+', '', string)

    if string in keyword.kwlist:
        string += '_element'

    return string
