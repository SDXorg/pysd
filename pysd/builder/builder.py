"""builder.py
Modified June 26 2015
James Houghton
james.p.houghton@gmail.com

This module contains everything that is needed to construct a system dynamics model
in python, using syntax that is compatible with the pysd model simulation functionality.

These functions could be used to construct a system dynamics model from scratch, in a
pinch. Due to the highly visual structure of system dynamics models, I still recommend
using an external model construction tool such as vensim or stella/iThink to build and
debug models.
"""

# Todo: Add a __doc__ function that summarizes the docstrings of the whole model
# Todo: Give the __doc__ function a 'short' and 'long' option
# Todo: Modify static functions to reference their own function attribute


import re
import keyword
from templates import templates
import numpy as np


class Builder(object):
    def __init__(self, outfilename, dictofsubs={}):
        """ The builder class

        Parameters
        ----------
        outfilename: <string> valid python filename
            including '.py'


        dictofsubs: dictionary
            numpy indices of translated subscript names and elements.
            {'apples':':', 'a1':0, 'a2':1,
            'pears':':', 'p1':0, 'p2':1}

        """
        # We specify the dictofsubs as a flat dictionary, because it is
        # cleaner and faster.

        self.filename = outfilename
        self.stocklist = []
        self.preamble = []
        self.body = []
        self.dictofsubs = dictofsubs
        self.preamble.append(templates['new file'].substitute())

        if dictofsubs: #Todo: check that the actual value passed to this function when there are no subscripts is an empty dict, or similar
            self.preamble.append(templates['subscript_dict'].substitute(dictofsubs=dictofsubs.__repr__()))
              
    def write(self):
        """ Writes out the model file """
        with open(self.filename, 'w') as outfile:
            [outfile.write(element) for element in self.preamble]
            [outfile.write(element) for element in self.body]

    def add_stock(self, identifier, sub, expression, initial_condition):
        """Adds a stock to the python model file based upon the interpreted expressions
        for the initial condition.

        Parameters
        ----------
        identifier: <string> valid python identifier
            Our translators are responsible for translating the model identifiers into
            something that python can use as a function name.

        expression: <string>
            This contains reference to all of the flows that

        initial_condition: <string>
            An expression that defines the value that the stock should take on when the model
            is initialized. This may be a constant, or a call to other model elements.
        subs is a string unlike in flaux where it's a list of strings. Because for now, a stock is only declared once.
        """

        #todo: consider the case where different flows work over different subscripts
        #todo: build a test case to test the above
        initial_condition = initial_condition.replace('\n','').replace('\t','') #todo: this is kluge
        if sub:
            size = getnumofelements(sub, self.dictofsubs)
            if re.search(';',initial_condition): #if the elements passed are an array of constants, need to format for numpy
                initial_condition = 'np.'+ np.array(np.mat(initial_condition.strip(';'))).__repr__()

            #todo: I don't like the fact that the array is applied even when it isnt needed - but thats for another day
            initial_condition += '*np.ones((%s))'%(','.join(map(str,size)))

        self.body.append(templates['stock'].substitute(identifier=identifier,
                                                       expression=expression,
                                                       initial_condition=initial_condition))
        self.stocklist.append(identifier)

    def add_flaux(self, identifier, sub, expression, doc=''):
        """Adds a flow or auxiliary element to the model.

        Parameters
        ----------
        identifier: <string> valid python identifier
            Our translators are responsible for translating the model identifiers into
            something that python can use as a function name.

        expression: list of strings
            Each element in the array is the equation that will be evaluated to fill
            the return array at the coordinates listed at corresponding locations
            in the `sub` dictionary

        sub: list of strings
            List of strings of subscript indices that correspond to the
            list of expressions
            ['a1,pears', 'a2,pears']

        doc: <string>
            The documentation string of the model

        Example
        -------
        assume we have some subscripts
            apples = [a1, a2, a3]
            pears = [p1, p2, p3]

        now sub list a list:
        sub = ['a1,pears', 'a2,pears']


        """
        # todo: evaluate if we should instead use syntax [['a1','pears'],['a2','pears']]
        # todo: build a test case to test

        # docstring = ('Type: Flow or Auxiliary\n        '+
        #              '\n        '.join(doc.split('\n')))
        docstring = ''


        # first work out the size of the array that the function will return
        if sub[0]!='': #todo: consider factoring this out if it is useful for the multiple flows
            #todo: why does the no-sub condition give [''] as the argument?
            size = getnumofelements(sub[0], self.dictofsubs)
            funcset = 'output = np.ndarray((%s))\n'%','.join(map(str,size)) #lines which encode the expressions for partially defined subscript pieces
            for expr, subi in zip(expression, sub):
                expr = expr.replace('\n','').replace('\t','').strip() #todo: we should clean the expressions before we get to this point...
                indices = ','.join(map(str,getelempos(subi,self.dictofsubs)))
                if re.search(';',expr): #if the elements passed are an array of constants, need to format for numpy
                    expr = 'np.'+np.array(np.mat(expr.strip(';'))).__repr__()
                    #todo: this might be an interesting way to identify and pull out of the function array constants
                funcset += '    output[%s] = %s\n'%(indices, expr)
        else:
            funcset = 'output = %s\n'%expression[0]


        funcstr = templates['flaux'].substitute(identifier=identifier,
                                                       expression=funcset,
                                                       docstring=docstring)
        self.body.append(funcstr)




    def add_lookup(self, identifier, valid_range, copair_list):
        """Constructs a function that implements a lookup.
        The function encodes the coordinate pairs as numeric values in the python file.

        Parameters
        ----------
        identifier: <string> valid python identifier
            Our translators are responsible for translating the model identifiers into
            something that python can use as a function name.

        range: <tuple>
            Minimum and maximum bounds on the lookup. Currently, we don't do anything
            with this, but in the future, may use it to implement some error checking.

        copair_list: a list of tuples, eg. [(0, 1), (1, 5), (2, 15)]
            The coordinates of the lookup formatted in coordinate pairs.

        """
        # todo: Add a docstring capability

        # in the future, we may want to check in bounds for the range. for now, lazy...
        xs, ys = zip(*copair_list)
        xs_str = str(list(xs))
        ys_str = str(list(ys))

        self.body.append(templates['lookup'].substitute(identifier=identifier,
                                                        xs_str=xs_str,
                                                        ys_str=ys_str))


    def add_n_delay(self, delay_input, delay_time, initial_value, order, sub):
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
        flowlist.append(add_flaux(identifier='%s_flow_1_of_%i'%(delay_name, order+1),
                                  sub=sub, expression=delay_input))

        for i in range(2, order+2):
            flowlist.append(add_flaux(identifier='%s_flow_%i_of_%i'%(delay_name, i, order+1),
                                      sub=sub,
                                      expression='self.%s_stock_%i_of_%i()/(1.*%s/%i)'%(
                                                  delay_name, i-1, order, delay_time, order)))

        for i in range(1, order+1):
            add_stock(identifier='%s_stock_%i_of_%i'%(delay_name, i, order),
                      sub=sub,
                      expression=flowlist[i-1]+' - '+flowlist[i],
                      initial_condition='%s * (%s / %i)'%(initial_value, delay_time, order))

        return flowlist[-1]


# these are module functions so that we can access them from the visitor

def getelempos(element, dictofsubs):
    """
    Helps for accessing elements of a

    Parameters
    ----------
    element
    dictofsubs

    Returns
    -------

    """
    # Todo: Make this accessible to the end user
    #  The end user will get an unnamed array, and will want to have access to
    #  members by name.


    position=[]
    elements=element.replace('!','').replace(' ', '').split(',')
    for element in elements:
        if element in dictofsubs.keys():
            position.append(':')
        else:
            for d in dictofsubs.itervalues():
                try:
                    position.append(d[element])
                except: pass

    return tuple(position)

def getnumofelements(element, dictofsubs):
    """

    Parameters
    ----------
    element <string of subscripts>

    returns a list of the sizes of the dimensions. A 4x3x6 array would return
    [4,3,6]

    """
    # todo: make this elementstr or something
    position=[]
    elements=element.replace('!','').replace('','').split(',')
    for element in elements:
        if element in dictofsubs.keys():
            position.append(len(dictofsubs[element]))
        else:
            for d in dictofsubs.itervalues():
                try:
                    (d[element])
                except: pass
                else:
                    position.append(len(d))

    return position






def add_lookup(filename, identifier, valid_range, copair_list):
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
#
# def add_Subscript(filename, identifier, expression):
#     docstring = ('Type: Subscript')
#     funcstr = ('    def subscript_%s(self):\n'%identifier +
#                '        """%s"""\n'%docstring +
#                '        return "%s" \n\n'%expression)
#
#     with open(filename, 'a') as outfile:
#         outfile.write(funcstr)
#
#     return 'self.%s()'%identifier


#
# def add_initial(filename, component):
#     """ Implement vensim's `INITIAL` command as a build-time function.
#         component cannot be a full expression, must be a reference to
#         a single external element.
#     """
#     if not re.search('[a-zA-Z]',component):
#         naked_component="number"
#     else:
#         naked_component = component.split("self.")[1]
#         naked_component = naked_component.split("()")[0]
#     funcstr = ('    def initial_%s(self, inval):                  \n'%naked_component +
#                '        if not hasattr(self.initial_%s, "value"): \n'%naked_component +
#                '            self.initial_%s.im_func.value = inval \n'%naked_component +
#                '        return self.initial_%s.value             \n\n'%naked_component
#               )
#     with open(filename, 'a') as outfile:
#         outfile.write(funcstr)
#
#     return 'self.initial_%s(%s)'%(naked_component, component)
#
#

# def add_initial(filename, component):
#     """ Implement vensim's `INITIAL` command as a build-time function.
#         component cannot be a full expression, must be a reference to
#         a single external element.
#     """
#
# #        This is a bit of a weird construction. For a simpler demo, try:
# #        class test:
# #            def method(self, inval):
# #                if not hasattr(self.method, 'value'):
# #                    self.method.im_func.value = inval
# #                return self.method.value
# #
# #        a = test()
# #        print a.method('hi')
# #        print a.method('there')
#
#     naked_component = component[5:-2]
#     funcstr = ('    def initial_%s(self, inval):                  \n'%naked_component +
#                '        if not hasattr(self.initial_%s, "value"): \n'%naked_component +
#                '            self.initial_%s.im_func.value = inval \n'%naked_component +
#                '        return self.initial_%s.value             \n\n'%naked_component
#               )
#     with open(filename, 'a') as outfile:
#         outfile.write(funcstr)
#
#     return 'self.initial_%s(%s)'%(naked_component, component)


#


#
#
#
#
# def add_n_smooth(filename, smooth_input, smooth_time, initial_value, order):
#     """Constructs stock and flow chains that implement the calculation of
#     a smoothing function.
#
#     delay_input: <string>
#         Reference to the model component that is the input to the smoothing function
#
#     smooth_time: <string>
#         Can be a number (in string format) or a reference to another model element
#         which will calculate the delay. This is calculated throughout the simulation
#         at runtime.
#
#     initial_value: <string>
#         This is used to initialize the stocks that are present in the delay. We
#         initialize the stocks with equal values so that the outflow in the first
#         timestep is equal to this value.
#
#     order: int
#         The number of stocks in the delay pipeline. As we construct the delays at
#         build time, this must be an integer and cannot be calculated from other
#         model components. Anything else will yield a ValueError.
#
#     """
#     try:
#         order = int(order)
#     except ValueError:
#         print "Order of delay must be an int. (Can't even be a reference to an int. Sorry...)"
#         raise
#
#          #depending in these cases on input to be formatted as 'self.smooth_input()' (or number)
#     naked_input = smooth_input[5:-2]
#     naked_smooth = smooth_time[5:-2] if smooth_time[:5] == 'self.' else smooth_time
#     smooth_name = '%s_smooth_%s'%(naked_input, naked_smooth)
#
#
#     flowlist = []
#     #use 1-based indexing for stocks in the delay chain so that (n of m) makes sense.
#     prev = smooth_input
#     current = 'self.%s_stock_1_of_%i()'%(smooth_name, order)
#     flowlist.append(add_flaux(filename,
#                               identifier='%s_flow_1_of_%i'%(smooth_name, order),
#                               expression='(%s - %s) / (1.*%s/%i)'%(
#                                            prev, current, smooth_time, order)))
#
#     for i in range(2, order+1):
#         prev = 'self.%s_stock_%i_of_%i()'%(smooth_name, i-1, order)
#         current = 'self.%s_stock_%i_of_%i()'%(smooth_name, i, order)
#         flowlist.append(add_flaux(filename,
#                                   identifier='%s_flow_%i_of_%i'%(smooth_name, i, order),
#                                   expression='(%s - %s)/(1.*%s/%i)'%(
#                                               prev, current, smooth_time, order)))
#
#     for i in range(1, order+1):
#         add_stock(filename,
#                   identifier='%s_stock_%i_of_%i'%(smooth_name, i, order),
#                   expression=flowlist[i-1],
#                   initial_condition=initial_value)
#
#     return flowlist[-1]
#
#

def make_python_identifier(string):
    """Takes an arbitrary string and creates a valid Python identifier.
    
    Identifiers must follow the convention outlined here:
    https://docs.python.org/2/reference/lexical_analysis.html#identifiers
    """
    # Todo: check for variable uniqueness -
    # perhaps maintain a list of current variables and their translations??

    string = string.lower()

    # remove leading and trailing whitespace
    string = string.strip()

    # Make spaces into underscores
    string = string.replace(' ', '_')

    # Make commas and brackets into underscores (mostly for subscript column names)
    string = string.replace(',', '__').replace('[','__')

    # Remove invalid characters
    string = re.sub('[^0-9a-zA-Z_]', '', string)

    # Remove leading characters until we find a letter or underscore
    string = re.sub('^[^a-zA-Z_]+', '', string)

    if string in keyword.kwlist:
        string += '_element'

    return string
