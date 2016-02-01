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
# If we have a function that defines a constant value (or a big, constructed
# numpy array) it may be better to have the array constructed once (outside of the
# function, perhaps as an attribute) than to construct and return it every time
# the function is called. (Of course, it may be that python detects this and does
# it for us - we should find out)
# Alternately, there may be a clever way to cache this so that we don't have to
# change the model file.
#

# Todo: Template separation is getting a bit out of control. Perhaps bring it back in?
# Todo: create a function that gets the subscript family name, given one of its elements
# this should be robust to the possibility that different families contain the same
# child, as would be the case with subranges. Probably means that we'll have to collect
# all of the subelements and pass them together to get the family name.


import re
import keyword
from templates import templates
import numpy as np


class Builder(object):
    def __init__(self, outfile_name, dictofsubs={}):
        """ The builder class

        Parameters
        ----------
        outfilename: <string> valid python filename
            including '.py'

        dictofsubs: dictionary
            # Todo: rewrite this once we settle on a schema

        """
        # Todo: check that the no-subscript value of dictofsubs is an empty dictionary.
        self.filename = outfile_name
        self.stocklist = []
        self.preamble = []
        self.body = []
        self.dictofsubs = dictofsubs
        self.preamble.append(templates['new file'].substitute())

        if dictofsubs:
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

        sub : basestring
            unlike in flaux where it's a list of strings. Because for now, a stock
            is only declared once.
        """

        # todo: consider the case where different flows work over different subscripts
        # todo: properly handle subscripts here
        # todo: build a test case to test the above
        # todo: force the sub parameter to be a list
        # todo: handle docstrings
        initial_condition = initial_condition.replace('\n','').replace('\t','').replace('[@@@]','')  # Todo:pull out
        if sub:
            if isinstance(sub, basestring): sub = [sub]  # Todo: rework
            directory, size = get_array_info(sub, self.dictofsubs)
            if re.search(';',initial_condition):  # format arrays for numpy
                initial_condition = 'np.'+ np.array(np.mat(initial_condition.strip(';'))).__repr__()

            # todo: I don't like the fact that the array is applied even when it isnt needed
            initial_condition += '*np.ones((%s))'%(','.join(map(str,size)))

        funcstr = templates['stock'].substitute(identifier=identifier,
                                                expression=expression.replace('[@@@]',''),
                                                initial_condition=initial_condition)

        if sub:  # this is super bad coding practice, should change it.
            funcstr += '%s.dimension_dir = '%identifier+directory.__repr__()+'\n'

        self.body.append(funcstr)
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
            list of expressions, and collectively define the shape of the output
            ['a1,pears', 'a2,pears']

        doc: <string>
            The documentation string of the model

        Returns
        -------
        identifier: <string>
            The name of the constructed function

        Example
        -------
        assume we have some subscripts
            apples = [a1, a2, a3]
            pears = [p1, p2, p3]

        now sub list a list:
        sub = ['a1,pears', 'a2,pears']
        """
        # todo: why does the no-sub condition give [''] as the argument?
        # todo: evaluate if we should instead use syntax [['a1','pears'],['a2','pears']]
        # todo: build a test case to test
        # todo: clean up this function
        # todo: add docstring handling

        docstring = ''

        if sub[0]!='': #todo: consider factoring this out if it is useful for the multiple flows
            directory, size = get_array_info(sub, self.dictofsubs)

            funcset = 'loc_dimension_dir = %s.dimension_dir \n'%identifier
            funcset += '    output = np.ndarray((%s))\n'%','.join(map(str,size)) #lines which encode the expressions for partially defined subscript pieces
            for expr, subi in zip(expression, sub):
                expr = expr.replace('\n','').replace('\t','').strip()  # todo: pull out
                indices = ','.join(map(str,getelempos(subi, self.dictofsubs)))
                #expr = expr.replace('\n', '').replace('\t', '').strip()  # todo: pull out
                #indices = ','.join(map(str,getelempos(subi, directory, self.dictofsubs)))
                if re.search(';',expr):  # if 2d array, format for numpy
                    expr = 'np.'+np.array(np.mat(expr.strip(';'))).__repr__()
                funcset += '    output[%s] = %s\n'%(indices, expr.replace('[@@@]','[%s]'%indices))
        else:
            funcset = 'loc_dimension_dir = 0 \n'
            funcset += '    output = %s\n'%expression[0]

        funcstr = templates['flaux'].substitute(identifier=identifier,
                                                expression=funcset,
                                                docstring=docstring)

        if sub[0] != '':  # todo: make less brittle
            funcstr += '%s.dimension_dir = '%identifier+directory.__repr__()+'\n'
        self.body.append(funcstr)

        return identifier

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

    def add_initial(self, component):
        """ Implement vensim's `INITIAL` command as a build-time function.
            component cannot be a full expression, must be a reference to
            a single external element.
        """
        if not re.search('[a-zA-Z]',component):
            naked_component="number"
            funcstr = ('\ndef initial_%s(inval):                       \n'%naked_component +
                    '    return inval             \n\n'
                    )

        else:
            naked_component = component.split("()")[0]
            funcstr = ('\ndef initial_%s(inval):                       \n'%naked_component +
                    '    if not hasattr(initial_%s, "value"): \n'%naked_component +
                    '        initial_%s.value = inval         \n'%naked_component +
                    '    return initial_%s.value             \n\n'%naked_component
                    )

        self.body.append(funcstr)
        return 'initial_%s(%s)'%(naked_component, component)

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

        Returns
        -------
        outflow: basestring
            Reference to the flow which contains the output of the delay process

        """
        try:
            order = int(order)
        except ValueError:
            print "Order of delay must be an int. (Can't even be a reference to an int. Sorry...)"
            raise

        # depending in these cases on input to be formatted as 'self.variable()' (or number)
        naked_input = delay_input[5:-2]
        naked_delay = delay_time[5:-2] if delay_time[:5] == 'self.' else delay_time
        delay_name = '%s_delay_%s'%(naked_input, naked_delay)


        flowlist = []
        # use 1-based indexing for stocks in the delay chain so that (n of m) makes sense.
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



    def add_n_smooth(self, smooth_input, smooth_time, initial_value, order, sub):
        """Constructs stock and flow chains that implement the calculation of
        a smoothing function.

        Parameters
        ----------
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

        sub: list of strings
            List of strings of subscript indices that correspond to the
            list of expressions, and collectively define the shape of the output
            See `builder.add_flaux` for more info

        Returns
        -------
        output: <basestring>
            Name of the stock which contains the smoothed version of the input

        """
        try:
            order = int(order)
        except ValueError:
            print "Order of delay must be an int. (Can't even be a reference to an int. Sorry...)"
            raise

        naked_input = smooth_input[:-2] #always a funciton
        naked_smooth = smooth_time[:-2] if smooth_time.endswith('()') else smooth_time #function or number
        smooth_name = '%s_smooth_%s'%(naked_input, naked_smooth)


        flowlist = []
        # use 1-based indexing for stocks in the delay chain so that (n of m) makes sense.
        prev = smooth_input
        current = '%s_stock_1_of_%i()'%(smooth_name, order)
        flowlist.append(self.add_flaux(identifier='%s_flow_1_of_%i'%(smooth_name, order),
                                       sub=sub,
                                       expression='(%s - %s) / (1.*%s/%i)'%(
                                               prev, current, smooth_time, order)))

        for i in range(2, order+1):
            prev = 'self.%s_stock_%i_of_%i()'%(smooth_name, i-1, order)
            current = 'self.%s_stock_%i_of_%i()'%(smooth_name, i, order)
            flowlist.append(self.add_flaux(identifier='%s_flow_%i_of_%i'%(smooth_name, i, order),
                                           sub=sub,
                                           expression='(%s - %s)/(1.*%s/%i)'%(
                                                  prev, current, smooth_time, order)))

        stocklist = []
        for i in range(1, order+1):
            stocklist.append(self.add_stock(identifier='%s_stock_%i_of_%i'%(smooth_name, i, order),
                                            sub=sub,
                                            expression=flowlist[i-1],
                                            initial_condition=initial_value))

        return stocklist[-1]


# these are module functions so that we can access them from other places

def getelempos(element, dictofsubs):
    """
    Helps for accessing elements of an array: given the subscript element names,
    returns the numerical ranges that correspond

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
            if isinstance(dictofsubs[element],dict):            
                position.append(':') 
            else:
                position.append(sorted(dictofsubs[element][:-1]))
        else: 
            for d in dictofsubs.itervalues(): 
                try: 
                    position.append(d[element]) 
                except: pass  
    return tuple(position)


def get_array_info(subs, dictofsubs):
    """
    Returns information needed to create and access members of the numpy array
    based upon the string names given to them in the model file.

    Parameters
    ----------
    subs : Array of strings of subscripts
        These should be all of the subscript names that are needed to create the array

    dictofsubs : dictionary

    returns
    -------
    A dictionary of the dimensions associating their names with their numpy indices
    directory = {'dimension name 1':0, 'dimension name 2':1}

    A list of the length of each dimension. Equivalently, the shape of the array:
    shape = [5,4]
    """

    # subscript references here are lists of array 'coordinate' names
    if isinstance(subs,list):
        element=subs[0]
    else:
        element=subs
    # we collect the references used in each dimension as a set, so we can compare contents
    position=[]
    directory={}
    dirpos=0
    elements=element.replace('!','').replace(' ','').split(',')
    for element in elements:
        if element in dictofsubs.keys():
            if isinstance(dictofsubs[element],list):
                dir,pos = (get_array_info(dictofsubs[element][-1],dictofsubs))
                position.append(pos[0])
                directory[dictofsubs[element][-1]]=dirpos
                dirpos+=1
            else:
                position.append(len(dictofsubs[element]))
                directory[element]=dirpos
                dirpos+=1

    # num_dimensions = len(subscript_references[0])
    # reference_sets = [set() for _ in range(num_dimensions)]
    # for subscript_reference in subscript_references:
    #     [reference_sets[i].add(element) for i, element in enumerate(subscript_reference)]
    # # `reference_sets` will now include a set for every dimension, containing the names used to
    # # access parts of that dimension
    #
    # directory = dict()
    # shape = np.zeros(num_dimensions)
    # for i, reference_set in enumerate(reference_sets):
    #     if len(reference_set) == 1:  # The element is almost certainly a subscript family name
    #         reference = list(reference_set)[0]
    #         if reference in dictofsubs.keys():
    #             shape[i] = len(dictofsubs[reference])
    #             directory[reference] = i
    #         else:  # Todo: handle the case of a one-element subscript here
    #             pass
        else:
            for famname,value in dictofsubs.iteritems():
                try:
                    (value[element])
                except: pass
                else:
                    position.append(len(value))
                    directory[famname]=dirpos
                    dirpos+=1
    return directory, position


def dict_find(in_dict, value):
    """ Helper function for looking up directory keys by their values.
     This isn't robust to repeated values

    Parameters
    ----------
    in_dict : dictionary
        A dictionary containing `value`

    value : any type
        What we wish to find in the dictionary

    Returns
    -------
    key: basestring
        The key at which the value can be found
    """
    # Todo: make this robust to repeated values
    # Todo: make this robust to missing values
    return in_dict.keys()[in_dict.values().index(value)]


def make_python_identifier(string):
    """
    Takes an arbitrary string and creates a valid Python identifier.

    Parameters
    ----------
    string : <basestring>
        The text to be converted into a valid python identifier

    Returns
    -------
    identifier : <string>
        A vaild python identifier based on the input string

    References
    ----------
    Identifiers must follow the convention outlined here:
        https://docs.python.org/2/reference/lexical_analysis.html#identifiers
    """
    # Todo: check for variable uniqueness
    #  perhaps maintain a list of current variables and their translations??
    # Todo: check that the output is actually a valid python identifier

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

    # Check that the string is not a python identifier
    if string in keyword.kwlist:
        string += '_element'

    return string
