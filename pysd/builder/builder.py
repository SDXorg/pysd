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

import textwrap
import autopep8
import re
import keyword
from templates import templates
import numpy as np


#
#
# class Builder(object):
#     def __init__(self, outfile_name, dictofsubs={}):
#         """ The builder class
#
#         Parameters
#         ----------
#         outfilename: <string> valid python filename
#             including '.py'
#
#         dictofsubs: dictionary
#             # Todo: rewrite this once we settle on a schema
#
#         """
#         # Todo: check that the no-subscript value of dictofsubs is an empty dictionary.
#         self.filename = outfile_name
#         self.stocklist = []
#         self.preamble = []
#         self.body = []
#         self.dictofsubs = dictofsubs
#         self.preamble.append(templates['new file'].substitute())
#
#         if dictofsubs:
#             self.preamble.append(templates['subscript_dict'].substitute(dictofsubs=dictofsubs.__repr__()))
#
#     def write(self):
#         """ Writes out the model file """
#         with open(self.filename, 'w') as outfile:
#             [outfile.write(element) for element in self.preamble]
#             [outfile.write(element) for element in self.body]

def create_base_array(subs_list, subscript_dict):
    """
    Given a list of subscript references,
    returns a base array that can be populated by these references

    Parameters
    ----------
    subs_list

    Returns
    -------
    base_array: string
        A string that

    >>> create_base_array([['Dim1', 'D'], ['Dim1', 'E'], ['Dim1', 'F']],
    ...                    {'Dim1': ['A', 'B', 'C'],
    ...                     'Dim2': ['D', 'E', 'F', 'G']})
    "xr.DataArray(data=np.empty([3, 3])*NaN, coords={'Dim2': ['D', 'E', 'F'], 'Dim1': ['A', 'B', 'C']})"

    # >>> create_base_array([['Dim1', 'A'], ['Dim1', 'B'], ['Dim1', 'C']],
    # ...                    {'Dim1': ['A', 'B', 'C']})

    """
    sub_names_list = subscript_dict.keys()
    sub_elems_list = [y for x in subscript_dict.values() for y in x]

    coords = dict()
    for subset in subs_list:
        for sub in subset:
            if sub in sub_names_list:
                if sub not in coords:
                    coords[sub] = subscript_dict[sub]
            elif sub in sub_elems_list:
                name = find_subscript_name(subscript_dict, sub)
                if name not in coords:
                    coords[name] = [sub]
                else:
                    if sub not in coords[name]:
                        coords[name] += [sub]

    return "xr.DataArray(data=np.empty(%(shape)s)*NaN, coords=%(coords)s)" % {
        'shape': repr(map(len, coords.values())),
        'coords': repr(coords)
    }


def build_element(element, subscript_dict):
    """
    Returns a string that has processed a single element dictionary
    Parameters
    ----------
    element
    subscript_dict

    Returns
    -------

    """
    if element['kind'] == 'constant':
        cache = "@cache('run')"
    elif element['kind'] == 'setup':
        cache = ''
    elif element['kind'] == 'component':
        cache = "@cache('step')"
    elif element['kind'] == 'macro':
        cache = ''

    if len(element['py_expr']) > 1:
        contents = "ret = %s\n" % create_base_array(element['subs'], subscript_dict)

        for sub, expr in zip(element['subs'], element['py_expr']):
            contents += 'ret.iloc[%(coord_dict)s] = %(expr)s\n' % {
                'coord_dict': repr(make_coord_dict(sub, subscript_dict)),
                'expr': expr}

        contents += "return ret"
    else:
        contents = "return %s" % element['py_expr'][0]

    indent = 8
    element.update({'cache': cache,
                    'ulines': '-' * len(element['real_name']),
                    'contents': contents.replace('\n',
                                                 '\n' + ' ' * indent)})  # indent lines 2 onward

    func = '''
    %(cache)s
    def %(py_name)s():
        """
        %(real_name)s
        %(ulines)s
        (%(py_name)s)
        %(unit)s

        %(doc)s
        """
        %(contents)s
        ''' % element
    return func


def build(elements, subscript_dict, outfile_name):
    """
    Takes in a list of model components


    Parameters
    ----------
    elements
    subscript_dict

    Returns
    -------

    """
    elements = merge_partial_elements(elements)

    functions = [build_element(element, subscript_dict) for element in elements]

    imports = []
    if subscript_dict:
        imports += ['import xarray as xr']

    text = """
    from __future__ import division
    import pysd.functions
    %(imports)s
    from pysd.builder import cache

    subscript_dict = %(subscript_dict)s

    %(functions)s

    """ % {'subscript_dict': repr(subscript_dict),
           'functions': '\n'.join(functions),
           'imports': '\n'.join(imports)}

    text = autopep8.fix_code(textwrap.dedent(text),
                             options={'aggressive': 10,
                                      'max_line_length': 99,
                                      'experimental': True})

    # this is used for testing
    if outfile_name == 'return':
        return text

    with open(outfile_name, 'w') as out:
        out.write(text)


def identify_subranges(subscript_dict):
    """

    Parameters
    ----------
    subscript_dict

    Returns
    -------

    Examples
    --------
    >>> identify_subranges({'Dim1': ['A', 'B', 'C', 'D', 'E', 'F'],
    ...                     'Range1': ['C', 'D', 'E']})
    {'Range1': ('Dim1', ['C', 'D', 'E'])}, {'Dim1'
    """

def add_stock(identifier, subs, expression, initial_condition):
    """
    Creates new model element dictionaries for the model elements associated
    with a stock.


    Parameters
    ----------
    identifier
    sub
    expression
    initial_condition

    Returns
    -------
    a string to use in place of the 'INTEG...' pieces in the element expression string,
    a list of additional model elements to add

    Examples
    --------
    >>> add_stock('stock_name', [], 'inflow_a', '10')

    """

    # create the stock initialization element
    init_element = {
        'py_name': '_init_%s' % identifier,
        'real_name': 'Implicit',
        'kind': 'setup',  # not explicitly specified in the model file, but must exist
        'py_expr': initial_condition,
        'subs': subs,
        'doc': 'Provides initial conditions for %s function' % identifier,
        'unit': 'See docs for %s' % identifier
    }

    ddt_element = {
        'py_name': '_d%s_dt' % identifier,
        'real_name': 'Implicit',
        'kind': 'component',
        'doc': 'Provides derivative for %s function' % identifier,
        'subs': subs,
        'unit': 'See docs for %s' % identifier,
        'py_expr': expression
    }

    return "_state['%s']" % identifier, [init_element, ddt_element]


def make_coord_dict(subs, subscript_dict):
    """
    This is for assisting with the lookup of a particular element, such that the output
    of this function would take the place of %s in this expression

    `variable.loc[%s]`

    Parameters
    ----------
    subs
    subscript_dict

    Returns
    -------

    Examples
    --------
    >>> make_coord_dict(['Dim1', 'D'], {'Dim1':['A','B','C'], 'Dim2':['D', 'E', 'F']})
    {'Dim2': ['D']}

    """
    sub_elems_list = [y for x in subscript_dict.values() for y in x]
    coordinates = {}
    for sub in subs:
        if sub in sub_elems_list:
            name = find_subscript_name(subscript_dict, sub)
            coordinates[name] = [sub]
    return coordinates


def find_subscript_name(subscript_dict, element):
    """
    Given a subscript dictionary, and a member of a subscript family,
    return the first key of which the

    Parameters
    ----------
    subscript_dict: dictionary
        Follows the {'subscript name':['list','of','subscript','elements']} format
    element: sting

    Returns
    -------

    Examples:
    >>> find_subscript_name({'Dim1': ['A', 'B'],
    ...                      'Dim2': ['C', 'D', 'E'],
    ...                      'Dim3': ['F', 'G', 'H', 'I']},
    ...                      'D')
    'Dim2'
    """
    for name, elements in subscript_dict.iteritems():
        if element in elements:
            return name


def merge_partial_elements(element_list):
    """
    merges model elements which collectively all define the model component,
    mostly for multidimensional subscripts


    Parameters
    ----------
    element_list

    Returns
    -------
    """
    outs = dict()  # output data structure
    for element in element_list:
        name = element['py_name']
        if name not in outs:
            outs[name] = {
                'py_name': element['py_name'],
                'real_name': element['real_name'],
                'doc': element['doc'],
                'py_expr': [element['py_expr']],  # in a list
                'unit': element['unit'],
                'subs': [element['subs']],
                'kind': element['kind']
            }

        else:
            outs[name]['doc'] = outs[name]['doc'] or element['doc']
            outs[name]['unit'] = outs[name]['unit'] or element['unit']
            outs[name]['py_expr'] += [element['py_expr']]
            outs[name]['subs'] += [element['subs']]

    return outs.values()

    # create the stock

    #
    #
    #
    # def add_stock(self, identifier, sub, expression, initial_condition):
    #     """Adds a stock to the python model file based upon the interpreted expressions
    #     for the initial condition.
    #
    #     Parameters
    #     ----------
    #     identifier: <string> valid python identifier
    #         Our translators are responsible for translating the model identifiers into
    #         something that python can use as a function name.
    #
    #     expression: <string>
    #         This contains reference to all of the flows that
    #
    #     initial_condition: <string>
    #         An expression that defines the value that the stock should take on when the model
    #         is initialized. This may be a constant, or a call to other model elements.
    #
    #     sub : basestring
    #         unlike in flaux where it's a list of strings. Because for now, a stock
    #         is only declared once.
    #     """
    #
    #     # todo: consider the case where different flows work over different subscripts
    #     # todo: properly handle subscripts here
    #     # todo: build a test case to test the above
    #     # todo: force the sub parameter to be a list
    #     # todo: handle docstrings
    #     initial_condition = initial_condition.replace('\n','').replace('\t','').replace('[@@@]','')  # Todo:pull out
    #     if sub:
    #         if isinstance(sub, basestring): sub = [sub]  # Todo: rework
    #         directory, size = get_array_info(sub, self.dictofsubs)
    #         if re.search(';',initial_condition):  # format arrays for numpy
    #             initial_condition = 'np.'+ np.array(np.mat(initial_condition.strip(';'))).__repr__()
    #
    #         # todo: I don't like the fact that the array is applied even when it isnt needed
    #         initial_condition += '*np.ones((%s))'%(','.join(map(str,size)))
    #
    #     funcstr = templates['stock'].substitute(identifier=identifier,
    #                                             expression=expression.replace('[@@@]',''),
    #                                             initial_condition=initial_condition)
    #
    #     if sub:  # this is super bad coding practice, should change it.
    #         funcstr += '%s.dimension_dir = '%identifier+directory.__repr__()+'\n'
    #
    #     self.body.append(funcstr)
    #     self.stocklist.append(identifier)
    #
    # def add_flaux(self, identifier, sub, expression, doc=''):
    #     """Adds a flow or auxiliary element to the model.
    #
    #     Parameters
    #     ----------
    #     identifier: <string> valid python identifier
    #         Our translators are responsible for translating the model identifiers into
    #         something that python can use as a function name.
    #
    #     expression: list of strings
    #         Each element in the array is the equation that will be evaluated to fill
    #         the return array at the coordinates listed at corresponding locations
    #         in the `sub` dictionary
    #
    #     sub: list of strings
    #         List of strings of subscript indices that correspond to the
    #         list of expressions, and collectively define the shape of the output
    #         ['a1,pears', 'a2,pears']
    #
    #     doc: <string>
    #         The documentation string of the model
    #
    #     Returns
    #     -------
    #     identifier: <string>
    #         The name of the constructed function
    #
    #     Example
    #     -------
    #     assume we have some subscripts
    #         apples = [a1, a2, a3]
    #         pears = [p1, p2, p3]
    #
    #     now sub list a list:
    #     sub = ['a1,pears', 'a2,pears']
    #     """
    #     # todo: why does the no-sub condition give [''] as the argument?
    #     # todo: evaluate if we should instead use syntax [['a1','pears'],['a2','pears']]
    #     # todo: build a test case to test
    #     # todo: clean up this function
    #     # todo: add docstring handling
    #
    #     docstring = ''
    #
    #     if sub[0]!='': #todo: consider factoring this out if it is useful for the multiple flows
    #         directory, size = get_array_info(sub, self.dictofsubs)
    #
    #         funcset = 'loc_dimension_dir = %s.dimension_dir \n'%identifier
    #         funcset += '    output = np.ndarray((%s))\n'%','.join(map(str,size)) #lines which encode the expressions for partially defined subscript pieces
    #         for expr, subi in zip(expression, sub):
    #             expr = expr.replace('\n','').replace('\t','').strip()  # todo: pull out
    #             indices = ','.join(map(str,getelempos(subi, self.dictofsubs)))
    #             if re.search(';',expr):  # if 2d array, format for numpy
    #                 expr = 'np.'+np.array(np.mat(expr.strip(';'))).__repr__()
    #             funcset += '    output[%s] = %s\n'%(indices, expr.replace('[@@@]','[%s]'%indices))
    #     else:
    #         funcset = 'loc_dimension_dir = 0 \n'
    #         funcset += '    output = %s\n'%expression[0]
    #
    #
    #     funcstr = templates['flaux'].substitute(identifier=identifier,
    #                                             expression=funcset,
    #                                             docstring=docstring)
    #
    #     if sub[0] != '':  # todo: make less brittle
    #         funcstr += '%s.dimension_dir = '%identifier+directory.__repr__()+'\n'
    #     self.body.append(funcstr)
    #
    #     return identifier


#     def add_lookup(self, identifier, valid_range, sub, copair_list):
#
#         """Constructs a function that implements a lookup.
#         The function encodes the coordinate pairs as numeric values in the python file.
#
#         Parameters
#         ----------
#         identifier: <string> valid python identifier
#             Our translators are responsible for translating the model identifiers into
#             something that python can use as a function name.
#
#         range: <tuple>
#             Minimum and maximum bounds on the lookup. Currently, we don't do anything
#             with this, but in the future, may use it to implement some error checking.
#
#         copair_list: a list of tuples, eg. [(0, 1), (1, 5), (2, 15)]
#             The coordinates of the lookup formatted in coordinate pairs.
#
#         """
#         # todo: Add a docstring capability
#
#         # in the future, we may want to check in bounds for the range. for now, lazy...
#         xs_str = []
#         ys_str = []
#         for i in copair_list:
#             xs, ys = zip(*i)
#             xs_str.append(str(list(xs)))
#             ys_str.append(str(list(ys)))
#
#         if sub_list[0]=='':
#             createxy=''
#             addendum=''
#
#         else:
#             createxy = ('%s.xs = np.ndarray((%s),object) \n'%(identifier,','.join(map(str,get_array_info(sub_list[0],self.dictofsubs)[1])))+
#                         '%s.ys = np.ndarray((%s),object) \n'%(identifier,','.join(map(str,get_array_info(sub_list[0],self.dictofsubs)[1]))))
#             addendum=  ('for i,j in np.ndenumerate(%s.xs): \n'%identifier+
#                         '    %s.xs[i]=j.split(",") \n'%identifier+
#                         '    for k,l in np.ndenumerate(%s.xs[i]): \n'%identifier+
#                         '        %s.xs[i][k[0]]=float(l) \n'%identifier+
#                         'for i,j in np.ndenumerate(%s.ys): \n'%identifier+
#                         '    %s.ys[i]=j.split(",") \n'%identifier+
#                         '    for k,l in np.ndenumerate(%s.ys[i]): \n'%identifier+
#                         '        %s.ys[i][k[0]]=float(l) \n'%identifier+
#                         'del i,j,k,l')
#         for i in range(len(copair_list)):
#             try:
#                 funcsub = ('[%s]')%(','.join(map(str,getelempos(sub_list[i],self.dictofsubs))))
#                 funcsub = re.sub(r'\[(:,*)*]*\]','',funcsub)
#                 if not funcsub:
#                     addendum = ''
#                     createxy = ''
#             except:
#                 funcsub = ''
#
#             if not addendum:
#                 createxy += ('%s.xs%s = %s \n'%(identifier,funcsub,xs_str[i])+
#                              '%s.ys%s = %s \n'%(identifier,funcsub,ys_str[i]))
#             else:
#                 createxy += ('%s.xs%s = ",".join(map(str,%s)) \n'%(identifier,funcsub,xs_str[i])+
#                              '%s.ys%s = ",".join(map(str,%s)) \n'%(identifier,funcsub,ys_str[i]))
#         funcstr = ('def %s(x):                                      \n'%identifier+
#                    '    try: localxs                                          \n'+
#                    '    except:                                               \n'+
#                    '        localxs = %s.xs                                \n'%identifier+
#                    '        localys = %s.ys                                \n'%identifier+
#                    '    return functions.lookup(x, localxs, localys)          \n'+
#                    '                                                          \n'+
#                    createxy+addendum+
#                    '                                                          \n'
#                   )
#
#         self.body.append(funcstr)
#
#     def add_initial(self, component):
#         """ Implement vensim's `INITIAL` command as a build-time function.
#             component cannot be a full expression, must be a reference to
#             a single external element.
#         """
#         if not re.search('[a-zA-Z]',component):
#             naked_component="number"
#             funcstr = ('\ndef initial_%s(inval):                       \n'%naked_component +
#                     '    return inval             \n\n'
#                     )
#
#         else:
#             naked_component = component.split("()")[0]
#             naked_component = naked_component.replace('functions.shorthander(', '')
#             funcstr = ('\ndef initial_%s(inval):                       \n'%naked_component +
#                     '    if not hasattr(initial_%s, "value"): \n'%naked_component +
#                     '        initial_%s.value = inval         \n'%naked_component +
#                     '    return initial_%s.value             \n\n'%naked_component
#                     )
#
#         self.body.append(funcstr)
#         return 'initial_%s(%s)'%(naked_component, component)
#
#     def add_n_delay(self, delay_input, delay_time, initial_value, order, sub):
#         """Constructs stock and flow chains that implement the calculation of
#         a delay.
#
#         delay_input: <string>
#             Reference to the model component that is the input to the delay
#
#         delay_time: <string>
#             Can be a number (in string format) or a reference to another model element
#             which will calculate the delay. This is calculated throughout the simulation
#             at runtime.
#
#         initial_value: <string>
#             This is used to initialize the stocks that are present in the delay. We
#             initialize the stocks with equal values so that the outflow in the first
#             timestep is equal to this value.
#
#         order: int
#             The number of stocks in the delay pipeline. As we construct the delays at
#             build time, this must be an integer and cannot be calculated from other
#             model components. Anything else will yield a ValueError.
#
#         Returns
#         -------
#         outflow: basestring
#             Reference to the flow which contains the output of the delay process
#
#         """
#         try:
#             order = int(order)
#         except ValueError:
#             print "Order of delay must be an int. (Can't even be a reference to an int. Sorry...)"
#             raise
#
#         # depending in these cases on input to be formatted as 'self.variable()' (or number)
#         naked_input = delay_input[:-2]
#         naked_delay = delay_time[:-2] if delay_time.endswith('()') else delay_time
#         delay_name = '%s_delay_%s'%(naked_input, naked_delay)
#
#         flowlist = []  # contains the identities of the flows, as strings
#         # use 1-based indexing for stocks in the delay chain so that (n of m) makes sense.
#         flowlist.append(self.add_flaux(identifier='%s_flow_1_of_%i'%(delay_name, order+1),
#                                        sub=sub,
#                                        expression=[delay_input]))
#
#         for i in range(2, order+2):
#             flowlist.append(self.add_flaux(identifier='%s_flow_%i_of_%i'%(delay_name, i, order+1),
#                                            sub=sub,
#                                            expression=['%s_stock_%i_of_%i()/(1.*%s/%i)'%(
#                                                   delay_name, i-1, order, delay_time, order)]))
#
#         for i in range(1, order+1):
#             self.add_stock(identifier='%s_stock_%i_of_%i'%(delay_name, i, order),
#                            sub=sub,
#                            expression=flowlist[i-1]+'() - '+flowlist[i]+'()',
#                            initial_condition='%s * (%s / %i)'%(initial_value, delay_time, order))
#
#         return flowlist[-1]+'()'
#
#
#
#     def add_n_smooth(self, smooth_input, smooth_time, initial_value, order, sub):
#         """Constructs stock and flow chains that implement the calculation of
#         a smoothing function.
#
#         Parameters
#         ----------
#         delay_input: <string>
#             Reference to the model component that is the input to the smoothing function
#
#         smooth_time: <string>
#             Can be a number (in string format) or a reference to another model element
#             which will calculate the delay. This is calculated throughout the simulation
#             at runtime.
#
#         initial_value: <string>
#             This is used to initialize the stocks that are present in the delay. We
#             initialize the stocks with equal values so that the outflow in the first
#             timestep is equal to this value.
#
#         order: int
#             The number of stocks in the delay pipeline. As we construct the delays at
#             build time, this must be an integer and cannot be calculated from other
#             model components. Anything else will yield a ValueError.
#
#         sub: list of strings
#             List of strings of subscript indices that correspond to the
#             list of expressions, and collectively define the shape of the output
#             See `builder.add_flaux` for more info
#
#         Returns
#         -------
#         output: <basestring>
#             Name of the stock which contains the smoothed version of the input
#
#         """
#         try:
#             order = int(order)
#         except ValueError:
#             print "Order of delay must be an int. (Can't even be a reference to an int. Sorry...)"
#             raise
#
#         naked_input = smooth_input[:-2] #always a funciton
#         naked_smooth = smooth_time[:-2] if smooth_time.endswith('()') else smooth_time #function or number
#         smooth_name = '%s_smooth_%s'%(naked_input, naked_smooth)
#
#
#         flowlist = []
#         # use 1-based indexing for stocks in the delay chain so that (n of m) makes sense.
#         prev = smooth_input
#         current = '%s_stock_1_of_%i()'%(smooth_name, order)
#         flowlist.append(self.add_flaux(identifier='%s_flow_1_of_%i'%(smooth_name, order),
#                                        sub=sub,
#                                        expression='(%s - %s) / (1.*%s/%i)'%(
#                                                prev, current, smooth_time, order)))
#
#         for i in range(2, order+1):
#             prev = 'self.%s_stock_%i_of_%i()'%(smooth_name, i-1, order)
#             current = 'self.%s_stock_%i_of_%i()'%(smooth_name, i, order)
#             flowlist.append(self.add_flaux(identifier='%s_flow_%i_of_%i'%(smooth_name, i, order),
#                                            sub=sub,
#                                            expression='(%s - %s)/(1.*%s/%i)'%(
#                                                   prev, current, smooth_time, order)))
#
#         stocklist = []
#         for i in range(1, order+1):
#             stocklist.append(self.add_stock(identifier='%s_stock_%i_of_%i'%(smooth_name, i, order),
#                                             sub=sub,
#                                             expression=flowlist[i-1],
#                                             initial_condition=initial_value))
#
#         return stocklist[-1]
#
#
# # these are module functions so that we can access them from other places
#
# def getelempos(element, dictofsubs):
#     """
#     Helps for accessing elements of an array: given the subscript element names,
#     returns the numerical ranges that correspond
#
#
#     Parameters
#     ----------
#     element
#     dictofsubs
#
#     Returns
#     -------
#
#
#     """
#     # Todo: Make this accessible to the end user
#     #  The end user will get an unnamed array, and will want to have access to
#     #  members by name.
#
#     position=[]
#     elements=element.replace('!','').replace(' ', '').split(',')
#     for element in elements:
#         if element in dictofsubs.keys():
#             if isinstance(dictofsubs[element],dict):
#                 position.append(':')
#             else:
#                 position.append(sorted(dictofsubs[element][:-1]))
#         else:
#             for d in dictofsubs.itervalues():
#                 try:
#                     position.append(d[element])
#                 except: pass
#     return tuple(position)
#
#
# def get_array_info(subs, dictofsubs):
#     """
#     Returns information needed to create and access members of the numpy array
#     based upon the string names given to them in the model file.
#
#     Parameters
#     ----------
#     subs : Array of strings of subscripts
#         These should be all of the subscript names that are needed to create the array
#
#     dictofsubs : dictionary
#
#     returns
#     -------
#     A dictionary of the dimensions associating their names with their numpy indices
#     directory = {'dimension name 1':0, 'dimension name 2':1}
#
#     A list of the length of each dimension. Equivalently, the shape of the array:
#     shape = [5,4]
#     """
#
#     # subscript references here are lists of array 'coordinate' names
#     if isinstance(subs,list):
#         element=subs[0]
#     else:
#         element=subs
#     # we collect the references used in each dimension as a set, so we can compare contents
#     position=[]
#     directory={}
#     dirpos=0
#     elements=element.replace('!','').replace(' ','').split(',')
#     for element in elements:
#         if element in dictofsubs.keys():
#             if isinstance(dictofsubs[element],list):
#                 dir,pos = (get_array_info(dictofsubs[element][-1],dictofsubs))
#                 position.append(pos[0])
#                 directory[dictofsubs[element][-1]]=dirpos
#                 dirpos+=1
#             else:
#                 position.append(len(dictofsubs[element]))
#                 directory[element]=dirpos
#                 dirpos+=1
#         else:
#             for famname,value in dictofsubs.iteritems():
#                 try:
#                     (value[element])
#                 except: pass
#                 else:
#                     position.append(len(value))
#                     directory[famname]=dirpos
#                     dirpos+=1
#     return directory, position
#
#
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

    Examples
    --------
    >>> dict_find({'Key1': 'A', 'Key2': 'B'})

    """
    # Todo: make this robust to repeated values
    # Todo: make this robust to missing values
    return in_dict.keys()[in_dict.values().index(value)]


#

def make_python_identifier(string, namespace=None, reserved_words=None,
                           convert='drop', handle='force'):
    """
    Takes an arbitrary string and creates a valid Python identifier.

    If the input string is in the namespace, return its value.

    If the python identifier created is already in the namespace,
    but the input string is not (ie, two similar strings resolve to
    the same python identifier)

    or if the identifier is a reserved word in the reserved_words
    list, or is a python default reserved word,
    adds _1, or if _1 is in the namespace, _2, etc.

    Parameters
    ----------
    string : <basestring>
        The text to be converted into a valid python identifier
    namespace : <dictionary>
        Map of existing translations into python safe identifiers.
        This is to ensure that two strings are not translated into
        the same python identifier
    reserved_words : <list of strings>
        List of words that are reserved (because they have other meanings
        in this particular program, such as also being the names of
        libraries, etc.
    convert : <string>
        Tells the function what to do with characters that are not
        valid in python identifiers
        - 'hex' implies that they will be converted to their hexidecimal
                representation. This is handy if you have variables that
                have a lot of reserved characters, or you don't want the
                name to be dependent on when things were added to the
                namespace
        - 'drop' implies that they will just be dropped altogether
    handle : <string>
        Tells the function how to deal with namespace conflicts
        - 'force' will create a representation which is not in conflict
                  by appending _n to the resulting variable where n is
                  the lowest number necessary to avoid a conflict
        - 'throw' will raise an exception

    Returns
    -------
    identifier : <string>
        A vaild python identifier based on the input string
    namespace : <dictionary>
        An updated map of the translations of words to python identifiers,
        including the passed in 'string'.

    Examples
    --------
    >>> make_python_identifier('Capital')
    ('capital', {'Capital': 'capital'})

    >>> make_python_identifier('multiple words')
    ('multiple_words', {'multiple words': 'multiple_words'})

    >>> make_python_identifier('multiple     spaces')
    ('multiple_spaces', {'multiple     spaces': 'multiple_spaces'})

    When the name is a python keyword, add '_1' to differentiate it
    >>> make_python_identifier('for')
    ('for_1', {'for': 'for_1'})

    Remove leading and trailing whitespace
    >>> make_python_identifier('  whitespace  ')
    ('whitespace', {'  whitespace  ': 'whitespace'})

    Remove most special characters outright:
    >>> make_python_identifier('H@t tr!ck')
    ('ht_trck', {'H@t tr!ck': 'ht_trck'})

    Replace special characters with their hex representations
    >>> make_python_identifier('H@t tr!ck', convert='hex')
    ('h40t_tr21ck', {'H@t tr!ck': 'h40t_tr21ck'})

    remove leading digits
    >>> make_python_identifier('123abc')
    ('abc', {'123abc': 'abc'})

    already in namespace
    >>> make_python_identifier('Variable$', namespace={'Variable$':'variable'})
    ('variable', {'Variable$': 'variable'})

    namespace conflicts
    >>> make_python_identifier('Variable$', namespace={'Variable@':'variable'})
    ('variable_1', {'Variable@': 'variable', 'Variable$': 'variable_1'})

    >>> make_python_identifier('Variable$', namespace={'Variable@':'variable', 'Variable%':'variable_1'})
    ('variable_2', {'Variable@': 'variable', 'Variable%': 'variable_1', 'Variable$': 'variable_2'})

    throw exception instead
    >>> make_python_identifier('Variable$', namespace={'Variable@':'variable'}, handle='throw')
    Traceback (most recent call last):
     ...
    NameError: variable already exists in namespace or is a reserved word


    References
    ----------
    Identifiers must follow the convention outlined here:
        https://docs.python.org/2/reference/lexical_analysis.html#identifiers
    """

    if namespace is None:
        namespace = {}

    if reserved_words is None:
        reserved_words = []

    if string in namespace:
        return namespace[string], namespace

    # create a working copy (and make it lowercase, while we're at it)
    s = string.lower()

    # remove leading and trailing whitespace
    s = s.strip()

    # Make spaces into underscores
    s = re.sub('[\\s\\t\\n]+', '_', s)

    if convert == 'hex':
        # Convert invalid characters to hex
        s = ''.join([c.encode("hex") if re.findall('[^0-9a-zA-Z_]', c) else c for c in s])

    elif convert == 'drop':
        # Remove invalid characters
        s = re.sub('[^0-9a-zA-Z_]', '', s)

    # Remove leading characters until we find a letter or underscore
    s = re.sub('^[^a-zA-Z_]+', '', s)

    # Check that the string is not a python identifier
    while (s in keyword.kwlist or
                   s in namespace.values() or
                   s in reserved_words):
        if handle == 'throw':
            raise NameError(s + ' already exists in namespace or is a reserved word')
        if handle == 'force':
            if re.match(".*?_\d+$", s):
                i = re.match(".*?_(\d+)$", s).groups()[0]
                s = s.strip('_' + i) + '_' + str(int(i) + 1)
            else:
                s += '_1'

    namespace[string] = s

    return s, namespace
