"""builder.py
Refactored May 26 2016
James Houghton
james.p.houghton@gmail.com

This is code to assemble a pysd model once all of the elements have
been translated from their native language into python compatible syntax.
There should be nothing in this file that has to know about either vensim or
xmile specific syntax.


"""


import textwrap
import autopep8
from _version import __version__
import utils


def build(elements, subscript_dict, namespace, outfile_name):
    """
    Takes in a list of model components

    Parameters
    ----------
    elements: list

    subscript_dict: dictionary

    outfile_name: string

    Returns
    -------

    """
    # Todo: deal with model level documentation
    # Todo: Make np import conditional on usage in the file
    # Todo: Make pysd functions import conditional on usage in the file
    # Todo: Make presence of subscript_dict instantiation conditional on usage
    # Todo: Sort elements (alphabetically? group stock funcs?)
    # Todo: do something better than hardcoding the time function
    elements = merge_partial_elements(elements)
    functions = [build_element(element, subscript_dict) for element in elements]


    text = '''
    """
    Python model %(outfile)s
    Translated using PySD version %(version)s
    """
    from __future__ import division
    import numpy as np
    from pysd import utils
    import xarray as xr

    from pysd.functions import cache
    from pysd import functions

    _subscript_dict = %(subscript_dict)s

    _namespace = %(namespace)s

    %(functions)s

    def time():
        return _t
    functions.time = time
    functions.initial_time = initial_time

    ''' % {'subscript_dict': repr(subscript_dict),
           'functions': '\n'.join(functions),
           'namespace': repr(namespace),
           'outfile': outfile_name,
           'version': __version__}

    text = autopep8.fix_code(textwrap.dedent(text),
                             options={'aggressive': 100,
                                      'max_line_length': 99,
                                      'experimental': True})

    # this is used for testing
    if outfile_name == 'return':
        return text

    with open(outfile_name, 'w') as out:
        out.write(text)


def build_element(element, subscript_dict):
    """
    Returns a string that has processed a single element dictionary
    Parameters
    ----------
    element: dictionary
        dictionary containing at least the elements:
        - kind: ['constant', 'setup', 'component', 'lookup']
            Different types of elements will be built differently
        - py_expr: string
            An expression that has been converted already into python syntax
        - subs: list of lists
            Each sublist contains coordinates for initialization of a particular
            part of a subscripted function, the list of subscripts vensim attaches to an equation

    subscript_dict: dictionary

    Returns
    -------

    """
    if element['kind'] == 'constant':
        cache_type = "@cache('run')"
    elif element['kind'] == 'setup':  # setups only get called once, so caching is wasted
        cache_type = ''
    elif element['kind'] == 'component':
        cache_type = "@cache('step')"
    elif element['kind'] == 'macro':
        cache_type = ''
    elif element['kind'] == 'lookup':  # lookups may be called with different values in a round
        cache_type = ''
    else:
        raise AttributeError("Bad value for 'kind'")

    if len(element['py_expr']) > 1:
        contents = 'return utils.xrmerge([%(das)s])' % {'das': ',\n'.join(element['py_expr'])}
    else:
        contents = 'return %(py_expr)s' % {'py_expr': element['py_expr'][0]}

    indent = 8
    element.update({'cache': cache_type,
                    'ulines': '-' * len(element['real_name']),
                    'contents': contents.replace('\n',
                                                 '\n' + ' ' * indent)})  # indent lines 2 onward
    element['doc'] = element['doc'].replace('\\', '\n    ')

    func = '''
    %(cache)s
    def %(py_name)s(%(arguments)s):
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
                'kind': element['kind'],
                'arguments': element['arguments']
            }

        else:
            outs[name]['doc'] = outs[name]['doc'] or element['doc']
            outs[name]['unit'] = outs[name]['unit'] or element['unit']
            outs[name]['py_expr'] += [element['py_expr']]
            outs[name]['subs'] += [element['subs']]
            outs[name]['arguments'] = element['arguments']

    return outs.values()


def add_stock(identifier, subs, expression, initial_condition, subscript_dict):
    """
    Creates new model element dictionaries for the model elements associated
    with a stock.


    Parameters
    ----------
    identifier
    subs
    expression
    initial_condition
    subscript_dict

    Returns
    -------
    a string to use in place of the 'INTEG...' pieces in the element expression string,
    a list of additional model elements to add

    Examples
    --------
    >>> add_stock('stock_name', [], 'inflow_a', '10')

    """
    # take care of cases when a float is passed as initialization for an array.
    # this might be better located in the translation function in the future.
    if subs and initial_condition.decode('unicode-escape').isnumeric():
        coords = utils.make_coord_dict(subs, subscript_dict, terse=False)
        dims = [utils.find_subscript_name(subscript_dict, sub) for sub in subs]
        shape = [len(coords[dim]) for dim in dims]
        initial_condition = textwrap.dedent("""\
            xr.DataArray(data=np.ones(%(shape)s)*%(value)s,
                         coords=%(coords)s,
                         dims=%(dims)s )""" % {
            'shape': shape,
            'value': initial_condition,
            'coords': repr(coords),
            'dims': repr(dims)})

    # create the stock initialization element
    init_element = {
        'py_name': '_init_%s' % identifier,
        'real_name': 'Implicit',
        'kind': 'setup',  # not explicitly specified in the model file, but must exist
        'py_expr': initial_condition,
        'subs': subs,
        'doc': 'Provides initial conditions for %s function' % identifier,
        'unit': 'See docs for %s' % identifier,
        'arguments': ''
    }

    ddt_element = {
        'py_name': '_d%s_dt' % identifier,
        'real_name': 'Implicit',
        'kind': 'component',
        'doc': 'Provides derivative for %s function' % identifier,
        'subs': subs,
        'unit': 'See docs for %s' % identifier,
        'py_expr': expression,
        'arguments': ''
    }

    return "_state['%s']" % identifier, [init_element, ddt_element]


def add_n_delay(delay_input, delay_time, initial_value, order, subs, subscript_dict):
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

    delayed_variable = delay_input[:-2]
    identifier = '_' + delayed_variable + '_delay'

    output_element = {
        'py_name': identifier,
        'real_name': 'Implicit',
        'kind': 'component',
        'doc': 'delayed value of %s' % delayed_variable,
        'subs': subs,
        'unit': 'See docs for %s' % identifier,
        'py_expr': "(%(stock)s / (%(delay)s / %(order)s)).loc[{'_delay': %(order)s-1}]" % {
            'stock': '_state["%s"]' % identifier,
            'delay': delay_time,
            'order': order,
        },
        'arguments': ''
    }

    coords = utils.make_coord_dict(subs, subscript_dict, terse=False)
    dims = [utils.find_subscript_name(subscript_dict, sub) for sub in subs]

    try:
        coords.update({'_delay': range(int(order))})
    except TypeError or ValueError:
        raise TypeError('Order of delay on %s must be an integer, instead recieved %s' % (
            identifier, str(order)))

    dims.append('_delay')

    shape = [len(coords[dim]) for dim in dims]
    initial_condition = textwrap.dedent("""\
            xr.DataArray(data=np.ones(%(shape)s)*%(value)s,
                         coords=%(coords)s,
                         dims=%(dims)s )*%(delay)s/%(order)s""" % {
        'shape': shape,
        'value': initial_value,
        'coords': repr(coords),
        'dims': repr(dims),
        'delay': delay_time,
        'order': order})

    init_element = {
        'py_name': '_init_%s' % identifier,
        'real_name': 'Implicit',
        'kind': 'setup',  # not explicitly specified in the model file, but must exist
        'py_expr': initial_condition,
        'subs': dims,
        'doc': 'Provides initial conditions for delay of %s function' % identifier,
        'unit': 'See docs for %s' % identifier,
        'arguments': ''
    }

    expression = textwrap.dedent("""\
        ((%(stock)s / (%(delay)s / %(order)s)).shift(**{'_delay': 1}).fillna(%(inval)s)
            - (%(stock)s / (%(delay)s / %(order)s)))""" % {
        'stock': '_state["%s"]' % identifier,
        'delay': delay_time,
        'order': order,
        'inval': delay_input
    })

    ddt_element = {
        'py_name': '_d%s_dt' % identifier,
        'real_name': 'Implicit',
        'kind': 'component',
        'doc': 'Provides derivative for %s function' % identifier,
        'subs': dims,
        'unit': 'See docs for %s' % identifier,
        'py_expr': expression,
        'arguments': ''
    }

    return ("%s()" % identifier,
            [init_element, ddt_element, output_element])


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

