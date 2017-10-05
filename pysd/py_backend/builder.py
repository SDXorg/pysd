"""builder.py
Refactored May 26 2016
James Houghton
james.p.houghton@gmail.com

This is code to assemble a pysd model once all of the elements have
been translated from their native language into python compatible syntax.
There should be nothing in this file that has to know about either vensim or
xmile specific syntax.
"""

from __future__ import absolute_import
import textwrap
import yapf
from .._version import __version__
from ..py_backend import utils
import os
import warnings
import pkg_resources

def build(elements, subscript_dict, namespace, outfile_name):
    """
    Actually constructs and writes the python representation of the model

    Parameters
    ----------
    elements: list
        Each element is a dictionary, with the various components needed to assemble
        a model component in python syntax. This will contain multiple entries for
        elements that have multiple definitions in the original file, and which need
        to be combined.

    subscript_dict: dictionary
        A dictionary containing the names of subscript families (dimensions) as keys, and
        a list of the possible positions within that dimension for each value

    namespace: dictionary
        Translation from original model element names (keys) to python safe
        function identifiers (values)

    outfile_name: string
        The name of the file to write the model to.
    """
    # Todo: deal with model level documentation
    # Todo: Make np, PySD.functions import conditional on usage in the file
    # Todo: Make presence of subscript_dict instantiation conditional on usage
    # Todo: Sort elements (alphabetically? group stock funcs?)
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

    from pysd.py_backend.functions import cache
    from pysd.py_backend import functions

    _subscript_dict = %(subscript_dict)s

    _namespace = %(namespace)s

    %(functions)s

    ''' % {'subscript_dict': repr(subscript_dict),
           'functions': '\n'.join(functions),
           #'namespace': '{\n' + '\n'.join(['%s: %s' % (key, namespace[key]) for key in
           #                                namespace.keys()]) + '\n}',
           'namespace': repr(namespace),
           'outfile': outfile_name,
           'version': __version__}

    style_file = os.path.dirname(os.path.realpath(__file__)) + '/output_style.yapf'
    text = text.replace('\t', '    ')
    text, changed = yapf.yapf_api.FormatCode(textwrap.dedent(text),
                                             style_config=style_file)

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
    elif element['kind'] in ['setup', 'stateful']:  # setups only get called once, caching is wasted
        cache_type = ''
    elif element['kind'] == 'component':
        cache_type = "@cache('step')"
    elif element['kind'] == 'stateful':
        cache_type = ''
    elif element['kind'] == 'lookup':  # lookups may be called with different values in a round
        cache_type = ''
    else:
        raise AttributeError("Bad value for 'kind'")

    if len(element['py_expr']) > 1:
        contents = 'return utils.xrmerge([%(das)s,])' % {'das': ',\n'.join(element['py_expr'])}
    else:
        contents = 'return %(py_expr)s' % {'py_expr': element['py_expr'][0]}

    indent = 8
    element.update({'cache': cache_type,
                    'ulines': '-' * len(element['real_name']),
                    'contents': contents.replace('\n',
                                                 '\n' + ' ' * indent)})  # indent lines 2 onward
    element['doc'] = element['doc'].replace('\\', '\n    ')

    if element['kind'] == 'stateful':
        func = '''
    %(py_name)s = %(py_expr)s
            ''' % {'py_name': element['py_name'], 'py_expr': element['py_expr'][0]}

    else:
        func = '''
    %(cache)s
    def %(py_name)s(%(arguments)s):
        """
        %(real_name)s

        %(unit)s
        
        %(kind)s

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
        if element['py_expr'] != "None":  # for
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

    return list(outs.values())


def add_stock(identifier, subs, expression, initial_condition, subscript_dict):
    """
    Creates new model element dictionaries for the model elements associated
    with a stock.

    Parameters
    ----------
    identifier: basestring
        the python-safe name of the stock

    subs: list
        a list of subscript elements

    expression: basestring
        The formula which forms the derivative of the stock

    initial_condition: basestring
        Formula which forms the initial condition for the stock

    subscript_dict: dictionary
        Dictionary describing the possible dimensions of the stock's subscripts

    Returns
    -------
    reference: string
        a string to use in place of the 'INTEG...' pieces in the element expression string,
        a reference to the stateful object
    new_structure: list

        list of additional model element dictionaries. When there are subscripts,
        constructs an external 'init' and 'ddt' function so that these can be appropriately
        aggregated

    """
    new_structure = []

    if len(subs) == 0:
        stateful_py_expr = 'functions.Integ(lambda: %s, lambda: %s)' % (expression,
                                                                        initial_condition)
    else:
        stateful_py_expr = 'functions.Integ(lambda: _d%s_dt(), lambda: _init_%s())' % (identifier,
                                                                                       identifier)

        try:
            decoded = initial_condition.decode('unicode-escape')
            initial_condition_numeric = decoded.isnumeric()
        except AttributeError:
            # I believe this should be okay for Py3 but should be checked
            initial_condition_numeric = initial_condition.isnumeric()

        if subs and initial_condition_numeric:
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
        new_structure.append({
            'py_name': '_init_%s' % identifier,
            'real_name': 'Implicit',
            'kind': 'setup',  # not explicitly specified in the model file, but must exist
            'py_expr': initial_condition,
            'subs': subs,
            'doc': 'Provides initial conditions for %s function' % identifier,
            'unit': 'See docs for %s' % identifier,
            'arguments': ''
        })

        new_structure.append({
            'py_name': '_d%s_dt' % identifier,
            'real_name': 'Implicit',
            'kind': 'component',
            'doc': 'Provides derivative for %s function' % identifier,
            'subs': subs,
            'unit': 'See docs for %s' % identifier,
            'py_expr': expression,
            'arguments': ''
        })

    # describe the stateful object
    stateful = {
        'py_name': 'integ_%s' % identifier,
        'real_name': 'Representation of  %s' % identifier,
        'doc': 'Integrates Expression %s' % expression,
        'py_expr': stateful_py_expr,
        'unit': 'None',
        'subs': '',
        'kind': 'stateful',
        'arguments': ''
    }

    new_structure.append(stateful)
    return "%s()" % stateful['py_name'], new_structure


def add_n_delay(delay_input, delay_time, initial_value, order, subs, subscript_dict):
    """
    Creates code to instantiate a stateful 'Delay' object,
    and provides reference to that object's output.

    The name of the stateful object is based upon the passed in parameters, so if
    there are multiple places where identical delay functions are referenced, the
    translated python file will only maintain one stateful object, and reference it
    multiple times.

    Parameters
    ----------

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

    order: string
        The number of stocks in the delay pipeline. As we construct the delays at
        build time, this must be an integer and cannot be calculated from other
        model components. Anything else will yield a ValueError.

    Returns
    -------
    reference: basestring
        reference to the delay object `__call__` method, which will return the output
        of the delay process

    new_structure: list
        list of element construction dictionaries for the builder to assemble
    """
    # the py name has to be unique to all the passed parameters, or if there are two things
    # that delay the output by different amounts, they'll overwrite the original function...

    stateful = {
        'py_name': utils.make_python_identifier('delay_%s_%s_%s_%s' % (delay_input,
                                                                       delay_time,
                                                                       initial_value,
                                                                       order))[0],
        'real_name': 'Delay of %s' % delay_input,
        'doc': 'Delay time: %s \n Delay initial value %s \n Delay order %s' % (
            delay_time, initial_value, order),
        'py_expr': 'functions.Delay(lambda: %s, lambda: %s, lambda: %s, lambda: %s)' % (
            delay_input, delay_time, initial_value, order),
        'unit': 'None',
        'subs': '',
        'kind': 'stateful',
        'arguments': ''
    }

    return "%s()" % stateful['py_name'], [stateful]


def add_n_smooth(smooth_input, smooth_time, initial_value, order, subs, subscript_dict):
    """Constructs stock and flow chains that implement the calculation of
        a smoothing function.

        Parameters
        ----------
        smooth_input: <string>
            Reference to the model component that is the input to the smoothing function

        smooth_time: <string>
            Can be a number (in string format) or a reference to another model element
            which will calculate the delay. This is calculated throughout the simulation
            at runtime.

        initial_value: <string>
            This is used to initialize the stocks that are present in the delay. We
            initialize the stocks with equal values so that the outflow in the first
            timestep is equal to this value.

        order: string
            The number of stocks in the delay pipeline. As we construct the delays at
            build time, this must be an integer and cannot be calculated from other
            model components. Anything else will yield a ValueError.

        subs: list of strings
            List of strings of subscript indices that correspond to the
            list of expressions, and collectively define the shape of the output
            See `builder.add_flaux` for more info

        Returns
        -------
        reference: basestring
            reference to the smooth object `__call__` method, which will return the output
            of the smooth process

        new_structure: list
            list of element construction dictionaries for the builder to assemble
        """

    stateful = {
        'py_name': utils.make_python_identifier('smooth_%s_%s_%s_%s' % (smooth_input,
                                                                        smooth_time,
                                                                        initial_value,
                                                                        order))[0],
        'real_name': 'Smooth of %s' % smooth_input,
        'doc': 'Smooth time: %s \n Smooth initial value %s \n Smooth order %s' % (
            smooth_time, initial_value, order),
        'py_expr': 'functions.Smooth(lambda: %s, lambda: %s, lambda: %s, lambda: %s)' % (
            smooth_input, smooth_time, initial_value, order),
        'unit': 'None',
        'subs': '',
        'kind': 'stateful',
        'arguments': ''
    }

    return "%s()" % stateful['py_name'], [stateful]


def add_n_trend(trend_input, average_time, initial_trend, subs, subscript_dict):
    """Trend.

        Parameters
        ----------
        trend_input: <string>

        average_time: <string>


        trend_initial: <string>

        subs: list of strings
            List of strings of subscript indices that correspond to the
            list of expressions, and collectively define the shape of the output
            See `builder.add_flaux` for more info

        Returns
        -------
        reference: basestring
            reference to the trend object `__call__` method, which will return the output
            of the trend process

        new_structure: list
            list of element construction dictionaries for the builder to assemble
        """

    stateful = {
        'py_name': utils.make_python_identifier('smooth_%s_%s_%s' % (trend_input,
                                                                        average_time,
                                                                        initial_trend))[0],
        'real_name': 'Smooth of %s' % trend_input,
        'doc': 'Trend average time: %s \n Trend initial value %s' % (
            average_time, initial_trend),
        'py_expr': 'functions.Trend(lambda: %s, lambda: %s, lambda: %s)' % (
            trend_input, average_time, initial_trend),
        'unit': 'None',
        'subs': '',
        'kind': 'stateful',
        'arguments': ''
    }

    return "%s()" % stateful['py_name'], [stateful]

def add_initial(initial_input):
    """
    Constructs a stateful object for handling vensim's 'Initial' functionality

    Parameters
    ----------
    initial_input: basestring
        The expression which will be evaluated, and the first value of which returned

    Returns
    -------
    reference: basestring
        reference to the Initial object `__call__` method,
        which will return the first calculated value of `initial_input`

    new_structure: list
        list of element construction dictionaries for the builder to assemble

    """
    stateful = {
        'py_name': utils.make_python_identifier('initial_%s' % initial_input)[0],
        'real_name': 'Smooth of %s' % initial_input,
        'doc': 'Returns the value taken on during the initialization phase',
        'py_expr': 'functions.Initial(lambda: %s)' % (
            initial_input),
        'unit': 'None',
        'subs': '',
        'kind': 'stateful',
        'arguments': ''
    }

    return "%s()" % stateful['py_name'], [stateful]


def add_macro(macro_name, filename, arg_names, arg_vals):
    """
    Constructs a stateful object instantiating a 'Macro'

    Parameters
    ----------
    macro_name: basestring
        python safe name for macro
    filename: basestring
        filepath to macro definition
    func_args: dict
        dictionary of values to be passed to macro
        {key: function}

    Returns
    -------
    reference: basestring
        reference to the Initial object `__call__` method,
        which will return the first calculated value of `initial_input`

    new_structure: list
        list of element construction dictionaries for the builder to assemble

    """
    func_args = '{ %s }' % ', '.join(["'%s': lambda: %s" % (key, val) for key, val in
                                      zip(arg_names, arg_vals)])

    stateful = {
        'py_name': 'macro_' + macro_name + '_' + '_'.join(
            [utils.make_python_identifier(f)[0] for f in arg_vals]),
        'real_name': 'Macro Instantiation of ' + macro_name,
        'doc': 'Instantiates the Macro',
        'py_expr': "functions.Macro('%s', %s, '%s')" % (filename, func_args, macro_name),
        'unit': 'None',
        'subs': '',
        'kind': 'stateful',
        'arguments': ''
    }

    return "%s()" % stateful['py_name'], [stateful]


def add_incomplete(var_name, dependencies):
    """
    Incomplete functions don't really need to be 'builders' as they
     add no new real structure, but it's helpful to have a function
     in which we can raise a warning about the incomplete equation
     at translate time.
    """
    warnings.warn('%s has no equation specified' %var_name,
                   SyntaxWarning, stacklevel=2)

    # first arg is `self` reference
    return "functions.incomplete(%s)" % ', '.join(dependencies[1:]), []


