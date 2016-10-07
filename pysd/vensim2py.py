"""
Translates vensim .mdl file to pieces needed by the builder module to write a python version of the
model. Everything that requires knowledge of vensim syntax should be in this file.
"""

import re
import parsimonious
from . import builder
from . import utils
import textwrap
import numpy as np


def get_file_sections(file_str):
    """
    This is where we separate out the macros from the rest of the model file.
    Working based upon documentation at: https://www.vensim.com/documentation/index.html?macros.htm

    Macros will probably wind up in their own python modules eventually.

    Parameters
    ----------
    file_str

    Returns
    -------
    entries: list of dictionaries
        Each dictionary represents a different section of the model file, either a macro,
        or the main body of the model file. The dictionaries contain various elements:
        - returns: list of strings
            represents what is returned from a macro (for macros) or empty for main model
        - params: list of strings
            represents what is passed into a macro (for macros) or empty for main model
        - name: string
            the name of the macro, or 'main' for main body of model
        - string: string
            string representing the model section
    Examples
    --------
    >>> get_file_sections(r'a~b~c| d~e~f| g~h~i|')
    [{'returns': [], 'params': [], 'name': 'main', 'string': 'a~b~c| d~e~f| g~h~i|'}]

    """
    file_structure_grammar = r"""
    file = encoding? (macro / main)+
    macro = ":MACRO:" _ name _ "(" _ (name _ ","? _)+ _ ":"? _ (name _ ","? _)* _ ")" ~r".+?(?=:END OF MACRO:)" ":END OF MACRO:"
    main = !":MACRO:" ~r".+(?!:MACRO:)"

    name = basic_id / escape_group
    basic_id = ~r"[a-zA-Z][a-zA-Z0-9_\s]*"

    # between quotes, either escaped quote or character that is not a quote
    escape_group = "\"" ( "\\\"" / ~r"[^\"]" )* "\""
    encoding = ~r"\{[^\}]*\}"

    _ = ~r"[\s\\]*"  # whitespace character
    """  # the leading 'r' for 'raw' in this string is important for handling backslashes properly

    parser = parsimonious.Grammar(file_structure_grammar)
    tree = parser.parse(file_str)

    class FileParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.entries = []
            self.visit(ast)

        def visit_main(self, n, vc):
            self.entries.append({'name'   : 'main',
                                 'params' : [],
                                 'returns': [],
                                 'string' : n.text.strip()})

        def visit_macro(self, n, vc):
            name = vc[2]
            params = vc[6]
            returns = vc[10]
            text = vc[13]
            self.entries.append({'name'   : name,
                                 'params' : [x.strip() for x in params.split(',')] if params else [],
                                 'returns': [x.strip() for x in
                                             returns.split(',')] if returns else [],
                                 'string' : text.strip()})

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text or ''

    return FileParser(tree).entries


def get_model_elements(model_str):
    """
    Takes in a string representing model text and splits it into elements

    I think we're making the assumption that all newline characters are removed...

    Parameters
    ----------
    model_str : string


    Returns
    -------
    entries : array of dictionaries
        Each dictionary contains the components of a different model element, separated into the
        equation, units, and docstring.

    Examples
    --------

    # Basic Parsing:
    >>> get_model_elements(r'a~b~c| d~e~f| g~h~i|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': 'f', 'unit': 'e', 'eqn': 'd'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]

    # Special characters are escaped within double-quotes:
    >>> get_model_elements(r'a~b~c| d~e"~"~f| g~h~i|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': 'f', 'unit': 'e"~"', 'eqn': 'd'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]
    >>> get_model_elements(r'a~b~c| d~e~"|"f| g~h~i|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': '"|"f', 'unit': 'e', 'eqn': 'd'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]

    # Double-quotes within escape groups are themselves escaped with backslashes:
    >>> get_model_elements(r'a~b~c| d~e"\\\"~"~f| g~h~i|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': 'f', 'unit': 'e"\\\\"~"', 'eqn': 'd'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]
    >>> get_model_elements(r'a~b~c| d~e~"\\\"|"f| g~h~i|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': '"\\\\"|"f', 'unit': 'e', 'eqn': 'd'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]
    >>> get_model_elements(r'a~b~c| d~e"x\\nx"~f| g~h~|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': 'f', 'unit': 'e"x\\\\nx"', 'eqn': 'd'}, {'doc': '', 'unit': 'h', 'eqn': 'g'}]

    # Todo: Handle model-level or section-level documentation
    >>> get_model_elements(r'*** .model doc ***~ Docstring!| d~e~f| g~h~i|')
    [{'doc': 'Docstring!', 'unit': '', 'eqn': ''}, {'doc': 'f', 'unit': 'e', 'eqn': 'd'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]

    # Handle control sections, returning appropriate docstring pieces
    >>> get_model_elements(r'a~b~c| ****.Control***~ Simulation Control Parameters | g~h~i|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]

    # Handle the model display elements (ignore them)
    >>> get_model_elements(r'a~b~c| d~e~f| \\\---///junk|junk~junk')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': 'f', 'unit': 'e', 'eqn': 'd'}]


    Notes
    -----
    - Tildes and pipes are not allowed in element docstrings, but we should still handle them there

    """

    model_structure_grammar = r"""
    model = (entry / section)+ sketch?
    entry = element "~" element "~" element ("~" element)? "|"
    section = element "~" element "|"
    sketch = ~r".*"  #anything

    # Either an escape group, or a character that is not tilde or pipe
    element = (escape_group / ~r"[^~|]")*

    # between quotes, either escaped quote or character that is not a quote
    escape_group = "\"" ( "\\\"" / ~r"[^\"]" )* "\""
    """
    parser = parsimonious.Grammar(model_structure_grammar)
    tree = parser.parse(model_str)

    class ModelParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.entries = []
            self.visit(ast)

        def visit_entry(self, n, vc):
            self.entries.append({'eqn' : vc[0].strip(),
                                 'unit': vc[2].strip(),
                                 'doc' : vc[4].strip(),
                                 'kind': 'entry'})

        def visit_section(self, n, vc):
            if vc[2].strip() != "Simulation Control Parameters":
                self.entries.append({'eqn' : '',
                                     'unit': '',
                                     'doc' : vc[2].strip(),
                                     'kind': 'section'})

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text or ''

    return ModelParser(tree).entries


def get_equation_components(equation_str):
    """
    Breaks down a string representing only the equation part of a model element.
    Recognizes the various types of model elements that may exist, and identifies them.

    Parameters
    ----------
    equation_str : basestring
        the first section in each model element - the full equation.

    Returns
    -------
    Returns a dictionary containing the following:

    real_name: basestring
        The name of the element as given in the original vensim file

    subs: list of strings
        list of subscripts or subscript elements

    expr: basestring

    kind: basestring
        What type of equation have we found?
        - *component* - normal model expression or constant
        - *lookup* - a lookup table
        - *subdef* - a subscript definition

    Examples
    --------
    >>> get_equation_components(r'constant = 25')
    {'expr': '25', 'kind': 'component', 'subs': [], 'real_name': 'constant'}

    Notes
    -----
    in this function we dont create python identifiers, we use real names.
    This is so that when everything comes back together, we can manage
    any potential namespace conflicts properly
    """

    component_structure_grammar = r"""
    entry = component / subscript_definition / lookup_definition
    component = name _ subscriptlist? _ "=" _ expression
    subscript_definition = name _ ":" _ subscript _ ("," _ subscript)*
    lookup_definition = name _ &"(" _ expression  # uses lookahead assertion to capture whole group

    name = basic_id / escape_group
    subscriptlist = '[' _ subscript _ ("," _ subscript)* _ ']'
    expression = ~r".*"  # expression could be anything, at this point.

    subscript = basic_id / escape_group

    basic_id = ~r"[a-zA-Z][a-zA-Z0-9_\s]*"
    escape_group = "\"" ( "\\\"" / ~r"[^\"]" )* "\""
    _ = ~r"[\s\\]*"  # whitespace character
    """

    # replace any amount of whitespace  with a single space
    equation_str = equation_str.replace('\\t', ' ')
    equation_str = re.sub(r"\s+", ' ', equation_str)

    parser = parsimonious.Grammar(component_structure_grammar)
    tree = parser.parse(equation_str)

    class ComponentParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.subscripts = []
            self.real_name = None
            self.expression = None
            self.kind = None
            self.visit(ast)

        def visit_subscript_definition(self, n, vc):
            self.kind = 'subdef'

        def visit_lookup_definition(self, n, vc):
            self.kind = 'lookup'

        def visit_component(self, n, vc):
            self.kind = 'component'

        def visit_name(self, n, vc):
            (name,) = vc
            self.real_name = name.strip()

        def visit_subscript(self, n, vc):
            (subscript,) = vc
            self.subscripts.append(subscript.strip())

        def visit_expression(self, n, vc):
            self.expression = n.text.strip()

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text

        def visit__(self, n, vc):
            return ' '

    parse_object = ComponentParser(tree)

    return {'real_name': parse_object.real_name,
            'subs'     : parse_object.subscripts,
            'expr'     : parse_object.expression,
            'kind'     : parse_object.kind}


def parse_units(units_str):
    """
    Extract and parse the units
    Extract the bounds over which the expression is assumed to apply.

    Parameters
    ----------
    units_str

    Returns
    -------

    Examples
    --------
    >>> parse_units('Widgets/Month [-10,10,1]')

    >>> parse_units('Month [0,?]')

    >>> parse_units('Widgets [0,100]')

    """
    return units_str


def parse_general_expression(element, namespace=None, subscript_dict=None):
    """
    Parses a normal expression
    # its annoying that we have to construct and compile the grammar every time...

    Parameters
    ----------
    element: dictionary

    namespace : dictionary

    subscript_dict : dictionary


    Returns
    -------
    translation

    new_elements: list of dictionaries
        If the expression contains builder functions, those builders will create new elements
        to add to our running list (that will eventually be output to a file) such as stock
        initialization and derivative funcs, etc.


    Examples
    --------
    >>> parse_general_expression({'expr': 'INTEG (FlowA, -10)',
    ...                           'py_name':'test_stock',
    ...                           'subs':None},
    ...                          {'FlowA': 'flowa'}),
    ({'kind': 'component', 'py_expr': "_state['test_stock']"},
     [{'kind': 'implicit',
       'subs': None,
       'doc': 'Provides initial conditions for test_stock function',
       'py_name': 'init_test_stock',
       'real_name': None,
       'unit': 'See docs for test_stock',
       'py_expr': '-10'},
      {'py_name': 'dtest_stock_dt',
       'kind': 'implicit',
       'py_expr': 'flowa',
       'real_name': None}])

    """
    if namespace is None:
        namespace = {}
    if subscript_dict is None:
        subscript_dict = {}

    functions = {
        # element-wise functions
        "abs"           : "abs", "integer": "int", "exp": "np.exp", "sin": "np.sin", "cos": "np.cos",
        "sqrt"          : "np.sqrt", "tan": "np.tan", "lognormal": "np.random.lognormal",
        "random normal" : "functions.bounded_normal", "poisson": "np.random.poisson", "ln": "np.log",
        "exprnd"        : "np.random.exponential", "_random uniform": "np.random.rand", "sum": "np.sum",
        "arccos"        : "np.arccos", "arcsin": "np.arcsin", "arctan": "np.arctan",
        "if then else"  : "functions.if_then_else", "step": "functions.step", "modulo": "np.mod",
        "pulse"         : "functions.pulse", "pulse train": "functions.pulse_train",
        "ramp"          : "functions.ramp", "min": "np.minimum", "max": "np.maximum",
        "active initial": "functions.active_initial", "xidz": "functions.xidz",
        "zidz"          : "functions.zidz",
        "random uniform": "functions.random_uniform",

        # vector functions
        "vmin"          : "np.min", "vmax": "np.max", "prod": "np.prod"
    }

    builders = {
        "integ"   : lambda expr, init: builder.add_stock(element['py_name'], element['subs'],
                                                         expr, init, subscript_dict),
        "delay1"  : lambda in_var, dtime: builder.add_n_delay(in_var, dtime, '0', '1',
                                                              element['subs'], subscript_dict),
        "delay1i" : lambda in_var, dtime, init: builder.add_n_delay(in_var, dtime, init, '1',
                                                                    element['subs'], subscript_dict),
        "delay3"  : lambda in_var, dtime: builder.add_n_delay(in_var, dtime, '0', '3',
                                                              element['subs'], subscript_dict),
        "delay3i" : lambda in_var, dtime, init: builder.add_n_delay(in_var, dtime, init, '3',
                                                                    element['subs'], subscript_dict),
        "delay n" : lambda in_var, dtime, init, order: builder.add_n_delay(in_var, dtime,
                                                                           init, order,
                                                                           element['subs'],
                                                                           subscript_dict),
        "smooth"  : lambda in_var, dtime: builder.add_n_smooth(in_var, dtime, '0', '1',
                                                               element['subs'], subscript_dict),
        "smoothi" : lambda in_var, dtime, init: builder.add_n_smooth(in_var, dtime, init, '1',
                                                                     element['subs'],
                                                                     subscript_dict),
        "smooth3" : lambda in_var, dtime: builder.add_n_smooth(in_var, dtime, '0', '3',
                                                               element['subs'], subscript_dict),
        "smooth3i": lambda in_var, dtime, init: builder.add_n_smooth(in_var, dtime, init, '3',
                                                                     element['subs'],
                                                                     subscript_dict),
        "smooth n": lambda in_var, dtime, init, order: builder.add_n_smooth(in_var, dtime,
                                                                            init, order,
                                                                            element['subs'],
                                                                            subscript_dict),
        "initial" : lambda initial_input: builder.add_initial(initial_input)
    }

    in_ops = {
        "+"    : "+", "-": "-", "*": "*", "/": "/", "^": "**", "=": "==", "<=": "<=", "<>": "!=",
        "<"    : "<", ">=": ">=", ">": ">",
        ":and:": " and ", ":or:": " or "}  # spaces important for word-based operators

    pre_ops = {
        "-": "-", ":not:": " not ",  # spaces important for word-based operators
        "+": " "  # space is important, so that and empty string doesn't slip through generic
    }

    # in the following, if lists are empty use non-printable character
    # everything needs to be escaped before going into the grammar, in case it includes quotes
    sub_names_list = [re.escape(x) for x in subscript_dict.keys()] or ['\\a']
    sub_elems_list = [re.escape(y) for x in subscript_dict.values() for y in x] or ['\\a']
    ids_list = [re.escape(x) for x in namespace.keys()] or ['\\a']
    in_ops_list = [re.escape(x) for x in in_ops.keys()]
    pre_ops_list = [re.escape(x) for x in pre_ops.keys()]

    expression_grammar = r"""
    expr_type = array / expr
    expr = _ pre_oper? _ (lookup_def / nested_build_call / build_call / call / parens / number / reference ) _ (in_oper _ expr)?


    lookup_def = ~r"(WITH\ LOOKUP)"I _ "(" _ reference _ "," _ "(" _  ("[" ~r"[^\]]*" "]" _ ",")?  ( "(" _ expr _ "," _ expr _ ")" ","? _ )+ _ ")" _ ")"
    call = (func / id) _ "(" _ (expr _ ","? _)* ")" # allows calls with no arguments
    nested_build_call = builder _ "(" _ call _ ")" # allows calls with no arguments
    build_call = builder _ "(" _ (expr _ ","? _)* ")" # allows calls with no arguments

    parens   = "(" _ expr _ ")"

    reference = id _ subscript_list?
    subscript_list = "[" _ ((sub_name / sub_element) _ ","? _)+ "]"

    array = (number _ ("," / ";")? _)+ !~r"."  # negative lookahead for anything other than an array
    number = ~r"\d+\.?\d*(e[+-]\d+)?"

    id = ~r"(%(ids)s)"I
    sub_name = ~r"(%(sub_names)s)"I  # subscript names (if none, use non-printable character)
    sub_element = ~r"(%(sub_elems)s)"I  # subscript elements (if none, use non-printable character)

    func = ~r"(%(funcs)s)"I  # functions (case insensitive)
    in_oper = ~r"(%(in_ops)s)"I  # infix operators (case insensitive)
    pre_oper = ~r"(%(pre_ops)s)"I  # prefix operators (case insensitive)
    builder = ~r"(%(builders)s)"I  # builder functions (case insensitive)

    _ = ~r"[\s\\]*"  # whitespace character
    """ % {
        # In the following, we have to sort keywords in decreasing order of length so that the
        # peg parser doesn't quit early when finding a partial keyword
        'sub_names': '|'.join(reversed(sorted(sub_names_list, key=len))),
        'sub_elems': '|'.join(reversed(sorted(sub_elems_list, key=len))),
        'ids'      : '|'.join(reversed(sorted(ids_list, key=len))),
        'funcs'    : '|'.join(reversed(sorted(functions.keys(), key=len))),
        'in_ops'   : '|'.join(reversed(sorted(in_ops_list, key=len))),
        'pre_ops'  : '|'.join(reversed(sorted(pre_ops_list, key=len))),
        'builders' : '|'.join(reversed(sorted(builders.keys(), key=len))),
    }

    parser = parsimonious.Grammar(expression_grammar)
    tree = parser.parse(element['expr'])

    class ExpressionParser(parsimonious.NodeVisitor):
        # Todo: at some point, we could make the 'kind' identification recursive on expression,
        # so that if an expression is passed into a builder function, the information
        # about whether it is a constant, or calls another function, goes with it.
        def __init__(self, ast):
            self.translation = ""
            self.kind = 'constant'  # change if we reference anything else
            self.new_structure = []
            self.visit(ast)

        def visit_expr_type(self, n, vc):
            s = ''.join(filter(None, vc)).strip()
            self.translation = s

        def visit_expr(self, n, vc):
            s = ''.join(filter(None, vc)).strip()
            self.translation = s
            return s

        def visit_func(self, n, vc):
            self.kind = 'component'
            return functions[n.text.lower()]

        def visit_in_oper(self, n, vc):
            return in_ops[n.text.lower()]

        def visit_pre_oper(self, n, vc):
            return pre_ops[n.text.lower()]

        def visit_reference(self, n, vc):
            self.kind = 'component'
            id_str = vc[0]
            return id_str + '()'

        def visit_id(self, n, vc):
            return namespace[n.text]

        def visit_lookup_def(self, n, vc):
            """ This exists because vensim has multiple ways of doing lookups.
            Which is frustrating."""
            x_val = vc[4]
            pairs = vc[11]
            mixed_list = pairs.replace('(', '').replace(')', '').split(',')
            xs = mixed_list[::2]
            ys = mixed_list[1::2]
            string = "functions.lookup(%(x)s, [%(xs)s], [%(ys)s])" % {
                'x' : x_val,
                'xs': ','.join(xs),
                'ys': ','.join(ys)
            }
            return string

        def visit_array(self, n, vc):
            if 'subs' in element and element['subs']:  # first test handles when subs is not defined
                coords = utils.make_coord_dict(element['subs'], subscript_dict, terse=False)
                dims = [utils.find_subscript_name(subscript_dict, sub) for sub in element['subs']]
                shape = [len(coords[dim]) for dim in dims]
                if ';' in n.text or ',' in n.text:
                    text = n.text.strip(';').replace(' ', '').replace(';', ',')
                    data = np.array([float(s) for s in text.split(',')]).reshape(shape)
                else:
                    data = np.tile(float(n.text), shape)
                datastr = np.array2string(data, separator=',').replace('\n', '').replace(' ', '')
                return textwrap.dedent("""\
                    xr.DataArray(data=%(datastr)s,
                                 coords=%(coords)s,
                                 dims=%(dims)s )""" % {
                    'datastr': datastr,
                    'coords' : repr(coords),
                    'dims'   : repr(dims)})

            else:
                return n.text.replace(' ', '')

        def visit_subscript_list(self, n, vc):
            refs = vc[2]
            subs = [x.strip() for x in refs.split(',')]
            coordinates = utils.make_coord_dict(subs, subscript_dict)
            if len(coordinates):
                return '.loc[%s]' % repr(coordinates)
            else:
                return ' '

        def visit_nested_build_call(self, n, vc):
            call = vc[0]
            args = vc[4]
            self.kind = 'component'
            name, structure = builders[call.strip().lower()](args)
            self.new_structure += structure
            return name

        def visit_build_call(self, n, vc):
            call = vc[0]
            args = vc[4]
            self.kind = 'component'
            arglist = [x.strip() for x in args.split(',')]
            name, structure = builders[call.strip().lower()](*arglist)
            self.new_structure += structure
            return name

        def visit__(self, n, vc):
            """ Handles whitespace characters"""
            return ''

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text

    parse_object = ExpressionParser(tree)

    return ({'py_expr'  : parse_object.translation,
             'kind'     : parse_object.kind,
             'arguments': ''},
            parse_object.new_structure)


def parse_lookup_expression(element):
    """ This syntax parses lookups that are defined with their own element """

    lookup_grammar = r"""
    lookup = _ "(" _ "[" ~r"[^\]]*" "]" _ "," _ ( "(" _ number _ "," _ number _ ")" ","? _ )+ ")"
    number = ("+"/"-")? ~r"\d+\.?\d*(e[+-]\d+)?"
    _ = ~r"[\s\\]*"  # whitespace character
    """
    parser = parsimonious.Grammar(lookup_grammar)
    tree = parser.parse(element['expr'])

    class LookupParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.translation = ""
            self.new_structure = []
            self.visit(ast)

        def visit__(self, n, vc):
            # remove whitespace
            return ''

        def visit_lookup(self, n, vc):
            pairs = vc[9]
            mixed_list = pairs.replace('(', '').replace(')', '').split(',')
            xs = mixed_list[::2]
            ys = mixed_list[1::2]
            string = "functions.lookup(x, [%(xs)s], [%(ys)s])" % {
                'xs': ','.join(xs),
                'ys': ','.join(ys)
            }
            self.translation = string

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text

    parse_object = LookupParser(tree)
    return {'py_expr'  : parse_object.translation,
            'arguments': 'x'}


def translate_vensim(mdl_file):
    """

    Parameters
    ----------
    mdl_file : basestring
        file path of a vensim model file to translate to python

    Returns
    -------

    Examples
    --------
    >>> translate_vensim('../tests/test-models/tests/subscript_3d_arrays/test_subscript_3d_arrays.mdl')

    #>>> translate_vensim('../../tests/test-models/tests/abs/test_abs.mdl')

    #>>> translate_vensim('../../tests/test-models/tests/exponentiation/exponentiation.mdl')

    #>>> translate_vensim('../../tests/test-models/tests/limits/test_limits.mdl')

    """
    # Todo: work out what to do with subranges
    # Todo: parse units string
    # Todo: handle macros

    with open(mdl_file, 'rU') as in_file:
        text = in_file.read()

    # extract model elements
    model_elements = []
    file_sections = get_file_sections(text.replace('\n', ''))
    for section in file_sections:
        if section['name'] == 'main':
            model_elements += get_model_elements(section['string'])

    # extract equation components
    model_docstring = ''
    for entry in model_elements:
        if entry['kind'] == 'entry':
            entry.update(get_equation_components(entry['eqn']))
        elif entry['kind'] == 'section':
            model_docstring += entry['doc']

    # make python identifiers and track for namespace conflicts
    namespace = {'TIME': 'time', 'Time': 'time'}  # Initialize with builtins
    for element in model_elements:
        if element['kind'] not in ['subdef', 'section']:
            element['py_name'], namespace = utils.make_python_identifier(element['real_name'],
                                                                         namespace)

    # Create a namespace for the subscripts
    # as these aren't used to create actual python functions, but are just labels on arrays,
    # they don't actually need to be python-safe
    subscript_dict = {e['real_name']: e['subs'] for e in model_elements if e['kind'] == 'subdef'}

    # Parse components to python syntax.
    for element in model_elements:
        if element['kind'] == 'component' and 'py_expr' not in element:
            # Todo: if there is new structure, it should be added to the namespace...
            translation, new_structure = parse_general_expression(element,
                                                                  namespace=namespace,
                                                                  subscript_dict=subscript_dict)
            element.update(translation)
            model_elements += new_structure

        elif element['kind'] == 'lookup':
            element.update(parse_lookup_expression(element))

    model_elements.append({'kind'     : 'component',
                           'subs'     : None,
                           'doc'      : 'The time of the model',
                           'py_name'  : 'time',
                           'real_name': 'TIME',
                           'unit'     : None,
                           'py_expr'  : '_t',
                           'arguments': ''})

    # define outfile name
    outfile_name = mdl_file.replace('.mdl', '.py')

    # send the pieces to be built
    build_elements = [e for e in model_elements if e['kind'] not in ['subdef', 'section']]
    builder.build(build_elements,
                  subscript_dict,
                  namespace,
                  outfile_name)

    return outfile_name
