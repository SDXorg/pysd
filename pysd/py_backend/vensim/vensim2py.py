"""
Translates vensim .mdl file to pieces needed by the builder module to write a python version of the
model. Everything that requires knowledge of vensim syntax should be in this file.
"""
from __future__ import absolute_import

import os
import re
import textwrap
import warnings
from io import open

import numpy as np
import parsimonious

from .. import functions as funcs
from ...py_backend import builder
from ...py_backend import utils


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

    # the leading 'r' for 'raw' in this string is important for handling backslashes properly
    file_structure_grammar = _include_common_grammar(r"""
    file = encoding? (macro / main)+
    macro = ":MACRO:" _ name _ "(" _ (name _ ","? _)+ _ ":"? _ (name _ ","? _)* _ ")" ~r".+?(?=:END OF MACRO:)" ":END OF MACRO:"
    main = !":MACRO:" ~r".+(?!:MACRO:)"
    encoding = ~r"\{[^\}]*\}"
    """)

    parser = parsimonious.Grammar(file_structure_grammar)
    tree = parser.parse(file_str)

    class FileParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.entries = []
            self.visit(ast)

        def visit_main(self, n, vc):
            self.entries.append({'name': '_main_',
                                 'params': [],
                                 'returns': [],
                                 'string': n.text.strip()})

        def visit_macro(self, n, vc):
            name = vc[2]
            params = vc[6]
            returns = vc[10]
            text = vc[13]
            self.entries.append({'name': name,
                                 'params': [x.strip() for x in params.split(',')] if params else [],
                                 'returns': [x.strip() for x in
                                             returns.split(',')] if returns else [],
                                 'string': text.strip()})

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

    model_structure_grammar = _include_common_grammar(r"""
    model = (entry / section)+ sketch?
    entry = element "~" element "~" element ("~" element)? "|"
    section = element "~" element "|"
    sketch = ~r".*"  #anything

    # Either an escape group, or a character that is not tilde or pipe
    element = (escape_group / ~r"[^~|]")*
    """)

    parser = parsimonious.Grammar(model_structure_grammar)
    tree = parser.parse(model_str)

    class ModelParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.entries = []
            self.visit(ast)

        def visit_entry(self, n, vc):
            units, lims = parse_units(vc[2].strip())
            self.entries.append({'eqn': vc[0].strip(),
                                 'unit': units,
                                 'lims': str(lims),
                                 'doc': vc[4].strip(),
                                 'kind': 'entry'})

        def visit_section(self, n, vc):
            if vc[2].strip() != "Simulation Control Parameters":
                self.entries.append({'eqn': '',
                                     'unit': '',
                                     'lims': '',
                                     'doc': vc[2].strip(),
                                     'kind': 'section'})

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text or ''

    return ModelParser(tree).entries


def _include_common_grammar(source_grammar):
    common_grammar = r"""
    name = basic_id / escape_group
    
    # This takes care of models with Unicode variable names
    basic_id = id_start id_continue*
    
    id_start = ~r"[\w]"IU
    id_continue = id_start / ~r"[0-9\'\$\s\_]"

    # between quotes, either escaped quote or character that is not a quote
    escape_group = "\"" ( "\\\"" / ~r"[^\"]" )* "\""
    
    _ = ~r"[\s\\]*"  # whitespace character
    """

    return r"""
    {source_grammar}
    
    {common_grammar}
    """.format(source_grammar=source_grammar, common_grammar=common_grammar)


def get_equation_components(equation_str, root_path):
    """
    Breaks down a string representing only the equation part of a model element.
    Recognizes the various types of model elements that may exist, and identifies them.

    Parameters
    ----------
    equation_str : basestring
        the first section in each model element - the full equation.

    root_path: basestring
        the root path of the vensim file (necessary to resolve external data file paths)

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
        - *data* -  a data variable

    Examples
    --------
    >>> get_equation_components(r'constant = 25')
    {'expr': '25', 'kind': 'component', 'subs': [], 'real_name': 'constant'}

    Notes
    -----
    in this function we don't create python identifiers, we use real names.
    This is so that when everything comes back together, we can manage
    any potential namespace conflicts properly
    """

    component_structure_grammar = _include_common_grammar(r"""
    entry = component / data_definition / test_definition / subscript_definition / lookup_definition
    component = name _ subscriptlist? _ "=" _ expression
    subscript_definition = name _ ":" _ (imported_subscript / literal_subscript)
    data_definition = name _ subscriptlist? _ keyword? _ ":=" _ expression
    lookup_definition = name _ &"(" _ expression  # uses lookahead assertion to capture whole group
    test_definition = name _ subscriptlist? _ &keyword _ expression

    name = basic_id / escape_group
    literal_subscript = subscript _ ("," _ subscript)*
    imported_subscript = func _ "(" _ (string _ ","? _)* ")"
    subscriptlist = '[' _ subscript _ ("," _ subscript)* _ ']'
    expression = ~r".*"  # expression could be anything, at this point.
    keyword = ":" _ basic_id _ ":"

    subscript = basic_id / escape_group
    func = basic_id
    string = "\'" ( "\\\'" / ~r"[^\']"IU )* "\'"
    """)

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
            self.keyword = None
            self.visit(ast)

        def visit_subscript_definition(self, n, vc):
            self.kind = 'subdef'

        def visit_lookup_definition(self, n, vc):
            self.kind = 'lookup'

        def visit_component(self, n, vc):
            self.kind = 'component'

        def visit_data_definition(self, n, vc):
            self.kind = 'data'

        def visit_test_definition(self, n, vc):
            self.kind = 'test'

        def visit_keyword(self, n, vc):
            self.keyword = n.text.strip()

        def visit_imported_subscript(self, n, vc):
            f_str = vc[0]
            args_str = vc[4]  # todo: make this less fragile?
            self.subscripts += get_external_data(f_str, args_str, root_path)

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
            'subs': parse_object.subscripts,
            'expr': parse_object.expression,
            'kind': parse_object.kind,
            'keyword': parse_object.keyword}


def get_external_data(func_str, args_str, root_path):
    # The py model file must be recompiled if external file subscripts change. This could be avoided
    # if we switch to function-defined subscript values instead of hard-coding them.
    f = subscript_functions[func_str.lower()]
    args = [x.strip().strip("\'") for x in args_str.split(',')]  # todo: make this less fragile?
    if args[0][0] == '?':
        args[0] = os.path.join(root_path, args[0][1:])
    return f(*args)


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
    ('Widgets/Month', (-10,10,1))

    >>> parse_units('Month [0,?]')
    ('Month', [-10, None])

    >>> parse_units('Widgets [0,100]')
    ('Widgets', (0, 100))

    >>> parse_units('Widgets')
    ('Widgets', (None, None))

    >>> parse_units('[0, 100]')
    ('', (0, 100))

    """
    if not len(units_str):
        return units_str, (None, None)

    if units_str[-1] == ']':
        units, lims = units_str.rsplit('[')  # type: str, str
    else:
        units = units_str
        lims = '?, ?]'

    lims = tuple([float(x) if x.strip() != '?' else None for x in lims.strip(']').split(',')])

    return units.strip(), lims


functions = {
    # element-wise functions
    "abs": "abs",
    "integer": "int",
    "exp": "np.exp",
    "sin": "np.sin",
    "cos": "np.cos",
    "sqrt": "np.sqrt",
    "tan": "np.tan",
    "lognormal": "np.random.lognormal",
    "random normal":
        "functions.bounded_normal",
    "poisson": "np.random.poisson",
    "ln": "np.log",
    "log": "functions.log",
    "exprnd": "np.random.exponential",
    "random uniform": "functions.random_uniform",
    "sum": "functions.sum",
    "arccos": "np.arccos",
    "arcsin": "np.arcsin",
    "arctan": "np.arctan",
    "tanh": "np.tanh",
    "sinh": "np.sinh",
    "cosh": "np.cosh",
    "if then else": "functions.if_then_else",
    "step": {
        "name": "functions.step",
        "parameters": [
            {"name": 'time', "type": 'time'},
            {"name": 'value'},
            {"name": 'tstep'}
        ]
    },
    "modulo": "np.mod",
    "pulse": {
        "name": "functions.pulse",
        "parameters": [
            {"name": 'time', "type": 'time'},
            {"name": 'start'},
            {"name": "duration"}
        ]
    },
    # time, start, duration, repeat_time, end
    "pulse train": {
        "name": "functions.pulse_train",
        "parameters": [
            {"name": 'time', "type": 'time'},
            {"name": 'start'},
            {"name": 'duration'},
            {"name": 'repeat_time'},
            {"name": 'end'}
        ]
    },
    "ramp": {
        "name": "functions.ramp",
        "parameters": [
            {"name": 'time', "type": 'time'},
            {"name": 'slope'},
            {"name": 'start'},
            {"name": 'finish', "optional": True}
        ]
    },
    "min": "np.minimum",
    "max": "np.maximum",
    # time, expr, init_val
    "active initial": {
        "name": "functions.active_initial",
        "parameters": [
            {"name": 'time', "type": 'time'},
            {"name": 'expr', "type": 'lambda'},
            {"name": 'init_val'}
        ]
    },
    "xidz": "functions.xidz",
    "zidz": "functions.zidz",
    "game": "",  # In the future, may have an actual `functions.game` pass through

    # vector functions
    "vmin": "functions.vmin",
    "vmax": "functions.vmax",
    "prod": "functions.prod",

}

subscript_functions = {
    "get xls subscript": funcs.get_xls_subscript,
    "get direct subscript": funcs.get_direct_subscript
}

data_ops = {
    'get data at time': '',
    'get data between times': '',
    'get data last time': '',
    'get data max': '',
    'get data min': '',
    'get data median': '',
    'get data mean': '',
    'get data stdv': '',
    'get data total points': ''
}

builders = {
    "integ": lambda element, subscript_dict, args: builder.add_stock(
        identifier=element['py_name'],
        subs=element['subs'],
        expression=args[0],
        initial_condition=args[1],
        subscript_dict=subscript_dict
    ),

    "delay1": lambda element, subscript_dict, args: builder.add_n_delay(
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[0],
        order='1',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "delay1i": lambda element, subscript_dict, args: builder.add_n_delay(
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[2],
        order='1',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "delay3": lambda element, subscript_dict, args: builder.add_n_delay(
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[0],
        order='3',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "delay3i": lambda element, subscript_dict, args: builder.add_n_delay(
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[2],
        order='3',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "delay fixed": lambda element, subscript_dict, args: builder.add_n_delay(
        delay_input=args[0],
        delay_time='round(' + args[1] + ' / time_step() ) * time_step()',
        initial_value=args[2],
        order=args[1] + ' / time_step()',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "delay n": lambda element, subscript_dict, args: builder.add_n_delay(
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[2],
        order=args[3],
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "smooth": lambda element, subscript_dict, args: builder.add_n_smooth(
        smooth_input=args[0],
        smooth_time=args[1],
        initial_value=args[0],
        order='1',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "smoothi": lambda element, subscript_dict, args: builder.add_n_smooth(
        smooth_input=args[0],
        smooth_time=args[1],
        initial_value=args[2],
        order='1',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "smooth3": lambda element, subscript_dict, args: builder.add_n_smooth(
        smooth_input=args[0],
        smooth_time=args[1],
        initial_value=args[0],
        order='3',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "smooth3i": lambda element, subscript_dict, args: builder.add_n_smooth(
        smooth_input=args[0],
        smooth_time=args[1],
        initial_value=args[2],
        order='3',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "smooth n": lambda element, subscript_dict, args: builder.add_n_smooth(
        smooth_input=args[0],
        smooth_time=args[1],
        initial_value=args[2],
        order=args[3],
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "trend": lambda element, subscript_dict, args: builder.add_n_trend(
        trend_input=args[0],
        average_time=args[1],
        initial_trend=args[2],
        subs=element['subs'],
        subscript_dict=subscript_dict),

    "get xls data": lambda element, subscript_dict, args: builder.add_data(
        identifier=element['py_name'],
        file=args[0],
        tab=args[1],
        time_row_or_col=args[2],
        cell=args[3],
        subs=element['subs'],
        subscript_dict=subscript_dict,
        keyword=element['keyword']
    ),

    "get xls constants": lambda element, subscript_dict, args: builder.add_ext_constant(
        identifier=element['py_name'],
        file=args[0],
        tab=args[1],
        cell=args[2],
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "get xls lookups": lambda element, subscript_dict, args: builder.add_ext_lookup(
        identifier=element['py_name'],
        file=args[0],
        tab=args[1],
        x_row_or_col=args[2],
        cell=args[3],
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "initial": lambda element, subscript_dict, args: builder.add_initial(args[0]),

    "a function of": lambda element, subscript_dict, args: builder.add_incomplete(
        element['real_name'], args)
}

builders['get direct data'] = builders['get xls data']  # Both are implemented identically in PySD
builders['get direct lookups'] = builders['get xls lookups']  # Both are implemented identically in PySD


def parse_general_expression(element, namespace=None, subscript_dict=None, macro_list=None):
    """
    Parses a normal expression
    # its annoying that we have to construct and compile the grammar every time...

    Parameters
    ----------
    element: dictionary

    namespace : dictionary

    subscript_dict : dictionary

    macro_list: list of dictionaries
        [{'name': 'M', 'py_name':'m', 'filename':'path/to/file', 'args':['arg1', 'arg2']}]

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

    in_ops = {
        "+": "+", "-": "-", "*": "*", "/": "/", "^": "**", "=": "==", "<=": "<=", "<>": "!=",
        "<": "<", ">=": ">=", ">": ">",
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
    if macro_list is not None and len(macro_list) > 0:
        macro_names_list = [re.escape(x['name']) for x in macro_list]
    else:
        macro_names_list = ['\\a']

    expression_grammar = r"""
    expr_type = array / expr / empty
    expr = _ pre_oper? _ (lookup_def / build_call / macro_call / call / lookup_call / parens / number / string / reference) _ in_oper_expr?

    in_oper_expr = (in_oper _ expr)
    lookup_def = ~r"(WITH\ LOOKUP)"I _ "(" _ expr _ "," _ "(" _  ("[" ~r"[^\]]*" "]" _ ",")?  ( "(" _ expr _ "," _ expr _ ")" _ ","? _ )+ _ ")" _ ")"
    lookup_call = (id _ subscript_list?) _ "(" _ (expr _ ","? _)* ")"  # these don't need their args parsed...
    call = func _ "(" _ (expr _ ","? _)* ")"  # these don't need their args parsed...
    build_call = builder _ "(" _ arguments _ ")"
    macro_call = macro _ "(" _ arguments _ ")"
    parens   = "(" _ expr _ ")"

    arguments = (expr _ ","? _)*

    reference = id _ subscript_list?
    subscript_list = "[" _ ((sub_name / sub_element) "!"? _ ","? _)+ "]"

    array = (number _ ("," / ";")? _)+ !~r"."  # negative lookahead for anything other than an array
    number = ~r"\d+\.?\d*(e[+-]\d+)?"
    string = "\'" ( "\\\'" / ~r"[^\']"IU )* "\'"

    id = ( basic_id / escape_group )
    basic_id = ~r"\w[\w\d_\s\']*"IU
    escape_group = "\"" ( "\\\"" / ~r"[^\"]"IU )* "\""
    
    sub_name = ~r"(%(sub_names)s)"IU  # subscript names (if none, use non-printable character)
    sub_element = ~r"(%(sub_elems)s)"IU  # subscript elements (if none, use non-printable character)

    func = ~r"(%(funcs)s)"IU  # functions (case insensitive)
    in_oper = ~r"(%(in_ops)s)"IU  # infix operators (case insensitive)
    pre_oper = ~r"(%(pre_ops)s)"IU  # prefix operators (case insensitive)
    builder = ~r"(%(builders)s)"IU  # builder functions (case insensitive)
    macro = ~r"(%(macros)s)"IU  # macros from model file (if none, use non-printable character)

    _ = ~r"[\s\\]*"  # whitespace character
    empty = "" # empty string
    """ % {
        # In the following, we have to sort keywords in decreasing order of length so that the
        # peg parser doesn't quit early when finding a partial keyword
        'sub_names': '|'.join(reversed(sorted(sub_names_list, key=len))),
        'sub_elems': '|'.join(reversed(sorted(sub_elems_list, key=len))),
        'funcs': '|'.join(reversed(sorted(functions.keys(), key=len))),
        'in_ops': '|'.join(reversed(sorted(in_ops_list, key=len))),
        'pre_ops': '|'.join(reversed(sorted(pre_ops_list, key=len))),
        'builders': '|'.join(reversed(sorted(builders.keys(), key=len))),
        'macros': '|'.join(reversed(sorted(macro_names_list, key=len)))
    }

    class ExpressionParser(parsimonious.NodeVisitor):
        # Todo: at some point, we could make the 'kind' identification recursive on expression,
        # so that if an expression is passed into a builder function, the information
        # about whether it is a constant, or calls another function, goes with it.
        def __init__(self, ast):
            self.translation = ""
            self.kind = 'constant'  # change if we reference anything else
            self.new_structure = []
            self.arguments = None
            self.in_oper = None
            self.visit(ast)

        def visit_expr_type(self, n, vc):
            s = ''.join(filter(None, vc)).strip()
            self.translation = s

        def visit_expr(self, n, vc):
            if self.in_oper:
                # This is rather inelegant, and possibly could be better implemented with a serious reorganization
                # of the grammar specification for general expressions.
                args = [x for x in vc if len(x.strip())]
                if len(args) == 3:
                    args = [''.join(args[0:2]), args[2]]
                if self.in_oper  == ' and ':
                    s = 'functions.and_(%s)' % ','.join(args)
                elif self.in_oper == ' or ':
                    s = 'functions.or_(%s)' % ','.join(args)
                else:
                    s = self.in_oper.join(args)
                self.in_oper = None
            else:
                s = ''.join(filter(None, vc)).strip()
            self.translation = s
            return s

        def visit_in_oper_expr(self, n, vc):
            # We have to pull out the internal operator because the Python "and" and "or" operator do not work with
            # numpy arrays or xarray DataArrays. We will later replace it with the functions.and_ or functions.or_.
            self.in_oper = vc[0]
            return ''.join(filter(None, vc[1:])).strip()

        def visit_call(self, n, vc):
            self.kind = 'component'
            function_name = vc[0].lower()
            arguments = [e.strip() for e in vc[4].split(",")]
            return builder.build_function_call(functions[function_name], arguments)

        def visit_in_oper(self, n, vc):
            return in_ops[n.text.lower()]

        def visit_pre_oper(self, n, vc):
            return pre_ops[n.text.lower()]

        def visit_reference(self, n, vc):
            self.kind = 'component'
            vc[0] += '()'
            return ''.join([x.strip(',') for x in vc])

        def visit_id(self, n, vc):
            return namespace[n.text.strip()]

        def visit_lookup_def(self, n, vc):
            """ This exists because vensim has multiple ways of doing lookups.
            Which is frustrating."""
            x_val = vc[4]
            pairs = vc[11]
            mixed_list = pairs.replace('(', '').replace(')', '').split(',')
            xs = mixed_list[::2]
            ys = mixed_list[1::2]
            string = "functions.lookup(%(x)s, [%(xs)s], [%(ys)s])" % {
                'x': x_val,
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
                    'coords': repr(coords),
                    'dims': repr(dims)})

            else:
                return n.text.replace(' ', '')

        def visit_subscript_list(self, n, vc):
            refs = vc[2]
            subs = [x.strip() for x in refs.split(',')]
            coordinates = utils.make_coord_dict(subs, subscript_dict)
            if len(coordinates):
                string = '.loc[%s].squeeze()' % repr(coordinates)
            else:
                string = ' '
            # Implements basic "!" subscript functionality in Vensim. Does NOT work for matrix diagonals in
            # FUNC(variable[sub1!,sub1!]) functions, nor with complex operations within the vector function
            # But works quite well for simple axis specifications, such as "SUM(variable[axis1, axis2!])
            axis = ['"%s"' % s.strip('!') for s in subs if s[-1] == '!']
            if axis:
                string += ', dim=(%s)' % ','.join(axis)
            return string

        def visit_build_call(self, n, vc):
            call = vc[0]
            arglist = vc[4]
            self.kind = 'component'

            builder_name = call.strip().lower()
            name, structure = builders[builder_name](element, subscript_dict, arglist)
            self.new_structure += structure

            if builder_name in ['get xls lookups', 'get direct lookups']:
                self.arguments = 'x'

            if builder_name == 'delay fixed':
                warnings.warn("Delay fixed only approximates solution, may not give the same "
                              "result as vensim")

            return name

        def visit_macro_call(self, n, vc):
            call = vc[0]
            arglist = vc[4]
            self.kind = 'component'
            py_name = utils.make_python_identifier(call)[0]
            macro = [x for x in macro_list if x['py_name'] == py_name][0]  # should match once
            name, structure = builder.add_macro(macro['py_name'], macro['file_name'],
                                                macro['params'], arglist)
            self.new_structure += structure
            return name

        def visit_arguments(self, n, vc):
            arglist = [x.strip(',') for x in vc]
            return arglist

        def visit__(self, n, vc):
            """ Handles whitespace characters"""
            return ''

        def visit_empty(self, n, vc):
            return 'None'

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text

    parser = parsimonious.Grammar(expression_grammar)

    tree = parser.parse(element['expr'])
    parse_object = ExpressionParser(tree)

    return ({'py_expr': parse_object.translation,
             'kind': parse_object.kind,
             'arguments': parse_object.arguments or ''},
            parse_object.new_structure)


def parse_lookup_expression(element):
    """ This syntax parses lookups that are defined with their own element """

    lookup_grammar = r"""
    lookup = _ "(" range? _ ( "(" _ number _ "," _ number _ ")" _ ","? _ )+ ")"
    number = ("+"/"-")? ~r"\d+\.?\d*(e[+-]\d+)?"
    _ = ~r"[\s\\]*"  # whitespace character
	range = _ "[" ~r"[^\]]*" "]" _ ","
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
            pairs = max(vc, key=len)
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
    return {'py_expr': parse_object.translation,
            'arguments': 'x'}


def translate_section(section, macro_list, root_path):
    model_elements = get_model_elements(section['string'])

    # extract equation components
    model_docstring = ''
    for entry in model_elements:
        if entry['kind'] == 'entry':
            entry.update(get_equation_components(entry['eqn'], root_path))
        elif entry['kind'] == 'section':
            model_docstring += entry['doc']

    # make python identifiers and track for namespace conflicts
    namespace = {'TIME': 'time', 'Time': 'time'}  # Initialize with builtins
    # add macro parameters when parsing a macro section
    for param in section['params']:
        name, namespace = utils.make_python_identifier(param, namespace)

    # add macro functions to namespace
    for macro in macro_list:
        if macro['name'] is not '_main_':
            name, namespace = utils.make_python_identifier(macro['name'], namespace)

    # add model elements
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
        if (element['kind'] == 'component' and 'py_expr' not in element) or element['kind'] == 'data':
            # Todo: if there is new structure, it should be added to the namespace...
            translation, new_structure = parse_general_expression(element,
                                                                  namespace=namespace,
                                                                  subscript_dict=subscript_dict,
                                                                  macro_list=macro_list)
            element.update(translation)
            model_elements += new_structure

        elif element['kind'] == 'lookup':
            element.update(parse_lookup_expression(element))

    # send the pieces to be built
    build_elements = [e for e in model_elements if e['kind'] not in ['subdef', 'test', 'section']]
    builder.build(build_elements,
                  subscript_dict,
                  namespace,
                  section['file_name'])

    return section['file_name']


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
    root_path = os.path.split(mdl_file)[0]
    with open(mdl_file, 'r', encoding='UTF-8') as in_file:
        text = in_file.read()

    outfile_name = mdl_file.replace('.mdl', '.py')
    out_dir = os.path.dirname(outfile_name)

    # extract model elements
    file_sections = get_file_sections(text.replace('\n', ''))
    for section in file_sections:
        if section['name'] == '_main_':
            section['file_name'] = outfile_name
        else:  # separate macro elements into their own files
            section['py_name'] = utils.make_python_identifier(section['name'])[0]
            section['file_name'] = out_dir + '/' + section['py_name'] + '.py'

    macro_list = [s for s in file_sections if s['name'] is not '_main_']

    for section in file_sections:
        translate_section(section, macro_list, root_path)

    return outfile_name
