"""
In this version:
build up construction functions as separate step in model loading


Todo:


"""

import re
import parsimonious
from pysd import builder


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

    Examples
    --------
    # normal model file with no macros
    >>> get_file_sections(r'a~b~c| d~e~f| g~h~i|')
    [{'returns': [], 'params': [], 'name': 'main', 'string': 'a~b~c| d~e~f| g~h~i|'}]

    # macro only
    >>> get_file_sections(':MACRO: MAC(z) a~b~c| :END OF MACRO:')
    [{'returns': [], 'params': ['z'], 'name': 'MAC', 'string': 'a~b~c|'}]

    # basic macro and model
    >>> get_file_sections(':MACRO: MAC(z) a~b~c| :END OF MACRO: d~e~f| g~h~i|')
    [{'returns': [], 'params': ['z'], 'name': 'MAC', 'string': 'a~b~c|'}, {'returns': [], 'params': [], 'name': 'main', 'string': 'd~e~f| g~h~i|'}]

    # multiple input parameters
    >>> get_file_sections(':MACRO: MAC(z, y) a~b~c| :END OF MACRO: d~e~f| g~h~i|')
    [{'returns': [], 'params': ['z', 'y'], 'name': 'MAC', 'string': 'a~b~c|'}, {'returns': [], 'params': [], 'name': 'main', 'string': 'd~e~f| g~h~i|'}]

    # macro with returns specified
    >>> get_file_sections(':MACRO: MAC(z, y :x, w) a~b~c| :END OF MACRO: d~e~f| g~h~i|')
    [{'returns': ['x', 'w'], 'params': ['z', 'y'], 'name': 'MAC', 'string': 'a~b~c|'}, {'returns': [], 'params': [], 'name': 'main', 'string': 'd~e~f| g~h~i|'}]



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

        def visit_main(self, n, text):
            self.entries.append({'name': 'main',
                                 'params': [],
                                 'returns': [],
                                 'string': n.text.strip()})

        def visit_macro(self, n, (m1, _1, name, _2, lp, _3, params, _4, cn, _5, returns,
                                  _6, rp, text, m2)):
            self.entries.append({'name': name,
                                 'params': [x.strip() for x in params.split(',')] if params else [],
                                 'returns': [x.strip() for x in returns.split(',')] if returns else [],
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

    # Handle encoding:
    >>> get_model_elements(r'{UTF-8}a~b~c| d~e~f| g~h~i|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': 'f', 'unit': 'e', 'eqn': 'd'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]
    >>> get_model_elements(r'a~b~c| d~e~f{special}| g~h~i|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': 'f{special}', 'unit': 'e', 'eqn': 'd'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]

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
    model = encoding? (entry / section)+ sketch?  # trailing `element` captures sketch info
    entry = element "~" element "~" element "|"  # these are what we want to capture
    section = element "~" element "|"  # section separators we don't capture
    sketch = ~r".*"  #anything that is not an end-of-file character

    # Either an escape group, or a character that is not tilde or pipe
    element = (escape_group / ~r"[^~|]")*

    # between quotes, either escaped quote or character that is not a quote
    escape_group = "\"" ( "\\\"" / ~r"[^\"]" )* "\""
    encoding = ~r"\{[^\}]*\}"
    """
    parser = parsimonious.Grammar(model_structure_grammar)
    tree = parser.parse(model_str)

    class ModelParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.entries = []
            self.visit(ast)

        def visit_entry(self, n, (eqn, _1, unit, _2, doc, _3)):
            self.entries.append({'eqn': eqn.strip(),
                                 'unit': unit.strip(),
                                 'doc': doc.strip()})

        def visit_section(self, n, (sect_marker, _1, text, _3)):
            if text.strip() != "Simulation Control Parameters":
                self.entries.append({'eqn': '',
                                     'unit': '',
                                     'doc': text.strip()})

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text or ''

    return ModelParser(tree).entries


def get_equation_components(equation_str):
    """
    Breaks down a string representing only the equation part of a model element.


    lookups:
    name, followed immediately by open parenthesis

    Parameters
    ----------
    eqn

    Returns
    -------
    real_name: basestring

    subscript_list: list of strings

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

    Parse cases with equal signs within them
    >>> get_equation_components(r'Boolean = IF THEN ELSE(1 = 1, 1, 0)')
    {'expr': 'IF THEN ELSE(1 = 1, 1, 0)', 'kind': 'component', 'subs': [], 'real_name': 'Boolean'}

    Shorten whitespaces
    >>> get_equation_components(r'''constant\t =
    ...                                           \t25\t ''')
    {'expr': '25', 'kind': 'component', 'subs': [], 'real_name': 'constant'}

    >>> get_equation_components(r'constant [Sub1, \\
    ...                                     Sub2] = 10, 12; 14, 16;')
    {'expr': '10, 12; 14, 16;', 'kind': 'component', 'subs': ['Sub1', 'Sub2'], 'real_name': 'constant'}

    Handle subscript definitions
    >>> get_equation_components(r'''Sub1: Entry 1, Entry 2, Entry 3 ''')
    {'expr': None, 'kind': 'subdef', 'subs': ['Entry 1', 'Entry 2', 'Entry 3'], 'real_name': 'Sub1'}

    Handle subscript references
    >>> get_equation_components(r'constant [Sub1, Sub2] = 10, 12; 14, 16;')
    {'expr': '10, 12; 14, 16;', 'kind': 'component', 'subs': ['Sub1', 'Sub2'], 'real_name': 'constant'}
    >>> get_equation_components(r'function [Sub1] = other function[Sub1]')
    {'expr': 'other function[Sub1]', 'kind': 'component', 'subs': ['Sub1'], 'real_name': 'function'}
    >>> get_equation_components(r'constant ["S1,b", "S1,c"] = 1, 2; 3, 4;')
    {'expr': '1, 2; 3, 4;', 'kind': 'component', 'subs': ['"S1,b"', '"S1,c"'], 'real_name': 'constant'}
    >>> get_equation_components(r'constant ["S1=b", "S1=c"] = 1, 2; 3, 4;')
    {'expr': '1, 2; 3, 4;', 'kind': 'component', 'subs': ['"S1=b"', '"S1=c"'], 'real_name': 'constant'}

    Handle lookup definitions:
    >>> get_equation_components(r'table([(0,-1)-(45,1)],(0,0),(5,0))')
    {'expr': '([(0,-1)-(45,1)],(0,0),(5,0))', 'kind': 'lookup', 'subs': [], 'real_name': 'table'}
    >>> get_equation_components(r'table2 ([(0,-1)-(45,1)],(0,0),(5,0))')
    {'expr': '([(0,-1)-(45,1)],(0,0),(5,0))', 'kind': 'lookup', 'subs': [], 'real_name': 'table2'}

    Handle pathological names:
    >>> get_equation_components(r'"silly-string" = 25')
    {'expr': '25', 'kind': 'component', 'subs': [], 'real_name': '"silly-string"'}
    >>> get_equation_components(r'"pathological\\-string" = 25')
    {'expr': '25', 'kind': 'component', 'subs': [], 'real_name': '"pathological\\\\-string"'}

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
    expression = ~r".*"  # expression could be anything, at this point. # Todo: should make this regex include newlines

    subscript = basic_id / escape_group

    basic_id = ~r"[a-zA-Z][a-zA-Z0-9_\s]*"
    escape_group = "\"" ( "\\\"" / ~r"[^\"]" )* "\""
    _ = ~r"[\s\\]*" #whitespace character
    """

    # replace any amount of whitespace  with a single space
    equation_str = re.sub('[\\s\\t\\\]+', ' ', equation_str)

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

        def visit_name(self, n, (name,)):
            self.real_name = name.strip()

        def visit_subscript(self, n, (subscript,)):
            self.subscripts.append(subscript.strip())

        def visit_expression(self, n, vc):
            self.expression = n.text.strip()

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text

    parse_object = ComponentParser(tree)

    return {'real_name': parse_object.real_name,
            'subs': parse_object.subscripts,
            'expr': parse_object.expression,
            'kind': parse_object.kind}


def parse_units(units_str):
    """

    Parameters
    ----------
    units_str

    Returns
    -------

    Examples
    --------
    """
    pass


def parse_general_expression(expression_string, namespace=None,
                             identifier=None, subscript_dict=None):
    """
    Parses a normal expression
    # its annoying that we have to construct and compile the grammar every time...

    Parameters
    ----------
    expression_string: <basestring>

    identifier: basestring
        The name of the element that we are parsing the expression of. This is
        mostly used in creating stock element structures.

    namespace : <dictionary>

    subscript_dict


    Returns
    -------
    translation

    new_elements: list of dictionaries. If the expression contains builder functions,
    those builders will create new elements to add to our running list (that will
    eventually be output to a file) such as stock initialization and derivative funcs,
    etc.

    Examples
    --------
    Parse Ids
    >>> parse_general_expression('StockA', namespace={'StockA': 'stocka'})
    {'constant': False, 'py_expr': 'stocka'}


    Parse numbers
    >>> parse_general_expression('20')
    {'constant': True, 'py_expr': '20'}

    >>> parse_general_expression('3.14159')
    {'constant': True, 'py_expr': '3.14159'}

    >>> parse_general_expression('1.3e-10')
    {'constant': True, 'py_expr': '1.3e-10'}

    >>> parse_general_expression('-1.3e+10')
    {'constant': True, 'py_expr': '-1.3e+10'}

    >>> parse_general_expression('1.3e+10')
    {'constant': True, 'py_expr': '1.3e+10'}

    >>> parse_general_expression('+3.14159')
    {'constant': True, 'py_expr': '3.14159'}


    General expressions / order of operations
    >>> parse_general_expression('10+3')
    {'constant': True, 'py_expr': '10+3'}

    >>> parse_general_expression('-10^3-2')
    {'constant': True, 'py_expr': '-10**3-2'}


    Parse build-in functions
    >>> parse_general_expression('Time^2', {})


    Parse function calls
    >>> parse_general_expression('ABS(StockA)', {'StockA': 'stocka'})
    {'constant': False, 'py_expr': 'abs(stocka)'}

    >>> parse_general_expression('If Then Else(A>B, 1, 0)', {'A': 'a', 'B':'b'})


    Parse construction functions
    >>> parse_general_expression('INTEG (FlowA, -10)', {'FlowA': 'flowa'})

    >>> parse_general_expression('Const * DELAY1(Variable, DelayTime)',
    ...                          {'Const':'const', 'Variable','variable'})

    >>> parse_general_expression('DELAY N(Inflow , delay , 0 , Order)',
    ...                          {'Const':'const', 'Variable','variable'})

    >>> parse_general_expression('SMOOTHI(Input, Adjustment Time, 0 )',
    ...                          {'Input':'input', 'Adjustment Time':'adjustment_time'})

    Parse pieces specific to subscripts
    >>> parse_general_expression('1, 2; 3, 4; 5, 6;')

    >>> parse_general_expression('StockA[Second Dimension Subscript, Third Dimension Subscript]',
    ...                          {'StockA': 'stocka'},
    ...                          {'Second Dimension Subscript': ['Column 1', 'Column 2'],
    ...                           'Third Dimension Subscript': ['Depth 1', 'Depth 2'],
    ...                           'One Dimensional Subscript': ['Entry 1', 'Entry 2', 'Entry 3']})

    >>> parse_general_expression('INTEG (Inflow A[sub_D1,sub_D2], Initials[sub_D1, sub_D2])',
    ...                          {'Initials': 'initials', 'Inflow A':'inflow_a'},
    ...                          {'sub_1D':['Entry 1', 'Entry 2', 'Entry 3'],
    ...                           'sub_2D':['Column 1', 'Column 2']})

    """
    if namespace is None:
        namespace = {}
    if subscript_dict is None:
        subscript_dict = {}

    functions = {"abs": "abs", "integer": "int", "exp": "np.exp",
                 "sin": "np.sin", "cos": "np.cos", "sqrt": "np.sqrt",
                 "tan": "np.tan", "lognormal": "np.random.lognormal",
                 "random normal": "functions.bounded_normal",
                 "poisson": "np.random.poisson", "ln": "np.log",
                 "exprnd": "np.random.exponential",
                 "random uniform": "np.random.rand",
                 "min": "np.minimum", "max": "np.maximum",  # these are element-wise
                 "vmin": "np.min", "vmax": "np.max",  # vector function
                 "prod": "np.prod",  # vector function
                 "sum": "np.sum", "arccos": "np.arccos",
                 "arcsin": "np.arcsin", "arctan": "np.arctan",
                 "if then else": "functions.if_then_else",
                 "step": "functions.step", "modulo": "np.mod", "pulse": "functions.pulse",
                 "pulse train": "functions.pulse_train", "ramp": "functions.ramp",
                 }

    builtins = {"time": "time"
                }

    # Todo: add functions in the builder class that interface to a generic delay, smooth, etc
    builders = {"integ": lambda expr, init: builder.add_stock(identifier, subscripts, expr, init),
                "delay1": lambda in_var, dtime: builder.add_n_delay(in_var, dtime, '0', '1'),
                "delay1i": lambda in_var, dtime, init: builder.add_n_delay(in_var, dtime, init, '1'),
                }

    in_ops = {"+": "+", "-": "-", "*": "*", "/": "/",  "^": "**",
              "=": "==", "<=": "<=", "<>": "!=", "<": "<",
              ">=": ">=", ">": ">",
              ":and:": "and", ":or:": "or",
              ",": ",", ";": ";"}  # Todo: check these

    pre_ops ={":not:": "not",
              "+": " ",  # space is important, so that and empty string doesn't slip through generic
              "-": "-"}

    sub_names_list = subscript_dict.keys() or ['\\a']  # if none, use non-printable character
    sub_elems_list = [y for x in subscript_dict.values() for y in x] or ['\\a']
    ids_list = namespace.keys() or ['\\a']
    in_ops_list = [re.escape(x) for x in in_ops.keys()]  # special characters need escaping
    pre_ops_list = [re.escape(x) for x in pre_ops.keys()]

    expression_grammar = r"""
    expr = _ pre_oper? _ (call / parens / number / reference / builtin) _ (in_oper _ expr)?

    call = (func / reference) _ "(" _ (expr _ ","? _)* ")" # allows calls with no arguments
    build_call = builder _ "(" _ (expr _ ","? _)* ")" # allows calls with no arguments
    parens   = "(" _ expr _ ")"

    reference = id _ subscript_list?
    subscript_list = "[" _ ((sub_name / sub_element) _ ","? _)+ "]"

    number = ~r"\d+\.?\d*(e[+-]\d+)?"

    id = %(ids)s
    sub_name = %(sub_names)s  # subscript names (if none, use non-printable character)
    sub_element = %(sub_elems)s  # subscript elements (if none, use non-printable character)

    func = ~r"(%(funcs)s)"I  # functions (case insensitive)
    in_oper = ~r"(%(in_ops)s)"I  # infix operators (case insensitive)
    pre_oper = ~r"(%(pre_ops)s)"I  # prefix operators (case insensitive)
    builder = ~r"(%(builders)s)"I # builder functions (case insensitive)
    builtin = ~r"(%(builtins)s)"I # build in functions (case insensitive)

    _ = ~r"[\s\\]*"  # whitespace character
    """ % {
        # In the following, we have to sort keywords in decreasing order of length so that the
        # peg parser doesn't quit early when finding a partial keyword
        'sub_names': ' / '.join(['"%s"' % n for n in reversed(sorted(sub_names_list, key=len))]),
        'sub_elems': ' / '.join(['"%s"' % n for n in reversed(sorted(sub_elems_list, key=len))]),
        'ids': '/'.join(['"%s"' % n for n in reversed(sorted(ids_list, key=len))]),
        # These are part of regex expressions, and may not need to be sorted, but just to be safe...
        'funcs': '|'.join(reversed(sorted(functions.keys(), key=len))),
        'in_ops': '|'.join(reversed(sorted(in_ops_list, key=len))),
        'pre_ops': '|'.join(reversed(sorted(pre_ops_list, key=len))),
        'builders': '|'.join(reversed(sorted(builders.keys(), key=len))),
        'builtins': '|'.join(reversed(sorted(builtins.keys(), key=len)))
    }

    #print expression_grammar

    parser = parsimonious.Grammar(expression_grammar)
    tree = parser.parse(expression_string)

    class ExpressionParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.translation = ""
            self.kind = 'constant' #set originally as constant, then change if we reference anything else
            self.visit(ast)
            self.new_structure = []

        def visit_expr(self, n, vc):
            s = ''.join(filter(None, vc)).strip()
            self.translation = s
            return s

        def visit_func(self, n, vc):
            self.kind = 'component'
            return functions[n.text.lower()]

        def visit_in_oper(self, n, vc):
            return in_ops[n.text]

        def visit_pre_oper(self, n, vc):
            return pre_ops[n.text]

        def visit_id(self, n, vc):
            self.kind = 'component'
            return namespace[n.text]

        def visit_build_call(self, n, (call, _1, lp, _2, args, _3, rp)):
            self.kind = 'component'
            name, self.new_structure += builders[call](args)
            return name

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text

    parse_object = ExpressionParser(tree)

    return {'py_expr': parse_object.translation,
            'kind': parse_object.kind},


def parse_lookup_expression():
    pass


def make_docstring(doc, unit, eqn):
    """
    Formats information to display in the python function's docstring

    Parameters
    ----------
    doc
    unit
    eqn

    Returns
    -------

    """
    return doc + '\n' + eqn + '\n' + unit


def build_model():
    """
    Take the various components that we have extracted/assembled, and call the
    builder

    There should be a



    Returns
    -------

    """

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
    #>>> translate_vensim('../../tests/test-models/tests/subscript_3d_arrays/test_subscript_3d_arrays.mdl')

    #>>> translate_vensim('../../tests/test-models/tests/abs/test_abs.mdl')

    >>> translate_vensim('../../tests/test-models/tests/exponentiation/exponentiation.mdl')

    #>>> translate_vensim('../../tests/test-models/tests/limits/test_limits.mdl')

    """
    with open(mdl_file, 'rU') as file:
            text = file.read()

    # extract model elements
    model_elements = []
    file_sections = get_file_sections(text.replace('\n',''))
    for section in file_sections:
        if section['name'] == 'main':
            model_elements += get_model_elements(section['string'])

    # extract equation components
    map(lambda e: e.update(get_equation_components(e['eqn'])),
        model_elements)

    # make python identifiers and track for namespace conflicts
    namespace = {}
    for element in model_elements:
        element['py_name'], namespace = builder.make_python_identifier(element['real_name'],
                                                                       namespace)

    # has to take out the various components of the subscripts and make them into a subscirpt
    # dictionary
    # rodo: make these just strings, not python safe ids.
    # subscript_dict = {}
    # for element in model_elements:
    #     if element['kind'] == 'subdef':
    #         subscript_elements = []
    #         for subelem in element['subs']:
    #             #py_name, namespace = builder.make_python_identifier(subelem, namespace)
    #             subscript_elements.append(py_name)
    #         subscript_dict[element['py_name']] = subscript_elements
    subscript_dict = {e['real_name']: e['subs'] for e in model_elements if e['kind'] == 'subdef'}

    # Todo: parse units string

    for element in model_elements:
        if element['kind'] == 'lookup':
            pass


    # Todo: translate expressions to python syntax
    for element in model_elements:
        if element['kind'] == 'component':
            parse_general_expression(element['expr'], namespace=namespace,
                                     subscript_dict=subscript_dict)


    # Todo: send pieces to the builder class

    print model_elements

