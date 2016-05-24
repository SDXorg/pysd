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

    # encoding
    >>> get_file_sections(r'{UTF-8} a~b~c| d~e~f| g~h~i|')
    [{'returns': [], 'params': [], 'name': 'main', 'string': 'a~b~c| d~e~f| g~h~i|'}]

    >>> get_file_sections(r'a~b~c| d~e~f{special}| g~h~i|') # allows the pattern in other locations
    [{'returns': [], 'params': [], 'name': 'main', 'string': 'a~b~c| d~e~f{special}| g~h~i|'}]
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
    entry = element "~" element "~" element "|"
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
        "abs": "abs", "integer": "int", "exp": "np.exp", "sin": "np.sin", "cos": "np.cos",
        "sqrt": "np.sqrt", "tan": "np.tan", "lognormal": "np.random.lognormal",
        "random normal": "functions.bounded_normal", "poisson": "np.random.poisson", "ln": "np.log",
        "exprnd": "np.random.exponential", "random uniform": "np.random.rand", "sum": "np.sum",
        "arccos": "np.arccos", "arcsin": "np.arcsin", "arctan": "np.arctan",
        "if then else": "functions.if_then_else", "step": "functions.step", "modulo": "np.mod",
        "pulse": "functions.pulse", "pulse train": "functions.pulse_train",
        "ramp": "functions.ramp", "min": "np.minimum", "max": "np.maximum",
        # vector functions
        "vmin": "np.min", "vmax": "np.max", "prod": "np.prod",
        }

    builtins = {"time": "time"
                }

    builders = {
        "integ": lambda expr, init: builder.add_stock(element['py_name'], element['subs'],
                                                              expr, init),
        "delay1": lambda in_var, dtime: builder.add_n_delay(in_var, dtime, '0', '1'),
        "delay1i": lambda in_var, dtime, init: builder.add_n_delay(in_var, dtime, init, '1'),
        # continue this pattern with the other delay functions and smooth functions
    }

    in_ops = {
        "+": "+", "-": "-", "*": "*", "/": "/",  "^": "**", "=": "==", "<=": "<=", "<>": "!=",
        "<": "<",">=": ">=", ">": ">", ":and:": "and", ":or:": "or"}  # Todo: make the semicolon into a [1,2],[3,4] type of array

    pre_ops ={
        "-": "-", ":not:": "not", "+": " ",  # space is important, so that and empty string doesn't slip through generic
        }

    sub_names_list = subscript_dict.keys() or ['\\a']  # if none, use non-printable character
    sub_elems_list = [y for x in subscript_dict.values() for y in x] or ['\\a']
    ids_list = namespace.keys() or ['\\a']
    in_ops_list = [re.escape(x) for x in in_ops.keys()]  # special characters need escaping
    pre_ops_list = [re.escape(x) for x in pre_ops.keys()]

    expression_grammar = r"""
    expr_type = array / expr
    expr = _ pre_oper? _ (build_call / call / parens / number / reference / builtin) _ (in_oper _ expr)?

    call = (func / reference) _ "(" _ (expr _ ","? _)* ")" # allows calls with no arguments
    build_call = builder _ "(" _ (expr _ ","? _)* ")" # allows calls with no arguments
    parens   = "(" _ expr _ ")"

    reference = id _ subscript_list?
    subscript_list = "[" _ ((sub_name / sub_element) _ ","? _)+ "]"

    array = (number _ ("," / ";")? _)+
    number = ~r"\d+\.?\d*(e[+-]\d+)?"

    id = %(ids)s
    sub_name = %(sub_names)s  # subscript names (if none, use non-printable character)
    sub_element = %(sub_elems)s  # subscript elements (if none, use non-printable character)

    func = ~r"(%(funcs)s)"I  # functions (case insensitive)
    in_oper = ~r"(%(in_ops)s)"I  # infix operators (case insensitive)
    pre_oper = ~r"(%(pre_ops)s)"I  # prefix operators (case insensitive)
    builder = ~r"(%(builders)s)"I  # builder functions (case insensitive)
    builtin = ~r"(%(builtins)s)"I  # build in functions (case insensitive)

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

    parser = parsimonious.Grammar(expression_grammar)
    tree = parser.parse(element['expr'])

    class ExpressionParser(parsimonious.NodeVisitor):
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
            return in_ops[n.text]

        def visit_pre_oper(self, n, vc):
            return pre_ops[n.text]

        def visit_id(self, n, vc):
            self.kind = 'component'
            return namespace[n.text]

        def visit_builtin(self, n, vc):
            return builtins[n.text.lower()]

        def visit_array(self, n, vc):
            text = n.text.strip(';').replace(' ','')  # remove trailing semi if exists
            if ';' in text:
                return '['+text.replace(';', '],[')+']'
            else:
                return text

        def visit_subscript_list(self, n, (lb, _1, refs, rb)):
            coordinates = {}
            subs = [x.strip() for x in refs.split(',')]
            for sub in subs:
                if sub in sub_elems_list:
                    name = find_subscript_name(subscript_dict, sub)
                    coordinates[name] = [sub]
            if len(coordinates):
                return '.loc[%s]'%repr(coordinates)
            else:
                return ' '

        def visit_build_call(self, n, (call, _1, lp, _2, args, rp)):
            self.kind = 'component'
            arglist = [x.strip() for x in args.split(',')]
            name, structure = builders[call.strip().lower()](*arglist)
            self.new_structure += structure
            return name

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text

    parse_object = ExpressionParser(tree)

    return ({'py_expr': parse_object.translation,
            'kind': parse_object.kind},
            parse_object.new_structure)


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
    pass


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
            outs[name]['kind'] = 'group'

    return outs.values()

def build_function_string(element, namespace, subscript_dict):
    """

    Parameters
    ----------
    element
    namespace
    subscript_dict

    Returns
    -------

    """


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
    >>> translate_vensim('../../tests/test-models/tests/subscript_3d_arrays/test_subscript_3d_arrays.mdl')

    #>>> translate_vensim('../../tests/test-models/tests/abs/test_abs.mdl')

    #>>> translate_vensim('../../tests/test-models/tests/exponentiation/exponentiation.mdl')

    #>>> translate_vensim('../../tests/test-models/tests/limits/test_limits.mdl')

    """
    with open(mdl_file, 'rU') as file:
            text = file.read()

    # extract model elements
    model_elements = []
    file_sections = get_file_sections(text.replace('\n', ''))
    for section in file_sections:
        if section['name'] == 'main':
            model_elements += get_model_elements(section['string'])
        # for now, ignoring macros

    # extract equation components
    map(lambda e: e.update(get_equation_components(e['eqn'])),
        model_elements)

    # make python identifiers and track for namespace conflicts
    namespace = {}
    for element in model_elements:
        if element['kind'] != 'subdef':
            element['py_name'], namespace = builder.make_python_identifier(element['real_name'],
                                                                           namespace)

    # Create a namespace for the subscripts
    # as these aren't used to create actual python functions, but are just labels on arrays,
    # they don't actually need to be python-safe
    subscript_dict = {e['real_name']: e['subs'] for e in model_elements if e['kind'] == 'subdef'}
    # need to be able to identify that subranges ARE subranges...
    subranges_dict = {}

    # Todo: parse units string

    for element in model_elements:
        if element['kind'] == 'lookup':
            pass


    # Todo: translate general expressions to python syntax
    for element in model_elements:
        if element['kind'] == 'component':
            translation, new_structure = parse_general_expression(element,
                                                   namespace=namespace,
                                                   subscript_dict=subscript_dict,
                                                   )
            element.update(translation)
            model_elements += new_structure


    model_groups = merge_partial_elements([e for e in model_elements if e['kind'] != 'subdef'])

    # Todo: Combine elements that share the same name
    # this is generally when functions or constants are defined in chunks,
    # such as for subscripts with more than 2 dimensions,
    # or for subscripted functions.

    # Todo: send pieces to the builder class

    print model_elements

"""
Elements Dictionary:
doc
eqn
expr
kind
py_name
real_name
subs
unit
"""