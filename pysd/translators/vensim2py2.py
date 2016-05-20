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
    ...                                           25 ''')
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
    expression = ~r".*"  # expression could be anything, at this point.

    subscript = basic_id / escape_group

    basic_id = ~r"[a-zA-Z][a-zA-Z0-9_\s]*"
    escape_group = "\"" ( "\\\"" / ~r"[^\"]" )* "\""
    _ = ~r"[\s\\]*" #whitespace character
    """

    # replace any amount of whitespace  with a single space
    equation_str = re.sub('[\s\t\\\]+', ' ', equation_str)

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


def parse_general_expression(expression_string, namespace=None, subscript_dict=None):
    """
    Parses a normal expression
    # its annoying that we have to construct and compile the grammar every time...

    Parameters
    ----------
    expression_string: <basestring>


    namespace : <dictionary>

    subscript_dict


    Returns
    -------
    new_elements

    Examples
    --------
    >>> parse_general_expression('INTEG ( FlowA, -10)', {'FlowA': 'flowa'})

    >>> parse_general_expression('INTEG ( FlowA, -10)', {'FlowA': 'flowa', 'StockA': 'stocka'})

    >>> parse_general_expression('ABS(StockA)', {'StockA': 'stocka'})

    >>> parse_general_expression('20', {'StockA': 'stocka'})

    >>> parse_general_expression('Time^2', {})

    """
    if namespace is None:
        namespace = {}
    if subscript_dict is None:
        subscript_dict = {}

    expr = expression_string.lower()

    # there is an issue here, which is that the subscript dict is using the already-translated
    # versions of the python identifiers... do we even need python safe versions of these?

    sub_names_list = subscript_dict.keys()
    sub_elems_list = [y for x in subscript_dict.values() for y in x]
    ids_list = set(namespace.keys()) - set(sub_elems_list) - set(sub_names_list)

    equation_grammar = """
    expr =

    reference = id subscript_list?
    id = %(ids)s  # variables that have already been identified
    sub_name = %(sub_names)s  # subscript names
    sub_element = %(sub_elems)s  # subscript elements

    _ = ~r"[\s\\]*"  # whitespace character
    """ % {
        # In the following, we have to sort keywords in decreasing order of length so that the
        # peg parser doesn't quit early when finding a partial keyword
        'sub_names': ' / '.join(['"%s"' % n for n in reversed(sorted(sub_names_list, key=len))]),
        'sub_elems': ' / '.join(['"%s"' % n for n in reversed(sorted(sub_elems_list, key=len))]),
        'ids': ' / '.join(['"%s"' % n for n in reversed(sorted(ids_list, key=len))])
    }

    print equation_grammar


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
    subscript_dict = {}
    for element in model_elements:
        if element['kind'] == 'subdef':
            subscript_elements = []
            for subelem in element['subs']:
                py_name, namespace = builder.make_python_identifier(subelem, namespace)
                subscript_elements.append(py_name)
            subscript_dict[element['py_name']] = subscript_elements

    # Todo: parse units string

    for element in model_elements:
        if element['kind'] == 'lookup':
            pass


    # Todo: translate expressions to python syntax
    for element in model_elements:
        if element['kind'] == 'component':
            pass


    # Todo: send pieces to the builder class

    print model_elements

