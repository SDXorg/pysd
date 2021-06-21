"""
These functions translate vensim .mdl file to pieces needed by the builder
module to write a python version of the model. Everything that requires
knowledge of vensim syntax should be here.
"""

import os
import re
from tkinter.font import names
import warnings
from io import open

import numpy as np
import parsimonious
from parsimonious.exceptions import IncompleteParseError, VisitationError, ParseError

from .. import builder, utils, external


def get_file_sections(file_str):
    """
    This is where we separate out the macros from the rest of the model file.
    Working based upon documentation at:
    https://www.vensim.com/documentation/index.html?macros.htm

    Macros will probably wind up in their own python modules eventually.

    Parameters
    ----------
    file_str

    Returns
    -------
    entries: list of dictionaries
        Each dictionary represents a different section of the model file,
        either a macro, or the main body of the model file. The
        dictionaries contain various elements:
        - returns: list of strings
            represents what is returned from a macro (for macros) or
            empty for main model
        - params: list of strings
            represents what is passed into a macro (for macros) or
            empty for main model
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
    file_structure_grammar = _include_common_grammar(
        r"""
    file = encoding? (macro / main)+
    macro = ":MACRO:" _ name _ "(" _ (name _ ","? _)+ _ ":"? _ (name _ ","? _)* _ ")" ~r".+?(?=:END OF MACRO:)" ":END OF MACRO:"
    main = !":MACRO:" ~r".+(?!:MACRO:)"
    encoding = ~r"\{[^\}]*\}"
    """
    )

    parser = parsimonious.Grammar(file_structure_grammar)
    tree = parser.parse(file_str)

    class FileParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.entries = []
            self.visit(ast)

        def visit_main(self, n, vc):
            self.entries.append(
                {
                    "name": "_main_",
                    "params": [],
                    "returns": [],
                    "string": n.text.strip(),
                }
            )

        def visit_macro(self, n, vc):
            name = vc[2]
            params = vc[6]
            returns = vc[10]
            text = vc[13]
            self.entries.append(
                {
                    "name": name,
                    "params": [x.strip() for x in params.split(",")] if params else [],
                    "returns": [x.strip() for x in returns.split(",")]
                    if returns
                    else [],
                    "string": text.strip(),
                }
            )

        def generic_visit(self, n, vc):
            return "".join(filter(None, vc)) or n.text or ""

    return FileParser(tree).entries


def get_model_elements(model_str):
    """
    Takes in a string representing model text and splits it into elements

    All newline characters were alreeady removed in a previous step.

    Parameters
    ----------
    model_str : string


    Returns
    -------
    entries : array of dictionaries
        Each dictionary contains the components of a different model element,
        separated into the equation, units, and docstring.

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

    model_structure_grammar = _include_common_grammar(
        r"""
    model = (entry / section)+ sketch?
    entry = element "~" element "~" doc ("~" element)? "|"
    section = element "~" element "|"
    sketch = ~r".*"  #anything

    # Either an escape group, or a character that is not tilde or pipe
    element = ( escape_group / ~r"[^~|]")*
    # Anything other that is not a tilde or pipe
    doc = (~r"[^~|]")*
    """
    )

    parser = parsimonious.Grammar(model_structure_grammar)
    tree = parser.parse(model_str)

    class ModelParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.entries = []
            self.visit(ast)

        def visit_entry(self, n, vc):
            units, lims = parse_units(vc[2].strip())
            self.entries.append(
                {
                    "eqn": vc[0].strip(),
                    "unit": units,
                    "lims": str(lims),
                    "doc": vc[4].strip(),
                    "kind": "entry",
                }
            )

        def visit_section(self, n, vc):
            if vc[2].strip() != "Simulation Control Parameters":
                self.entries.append(
                    {
                        "eqn": "",
                        "unit": "",
                        "lims": "",
                        "doc": vc[2].strip(),
                        "kind": "section",
                    }
                )

        def generic_visit(self, n, vc):
            return "".join(filter(None, vc)) or n.text or ""

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
    """.format(
        source_grammar=source_grammar, common_grammar=common_grammar
    )


def get_equation_components(equation_str, root_path=None):
    """
    Breaks down a string representing only the equation part of a model
    element. Recognizes the various types of model elements that may exist,
    and identifies them.

    Parameters
    ----------
    equation_str : basestring
        the first section in each model element - the full equation.

    root_path: basestring
        the root path of the vensim file (necessary to resolve external
        data file paths)

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

    keyword: basestring or None

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

    imp_subs_func_list = [
        "get xls subscript",
        "get direct subscript",
        "get_xls_subscript",
        "get_direct_subscript",
    ]

    component_structure_grammar = _include_common_grammar(
        r"""
    entry = component / data_definition / test_definition / subscript_definition / lookup_definition / subscript_copy
    component = name _ subscriptlist? _ "=" "="? _ expression
    subscript_definition = name _ ":" _ (imported_subscript / literal_subscript / numeric_range) _ subscript_mapping_list?
    data_definition = name _ subscriptlist? _ keyword? _ ":=" _ expression
    lookup_definition = name _ subscriptlist? &"(" _ expression  # uses lookahead assertion to capture whole group
    test_definition = name _ subscriptlist? _ &keyword _ expression
    subscript_copy = name _ "<->" _ name_mapping

    name = basic_id / escape_group

    literal_subscript = index_list
    imported_subscript = imp_subs_func _ "(" _ (string _ ","? _)* ")"
    numeric_range = _ (range / value) _ ("," _ (range / value) _)*
    value = _ sequence_id _
    range = "(" _ sequence_id _ "-" _ sequence_id _ ")"
    subscriptlist = '[' _ index_list _ ']'
    subscript_mapping_list = "->" _ subscript_mapping _ ("," _ subscript_mapping _)*
    subscript_mapping = (_ name_mapping _) / (_ "(" _ name_mapping _ ":" _ index_list _")"  )

    expression = ~r".*"  # expression could be anything, at this point.
    keyword = ":" _ basic_id _ ":"
    index_list = subscript _ ("," _ subscript _)*
    name_mapping = basic_id / escape_group
    sequence_id = _ basic_id _
    subscript = basic_id / escape_group
    imp_subs_func = ~r"(%(imp_subs)s)"IU
    string = "\'" ( "\\\'" / ~r"[^\']"IU )* "\'"
    """
        % {"imp_subs": "|".join(imp_subs_func_list)}
    )

    # replace any amount of whitespace  with a single space
    equation_str = equation_str.replace("\\t", " ")
    equation_str = re.sub(r"\s+", " ", equation_str)

    parser = parsimonious.Grammar(component_structure_grammar)

    class ComponentParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.subscripts = []
            self.subscripts_compatibility = {}
            self.real_name = None
            self.expression = None
            self.kind = None
            self.keyword = None
            self.visit(ast)

        def visit_subscript_definition(self, n, vc):
            self.kind = "subdef"

        def visit_lookup_definition(self, n, vc):
            self.kind = "lookup"

        def visit_component(self, n, vc):
            self.kind = "component"

        def visit_data_definition(self, n, vc):
            self.kind = "data"

        def visit_test_definition(self, n, vc):
            self.kind = "test"

        def visit_keyword(self, n, vc):
            self.keyword = n.text.strip()

        def visit_imported_subscript(self, n, vc):
            # TODO: make this less fragile
            args = [x.strip().strip("'") for x in vc[4].split(",")]
            self.subscripts += external.ExtSubscript(*args, root=root_path).subscript

        def visit_subscript_copy(self, n, vc):
            self.kind = "subdef"
            subs_copy1 = vc[4].strip()
            subs_copy2 = vc[0].strip()

            if subs_copy1 not in self.subscripts_compatibility:
                self.subscripts_compatibility[subs_copy1] = []

            if subs_copy2 not in self.subscripts_compatibility:
                self.subscripts_compatibility[subs_copy2] = []

            self.subscripts_compatibility[subs_copy1].append(subs_copy2)
            self.subscripts_compatibility[subs_copy2].append(subs_copy1)

        def visit_subscript_mapping(self, n, vc):

            warnings.warn(
                "\n Subscript mapping detected."
                + "This feature works only in some simple cases."
            )

            if ":" in str(vc):
                # Obtain subscript name and split by : and (
                name_mapped = str(vc).split(":")[0].split("(")[1]
            else:
                (name_mapped,) = vc

            if self.real_name not in self.subscripts_compatibility:
                self.subscripts_compatibility[self.real_name] = []
            self.subscripts_compatibility[self.real_name].append(name_mapped.strip())

        def visit_range(self, n, vc):
            subs_start = vc[2].strip()
            subs_end = vc[6].strip()
            if subs_start == subs_end:
                raise ValueError(
                    "Only different subscripts are valid in a numeric range, error in expression:\n\t %s\n"
                    % (equation_str)
                )

            # get the common prefix and the starting and
            # ending number of the numeric range
            subs_start = re.findall(r"\d+|\D+", subs_start)
            subs_end = re.findall(r"\d+|\D+", subs_end)
            prefix_start = "".join(subs_start[:-1])
            prefix_end = "".join(subs_end[:-1])
            num_start = int(subs_start[-1])
            num_end = int(subs_end[-1])

            if not (prefix_start) or not (prefix_end):
                raise ValueError(
                    "A numeric range must contain at least one letter, error in expression:\n\t %s\n"
                    % (equation_str)
                )
            if num_start > num_end:
                raise ValueError(
                    "The number of the first subscript value must be lower than the second subscript value in a subscript numeric range, error in expression:\n\t %s\n"
                    % (equation_str)
                )
            if (
                prefix_start != prefix_end
                or subs_start[0].isdigit()
                or subs_end[0].isdigit()
            ):
                raise ValueError(
                    "Only matching names ending in numbers are valid, error in expression:\n\t %s\n"
                    % (equation_str)
                )

            for i in range(num_start, num_end + 1):
                s = prefix_start + str(i)
                self.subscripts.append(s.strip())

        def visit_value(self, n, vc):
            self.subscripts.append(vc[1].strip())

        def visit_name(self, n, vc):
            (name,) = vc
            self.real_name = name.strip()
            return self.real_name

        def visit_subscript(self, n, vc):
            (subscript,) = vc
            self.subscripts.append(subscript.strip())
            return subscript.strip()

        def visit_expression(self, n, vc):
            self.expression = n.text.strip()

        def generic_visit(self, n, vc):
            return "".join(filter(None, vc)) or n.text

        def visit__(self, n, vc):
            return " "

    try:
        tree = parser.parse(equation_str)
        parse_object = ComponentParser(tree)
    except (IncompleteParseError, VisitationError, ParseError) as err:
        # this way we get the element name and equation and is easier
        # to detect the error in the model file
        raise ValueError(
            err.args[0] + "\n\n"
            "\nError when parsing definition:\n\t %s\n\n"
            "probably used definition is not integrated..."
            "\nSee parsimonious output above." % (equation_str)
        )

    return {
        "real_name": parse_object.real_name,
        "subs": parse_object.subscripts,
        "subs_compatibility": parse_object.subscripts_compatibility,
        "expr": parse_object.expression,
        "kind": parse_object.kind,
        "keyword": parse_object.keyword,
    }


def parse_sketch_line(module, namespace):

    # not all possibilities can be tested, so this should be considered experimental for now
    sketch_grammar = _include_common_grammar(
        r"""
            line = var_definition / module_intro / module_title / module_definition / arrow / flow / plot_var / anything
            
            module_intro = ~r"\s*Sketch.*?names$" / ~r"^V300.*?ignored$"
            module_title = "*" module_name
            module_name = ~r"(?<=\*)[^\n]+$"
            module_definition = "$" color "," ~r"\d" "," font_properties "|" ((color / weird_stuff) "|")* module_code
            var_definition = var_code "," var_number "," var_name "," position "," var_box_type "," hide_level "," var_face "," var_word_position "," var_thickness "," var_rest_conf ","? ((weird_stuff / color) ",")* font_properties?
            #line_of_symbols = ~r"^[^\w]+$"
            
            # elements used in a line defining the properties of a variable or stock (most are digits, but identifying them now may be useful for further parsing)
            var_name = element #( basic_id / escape_group )
            var_name = ~r"(?<=,)[^,]+(?=,)"
            var_number = digit               
            var_box_type = ~r"(?<=,)\d+,\d+,\d+,\d+(?=,)" # improve this regex
            hide_level = digit
            var_face = digit
            var_word_position = ~r"(?<=,)\-*\d+(?=,)"         
            var_thickness = digit
            var_rest_conf = digit "," ~r"\d+"
            
            arrow = arrow_code "," digit "," origin_var "," destination_var "," (digit ",")+ (weird_stuff ",")?  ((color ",") / ("," ~r"\d+") / (font_properties "," ~"\d+"))* "|(" position ")|"
            
            # arrow origin and destination (this may be useful if further parsing is required)
            origin_var = digit
            destination_var = digit

            # flow arrows
            flow = source_or_sink_or_plot / flow_arrow
            
            # if you want to extend the parsing, these two would be a good starting point (they are followed by "anything")
            source_or_sink_or_plot = multipurpose_code "," anything
            flow_arrow =  flow_arrow_code "," anything
            
            # Variable names after a line with a plot definition, comments, anything really
            plot_var = anything # ~r"^[^,]+$" # if the comment contains commas, it's going to be problematic
            
            # fonts
            font_properties = font_name? "|" font_size "|" font_style? "|" color  # if the B from an RGB is either 1 or 2 (very unlikely), the parser may not be able to tell the begining of the next line
            font_style =  "B" / "I" / "U" / "S" / "V"  # italics, bold, underline, etc
            font_size =  ~r"\d+"  # this needs to be made a regex to match any font
            #font_name = ~r"(%(fonts)s)"IU
            font_name = ~r"(?<=,)[^\|\d]+(?=\|)"

            # this may be useful if further parsing is required
            position = ~r"-*\d+,-*\d+" # x and y
            
            # comma separated value/s
            digit = ~r"(?<=,)\d+(?=,)"
            
            # rgb color
            color = ~r"((?<!\d|\.)([0-9]?[0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])(?!\d|\.) *[-] *){2}(?<!\d|\.)([0-9]?[0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])(?!\d|\.)" # rgb as in 255-255-255
            
            # lines that start with specific numbers (1:arrows, 11:flow_arrow, 12:)
            arrow_code = ~r"^1(?=,)"
            flow_arrow_code = ~r"^11(?=,)"
            var_code = ~r"^10(?=,)"
            multipurpose_code = ~r"^12(?=,)" # source, sink, plot, comment
            
            # code at the end of module definitions
            module_code = "96,96" "," ~r"\d{1,3}" "," "0"
            
            weird_stuff = ~r"\-1\-\-1\-\-1"
            anything = ~r".*" # this one is dangerous, if something is not parsed before, it will fall here, with the risk of missing it
            """
    )

    parser = parsimonious.Grammar(sketch_grammar)

    class SketchParser(parsimonious.NodeVisitor):
        def __init__(self, ast, namespace):
            self.namespace = namespace
            self.entry = {}
            self.visit(ast)

        """
        def visit_line(self, n, vc):
            return vc[0]
        """

        def visit_module_name(self, n, vc):
            self.entry.update(
                {
                    "new_module": True,
                    "module_name": n.text,
                    "variable": False,
                    "variable_name": "",
                }
            )

        def visit_var_name(self, n, vc):
            if n.text in self.namespace.keys():
                self.entry.update(
                    {
                        "new_module": False,
                        "module_name": "",
                        "variable": True,
                        "variable_name": self.namespace[n.text],
                    }
                )
            else:
                message = "\n{} is in the sketch but not in the namespace.\n".format(
                    n.text
                )
                warnings.warn(message)

        def generic_visit(self, n, vc):
            if not self.entry:
                self.entry.update(
                    {
                        "new_module": False,
                        "module_name": "",
                        "variable": False,
                        "variable_name": "",
                    }
                )

    try:
        tree = parser.parse(module)
        return SketchParser(tree, namespace=namespace).entry
    except (IncompleteParseError, VisitationError, ParseError) as err:
        if isinstance(err, VisitationError):
            raise ("Something went wrong while traversing a parse tree", err)
        else:
            raise ValueError(
                (
                    err.args[0] + "\n\n"
                    "\nError when parsing definition:\n\t %s\n\n"
                    "probably used definition is not integrated..."
                    "\nSee parsimonious output above." % (module)
                )
            )


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

    if units_str[-1] == "]":
        units, lims = units_str.rsplit("[")  # type: str, str
    else:
        units = units_str
        lims = "?, ?]"

    lims = tuple(
        [float(x) if x.strip() != "?" else None for x in lims.strip("]").split(",")]
    )

    return units.strip(), lims


functions = {
    # element-wise functions
    "abs": "abs",
    "integer": "int",
    "modulo": {"name": "np.mod", "module": "numpy"},
    "min": {"name": "np.minimum", "module": "numpy"},
    "max": {"name": "np.maximum", "module": "numpy"},
    "exp": {"name": "np.exp", "module": "numpy"},
    "sin": {"name": "np.sin", "module": "numpy"},
    "cos": {"name": "np.cos", "module": "numpy"},
    "tan": {"name": "np.tan", "module": "numpy"},
    "arcsin": {"name": "np.arcsin", "module": "numpy"},
    "arccos": {"name": "np.arccos", "module": "numpy"},
    "arctan": {"name": "np.arctan", "module": "numpy"},
    "sinh": {"name": "np.sinh", "module": "numpy"},
    "cosh": {"name": "np.cosh", "module": "numpy"},
    "tanh": {"name": "np.tanh", "module": "numpy"},
    "sqrt": {"name": "np.sqrt", "module": "numpy"},
    "xidz": {"name": "xidz", "module": "functions"},
    "zidz": {"name": "zidz", "module": "functions"},
    "ln": {"name": "np.log", "module": "numpy"},
    "log": {"name": "log", "module": "functions"},
    "lognormal": {"name": "np.random.lognormal", "module": "numpy"},
    "random normal": {"name": "bounded_normal", "module": "functions"},
    "poisson": {"name": "np.random.poisson", "module": "numpy"},
    "exprnd": {"name": "np.random.exponential", "module": "numpy"},
    "random 0 1": {"name": "random_0_1", "module": "functions"},
    "random uniform": {"name": "random_uniform", "module": "functions"},
    "if then else": {
        "name": "if_then_else",
        "parameters": [
            {"name": "condition"},
            {"name": "val_if_true", "type": "lambda"},
            {"name": "val_if_false", "type": "lambda"},
        ],
        "module": "functions",
    },
    "step": {
        "name": "step",
        "parameters": [
            {"name": "time", "type": "time"},
            {"name": "value"},
            {"name": "tstep"},
        ],
        "module": "functions",
    },
    "pulse": {
        "name": "pulse",
        "parameters": [
            {"name": "time", "type": "time"},
            {"name": "start"},
            {"name": "duration"},
        ],
        "module": "functions",
    },
    # time, start, duration, repeat_time, end
    "pulse train": {
        "name": "pulse_train",
        "parameters": [
            {"name": "time", "type": "time"},
            {"name": "start"},
            {"name": "duration"},
            {"name": "repeat_time"},
            {"name": "end"},
        ],
        "module": "functions",
    },
    "ramp": {
        "name": "ramp",
        "parameters": [
            {"name": "time", "type": "time"},
            {"name": "slope"},
            {"name": "start"},
            {"name": "finish", "optional": True},
        ],
        "module": "functions",
    },
    "active initial": {
        "name": "active_initial",
        "parameters": [
            {"name": "time", "type": "time"},
            {"name": "expr", "type": "lambda"},
            {"name": "init_val"},
        ],
        "module": "functions",
    },
    "game": "",  # In the future, may have an actual `functions.game` to pass
    # vector functions
    "sum": {"name": "sum", "module": "functions"},
    "prod": {"name": "prod", "module": "functions"},
    "vmin": {"name": "vmin", "module": "functions"},
    "vmax": {"name": "vmax", "module": "functions"},
    # TODO functions/stateful objects to be added
    # https://github.com/JamesPHoughton/pysd/issues/154
    "forecast": {
        "name": "not_implemented_function",
        "module": "functions",
        "original_name": "FORECAST",
    },
    "invert matrix": {
        "name": "not_implemented_function",
        "module": "functions",
        "original_name": "INVERT MATRIX",
    },
    "get time value": {
        "name": "not_implemented_function",
        "module": "functions",
        "original_name": "GET TIME VALUE",
    },
}

# list of fuctions that accept a dimension to apply over
vectorial_funcs = ["sum", "prod", "vmax", "vmin"]

# other functions
functions_utils = {
    "lookup": {"name": "lookup", "module": "functions"},
    "round": {"name": "round_", "module": "utils"},
    "rearrange": {"name": "rearrange", "module": "utils"},
    "DataArray": {"name": "xr.DataArray", "module": "xarray"},
}

data_ops = {
    "get data at time": "",
    "get data between times": "",
    "get data last time": "",
    "get data max": "",
    "get data min": "",
    "get data median": "",
    "get data mean": "",
    "get data stdv": "",
    "get data total points": "",
}

builders = {
    "integ": lambda element, subscript_dict, args: builder.add_stock(
        identifier=element["py_name"],
        expression=args[0],
        initial_condition=args[1],
        subs=element["subs"],
    ),
    "delay1": lambda element, subscript_dict, args: builder.add_delay(
        identifier=element["py_name"],
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[0],
        order="1",
        subs=element["subs"],
    ),
    "delay1i": lambda element, subscript_dict, args: builder.add_delay(
        identifier=element["py_name"],
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[2],
        order="1",
        subs=element["subs"],
    ),
    "delay3": lambda element, subscript_dict, args: builder.add_delay(
        identifier=element["py_name"],
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[0],
        order="3",
        subs=element["subs"],
    ),
    "delay3i": lambda element, subscript_dict, args: builder.add_delay(
        identifier=element["py_name"],
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[2],
        order="3",
        subs=element["subs"],
    ),
    "delay fixed": lambda element, subscript_dict, args: builder.add_delay_f(
        identifier=element["py_name"],
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[2],
    ),
    "delay n": lambda element, subscript_dict, args: builder.add_n_delay(
        identifier=element["py_name"],
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[2],
        order=args[3],
        subs=element["subs"],
    ),
    "sample if true": lambda element, subscript_dict, args: builder.add_sample_if_true(
        identifier=element["py_name"],
        condition=args[0],
        actual_value=args[1],
        initial_value=args[2],
    ),
    "smooth": lambda element, subscript_dict, args: builder.add_n_smooth(
        identifier=element["py_name"],
        smooth_input=args[0],
        smooth_time=args[1],
        initial_value=args[0],
        order="1",
        subs=element["subs"],
    ),
    "smoothi": lambda element, subscript_dict, args: builder.add_n_smooth(
        identifier=element["py_name"],
        smooth_input=args[0],
        smooth_time=args[1],
        initial_value=args[2],
        order="1",
        subs=element["subs"],
    ),
    "smooth3": lambda element, subscript_dict, args: builder.add_n_smooth(
        identifier=element["py_name"],
        smooth_input=args[0],
        smooth_time=args[1],
        initial_value=args[0],
        order="3",
        subs=element["subs"],
    ),
    "smooth3i": lambda element, subscript_dict, args: builder.add_n_smooth(
        identifier=element["py_name"],
        smooth_input=args[0],
        smooth_time=args[1],
        initial_value=args[2],
        order="3",
        subs=element["subs"],
    ),
    "smooth n": lambda element, subscript_dict, args: builder.add_n_smooth(
        identifier=element["py_name"],
        smooth_input=args[0],
        smooth_time=args[1],
        initial_value=args[2],
        order=args[3],
        subs=element["subs"],
    ),
    "trend": lambda element, subscript_dict, args: builder.add_n_trend(
        identifier=element["py_name"],
        trend_input=args[0],
        average_time=args[1],
        initial_trend=args[2],
        subs=element["subs"],
    ),
    "get xls data": lambda element, subscript_dict, args: builder.add_ext_data(
        identifier=element["py_name"],
        file_name=args[0],
        tab=args[1],
        time_row_or_col=args[2],
        cell=args[3],
        subs=element["subs"],
        subscript_dict=subscript_dict,
        keyword=element["keyword"],
    ),
    "get xls constants": lambda element, subscript_dict, args: builder.add_ext_constant(
        identifier=element["py_name"],
        file_name=args[0],
        tab=args[1],
        cell=args[2],
        subs=element["subs"],
        subscript_dict=subscript_dict,
    ),
    "get xls lookups": lambda element, subscript_dict, args: builder.add_ext_lookup(
        identifier=element["py_name"],
        file_name=args[0],
        tab=args[1],
        x_row_or_col=args[2],
        cell=args[3],
        subs=element["subs"],
        subscript_dict=subscript_dict,
    ),
    "initial": lambda element, subscript_dict, args: builder.add_initial(args[0]),
    "a function of": lambda element, subscript_dict, args: builder.add_incomplete(
        element["real_name"], args
    ),
}

# direct and xls methods are identically implemented in PySD
builders["get direct data"] = builders["get xls data"]
builders["get direct lookups"] = builders["get xls lookups"]
builders["get direct constants"] = builders["get xls constants"]

# expand dictionaries to detect _ in Vensim def
utils.add_entries_underscore(functions, data_ops, builders)


def parse_general_expression(
    element,
    namespace={},
    subscript_dict={},
    macro_list=None,
    elements_subs_dict={},
    subs_compatibility={},
):
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

    elements_subs_dict : dictionary
        The dictionary with element python names as keys and their merged
        subscripts as values.

    subs_compatibility : dictionary
        The dictionary storing the mapped subscripts

    Returns
    -------
    translation

    new_elements: list of dictionaries
        If the expression contains builder functions, those builders will
        create new elements to add to our running list (that will eventually
        be output to a file) such as stock initialization and derivative
        funcs, etc.


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

    # spaces important for word-based operators
    in_ops = {
        "+": "+",
        "-": "-",
        "*": "*",
        "/": "/",
        "^": "**",
        "=": "==",
        "<=": "<=",
        "<>": "!=",
        "<": "<",
        ">=": ">=",
        ">": ">",
        ":and:": " and ",
        ":or:": " or ",
    }

    pre_ops = {"-": "-", "+": " ", ":not:": " not "}

    # in the following, if lists are empty use non-printable character
    # everything needs to be escaped before going into the grammar,
    # in case it includes quotes
    sub_names_list = [re.escape(x) for x in subscript_dict.keys()] or ["\\a"]
    sub_elems_list = [
        re.escape(y).replace('"', "") for x in subscript_dict.values() for y in x
    ] or ["\\a"]
    in_ops_list = [re.escape(x) for x in in_ops.keys()]
    pre_ops_list = [re.escape(x) for x in pre_ops.keys()]
    if macro_list is not None and len(macro_list) > 0:
        macro_names_list = [re.escape(x["name"]) for x in macro_list]
    else:
        macro_names_list = ["\\a"]

    expression_grammar = _include_common_grammar(
        r"""
    expr_type = array / expr / empty
    expr = _ pre_oper? _ (lookup_with_def / build_call / macro_call /
                          call / lookup_call / parens / number / string / reference)
           _ (in_oper _ expr)?

    lookup_with_def = ~r"(WITH\ LOOKUP)"I _ "(" _ expr _ "," _ "(" _  ("[" ~r"[^\]]*" "]" _ ",")?  ( "(" _ expr _ "," _ expr _ ")" _ ","? _ )+ _ ")" _ ")"

    lookup_call = lookup_call_subs _ parens
    lookup_call_subs = (id _ subscript_list) / id # check first for subscript

    number = ("+"/"-")? ~r"\d+\.?\d*(e[+-]\d+)?"
    range = _ "[" ~r"[^\]]*" "]" _ ","

    arguments = (expr _ ","? _)*
    parens   = "(" _ expr _ ")"

    call = func _ "(" _ arguments _ ")"
    build_call = builder _ "(" _ arguments _ ")"
    macro_call = macro _ "(" _ arguments _ ")"

    reference = (id _ subscript_list) / id  # check first for subscript
    subscript_list = "[" _ ~"\""? _ (subs _ ~"\""? _ "!"? _ ","? _)+ _ "]"

    array = (number _ ("," / ";")? _)+ !~r"."  # negative lookahead for anything other than an array
    string = "\'" ( "\\\'" / ~r"[^\']"IU )* "\'"

    id = ( basic_id / escape_group )

    subs = ~r"(%(subs)s)"IU  # subscript names and elements (if none, use non-printable character)
    func = ~r"(%(funcs)s)"IU  # functions (case insensitive)
    in_oper = ~r"(%(in_ops)s)"IU  # infix operators (case insensitive)
    pre_oper = ~r"(%(pre_ops)s)"IU  # prefix operators (case insensitive)
    builder = ~r"(%(builders)s)"IU  # builder functions (case insensitive)
    macro = ~r"(%(macros)s)"IU  # macros from model file (if none, use non-printable character)

    empty = "" # empty string
    """
        % {
            # In the following, we have to sort keywords in decreasing order of length so that the
            # peg parser doesn't quit early when finding a partial keyword
            "subs": "|".join(
                reversed(sorted(sub_names_list + sub_elems_list, key=len))
            ),
            "funcs": "|".join(reversed(sorted(functions.keys(), key=len))),
            "in_ops": "|".join(reversed(sorted(in_ops_list, key=len))),
            "pre_ops": "|".join(reversed(sorted(pre_ops_list, key=len))),
            "builders": "|".join(reversed(sorted(builders.keys(), key=len))),
            "macros": "|".join(reversed(sorted(macro_names_list, key=len))),
        }
    )

    parser = parsimonious.Grammar(expression_grammar)

    class ExpressionParser(parsimonious.NodeVisitor):
        # TODO: at some point, we could make the 'kind' identification
        # recursive on expression, so that if an expression is passed into
        # a builder function, the information  about whether it is a constant,
        # or calls another function, goes with it.

        def __init__(self, ast):
            self.translation = ""
            self.subs = None  # the subscript list if given
            self.lookup_subs = []
            self.apply_dim = set()  # the dimensions with ! if given
            self.kind = "constant"  # change if we reference anything else
            self.new_structure = []
            self.append = ""
            self.lookup_append = []
            self.arguments = None
            self.in_oper = None
            self.args = []
            self.visit(ast)

        def visit_expr_type(self, n, vc):
            s = "".join(filter(None, vc)).strip()
            self.translation = s

        def visit_expr(self, n, vc):
            s = "".join(filter(None, vc)).strip()
            self.translation = s
            return s

        def visit_call(self, n, vc):
            self.kind = "component"

            function_name = vc[0].lower()
            arguments = vc[4]

            # add dimensions as last argument
            if self.apply_dim and function_name in vectorial_funcs:
                arguments += ["dim=" + str(tuple(self.apply_dim))]
                self.apply_dim = set()

            return builder.build_function_call(functions[function_name], arguments)

        def visit_in_oper(self, n, vc):
            return in_ops[n.text.lower()]

        def visit_pre_oper(self, n, vc):
            return pre_ops[n.text.lower()]

        def visit_reference(self, n, vc):
            self.kind = "component"

            py_expr = vc[0] + "()" + self.append
            self.append = ""

            if self.subs:
                if elements_subs_dict[vc[0]] != self.subs:
                    py_expr = builder.build_function_call(
                        functions_utils["rearrange"],
                        [py_expr, repr(self.subs), "_subscript_dict"],
                    )

                mapping = self.subs.copy()
                for i, sub in enumerate(self.subs):
                    if sub in subs_compatibility:
                        for compatible in subs_compatibility[sub]:
                            if compatible in element["subs"]:
                                mapping[i] = compatible

                if self.subs != mapping:
                    py_expr = builder.build_function_call(
                        functions_utils["rearrange"],
                        [py_expr, repr(mapping), "_subscript_dict"],
                    )

                self.subs = None

            return py_expr

        def visit_lookup_call_subs(self, n, vc):
            # necessary if a lookup dimension is subselected but we have
            # other reference objects as arguments
            self.lookup_append.append(self.append)
            self.append = ""

            # recover subs for lookup to avoid using them for arguments
            if self.subs:
                self.lookup_subs.append(self.subs)
                self.subs = None
            else:
                self.lookup_subs.append(None)

            return vc[0]

        def visit_lookup_call(self, n, vc):
            lookup_append = self.lookup_append.pop()
            lookup_subs = self.lookup_subs.pop()
            py_expr = "".join([x.strip(",") for x in vc]) + lookup_append

            if lookup_subs and elements_subs_dict[vc[0]] != lookup_subs:
                dims = [
                    utils.find_subscript_name(subscript_dict, sub)
                    for sub in lookup_subs
                ]
                return builder.build_function_call(
                    functions_utils["rearrange"],
                    [py_expr, repr(dims), "_subscript_dict"],
                )

            return py_expr

        def visit_id(self, n, vc):
            return namespace[n.text.strip()]

        def visit_lookup_with_def(self, n, vc):
            """This exists because vensim has multiple ways of doing lookups.
            Which is frustrating."""
            self.kind = "lookup"
            x_val = vc[4]
            pairs = vc[11]
            mixed_list = pairs.replace("(", "").replace(")", "").split(",")
            xs = mixed_list[::2]
            ys = mixed_list[1::2]
            arguments = [x_val, "[" + ",".join(xs) + "]", "[" + ",".join(ys) + "]"]
            return builder.build_function_call(functions_utils["lookup"], arguments)

        def visit_array(self, n, vc):
            # first test handles when subs is not defined
            if "subs" in element and element["subs"]:
                coords = utils.make_coord_dict(
                    element["subs"], subscript_dict, terse=False
                )
                if ";" in n.text or "," in n.text:
                    text = n.text.strip(";").replace(" ", "").replace(";", ",")
                    data = np.array([float(s) for s in text.split(",")])
                    data = data.reshape(utils.compute_shape(coords))
                    datastr = (
                        np.array2string(data, separator=",")
                        .replace("\n", "")
                        .replace(" ", "")
                    )
                else:
                    datastr = n.text

                # Following implementation makes cleaner the model file
                if list(coords) == element["subs"]:
                    if len(coords) == len(subscript_dict):
                        # variable is defined with all subscrips
                        return builder.build_function_call(
                            functions_utils["DataArray"],
                            [datastr, "_subscript_dict", repr(list(coords))],
                        )

                from_dict, no_from_dict = [], {}
                for coord, sub in zip(coords, element["subs"]):
                    # find dimensions can be retrieved from _subscript_dict
                    if coord == sub:
                        from_dict.append(coord)
                    else:
                        no_from_dict[coord] = coords[coord]

                if from_dict and no_from_dict:
                    # some dimensons can be retrieved from _subscript_dict
                    coordsp = (
                        "{**{dim: _subscript_dict[dim] for dim in %s}, " % from_dict
                        + repr(no_from_dict)[1:]
                    )
                elif from_dict:
                    # all dimensons can be retrieved from _subscript_dict
                    coordsp = "{dim: _subscript_dict[dim] for dim in %s}" % from_dict
                else:
                    # no dimensons can be retrieved from _subscript_dict
                    coordsp = repr(no_from_dict)

                return builder.build_function_call(
                    functions_utils["DataArray"], [datastr, coordsp, repr(list(coords))]
                )

            else:
                return n.text.replace(" ", "")

        def visit_subscript_list(self, n, vc):
            refs = vc[4]
            subs = [x.strip() for x in refs.split(",")]
            coordinates = [
                sub if sub not in subscript_dict and sub[-1] != "!" else False
                for sub in subs
            ]

            # Implements basic "!" subscript functionality in Vensim.
            # Does NOT work for matrix diagonals in
            # FUNC(variable[sub1!,sub1!]) functions
            self.apply_dim.update(["%s" % s.strip("!") for s in subs if s[-1] == "!"])

            if any(coordinates):
                coords, subs2 = [], []
                for coord, sub in zip(coordinates, subs):
                    if coord:
                        # subset coord
                        coords.append("'%s'" % coord)
                    else:
                        # do not subset coord
                        coords.append(":")
                        subs2.append(sub.strip("!"))

                if subs2:
                    self.subs = subs2

                self.append = ".loc[%s].reset_coords(drop=True)" % (", ".join(coords))

            else:
                self.subs = ["%s" % s.strip("!") for s in subs]

            return ""

        def visit_build_call(self, n, vc):
            # use only the dict with the final subscripts
            # needed for the good working of externals

            subs = {
                k: subscript_dict[k] for k in elements_subs_dict[element["py_name"]]
            }
            self.kind = "component"
            builder_name = vc[0].strip().lower()
            name, structure = builders[builder_name](element, subs, vc[4])

            self.new_structure += structure

            if "lookups" in builder_name:
                self.arguments = "x"
                self.kind = "lookup"
            elif "constant" in builder_name:
                # External constants
                self.kind = "constant"
            elif "data" in builder_name:
                # External data
                self.kind = "component_ext_data"

            return name

        def visit_macro_call(self, n, vc):
            call = vc[0]
            arglist = vc[4]
            self.kind = "component"
            py_name = utils.make_python_identifier(call)[0]
            macro = [x for x in macro_list if x["py_name"] == py_name][
                0
            ]  # should match once
            name, structure = builder.add_macro(
                macro["py_name"], macro["file_name"], macro["params"], arglist
            )
            self.new_structure += structure
            return name

        def visit_arguments(self, n, vc):
            arglist = [x.strip(",") for x in vc]
            return arglist

        def visit__(self, n, vc):
            """Handles whitespace characters"""
            return ""

        def visit_empty(self, n, vc):
            return "None"

        def generic_visit(self, n, vc):
            return "".join(filter(None, vc)) or n.text

    try:
        tree = parser.parse(element["expr"])
        parse_object = ExpressionParser(tree)
    except (IncompleteParseError, VisitationError, ParseError) as err:
        # this way we get the element name and equation and is easier
        # to detect the error in the model file
        raise ValueError(
            err.args[0] + "\n\n"
            "\nError when parsing %s with equation\n\t %s\n\n"
            "probably a used function is not integrated..."
            "\nSee parsimonious output above." % (element["real_name"], element["eqn"])
        )

    return (
        {
            "py_expr": parse_object.translation,
            "kind": parse_object.kind,
            "arguments": parse_object.arguments or "",
        },
        parse_object.new_structure,
    )


def parse_lookup_expression(element, subscript_dict):
    """This syntax parses lookups that are defined with their own element"""

    lookup_grammar = r"""
    lookup = _ "(" _ (regularLookup / excelLookup) _ ")"
    regularLookup = range? _ ( "(" _ number _ "," _ number _ ")" _ ","? _ )+
    excelLookup = ~"GET( |_)(XLS|DIRECT)( |_)LOOKUPS"I _ "(" (args _ ","? _)+ ")"
    args = ~r"[^,()]*"
    number = ("+"/"-")? ~r"\d+\.?\d*(e[+-]\d+)?"
    _ =  ~r"[\s\\]*" #~r"[\ \t\n]*" #~r"[\s\\]*"  # whitespace character
    range = _ "[" ~r"[^\]]*" "]" _ ","
    """
    parser = parsimonious.Grammar(lookup_grammar)
    tree = parser.parse(element["expr"])

    class LookupParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.translation = ""
            self.new_structure = []
            self.visit(ast)

        def visit__(self, n, vc):
            # remove whitespace
            return ""

        def visit_regularLookup(self, n, vc):

            pairs = max(vc, key=len)
            mixed_list = pairs.replace("(", "").replace(")", "").split(",")
            xs = mixed_list[::2]
            ys = mixed_list[1::2]
            arguments = ["x", "[" + ",".join(xs) + "]", "[" + ",".join(ys) + "]"]
            self.translation = builder.build_function_call(
                functions_utils["lookup"], arguments
            )

        def visit_excelLookup(self, n, vc):
            arglist = vc[3].split(",")
            arglist = [arg.replace("\\ ", "") for arg in arglist]
            trans, structure = builders["get xls lookups"](
                element, subscript_dict, arglist
            )

            self.translation = trans
            self.new_structure += structure

        def generic_visit(self, n, vc):
            return "".join(filter(None, vc)) or n.text

    parse_object = LookupParser(tree)
    return (
        {"py_expr": parse_object.translation, "arguments": "x"},
        parse_object.new_structure,
    )


def translate_section(section, macro_list, sketch, root_path):

    model_elements = get_model_elements(section["string"])

    # extract equation components
    model_docstring = ""
    for entry in model_elements:
        if entry["kind"] == "entry":
            entry.update(get_equation_components(entry["eqn"], root_path))
        elif entry["kind"] == "section":
            model_docstring += entry["doc"]

    # make python identifiers and track for namespace conflicts
    namespace = {"TIME": "time", "Time": "time"}  # Initialize with builtins
    # add macro parameters when parsing a macro section
    for param in section["params"]:
        name, namespace = utils.make_python_identifier(param, namespace)

    # add macro functions to namespace
    for macro in macro_list:
        if macro["name"] != "_main_":
            name, namespace = utils.make_python_identifier(macro["name"], namespace)

    # Create a namespace for the subscripts as these aren't used to
    # create actual python functions, but are just labels on arrays,
    # they don't actually need to be python-safe
    # Also creates a dictionary with all the subscript that are mapped

    subscript_dict = {}
    subs_compatibility_dict = {}
    for e in model_elements:
        if e["kind"] == "subdef":
            subscript_dict[e["real_name"]] = e["subs"]
            for compatible in e["subs_compatibility"]:
                if compatible in subs_compatibility_dict:
                    subs_compatibility_dict[compatible].update(
                        e["subs_compatibility"][compatible]
                    )
                else:
                    subs_compatibility_dict[compatible] = set(
                        e["subs_compatibility"][compatible]
                    )
                # check if copy
                if not subscript_dict[compatible]:
                    # copy subscript to subscript_dict
                    subscript_dict[compatible] = subscript_dict[
                        e["subs_compatibility"][compatible][0]
                    ]

    elements_subs_dict = {}
    # add model elements
    for element in model_elements:
        if element["kind"] not in ["subdef", "section"]:
            element["py_name"], namespace = utils.make_python_identifier(
                element["real_name"], namespace
            )
            # dictionary to save the subscripts of each element so we can avoid
            # using utils.rearrange when calling them with the same dimensions
            if element["py_name"] in elements_subs_dict:
                elements_subs_dict[element["py_name"]].append(element["subs"])
            else:
                elements_subs_dict[element["py_name"]] = [element["subs"]]

    elements_subs_dict = {
        el: utils.make_merge_list(elements_subs_dict[el], subscript_dict)
        for el in elements_subs_dict
    }

    if sketch:
        # TODO how about macros??? are they also put in the sketch?
        modules_list = []  # list of all modules
        module_elements = {}
        # from the sketch it is not apparent what distinguishes a normal variable from a ghost variable (apart from the color, which is not a reliable)
        ghost_variables = {}
        sketch = list(map(lambda x: x.strip(), sketch.split("\\\\\\---/// ")))
        for module in sketch:
            if module:
                for sketch_line in module.split("\n"):
                    line = parse_sketch_line(sketch_line.strip(), namespace)
                    # When a module name is found, the "new_module" becomes True.
                    # When a variable name is found, the "new_module" is set back to False
                    if line["new_module"]:
                        # remove characters that are not [a-zA-Z0-9_] from the module name
                        module_name = re.sub(
                            r"[\W]+", "", line["module_name"].replace(" ", "_")
                        ).lstrip(
                            "0123456789"
                        )  # there's probably a more elegant way to do it with regex
                        if module_name not in modules_list:
                            modules_list.append(module_name)
                    else:
                        if line["variable_name"]:
                            if line["variable_name"] not in module_elements.keys():
                                module_elements[line["variable_name"]] = module_name
                            else:
                                if line["variable_name"] not in ghost_variables.keys():
                                    ghost_variables[line["variable_name"]] = [
                                        module_elements[line["variable_name"]],
                                        module_name,
                                    ]
                                else:
                                    ghost_variables[line["variable_name"]].append(
                                        module_name
                                    )

    # Parse components to python syntax.
    for element in model_elements:
        if (element["kind"] == "component" and "py_expr" not in element) or element[
            "kind"
        ] == "data":
            # TODO: if there is new structure,
            # it should be added to the namespace...
            translation, new_structure = parse_general_expression(
                element,
                namespace=namespace,
                subscript_dict=subscript_dict,
                macro_list=macro_list,
                elements_subs_dict=elements_subs_dict,
                subs_compatibility=subs_compatibility_dict,
            )
            element.update(translation)
            model_elements += new_structure

        elif element["kind"] == "lookup":
            translation, new_structure = parse_lookup_expression(
                element, subscript_dict
            )
            element.update(translation)
            model_elements += new_structure

    # send the pieces to be built
    build_elements = [
        e for e in model_elements if e["kind"] not in ["subdef", "test", "section"]
    ]
    builder.build(build_elements, subscript_dict, namespace, section["file_name"])

    return section["file_name"]


def split_sketch(text):
    """

    Parameters
    ----------
    text : basestring
        Full model as a string

    Returns
    -------
    text: string
        Model file without sketch

    sketch: string
        Model sketch

    """
    split_model = text.split("\\\\\\---///", 1)

    try:
        text = split_model[0]
        sketch = split_model[1]
        # remove plots section, if it exists
        sketch = sketch.split("///---\\\\\\")[0]
    except:
        raise ("Your model does not have a sketch.")

    return text, sketch


def translate_vensim(mdl_file, split_modules):
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

    """

    root_path = os.path.split(mdl_file)[0]
    with open(mdl_file, "r", encoding="UTF-8") as in_file:
        text = in_file.read()

    # check for model extension
    if ".mdl" not in mdl_file and ".MDL" not in mdl_file:
        raise ValueError(
            "The file to translate, "
            + mdl_file
            + " is not a vensim model. It must end with mdl or MDL extension."
        )
    outfile_name = mdl_file.replace(".mdl", ".py").replace(".MDL", ".py")
    out_dir = os.path.dirname(outfile_name)

    if split_modules:
        text, sketch = split_sketch(text)
    else:
        sketch = ""

    file_sections = get_file_sections(text.replace("\n", ""))

    for section in file_sections:
        if section["name"] == "_main_":
            section["file_name"] = outfile_name
        else:  # separate macro elements into their own files
            section["py_name"] = utils.make_python_identifier(section["name"])[0]
            section["file_name"] = out_dir + "/" + section["py_name"] + ".py"

    macro_list = [s for s in file_sections if s["name"] != "_main_"]

    for section in file_sections:
        translate_section(section, macro_list, sketch, root_path)

    return outfile_name
