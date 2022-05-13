"""
The Element class allows parsing the LHS of a model equation.
Depending on the LHS value, either a SubscriptRange object or a Component
object will be returned. There are 4 components types:

- Component: Regular component, defined with '='.
- UnchangeableConstant: Unchangeable constant, defined with '=='.
- Data: Data component, defined with ':='
- Lookup: Lookup component, defined with '()'

Lookup components have their own parser for the RHS of the expression,
while the other 3 components share the same parser. The final result
from a parsed component can be exported to an AbstractComponent object
in order to build a model in other programming languages.
"""
import re
from typing import Union, Tuple, List
import warnings
import parsimonious
import numpy as np

from ..structures.abstract_model import\
    AbstractData, AbstractLookup, AbstractComponent,\
    AbstractUnchangeableConstant, AbstractSubscriptRange

from . import vensim_utils as vu
from .vensim_structures import structures, parsing_ops


class Element():
    """
    Element object allows parsing the the LHS of the Vensim
    expressions.

    Parameters
    ----------
    equation: str
        Original equation in the Vensim file.

    units: str
        The units of the element with the limits, i.e., the content after
        the first '~' symbol.

    documentation: str
        The comment of the element, i.e., the content after the second
        '~' symbol.

    """

    def __init__(self, equation: str, units: str, documentation: str):
        self.equation = equation
        self.units, self.limits = self._parse_units(units)
        self.documentation = documentation

    def __str__(self):  # pragma: no cover
        return "Model element:\n\t%s\nunits: %s\ndocs: %s\n" % (
            self.equation, self.units, self.documentation)

    @property
    def _verbose(self) -> str:  # pragma: no cover
        """Get element information."""
        return self.__str__()

    @property
    def verbose(self):  # pragma: no cover
        """Print element information."""
        print(self._verbose)

    def _parse_units(self, units_str: str) -> Tuple[str, tuple]:
        """Separate the limits from the units."""
        # TODO improve units parsing: parse them when parsing the section
        # elements
        if not units_str:
            return "", None

        if units_str.endswith("]"):
            units, lims = units_str.rsplit("[")  # types: str, str
        else:
            return units_str, None

        lims = tuple(
            [
                float(x) if x.strip() != "?" else None
                for x in lims.strip("]").split(",")
            ]
        )
        return units.strip(), lims

    def parse(self) -> object:
        """
        Parse an Element object with parsimonious using the grammar given in
        'parsing_grammars/element_object.peg' and the class
        ElementsComponentVisitor to visit the parsed expressions.

        Splits the LHS from the RHS of the equation. If the returned
        object is a SubscriptRange, no more parsing is needed. Otherwise,
        the RHS of the returned object (Component) should be parsed
        to get the AbstractSyntax Tree.

        Returns
        -------
        self.component: SubscriptRange or Component
            The subscript range definition object or component object.

        """
        tree = vu.Grammar.get("element_object").parse(self.equation)
        self.component = ElementsComponentVisitor(tree).component
        self.component.units = self.units
        self.component.limits = self.limits
        self.component.documentation = self.documentation
        return self.component


class ElementsComponentVisitor(parsimonious.NodeVisitor):
    """Visit model element definition to get the component object."""

    def __init__(self, ast):
        self.mapping = []
        self.subscripts = []
        self.subscripts_except = []
        self.subscripts_except_groups = []
        self.name = None
        self.expression = None
        self.keyword = None
        self.visit(ast)

    def visit_subscript_definition(self, n, vc):
        self.component = SubscriptRange(
            self.name, self.subscripts, self.mapping)

    def visit_lookup_definition(self, n, vc):
        self.component = Lookup(
            self.name,
            (self.subscripts, self.subscripts_except_groups),
            self.expression
        )

    def visit_unchangeable_constant(self, n, vc):
        self.component = UnchangeableConstant(
            self.name,
            (self.subscripts, self.subscripts_except_groups),
            self.expression
        )

    def visit_component(self, n, vc):
        self.component = Component(
            self.name,
            (self.subscripts, self.subscripts_except_groups),
            self.expression
        )

    def visit_data_definition(self, n, vc):
        self.component = Data(
            self.name,
            (self.subscripts, self.subscripts_except_groups),
            self.keyword,
            self.expression
        )

    def visit_keyword(self, n, vc):
        self.keyword = n.text.strip()[1:-1].lower().replace(" ", "_")

    def visit_imported_subscript(self, n, vc):
        self.subscripts = {
            arg_name: argument.strip().strip("'")
            for arg_name, argument
            in zip(
                ("file", "tab", "firstcell", "lastcell", "prefix"),
                vc[4].split(",")
            )
        }

    def visit_subscript_copy(self, n, vc):
        self.component = SubscriptRange(self.name, vc[4].strip())

    def visit_subscript_mapping(self, n, vc):
        if ":" in str(vc):
            # TODO: ensure the correct working of this condition adding
            # full integration tests
            warnings.warn(
                "\nSubscript mapping detected. "
                + "This feature works only for simple cases."
            )
            # Obtain subscript name and split by : and (
            self.mapping.append(str(vc).split(":")[0].split("(")[1].strip())
        else:
            self.mapping.append(vc[0].strip())

    def visit_subscript_range(self, n, vc):
        subs_start = re.findall(r"\d+|\D+", vc[2].strip())
        subs_end = re.findall(r"\d+|\D+", vc[6].strip())
        prefix_start, num_start = "".join(subs_start[:-1]), int(subs_start[-1])
        prefix_end, num_end = "".join(subs_end[:-1]), int(subs_end[-1])

        if not prefix_start or not prefix_end:
            raise ValueError(
                "\nA numeric range must contain at least one letter.")
        elif num_start >= num_end:
            raise ValueError(
                "\nThe number of the first subscript value must be "
                "lower than the second subscript value in a "
                "subscript numeric range.")
        elif prefix_start != prefix_end:
            raise ValueError(
                "\nOnly matching names ending in numbers are valid.")

        self.subscripts += [
            prefix_start + str(i) for i in range(num_start, num_end + 1)
            ]

    def visit_name(self, n, vc):
        self.name = vc[0].strip()

    def visit_subscript(self, n, vc):
        self.subscripts.append(n.text.strip())

    def visit_subscript_except(self, n, vc):
        self.subscripts_except.append(n.text.strip())

    def visit_subscript_except_group(self, n, vc):
        self.subscripts_except_groups.append(self.subscripts_except.copy())
        self.subscripts_except = []

    def visit_expression(self, n, vc):
        self.expression = n.text.strip()

    def generic_visit(self, n, vc):
        return "".join(filter(None, vc)) or n.text


class SubscriptRange():
    """
    Subscript range definition, defined by ":" or "<->" in Vensim.
    """

    def __init__(self, name: str, definition: Union[List[str], str, dict],
                 mapping: List[str] = []):
        self.name = name
        self.definition = definition
        self.mapping = mapping

    def __str__(self):  # pragma: no cover
        return "\nSubscript range definition:  %s\n\t%s\n" % (
            self.name,
            "%s <- %s" % (self.definition, self.mapping)
            if self.mapping else self.definition)

    @property
    def _verbose(self) -> str:  # pragma: no cover
        """Get subscript range information."""
        return self.__str__()

    @property
    def verbose(self):  # pragma: no cover
        """Print subscript range information."""
        print(self._verbose)

    def get_abstract_subscript_range(self) -> AbstractSubscriptRange:
        """
        Instantiates an AbstractSubscriptRange object used for building.
        This method is automatically called by the Sections's 
        get_abstract_section method.

        Returns
        -------
        AbstractSubscriptRange: AbstractSubscriptRange
          AbstractSubscriptRange object that can be used for building
          the model in another programming language.

        """
        return AbstractSubscriptRange(
            name=self.name,
            subscripts=self.definition,
            mapping=self.mapping
        )


class Component():
    """
    Model component defined by "name = expr" in Vensim.

    Parameters
    ----------
    name: str
        The original name of the component.

    subscripts: tuple
        Tuple of length two with first argument the list of subscripts
        in the variable definition and the second argument the list of
        subscripts list that appears after :EXCEPT: keyword (if used).

    expression: str
        The RHS of the element, expression to parse.

    """
    _kind = "Model component"

    def __init__(self, name: str, subscripts: Tuple[list, list],
                 expression: str):
        self.name = name
        self.subscripts = subscripts
        self.expression = expression

    def __str__(self):  # pragma: no cover
        text = "\n%s definition: %s" % (self._kind, self.name)
        text += "\nSubscrips: %s" % repr(self.subscripts[0])\
            if self.subscripts[0] else ""
        text += "  EXCEPT  %s" % repr(self.subscripts[1])\
            if self.subscripts[1] else ""
        text += "\n\t%s" % self._expression
        return text

    @property
    def _expression(self):  # pragma: no cover
        if hasattr(self, "ast"):
            return str(self.ast).replace("\n", "\n\t")

        else:
            return self.expression.replace("\n", "\n\t")

    @property
    def _verbose(self) -> str:  # pragma: no cover
        """Get component information."""
        return self.__str__()

    @property
    def verbose(self):  # pragma: no cover
        """Print component information."""
        print(self._verbose)

    def parse(self) -> None:
        """
        Parse component object with parsimonious using the grammar given
        in 'parsin_grammars/components.peg' and the class EquationVisitor
        to visit the RHS of the expressions.

        """
        tree = vu.Grammar.get("components", parsing_ops).parse(self.expression)
        self.ast = EquationVisitor(tree).translation

        if isinstance(self.ast, structures["get_xls_lookups"]):
            self.lookup = True
        else:
            self.lookup = False

    def get_abstract_component(self) -> Union[AbstractComponent,
                                              AbstractLookup]:
        """
        Get Abstract Component used for building. This method is
        automatically called by Sections's get_abstract_section.

        Returns
        -------
        AbstractComponent: AbstractComponent or AbstractLookup
          Abstract Component object that can be used for building
          the model in another language. If the component equations
          includes external lookups (GET XLS/DIRECT LOOKUPS)
          AbstractLookup class will be used

        """
        if self.lookup:
            # get lookups equations
            return AbstractLookup(subscripts=self.subscripts, ast=self.ast)
        else:
            return AbstractComponent(subscripts=self.subscripts, ast=self.ast)


class UnchangeableConstant(Component):
    """
    Unchangeable constant defined by "name == expr" in Vensim.
    This class is a soon of Component.

    Parameters
    ----------
    name: str
        The original name of the component.

    subscripts: tuple
        Tuple of length two with first argument the list of subscripts
        in the variable definition and the second argument the list of
        subscripts list that appears after :EXCEPT: keyword (if used).

    expression: str
        The RHS of the element, expression to parse.

    """
    _kind = "Unchangeable constant component"

    def __init__(self, name: str, subscripts: Tuple[list, list],
                 expression: str):
        super().__init__(name, subscripts, expression)

    def get_abstract_component(self) -> AbstractUnchangeableConstant:
        """
        Get Abstract Component used for building. This method is
        automatically called by Sections's get_abstract_section.

        Returns
        -------
        AbstractComponent: AbstractUnchangeableConstant
          Abstract Component object that can be used for building
          the model in another language.

        """
        return AbstractUnchangeableConstant(
            subscripts=self.subscripts, ast=self.ast)


class Lookup(Component):
    """
    Lookup component, defined by "name(expr)" in Vensim.
    This class is a soon of Component.

    Parameters
    ----------
    name: str
        The original name of the component.

    subscripts: tuple
        Tuple of length two with first argument the list of subscripts
        in the variable definition and the second argument the list of
        subscripts list that appears after :EXCEPT: keyword (if used).

    expression: str
        The RHS of the element, expression to parse.

    """
    _kind = "Lookup component"

    def __init__(self, name: str, subscripts: Tuple[list, list],
                 expression: str):
        super().__init__(name, subscripts, expression)

    def parse(self) -> None:
        """
        Parse component object with parsimonious using the grammar given
        in 'parsin_grammars/lookups.peg' and the class LookupsVisitor
        to visit the RHS of the expressions.
        """
        tree = vu.Grammar.get("lookups").parse(self.expression)
        self.ast = LookupsVisitor(tree).translation

    def get_abstract_component(self) -> AbstractLookup:
        """
        Get Abstract Component used for building. This method is
        automatically called by Sections's get_abstract_section.

        Returns
        -------
        AbstractComponent: AbstractLookup
          Abstract Component object that can be used for building
          the model in another language.

        """
        return AbstractLookup(subscripts=self.subscripts, ast=self.ast)


class Data(Component):
    """
    Data component, defined by "name := expr" in Vensim.
    This class is a soon of Component.

    Parameters
    ----------
    name: str
        The original name of the component.

    subscripts: tuple
        Tuple of length two with first argument the list of subscripts
        in the variable definition and the second argument the list of
        subscripts list that appears after :EXCEPT: keyword (if used).

    keyword: str
        The keyword used befor the ":=" symbol, it could be ('interpolate',
        'raw', 'hold_backward', 'look_forward')

    expression: str
        The RHS of the element, expression to parse.

    """
    _kind = "Data component"

    def __init__(self, name: str, subscripts: Tuple[list, list],
                 keyword: str, expression: str):
        super().__init__(name, subscripts, expression)
        self.keyword = keyword

    def __str__(self):  # pragma: no cover
        text = "\n%s definition: %s" % (self._kind, self.name)
        text += "\nSubscrips: %s" % repr(self.subscripts[0])\
            if self.subscripts[0] else ""
        text += "  EXCEPT  %s" % repr(self.subscripts[1])\
            if self.subscripts[1] else ""
        text += "\nKeyword: %s" % self.keyword if self.keyword else ""
        text += "\n\t%s" % self._expression
        return text

    def parse(self) -> None:
        """
        Parse component object with parsimonious using the grammar given
        in 'parsin_grammars/components.peg' and the class EquationVisitor
        to visit the RHS of the expressions.

        If the expression is None, then de data will be readen from a
        VDF file in Vensim.

        """
        if not self.expression:
            # empty data vars, read from vdf file
            self.ast = structures["data"]()
        else:
            super().parse()

    def get_abstract_component(self) -> AbstractData:
        """
        Get Abstract Component used for building. This method is
        automatically called by Sections's get_abstract_section.

        Returns
        -------
        AbstractComponent: AbstractData
          Abstract Component object that can be used for building
          the model in another language.

        """
        return AbstractData(
            subscripts=self.subscripts, ast=self.ast, keyword=self.keyword)


class LookupsVisitor(parsimonious.NodeVisitor):
    """Visit the elements of a lookups to get the AST"""
    def __init__(self, ast):
        self.translation = None
        self.visit(ast)

    def visit_limits(self, n, vc):
        return n.text.strip()[:-1].replace(")-(", "),(")

    def visit_regularLookup(self, n, vc):
        if vc[0]:
            xy_limits = np.array(eval(vc[0]))
        else:
            xy_limits = np.full((2, 2), np.nan)

        values = np.array((eval(vc[2])))
        values = values[np.argsort(values[:, 0])]

        self.translation = structures["lookup"](
            x=tuple(values[:, 0]),
            y=tuple(values[:, 1]),
            x_limits=tuple(xy_limits[:, 0]),
            y_limits=tuple(xy_limits[:, 1]),
            type="interpolate"
        )

    def visit_excelLookup(self, n, vc):
        arglist = vc[3].split(",")

        self.translation = structures["get_xls_lookups"](
            file=eval(arglist[0]),
            tab=eval(arglist[1]),
            x_row_or_col=eval(arglist[2]),
            cell=eval(arglist[3])
        )

    def generic_visit(self, n, vc):
        return "".join(filter(None, vc)) or n.text


class EquationVisitor(parsimonious.NodeVisitor):
    """Visit the elements of a equation to get the AST"""
    def __init__(self, ast):
        self.translation = None
        self.elements = {}
        self.subs = None  # the subscripts if given
        self.negatives = set()
        self.visit(ast)

    def visit_expr_type(self, n, vc):
        self.translation = self.elements[vc[0]]

    def visit_final_expr(self, n, vc):
        # expressions with logical binary operators (:AND:, :OR:)
        return vu.split_arithmetic(
            structures["logic"], parsing_ops["logic_ops"],
            "".join(vc).strip(), self.elements)

    def visit_logic_expr(self, n, vc):
        # expressions with logical unitary operators (:NOT:)
        id = vc[2]
        if vc[0].lower() == ":not:":
            id = self.add_element(structures["logic"](
                [":NOT:"],
                (self.elements[id],)
                ))
        return id

    def visit_comp_expr(self, n, vc):
        # expressions with comparisons (=, <>, <, <=, >, >=)
        return vu.split_arithmetic(
            structures["logic"], parsing_ops["comp_ops"],
            "".join(vc).strip(), self.elements)

    def visit_add_expr(self, n, vc):
        # expressions with additions (+, -)
        return vu.split_arithmetic(
            structures["arithmetic"], parsing_ops["add_ops"],
            "".join(vc).strip(), self.elements)

    def visit_prod_expr(self, n, vc):
        # expressions with products (*, /)
        return vu.split_arithmetic(
            structures["arithmetic"], parsing_ops["prod_ops"],
            "".join(vc).strip(), self.elements)

    def visit_exp_expr(self, n, vc):
        # expressions with exponentials (^)
        return vu.split_arithmetic(
            structures["arithmetic"], parsing_ops["exp_ops"],
            "".join(vc).strip(), self.elements, self.negatives)

    def visit_neg_expr(self, n, vc):
        id = vc[2]
        if vc[0] == "-":
            if isinstance(self.elements[id], (float, int)):
                self.elements[id] = -self.elements[id]
            else:
                self.negatives.add(id)
        return id

    def visit_call(self, n, vc):
        func = self.elements[vc[0]]
        args = self.elements[vc[4]]
        if func.reference in structures:
            return self.add_element(structures[func.reference](*args))
        else:
            return self.add_element(structures["call"](func, args))

    def visit_reference(self, n, vc):
        id = self.add_element(structures["reference"](
            vc[0].lower().replace(" ", "_"), self.subs))
        self.subs = None
        return id

    def visit_limits(self, n, vc):
        return self.add_element(n.text.strip()[:-1].replace(")-(", "),("))

    def visit_lookup_with_def(self, n, vc):
        if vc[10]:
            xy_limits = np.array(eval(self.elements[vc[10]]))
        else:
            xy_limits = np.full((2, 2), np.nan)

        values = np.array((eval(vc[11])))
        values = values[np.argsort(values[:, 0])]

        lookup = structures["lookup"](
            x=tuple(values[:, 0]),
            y=tuple(values[:, 1]),
            x_limits=tuple(xy_limits[:, 0]),
            y_limits=tuple(xy_limits[:, 1]),
            type="interpolate"
        )

        return self.add_element(structures["with_lookup"](
            self.elements[vc[4]], lookup))

    def visit_array(self, n, vc):
        if ";" in n.text or "," in n.text:
            return self.add_element(np.squeeze(np.array(
                [row.split(",") for row in n.text.strip(";").split(";")],
                dtype=float)))
        else:
            return self.add_element(eval(n.text))

    def visit_subscript_list(self, n, vc):
        subs = [x.strip() for x in vc[2].split(",")]
        self.subs = structures["subscripts_ref"](subs)
        return ""

    def visit_name(self, n, vc):
        return n.text.strip()

    def visit_expr(self, n, vc):
        if vc[0] not in self.elements:
            return self.add_element(eval(vc[0]))
        else:
            return vc[0]

    def visit_string(self, n, vc):
        return self.add_element(eval(n.text))

    def visit_arguments(self, n, vc):
        arglist = tuple(x.strip(",") for x in vc)
        return self.add_element(tuple(
            self.elements[arg] if arg in self.elements
            else eval(arg) for arg in arglist))

    def visit_parens(self, n, vc):
        return vc[2]

    def visit__(self, n, vc):
        # handles whitespace characters
        return ""

    def visit_nan(self, n, vc):
        return self.add_element(np.nan)

    def generic_visit(self, n, vc):
        return "".join(filter(None, vc)) or n.text

    def add_element(self, element):
        return vu.add_element(self.elements, element)
