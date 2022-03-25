import re
from typing import Union, Tuple, List
import warnings
import parsimonious
import numpy as np

from ..structures.abstract_model import\
    AbstractData, AbstractLookup, AbstractComponent,\
    AbstractUnchangeableConstant
from parsimonious.exceptions import IncompleteParseError,\
                                    VisitationError,\
                                    ParseError

from . import vensim_utils as vu
from .vensim_structures import structures, parsing_ops


class Element():
    """Model element parsed definition"""

    def __init__(self, equation: str, units: str, documentation: str):
        self.equation = equation
        self.units, self.range = self._parse_units(units)
        self.documentation = documentation

    def __str__(self):  # pragma: no cover
        return "Model element:\n\t%s\nunits: %s\ndocs: %s\n" % (
            self.equation, self.units, self.documentation)

    @property
    def _verbose(self) -> str:  # pragma: no cover
        """Get model information"""
        return self.__str__()

    @property
    def verbose(self):  # pragma: no cover
        """Print model information"""
        print(self._verbose)

    def _parse_units(self, units_str: str) -> Tuple[str, tuple]:
        """Split the range from the units"""
        # TODO improve units parsing: move to _parse_section_elements
        if not units_str:
            return "", (None, None)

        if units_str.endswith("]"):
            units, lims = units_str.rsplit("[")  # types: str, str
        else:
            units = units_str
            lims = "?, ?]"

        lims = tuple(
            [
                float(x) if x.strip() != "?" else None
                for x in lims.strip("]").split(",")
            ]
        )
        return units, lims

    def _parse(self) -> object:
        """Parse model element to get the component object"""
        tree = vu.Grammar.get("element_object").parse(self.equation)
        self.component = ElementsComponentParser(tree).component
        self.component.units = self.units
        self.component.range = self.range
        self.component.documentation = self.documentation
        return self.component


class ElementsComponentParser(parsimonious.NodeVisitor):
    """Visit model element definition to get the component object"""

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

        warnings.warn(
            "\n Subscript mapping detected."
            + "This feature works only in some simple cases."
        )

        if ":" in str(vc):
            # TODO: add test for this condition
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

    def visit__(self, n, vc):
        # TODO check if necessary when finished
        return " "


class SubscriptRange():
    """Subscript range definition, defined by ":" or "<->" in Vensim."""

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
        """Get model information"""
        return self.__str__()

    @property
    def verbose(self):  # pragma: no cover
        """Print model information"""
        print(self._verbose)


class Component():
    """Model component defined by "name = expr" in Vensim."""
    kind = "Model component"

    def __init__(self, name: str, subscripts: Tuple[list, list],
                 expression: str):
        self.name = name
        self.subscripts = subscripts
        self.expression = expression

    def __str__(self):  # pragma: no cover
        text = "\n%s definition: %s" % (self.kind, self.name)
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
        return self.__str__()

    @property
    def verbose(self):  # pragma: no cover
        print(self._verbose)

    def _parse(self) -> None:
        """Parse model component to get the AST"""
        try:
            tree = vu.Grammar.get("components", parsing_ops).parse(
                self.expression)
        except (IncompleteParseError, ParseError) as err:
            raise ValueError(
                err.args[0] + "\n\n"
                "\nError when parsing definition:\n\t %s\n\n"
                "probably used definition is invalid or not integrated..."
                "\nSee parsimonious output above." % self.expression
            )
        try:
            self.ast = EquationParser(tree).translation
        except VisitationError as err:
            raise ValueError(
                err.args[0] + "\n\n"
                "\nError when visiting definition:\n\t %s\n\n"
                "probably used definition is invalid or not integrated..."
                "\nSee parsimonious output above." % self.expression
            )

        if isinstance(self.ast, structures["get_xls_lookups"]):
            self.lookup = True
        else:
            self.lookup = False

    def get_abstract_component(self) -> Union[AbstractComponent,
                                              AbstractLookup]:
        """Get Abstract Component used for building"""
        if self.lookup:
            # get lookups equations
            return AbstractLookup(subscripts=self.subscripts, ast=self.ast)
        else:
            return AbstractComponent(subscripts=self.subscripts, ast=self.ast)


class UnchangeableConstant(Component):
    """Unchangeable constant defined by "name == expr" in Vensim."""
    kind = "Unchangeable constant component"

    def __init__(self, name: str, subscripts: Tuple[list, list],
                 expression: str):
        super().__init__(name, subscripts, expression)

    def get_abstract_component(self) -> AbstractUnchangeableConstant:
        """Get Abstract Component used for building"""
        return AbstractUnchangeableConstant(
            subscripts=self.subscripts, ast=self.ast)


class Lookup(Component):
    """Lookup variable, defined by "name(expr)" in Vensim."""
    kind = "Lookup component"

    def __init__(self, name: str, subscripts: Tuple[list, list],
                 expression: str):
        super().__init__(name, subscripts, expression)

    def _parse(self) -> None:
        """Parse model component to get the AST"""
        tree = vu.Grammar.get("lookups").parse(self.expression)
        self.ast = LookupsParser(tree).translation

    def get_abstract_component(self) -> AbstractLookup:
        """Get Abstract Component used for building"""
        return AbstractLookup(subscripts=self.subscripts, ast=self.ast)


class Data(Component):
    """Data variable, defined by "name := expr" in Vensim."""
    kind = "Data component"

    def __init__(self, name: str, subscripts: Tuple[list, list],
                 keyword: str, expression: str):
        super().__init__(name, subscripts, expression)
        self.keyword = keyword

    def __str__(self):  # pragma: no cover
        text = "\n%s definition: %s" % (self.kind, self.name)
        text += "\nSubscrips: %s" % repr(self.subscripts[0])\
            if self.subscripts[0] else ""
        text += "  EXCEPT  %s" % repr(self.subscripts[1])\
            if self.subscripts[1] else ""
        text += "\nKeyword: %s" % self.keyword if self.keyword else ""
        text += "\n\t%s" % self._expression
        return text

    def _parse(self) -> None:
        """Parse model component to get the AST"""
        if not self.expression:
            # empty data vars, read from vdf file
            self.ast = structures["data"]()
        else:
            super()._parse()

    def get_abstract_component(self) -> AbstractData:
        """Get Abstract Component used for building"""
        return AbstractData(
            subscripts=self.subscripts, ast=self.ast, keyword=self.keyword)


class LookupsParser(parsimonious.NodeVisitor):
    """Visit the elements of a lookups to get the AST"""
    def __init__(self, ast):
        self.translation = None
        self.visit(ast)

    def visit_range(self, n, vc):
        return n.text.strip()[:-1].replace(")-(", "),(")

    def visit_regularLookup(self, n, vc):
        if vc[0]:
            xy_range = np.array(eval(vc[0]))
        else:
            xy_range = np.full((2, 2), np.nan)

        values = np.array((eval(vc[2])))
        values = values[np.argsort(values[:, 0])]

        self.translation = structures["lookup"](
            x=tuple(values[:, 0]),
            y=tuple(values[:, 1]),
            x_range=tuple(xy_range[:, 0]),
            y_range=tuple(xy_range[:, 1]),
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


class EquationParser(parsimonious.NodeVisitor):
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

    def visit_range(self, n, vc):
        return self.add_element(n.text.strip()[:-1].replace(")-(", "),("))

    def visit_lookup_with_def(self, n, vc):
        if vc[10]:
            xy_range = np.array(eval(self.elements[vc[10]]))
        else:
            xy_range = np.full((2, 2), np.nan)

        values = np.array((eval(vc[11])))
        values = values[np.argsort(values[:, 0])]

        lookup = structures["lookup"](
            x=tuple(values[:, 0]),
            y=tuple(values[:, 1]),
            x_range=tuple(xy_range[:, 0]),
            y_range=tuple(xy_range[:, 1]),
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

    def visit_empty(self, n, vc):
        return self.add_element(None)

    def generic_visit(self, n, vc):
        return "".join(filter(None, vc)) or n.text

    def add_element(self, element):
        return vu.add_element(self.elements, element)
