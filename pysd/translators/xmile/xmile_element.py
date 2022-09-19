"""
The Element class child classes alow parsing the expressions of a
given model element. There are 3 tipes of elements:

- Flows and auxiliars (Flaux class): Regular elements, defined with
  <flow> or <aux>.
- Gfs (Gf class): Lookup elements, defined with <gf>.
- Stocks (Stock class): Data component, defined with <stock>

Moreover, a 4 type element is defined ControlElement, which allows parsing
the values of the model control variables (time step, initialtime, final time).

The final result from a parsed element can be exported to an
AbstractElement object in order to build a model in other language.
"""
import re
from typing import Tuple, Union, List
from lxml import etree
import parsimonious
import numpy as np

from ..structures.abstract_model import\
    AbstractElement, AbstractControlElement,\
    AbstractLookup, AbstractComponent, AbstractSubscriptRange

from ..structures.abstract_expressions import AbstractSyntax

from . import xmile_utils as vu
from .xmile_structures import structures, parsing_ops


class Element():
    """
    Element class. This class provides the shared methods for its
    children: Flaux, Gf, Stock, and ControlElement.

    Parameters
    ----------
    node: etree._Element
        The element node content.

    ns: dict
        The namespace of the section.

    subscripts: dict
        The subscript dictionary of the section, necessary to parse
        some subscripted elements.

    """
    _interp_methods = {
        "continuous": "interpolate",
        "extrapolate": "extrapolate",
        "discrete": "hold_backward"
    }

    _kind = "Element"

    def __init__(self, node: etree._Element, ns: dict, subscripts):
        self.node = node
        self.ns = ns
        self.name = node.attrib["name"].replace("\\n", " ")
        self.units = self._get_xpath_text(node, "ns:units") or ""
        self.documentation = self._get_xpath_text(node, "ns:doc") or ""
        self.limits = (None, None)
        self.components = []
        self.subscripts = subscripts

    def __str__(self):  # pragma: no cover
        text = "\n%s definition: %s" % (self._kind, self.name)
        text += "\nSubscrips: %s" % repr(self.subscripts)\
            if self.subscripts else ""
        text += "\n\t%s" % self._expression
        return text

    @property
    def _expression(self):  # pragma: no cover
        if hasattr(self, "ast"):
            return str(self.ast).replace("\n", "\n\t")

        else:
            return self.node.text.replace("\n", "\n\t")

    @property
    def _verbose(self) -> str:  # pragma: no cover
        """Get element information."""
        return self.__str__()

    @property
    def verbose(self):  # pragma: no cover
        """Print element information to standard output."""
        print(self._verbose)

    def _get_xpath_text(self, node: etree._Element,
                        xpath: str) -> Union[str, None]:
        """Safe access of occassionally missing text"""
        try:
            return node.xpath(xpath, namespaces=self.ns)[0].text
        except IndexError:
            return None

    def _get_xpath_attrib(self, node: etree._Element,
                          xpath: str, attrib: str) -> Union[str, None]:
        """Safe access of occassionally missing attributes"""
        # defined here to take advantage of NS in default
        try:
            return node.xpath(xpath, namespaces=self.ns)[0].attrib[attrib]
        except IndexError:
            return None

    def _get_limits(self) -> Tuple[Union[None, str], Union[None, str]]:
        """Get the limits of the element"""
        lims = (
            self._get_xpath_attrib(self.node, 'ns:range', 'min'),
            self._get_xpath_attrib(self.node, 'ns:range', 'max')
        )
        return tuple(float(x) if x is not None else x for x in lims)

    def _parse_lookup_xml_node(self, node: etree._Element) -> AbstractSyntax:
        """
        Parse lookup definition

        Returns
        -------
        AST: AbstractSyntax

        """
        ys_node = node.xpath('ns:ypts', namespaces=self.ns)[0]
        ys = np.fromstring(
            ys_node.text,
            dtype=float,
            sep=ys_node.attrib['sep'] if 'sep' in ys_node.attrib else ','
        )
        xscale_node = node.xpath('ns:xscale', namespaces=self.ns)
        if len(xscale_node) > 0:
            xmin = xscale_node[0].attrib['min']
            xmax = xscale_node[0].attrib['max']
            xs = np.linspace(float(xmin), float(xmax), len(ys))
        else:
            xs_node = node.xpath('ns:xpts', namespaces=self.ns)[0]
            xs = np.fromstring(
                xs_node.text,
                dtype=float,
                sep=xs_node.attrib['sep'] if 'sep' in xs_node.attrib else ','
            )

        interp = node.attrib['type'] if 'type' in node.attrib else 'continuous'

        return structures["lookup"](
            x=tuple(xs[np.argsort(xs)]),
            y=tuple(ys[np.argsort(xs)]),
            x_limits=(np.min(xs), np.max(xs)),
            y_limits=(np.min(ys), np.max(ys)),
            type=self._interp_methods[interp]
        )

    def parse(self) -> None:
        """Parse all the components of an element"""
        if self.node.xpath("ns:element", namespaces=self.ns):
            # defined in several equations each with one subscript
            for subnode in self.node.xpath("ns:element", namespaces=self.ns):
                self.components.append(
                    ((subnode.attrib["subscript"].split(","), []),
                     self._parse_component(subnode)[0])
                )
        else:
            # get the subscripts from element
            subscripts = [
                subnode.attrib["name"]
                for subnode
                in self.node.xpath("ns:dimensions/ns:dim", namespaces=self.ns)
            ]
            parsed = self._parse_component(self.node)
            if len(parsed) == 1:
                # element defined with one equation
                self.components = [((subscripts, []),  parsed[0])]
            else:
                # element defined in several equations, but only the general
                # subscripts are given, save each equation with its
                # subscrtipts
                subs_list = self.subscripts[subscripts[0]]
                self.components = [
                    (([subs], []), parsed_i) for subs, parsed_i in
                    zip(subs_list, parsed)
                ]

    def _smile_parser(self, expression: str) -> AbstractSyntax:
        """
        Parse expression with parsimonious.

        Returns
        -------
        AST: AbstractSyntax

        """
        tree = vu.Grammar.get("equations", parsing_ops).parse(
            expression.strip())
        return EquationVisitor(tree).translation

    def _get_empty_abstract_element(self) -> AbstractElement:
        """
        Get empty Abstract used for building

        Returns
        -------
        AbstractElement
        """
        return AbstractElement(
            name=self.name,
            units=self.units,
            limits=self.limits,
            documentation=self.documentation,
            components=[])


class Flaux(Element):
    """
    Flow or auxiliary variable definde by <flow> or <aux> in Xmile.

    Parameters
    ----------
    node: etree._Element
        The element node content.

    ns: dict
        The namespace of the section.

    subscripts: dict
        The subscript dictionary of the section, necessary to parse
        some subscripted elements.

    """
    _kind = "Flaux"

    def __init__(self, node, ns, subscripts):
        super().__init__(node, ns, subscripts)
        self.limits = self._get_limits()

    def _parse_component(self, node: etree._Element) -> List[AbstractSyntax]:
        """
        Parse one Flaux component

        Returns
        -------
        AST: AbstractSyntax

        """
        asts = []
        for eqn in node.xpath('ns:eqn', namespaces=self.ns):
            # Replace new lines with space, and replace 2 or more spaces with
            # single space. Then ensure there is no space at start or end of
            # equation
            eqn = re.sub(r"(\s{2,})", " ", eqn.text.replace("\n", ' ')).strip()
            ast = self._smile_parser(eqn)

            gf_node = self.node.xpath("ns:gf", namespaces=self.ns)
            if len(gf_node) > 0:
                ast = structures["inline_lookup"](
                    ast, self._parse_lookup_xml_node(gf_node[0]))
            asts.append(ast)

        return asts

    def get_abstract_element(self) -> AbstractElement:
        """
        Get Abstract Element used for building. This method is
        automatically called by Sections's get_abstract_section.

        Returns
        -------
        AbstractElement: AbstractElement
          Abstract Element object that can be used for building
          the model in another language. It contains a list of
          AbstractComponents with the Abstract Syntax Tree of each of
          the expressions.

        """
        ae = self._get_empty_abstract_element()
        for component in self.components:
            ae.components.append(AbstractComponent(
                subscripts=component[0],
                ast=component[1]))
        return ae


class Gf(Element):
    """
    Gf variable (lookup) definde by <gf> in Xmile.

    Parameters
    ----------
    node: etree._Element
        The element node content.

    ns: dict
        The namespace of the section.

    subscripts: dict
        The subscript dictionary of the section, necessary to parse
        some subscripted elements.

    """
    _kind = "Gf component"

    def __init__(self, node, ns, subscripts):
        super().__init__(node, ns, subscripts)
        self.limits = self.get_limits()

    def get_limits(self) -> Tuple[Union[None, str], Union[None, str]]:
        """Get the limits of the Gf element"""
        lims = (
            self._get_xpath_attrib(self.node, 'ns:yscale', 'min'),
            self._get_xpath_attrib(self.node, 'ns:yscale', 'max')
        )
        return tuple(float(x) if x is not None else x for x in lims)

    def _parse_component(self, node: etree._Element) -> AbstractSyntax:
        """
        Parse one Gf component

        Returns
        -------
        AST: AbstractSyntax

        """
        return [self._parse_lookup_xml_node(self.node)]

    def get_abstract_element(self) -> AbstractElement:
        """
        Get Abstract Element used for building. This method is
        automatically called by Sections's get_abstract_section.

        Returns
        -------
        AbstractElement: AbstractElement
          Abstract Element object that can be used for building
          the model in another language. It contains a list of
          AbstractComponents with the Abstract Syntax Tree of each of
          the expressions.

        """
        ae = self._get_empty_abstract_element()
        for component in self.components:
            ae.components.append(AbstractLookup(
                subscripts=component[0],
                ast=component[1]))
        return ae


class Stock(Element):
    """
    Stock variable definde by <stock> in Xmile.

    Parameters
    ----------
    node: etree._Element
        The element node content.

    ns: dict
        The namespace of the section.

    subscripts: dict
        The subscript dictionary of the section, necessary to parse
        some subscripted elements.

    """

    _kind = "Stock component"

    def __init__(self, node, ns, subscripts):
        super().__init__(node, ns, subscripts)
        self.limits = self._get_limits()

    def _parse_component(self, node) -> AbstractSyntax:
        """
        Parse one Stock component

        Returns
        -------
        AST: AbstractSyntax

        """
        # Parse each flow equations
        inflows = [
            self._smile_parser(inflow.text)
            for inflow in self.node.xpath('ns:inflow', namespaces=self.ns)]
        outflows = [
            self._smile_parser(outflow.text)
            for outflow in self.node.xpath('ns:outflow', namespaces=self.ns)]

        if inflows:
            # stock has inflows
            expr = ["+"] * (len(inflows)-1) + ["-"] * len(outflows)
        elif outflows:
            # stock has no inflows but outflows
            outflows[0] = structures["negative"](outflows[0])
            expr = ["-"] * (len(outflows)-1)
        else:
            # stock is constant
            expr = []
            inflows = [0]

        if expr:
            # stock has more than one flow
            flows = structures["arithmetic"](expr, inflows+outflows)
        else:
            # stock has only one flow
            flows = inflows[0] if inflows else outflows[0]

        # Read the initial value equation for stock element
        initial = self._smile_parser(self._get_xpath_text(self.node, 'ns:eqn'))

        return [structures["stock"](flows, initial)]

    def get_abstract_element(self) -> AbstractElement:
        """
        Get Abstract Element used for building. This method is
        automatically called by Sections's get_abstract_section.

        Returns
        -------
        AbstractElement: AbstractElement
          Abstract Element object that can be used for building
          the model in another language. It contains a list of
          AbstractComponents with the Abstract Syntax Tree of each of
          the expressions.

        """
        ae = self._get_empty_abstract_element()
        for component in self.components:
            ae.components.append(AbstractComponent(
                subscripts=component[0],
                ast=component[1]))
        return ae


class ControlElement(Element):
    """Control variable (lookup)"""
    _kind = "Control variable"

    def __init__(self, name, units, documentation, eqn):
        self.name = name
        self.units = units
        self.documentation = documentation
        self.limits = (None, None)
        self.eqn = eqn

    def parse(self) -> None:
        """
        Parse control elment.

        Returns
        -------
        AST: AbstractSyntax

        """
        self.ast = self._smile_parser(self.eqn)

    def get_abstract_element(self) -> AbstractElement:
        """
        Get Abstract Element used for building. This method is
        automatically called by Sections's get_abstract_section.

        Returns
        -------
        AbstractElement: AbstractElement
          Abstract Element object that can be used for building
          the model in another language. It contains an AbstractComponent
          with the Abstract Syntax Tree of the expression.

        """
        return AbstractControlElement(
            name=self.name,
            units=self.units,
            limits=self.limits,
            documentation=self.documentation,
            components=[
                AbstractComponent(subscripts=([], []), ast=self.ast)
            ]
        )


class SubscriptRange():
    """Subscript range definition."""

    def __init__(self, name: str, definition: List[str],
                 mapping: List[str] = []):
        self.name = name
        self.definition = definition
        self.mapping = mapping

    def __str__(self):  # pragma: no cover
        return "\nSubscript range definition:  %s\n\t%s\n" % (
            self.name,
            self.definition)

    @property
    def _verbose(self) -> str:  # pragma: no cover
        """Get subscript range information."""
        return self.__str__()

    @property
    def verbose(self):  # pragma: no cover
        """Print subscript range information to standard output."""
        print(self._verbose)

    def get_abstract_subscript_range(self) -> AbstractSubscriptRange:
        """
        Get Abstract Subscript Range used for building. This method is
        automatically called by Sections's get_abstract_section.

        Returns
        -------
        AbstractSubscriptRange: AbstractSubscriptRange
          Abstract Subscript Range object that can be used for building
          the model in another language.

        """
        return AbstractSubscriptRange(
            name=self.name,
            subscripts=self.definition,
            mapping=self.mapping
        )


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

    def visit_logic2_expr(self, n, vc):
        # expressions with logical binary operators (and, or)
        return vu.split_arithmetic(
            structures["logic"], parsing_ops["logic_ops"],
            "".join(vc).strip(), self.elements)

    def visit_logic_expr(self, n, vc):
        # expressions with logical unitary operators (not)
        id = vc[2]
        if vc[0].lower() == "not":
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

    def visit_mod_expr(self, n, vc):
        # modulo expressions (mod)
        if vc[1].lower().startswith("mod"):
            return self.add_element(
                structures["call"](
                    structures["reference"]("modulo"),
                    (self.elements[vc[0]], self.elements[vc[1][3:]])
                ))
        else:
            return vc[0]

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
            func_str = structures[func.reference]
            if isinstance(func_str, dict):
                return self.add_element(func_str[len(args)](*args))
            else:
                return self.add_element(func_str(*args))
        else:
            return self.add_element(structures["call"](func, args))

    def visit_conditional_statement(self, n, vc):
        return self.add_element(structures["if_then_else"](
            self.elements[vc[2]],
            self.elements[vc[6]],
            self.elements[vc[10]]))

    def visit_reference(self, n, vc):
        id = self.add_element(structures["reference"](
            vc[0].lower().replace(" ", "_").strip("\""), self.subs))
        self.subs = None
        return id

    def visit_array(self, n, vc):
        if ";" in n.text or "," in n.text:
            return self.add_element(np.squeeze(np.array(
                [row.split(",") for row in n.text.strip(";").split(";")],
                dtype=float)))
        else:
            return self.add_element(eval(n.text))

    def visit_subscript_list(self, n, vc):
        subs = [x.strip().replace("_", " ") for x in vc[2].split(",")]
        self.subs = structures["subscripts_ref"](subs)
        return ""

    def visit_name(self, n, vc):
        return n.text.strip()

    def visit_expr(self, n, vc):
        if vc[0] not in self.elements:
            return self.add_element(eval(vc[0]))
        else:
            return vc[0]

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

    def generic_visit(self, n, vc):
        return "".join(filter(None, vc)) or n.text

    def add_element(self, element):
        return vu.add_element(self.elements, element)
