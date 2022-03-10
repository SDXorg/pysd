import re
import parsimonious
import numpy as np

from ..structures.abstract_model import AbstractElement, AbstractLookup,\
                                        AbstractComponent

from . import xmile_utils as vu
from .xmile_structures import structures, parsing_ops


class Element():

    interp_methods = {
        "continuous": "interpolate",
        "extrapolate": "extrapolate",
        "discrete": "hold_backward"
    }

    def __init__(self, node, ns):
        self.node = node
        self.ns = ns
        self.name = node.attrib["name"]
        self.units = self.get_xpath_text(node, "ns:units") or ""
        self.documentation = self.get_xpath_text(node, "ns:doc") or ""
        self.limits = (None, None)
        self.components = []

    def __str__(self):
        text = "\n%s definition: %s" % (self.kind, self.name)
        text += "\nSubscrips: %s" % repr(self.subscripts)\
            if self.subscripts else ""
        text += "\n\t%s" % self._expression
        return text

    @property
    def _expression(self):
        if hasattr(self, "ast"):
            return str(self.ast).replace("\n", "\n\t")

        else:
            return self.node.text.replace("\n", "\n\t")

    @property
    def _verbose(self):
        return self.__str__()

    @property
    def verbose(self):
        print(self._verbose)

    def get_xpath_text(self, node, xpath):
        """ Safe access of occassionally missing text"""
        try:
            return node.xpath(xpath, namespaces=self.ns)[0].text
        except IndexError:
            return None

    def get_xpath_attrib(self, node, xpath, attrib):
        """ Safe access of occassionally missing attributes"""
        # defined here to take advantage of NS in default
        try:
            return node.xpath(xpath, namespaces=self.ns)[0].attrib[attrib]
        except IndexError:
            return None

    def get_lims(self):
        lims = (
            self.get_xpath_attrib(self.node, 'ns:range', 'min'),
            self.get_xpath_attrib(self.node, 'ns:range', 'max')
        )
        return tuple(float(x) if x is not None else x for x in lims)

    def parse_lookup_xml_node(self, node):
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
            x_range=(np.min(xs), np.max(xs)),
            y_range=(np.min(ys), np.max(ys)),
            type=self.interp_methods[interp]
        )

    def _parse(self):
        if self.node.xpath("ns:element", namespaces=self.ns):
            for subnode in self.node.xpath("ns:element", namespaces=self.ns):
                self.components.append(
                    ((subnode.attrib["subscript"].split(","), []),
                     self._parse_component(subnode))
                )
        else:
            subscripts = []
            for subnode in self.node.xpath("ns:dimensions/ns:dim", namespaces=self.ns):
                subscripts.append(subnode.attrib["name"])
            self.components = [
                ((subscripts, []),
                 self._parse_component(self.node))
            ]

    def smile_parser(self, expression):
        tree = vu.Grammar.get("equations", parsing_ops).parse(expression)
        return EquationParser(tree).translation

    def get_abstract_element(self):
        return AbstractElement(
            name=self.name,
            units=self.units,
            range=self.limits,
            documentation=self.documentation,
            components=[])


class Flaux(Element):
    """Flow or auxiliary variable"""
    def __init__(self, node, ns):
        super().__init__(node, ns)
        self.limits = self.get_lims()

    @property
    def _verbose(self):
        return self.__str__()

    @property
    def verbose(self):
        print(self._verbose)

    def _parse_component(self, node):
        eqn = self.get_xpath_text(node, 'ns:eqn')

        # Replace new lines with space, and replace 2 or more spaces with
        # single space. Then ensure there is no space at start or end of
        # equation
        eqn = re.sub(r"(\s{2,})", " ", eqn.replace("\n", ' ')).strip()
        ast = self.smile_parser(eqn)

        gf_node = self.node.xpath("ns:gf", namespaces=self.ns)
        if len(gf_node) > 0:
            ast = structures["inline_lookup"](
                ast, self.parse_lookup_xml_node(gf_node[0]))

        return ast

    def get_abstract_component(self):
        ae = self.get_abstract_element()
        for component in self.components:
            ae.components.append(AbstractComponent(
                subscripts=component[0],
                ast=component[1]))
        return ae


class Gf(Element):
    """Gf variable (lookup)"""
    kind = "Gf component"

    def __init__(self, node, ns):
        super().__init__(node, ns)
        self.limits = self.get_lims()

    def get_lims(self):
        lims = (
            self.get_xpath_attrib(self.node, 'ns:yscale', 'min'),
            self.get_xpath_attrib(self.node, 'ns:yscale', 'max')
        )
        return tuple(float(x) if x is not None else x for x in lims)

    def _parse_component(self, node):
        return self.parse_lookup_xml_node(self.node)

    def get_abstract_component(self):
        ae = self.get_abstract_element()
        for component in self.components:
            ae.components.append(AbstractLookup(
                subscripts=component[0],
                ast=component[1]))
        return ae


class Stock(Element):
    """Stock component (Integ)"""
    kind = "Stock component"

    def __init__(self, node, ns):
        super().__init__(node, ns)
        self.limits = self.get_lims()

    def _parse_component(self, node):
        # Parse each flow equations
        inflows = [
            self.smile_parser(inflow.text)
            for inflow in self.node.xpath('ns:inflow', namespaces=self.ns)]
        outflows = [
            self.smile_parser(outflow.text)
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
        initial = self.smile_parser(self.get_xpath_text(self.node, 'ns:eqn'))

        return structures["stock"](flows, initial)

    def get_abstract_component(self):
        ae = self.get_abstract_element()
        for component in self.components:
            ae.components.append(AbstractComponent(
                subscripts=component[0],
                ast=component[1]))
        return ae


class ControlElement(Element):
    """Control variable (lookup)"""
    kind = "Control bvariable"

    def __init__(self, name, units, documentation, eqn):
        self.name = name
        self.units = units
        self.documentation = documentation
        self.limits = (None, None)
        self.eqn = eqn

    def _parse(self):
        self.ast = self.smile_parser(self.eqn)

    def get_abstract_component(self):
        ae = self.get_abstract_element()
        ae.components.append(AbstractComponent(
            subscripts=([], []),
            ast=self.ast))
        return ae


class SubscriptRange():
    """Subscript range definition."""

    def __init__(self, name, definition, mapping=[]):
        self.name = name
        self.definition = definition
        self.mapping = mapping

    def __str__(self):
        return "\nSubscript range definition:  %s\n\t%s\n" % (
            self.name,
            self.definition)

    @property
    def _verbose(self):
        return self.__str__()

    @property
    def verbose(self):
        print(self._verbose)


class EquationParser(parsimonious.NodeVisitor):
    def __init__(self, ast):
        self.translation = None
        self.elements = {}
        self.subs = None  # the subscripts if given
        self.negatives = set()
        self.visit(ast)

    def visit_expr_type(self, n, vc):
        self.translation = self.elements[vc[0]]

    def visit_logic2_expr(self, n, vc):
        return vu.split_arithmetic(
            structures["logic"], parsing_ops["logic_ops"],
            "".join(vc).strip(), self.elements)

    def visit_logic_expr(self, n, vc):
        id = vc[2]
        if vc[0].lower() == "not":
            id = self.add_element(structures["logic"](
                [":NOT:"],
                (self.elements[id],)
                ))
        return id

    def visit_comp_expr(self, n, vc):
        return vu.split_arithmetic(
            structures["logic"], parsing_ops["comp_ops"],
            "".join(vc).strip(), self.elements)

    def visit_add_expr(self, n, vc):
        return vu.split_arithmetic(
            structures["arithmetic"], parsing_ops["add_ops"],
            "".join(vc).strip(), self.elements)

    def visit_prod_expr(self, n, vc):
        return vu.split_arithmetic(
            structures["arithmetic"], parsing_ops["prod_ops"],
            "".join(vc).strip(), self.elements)

    def visit_exp_expr(self, n, vc):
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
        """Handles whitespace characters"""
        return ""

    def visit_nan(self, n, vc):
        return "np.nan"

    def visit_empty(self, n, vc):
        #warnings.warn(f"Empty expression for '{element['real_name']}''.")
        return self.add_element(None)

    def generic_visit(self, n, vc):
        return "".join(filter(None, vc)) or n.text

    def add_element(self, element):
        return vu.add_element(self.elements, element)
