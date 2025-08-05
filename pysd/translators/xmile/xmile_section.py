"""
The Section class allows parsing a model section into Elements. The
final result can be exported to an AbstractSection class in order to
build a model in other language. A section could be either the main model
(without the macros), or a macro definition (not supported yet for Xmile).
"""
from typing import List, Union
from lxml import etree
from pathlib import Path

from ..structures.abstract_model import AbstractSection

from .xmile_element import ControlElement, SubscriptRange, Aux, Flow, Gf, Stock


class Section():
    """
    Section object allows parsing the elements of that section.

    Parameters
    ----------
    name: str
        Section name. '__main__' for the main section or the macro name.

    path: pathlib.Path
        Section path. It should be the model name for main  section and
        the clean macro name for a macro.

    section_type: str ('main' or 'macro')
        The section type.

    params: list
        List of params that takes the section. In the case of main
        section it will be an empty list.

    returns: list
        List of variables that returns the section. In the case of main
        section it will be an empty list.

    content_root: etree._Element
        Section parsed tree content.

    namespace: str
        The namespace of the section given after parsing its content
        with etree.

    split: bool
        If split is True the created section will split the variables
        depending on the views_dict.

    views_dict: dict
        The dictionary of the views. Giving the variables classified at
        any level in order to split them by files.

    """
    _control_vars = ["initial_time", "final_time", "time_step", "saveper"]

    def __init__(self, name: str, path: Path, section_type: str,
                 params: List[str], returns: List[str],
                 content_root: etree._Element, namespace: str, split: bool,
                 views_dict: Union[dict, None]):
        self.name = name
        self.path = path
        self.type = section_type
        self.params = params
        self.returns = returns
        self.content = content_root
        self.ns = {"ns": namespace}
        self.split = split
        self.views_dict = views_dict
        self.elements = None
        self.behaviors = {}

    def __str__(self):  # pragma: no cover
        return "\nSection: %s\n" % self.name

    @property
    def _verbose(self) -> str:  # pragma: no cover
        """Get section information."""
        text = self.__str__()
        if self.elements:
            for element in self.elements:
                text += element._verbose
        else:
            text += self.content

        return text

    @property
    def verbose(self):  # pragma: no cover
        """Print section information to standard output."""
        print(self._verbose)

    def parse(self, parse_all: bool = True) -> None:
        """
        Parse section object. The subscripts of the section will be added
        to self subscripts. The variables defined as Flows, Auxiliary, Gf,
        and Stock will be converted in XmileElements. The control variables,
        if the section is __main__, will be converted to a ControlElement.

        Parameters
        ----------
        parse_all: bool (optional)
            If True then the created VensimElement objects will be
            automatically parsed. Otherwise, this objects will only be
            added to self.elements but not parser. Default is True.

        """
        # parse subscripts and components
        self.subscripts = self._parse_subscripts()
        self.components = self._parse_components()

        if self.name == "__main__":
            # parse control variables
            self.components += self._parse_control_vars()

        # Parse behavior section
        self.behaviors.update(self._parse_behavior())

        if parse_all:
            [component.parse(self.behaviors) for component in self.components]

        # define elements for printting information
        self.elements = self.subscripts + self.components

    def _parse_behavior(self) -> List[SubscriptRange]:
        """Parse the behavior the section."""
        behaviors = {
            'non_negative_stock': False,
            'non_negative_flow': False
        }
        parsed_bhs = self.content.xpath("ns:behavior", namespaces=self.ns)
        if not parsed_bhs:
            return behaviors

        if parsed_bhs[0].xpath('ns:non_negative', namespaces=self.ns):
            behaviors['non_negative_stock'] = True
            behaviors['non_negative_flow'] = True
            return behaviors

        bhs_stock = parsed_bhs[0].xpath('ns:stock', namespaces=self.ns)
        bhs_flow = parsed_bhs[0].xpath('ns:flow', namespaces=self.ns)

        if bhs_stock and bhs_stock[0].xpath('ns:non_negative', namespaces=self.ns):
            behaviors['non_negative_stock'] = True

        if bhs_flow and bhs_flow[0].xpath('ns:non_negative', namespaces=self.ns):
            behaviors['non_negative_flow'] = True

        return behaviors

    def _parse_subscripts(self) -> List[SubscriptRange]:
        """Parse the subscripts of the section."""

        def get_subs(dim_node: etree._Element) -> List[str]:
            """get the subscripts of a given dimension node"""
            if 'size' in dim_node.attrib:
                # subscripts defined with size, no name given
                return [
                    str(i)
                    for i in range(1, int(dim_node.attrib['size'])+1)
                ]
            else:
                # subscripts defined with names
                return [
                    sub.attrib["name"]
                    for sub in dim_node.xpath("ns:elem", namespaces=self.ns)
                ]

        subscripts = [
            SubscriptRange(node.attrib["name"], get_subs(node), [])
            # no subscript mapping implemented
            for node
            in self.content.xpath("ns:dimensions/ns:dim", namespaces=self.ns)
        ]
        self.subscripts_dict = {
            subr.name: subr.definition for subr in subscripts}
        return subscripts

    def _parse_components(self) -> List[Union[Flow, Aux, Gf, Stock]]:
        """
        Parse model components. Four groups defined:
        Aux: auxiliary variables
        Flow: flows
        Gf: lookups
        Stock: integs

        """
        # Add auxiliary variables
        components = [
            Aux(node, self.ns, self.subscripts_dict)
            for node in self.content.xpath(
                "ns:model/ns:variables/ns:aux",
                namespaces=self.ns)
            if node.attrib["name"].lower().replace(" ", "_")
            not in self._control_vars]

        # Add flows
        components += [
            Flow(node, self.ns, self.subscripts_dict)
            for node in self.content.xpath(
                "ns:model/ns:variables/ns:flow",
                namespaces=self.ns)
            if node.attrib["name"].lower().replace(" ", "_")
            not in self._control_vars]

        # Add lookups
        components += [
            Gf(node, self.ns, self.subscripts_dict)
            for node in self.content.xpath(
                "ns:model/ns:variables/ns:gf",
                namespaces=self.ns)
            ]

        # Add stocks
        components += [
            Stock(node, self.ns, self.subscripts_dict)
            for node in self.content.xpath(
                "ns:model/ns:variables/ns:stock",
                namespaces=self.ns)
            ]

        return components

    def _parse_control_vars(self) -> List[ControlElement]:
        """Parse control vars and rename them with Vensim standard."""

        # Read the start time of simulation
        node = self.content.xpath('ns:sim_specs', namespaces=self.ns)[0]
        time_units = node.attrib['time_units']\
            if 'time_units' in node.attrib else ""

        control_vars = []

        # initial time of the simulation
        control_vars.append(ControlElement(
            name="INITIAL TIME",
            units=time_units,
            documentation="The initial time for the simulation.",
            eqn=node.xpath("ns:start", namespaces=self.ns)[0].text
        ))

        # final time of the simulation
        control_vars.append(ControlElement(
            name="FINAL TIME",
            units=time_units,
            documentation="The final time for the simulation.",
            eqn=node.xpath("ns:stop", namespaces=self.ns)[0].text
        ))

        # time step of simulation
        dt_node = node.xpath("ns:dt", namespaces=self.ns)[0]
        dt_eqn = "1/(" + dt_node.text + ")" if "reciprocal" in dt_node.attrib\
            and dt_node.attrib["reciprocal"].lower() == "true"\
            else dt_node.text

        control_vars.append(ControlElement(
            name="TIME STEP",
            units=time_units,
            documentation="The time step for the simulation.",
            eqn=dt_eqn
        ))

        # saving time of the simulation = time step
        control_vars.append(ControlElement(
            name="SAVEPER",
            units=time_units,
            documentation="The save time step for the simulation.",
            eqn="time_step"
        ))

        return control_vars

    def get_abstract_section(self) -> AbstractSection:
        """
        Get Abstract Section used for building. This, method should be
        called after parsing the section (self.parse). This method is
        automatically called by Model's get_abstract_model and
        automatically generates the AbstractSubscript ranges and merge
        the components in elements calling also the get_abstract_components
        method from each model component.

        Returns
        -------
        AbstractSection: AbstractSection
          Abstract Section object that can be used for building the model
          in another language.

        """
        return AbstractSection(
            name=self.name,
            path=self.path,
            type=self.type,
            params=self.params,
            returns=self.returns,
            subscripts=[
                subs_range.get_abstract_subscript_range()
                for subs_range in self.subscripts
            ],
            elements=[
                element.get_abstract_element()
                for element in self.components
            ],
            constraints=[],
            test_inputs=[],
            split=self.split,
            views_dict=self.views_dict
        )
