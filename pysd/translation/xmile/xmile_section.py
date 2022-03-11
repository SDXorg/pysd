from typing import List, Union
from lxml import etree
from pathlib import Path

from ..structures.abstract_model import\
    AbstractSubscriptRange, AbstractSection

from .xmile_element import ControlElement, SubscriptRange, Flaux, Gf, Stock


class FileSection():  # File section dataclass

    control_vars = ["initial_time", "final_time", "time_step", "saveper"]

    def __init__(self, name: str, path: Path, section_type: str,
                 params: List[str], returns: List[str],
                 content_root: etree._Element, namespace: str, split: bool,
                 views_dict: Union[dict, None]
                 ):
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

    def __str__(self):  # pragma: no cover
        return "\nFile section: %s\n" % self.name

    @property
    def _verbose(self) -> str:  # pragma: no cover
        """Get model information"""
        text = self.__str__()
        if self.elements:
            for element in self.elements:
                text += element._verbose
        else:
            text += self.content

        return text

    @property
    def verbose(self):  # pragma: no cover
        """Print model information"""
        print(self._verbose)

    def _parse(self) -> None:
        """Parse the section"""
        # parse subscripts and components
        self.subscripts = self._parse_subscripts()
        self.components = self._parse_components()

        if self.name == "__main__":
            # parse control variables
            self.components += self._parse_control_vars()

        # define elements for printting information
        self.elements = self.subscripts + self.components

    def _parse_subscripts(self) -> List[SubscriptRange]:
        """Parse the subscripts of the section"""
        subscripts = [
            SubscriptRange(
                node.attrib["name"],
                [
                    sub.attrib["name"]
                    for sub in node.xpath("ns:elem", namespaces=self.ns)
                ],
                [])   # no subscript mapping implemented
            for node
            in self.content.xpath("ns:dimensions/ns:dim", namespaces=self.ns)
        ]
        self.subscripts_dict = {
            subr.name: subr.definition for subr in subscripts}
        return subscripts

    def _parse_control_vars(self) -> List[ControlElement]:
        """Parse control vars and rename them with Vensim standard"""

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

        [component._parse() for component in control_vars]
        return control_vars

    def _parse_components(self) -> List[Union[Flaux, Gf, Stock]]:
        """
        Parse model components. Three groups defined:
        Flaux: flows and auxiliary variables
        Gf: lookups
        Stock: integs
        """

        # Add flows and auxiliary variables
        components = [
            Flaux(node, self.ns, self.subscripts_dict)
            for node in self.content.xpath(
                "ns:model/ns:variables/ns:aux|ns:model/ns:variables/ns:flow",
                namespaces=self.ns)
            if node.attrib["name"].lower().replace(" ", "_")
            not in self.control_vars]

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

        [component._parse() for component in components]
        return components

    def get_abstract_section(self) -> AbstractSection:
        """
        Get Abstract Section used for building

        Returns
        -------
        AbstractSection
        """
        return AbstractSection(
            name=self.name,
            path=self.path,
            type=self.type,
            params=self.params,
            returns=self.returns,
            subscripts=self.solve_subscripts(),
            elements=[
                element.get_abstract_element()
                for element in self.components
            ],
            split=self.split,
            views_dict=self.views_dict
        )

    def solve_subscripts(self) -> List[AbstractSubscriptRange]:
        """Convert the subscript ranges to Abstract Subscript Ranges"""
        return [AbstractSubscriptRange(
            name=subs_range.name,
            subscripts=subs_range.definition,
            mapping=subs_range.mapping
        ) for subs_range in self.subscripts]
