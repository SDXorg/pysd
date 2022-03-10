from typing import List, Union
from pathlib import Path

from ..structures.abstract_model import\
    AbstractElement, AbstractSubscriptRange,  AbstractSection

from .xmile_element import ControlElement, SubscriptRange, Flaux, Gf, Stock


class FileSection():  # File section dataclass

    control_vars = ["initial_time", "final_time", "time_step", "saveper"]

    def __init__(self, name: str, path: Path, type: str,
                 params: List[str], returns: List[str],
                 content_root: str, namespace: str, split: bool,
                 views_dict: Union[dict, None]
                 ) -> object:
        self.name = name
        self.path = path
        self.type = type
        self.params = params
        self.returns = returns
        self.content = content_root
        self.ns = {"ns": namespace}
        self.split = split
        self.views_dict = views_dict
        self.elements = None

    def __str__(self):
        return "\nFile section: %s\n" % self.name

    @property
    def _verbose(self):
        text = self.__str__()
        if self.elements:
            for element in self.elements:
                text += element._verbose
        else:
            text += self.content

        return text

    @property
    def verbose(self):
        print(self._verbose)

    def _parse(self):
        self.subscripts = self._parse_subscripts()
        self.components = self._parse_components()
        if self.name == "__main__":
            self.components += self._parse_control_vars()
        self.elements = self.subscripts + self.components

    def _parse_subscripts(self):
        """Parse the subscripts of the section"""
        subscripts_ranges = []
        path = "ns:dimensions/ns:dim"
        for node in self.content.xpath(path, namespaces=self.ns):
            name = node.attrib["name"]
            subscripts = [
                sub.attrib["name"]
                for sub in node.xpath("ns:elem", namespaces=self.ns)
            ]
            subscripts_ranges.append(SubscriptRange(name, subscripts, []))
        return subscripts_ranges

    def _parse_control_vars(self):

        # Read the start time of simulation
        node = self.content.xpath('ns:sim_specs', namespaces=self.ns)[0]
        time_units = node.attrib['time_units'] if 'time_units' in node.attrib else ""

        control_vars = []

        control_vars.append(ControlElement(
            name="INITIAL TIME",
            units=time_units,
            documentation="The initial time for the simulation.",
            eqn=node.xpath("ns:start", namespaces=self.ns)[0].text
        ))

        control_vars.append(ControlElement(
            name="FINAL TIME",
            units=time_units,
            documentation="The final time for the simulation.",
            eqn=node.xpath("ns:stop", namespaces=self.ns)[0].text
        ))

        # Read the time step of simulation
        dt_node = node.xpath("ns:dt", namespaces=self.ns)

        # Use default value for time step if `dt` is not specified in model
        dt_eqn = "1"
        if len(dt_node) > 0:
            dt_node = dt_node[0]
            dt_eqn = dt_node.text
            # If reciprocal mode are defined for `dt`, we should inverse value
            if "reciprocal" in dt_node.attrib\
              and dt_node.attrib["reciprocal"].lower() == "true":
                dt_eqn = "1/(" + dt_eqn + ")"

        control_vars.append(ControlElement(
            name="TIME STEP",
            units=time_units,
            documentation="The time step for the simulation.",
            eqn=dt_eqn
        ))

        control_vars.append(ControlElement(
            name="SAVEPER",
            units=time_units,
            documentation="The save time step for the simulation.",
            eqn="time_step"
        ))

        [component._parse() for component in control_vars]
        return control_vars

    def _parse_components(self):

        # Add flows and auxiliary variables
        components = [
            Flaux(node, self.ns)
            for node in self.content.xpath(
                "ns:model/ns:variables/ns:aux|ns:model/ns:variables/ns:flow",
                namespaces=self.ns)
            if node.attrib["name"].lower().replace(" ", "_")
            not in self.control_vars]

        # Add lookups
        components += [
            Gf(node, self.ns)
            for node in self.content.xpath(
                "ns:model/ns:variables/ns:gf",
                namespaces=self.ns)
            ]

        # Add stocks
        components += [
            Stock(node, self.ns)
            for node in self.content.xpath(
                "ns:model/ns:variables/ns:stock",
                namespaces=self.ns)
            ]

        [component._parse() for component in components]
        return components

    def get_abstract_section(self):
        return AbstractSection(
            name=self.name,
            path=self.path,
            type=self.type,
            params=self.params,
            returns=self.returns,
            subscripts=self.solve_subscripts(),
            elements=[
                component.get_abstract_component()
                for component in self.components
            ],
            split=self.split,
            views_dict=self.views_dict
        )

    def solve_subscripts(self):
        return [AbstractSubscriptRange(
            name=subs_range.name,
            subscripts=subs_range.definition,
            mapping=subs_range.mapping
        ) for subs_range in self.subscripts]
