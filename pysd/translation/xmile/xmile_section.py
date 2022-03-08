from typing import List, Union
from pathlib import Path

from ..structures.abstract_model import\
    AbstractElement, AbstractSubscriptRange,  AbstractSection

from .xmile_element import SubscriptRange, Flaux, Gf, Stock


class FileSection():  # File section dataclass

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

    def _parse_components(self):
        components = []

        flaux_xpath = "ns:model/ns:variables/ns:aux|"\
                      "ns:model/ns:variables/ns:flow"
        for node in self.conten.xpath(flaux_xpath, namespace=self.ns):
            # flows and auxiliary variables
            components.append(Flaux(node, self.ns))

        gf_xpath = "ns:model/ns:variables/ns:gf"
        for node in self.conten.xpath(gf_xpath, namespace=self.ns):
            # Lookups
            components.append(Gf(node, self.ns))

        stock_xpath = "ns:model/ns:variables/ns:stock"
        for node in self.conten.xpath(stock_xpath, namespace=self.ns):
            # Integs (stocks)
            components.append(Stock(node, self.ns))

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
            elements=self.merge_components(),
            split=self.split,
            views_dict=self.views_dict
        )

    def solve_subscripts(self):
        return [AbstractSubscriptRange(
            name=subs_range.name,
            subscripts=subs_range.definition,
            mapping=subs_range.mapping
        ) for subs_range in self.subscripts]

    def merge_components(self):
        merged = {}
        for component in self.components:
            name = component.name.lower().replace(" ", "_")
            if name not in merged:
                merged[name] = AbstractElement(
                    name=component.name,
                    components=[])

            if component.units:
                merged[name].units = component.units
            if component.limits[0] is not None\
              or component.limits[1] is not None:
                merged[name].range = component.limits
            if component.documentation:
                merged[name].documentation = component.documentation

            merged[name].components.append(component.get_abstract_component())

        return list(merged.values())
