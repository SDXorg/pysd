from typing import List, Union
from pathlib import Path
import parsimonious

from ..structures.abstract_model import\
    AbstractElement, AbstractSubscriptRange,  AbstractSection

from . import vensim_utils as vu
from .vensim_element import Element, SubscriptRange, Component


class FileSection():  # File section dataclass

    def __init__(self, name: str, path: Path, type: str,
                 params: List[str], returns: List[str],
                 content: str, split: bool, views_dict: Union[dict, None]
                 ) -> object:
        self.name = name
        self.path = path
        self.type = type
        self.params = params
        self.returns = returns
        self.content = content
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
        tree = vu.Grammar.get("section_elements").parse(self.content)
        self.elements = SectionElementsParser(tree).entries
        self.elements = [element._parse() for element in self.elements]
        # split subscript from other components
        self.subscripts = [
            element for element in self.elements
            if isinstance(element, SubscriptRange)
        ]
        self.components = [
            element for element in self.elements
            if isinstance(element, Component)
        ]
        # reorder element list for better printing
        self.elements = self.subscripts + self.components

        [component._parse() for component in self.components]

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


class SectionElementsParser(parsimonious.NodeVisitor):
    # TODO include units parsing
    def __init__(self, ast):
        self.entries = []
        self.visit(ast)

    def visit_entry(self, n, vc):
        self.entries.append(
            Element(
                equation=vc[0].strip(),
                units=vc[2].strip(),
                documentation=vc[4].strip(),
            )
        )

    def generic_visit(self, n, vc):
        return "".join(filter(None, vc)) or n.text or ""
