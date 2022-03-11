from typing import List, Union
from pathlib import Path
import parsimonious

from ..structures.abstract_model import\
    AbstractElement, AbstractSubscriptRange, AbstractSection

from . import vensim_utils as vu
from .vensim_element import Element, SubscriptRange, Component


class FileSection():  # File section dataclass

    def __init__(self, name: str, path: Path, section_type: str,
                 params: List[str], returns: List[str],
                 content: str, split: bool, views_dict: Union[dict, None]
                 ):
        self.name = name
        self.path = path
        self.type = section_type
        self.params = params
        self.returns = returns
        self.content = content
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
        # parse the section to get the elements
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
            elements=self.merge_components(),
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

    def merge_components(self) -> List[AbstractElement]:
        """Merge model components by their name"""
        merged = {}
        for component in self.components:
            # get a safe name to merge (case and white/underscore sensitivity)
            name = component.name.lower().replace(" ", "_")
            if name not in merged:
                # create new element if it is the first component
                merged[name] = AbstractElement(
                    name=component.name,
                    components=[])

            if component.units:
                # add units to element data
                merged[name].units = component.units
            if component.range != (None, None):
                # add range to element data
                merged[name].range = component.range
            if component.documentation:
                # add documentation to element data
                merged[name].documentation = component.documentation

            # add AbstractComponent to the list of components
            merged[name].components.append(component.get_abstract_component())

        return list(merged.values())


class SectionElementsParser(parsimonious.NodeVisitor):
    """
    Visit section elements to get their equation units and documentation.
    """
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
