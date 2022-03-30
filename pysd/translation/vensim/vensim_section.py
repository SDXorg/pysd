"""
The Section class allows parsing a model section into Elements using the
SectionElementsVisitor. The final result can be exported to an
AbstractSection class in order to build a model in other language.
A section could be either the main model (without the macros), or a
macro definition.
"""
from typing import List, Union
from pathlib import Path
import parsimonious

from ..structures.abstract_model import AbstractElement, AbstractSection

from . import vensim_utils as vu
from .vensim_element import Element, SubscriptRange, Component


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

    content: str
        Section content as string.

    split: bool
        If split is True the created section will split the variables
        depending on the views_dict.

    views_dict: dict
        The dictionary of the views. Giving the variables classified at
        any level in order to split them by files.

    """
    def __init__(self, name: str, path: Path, section_type: str,
                 params: List[str], returns: List[str],
                 content: str, split: bool, views_dict: Union[dict, None]):
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
        """Print section information."""
        print(self._verbose)

    def parse(self, parse_all: bool = True) -> None:
        """
        Parse section object with parsimonious using the grammar given in
        'parsin_grammars/section_elements.peg' and the class
        SectionElementsVisitor to visit the parsed expressions.

        This will break the section (__main__ or macro) in VensimElements,
        which are each model expression LHS and RHS with already parsed
        units and description.

        Parameters
        ----------
        parse_all: bool (optional)
            If True then the created VensimElement objects will be
            automatically parsed. Otherwise, this objects will only be
            added to self.elements but not parser. Default is True.

        """
        # parse the section to get the elements
        tree = vu.Grammar.get("section_elements").parse(self.content)
        self.elements = SectionElementsParser(tree).entries

        if parse_all:
            # parse all elements
            self.elements = [element.parse() for element in self.elements]

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

            [component.parse() for component in self.components]

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
            elements=self._merge_components(),
            split=self.split,
            views_dict=self.views_dict
        )

    def _merge_components(self) -> List[AbstractElement]:
        """Merge model components by their name."""
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
