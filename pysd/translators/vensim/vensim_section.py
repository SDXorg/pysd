"""
The Section class allows parsing a model section into Elements using the
SectionElementsVisitor. The final result can be exported to an
AbstractSection class in order to build a model in another programming
language.
A section is either the main model (without the macros), or a
macro definition.
"""
from typing import List, Union
from pathlib import Path
import parsimonious

from ..structures.abstract_model import\
    AbstractElement, AbstractControlElement, AbstractSection

from . import vensim_utils as vu
from .vensim_element import Element, SubscriptRange, Component,\
                            Constraint, TestInput


class Section():
    """
    The Section class allows parsing the elements of a model section.

    Parameters
    ----------
    name: str
        Section name. That is, '__main__' for the main section, and the
        macro name for macros.

    path: pathlib.Path
        Section path. The model name for the main section and the clean
        macro name for a macro.

    section_type: str ('main' or 'macro')
        The section type.

    params: list
        List of parameters that the section takes. If it is the main
        section, it will be an empty list.

    returns: list
        List of variables that returns the section. If it is the main
        section, it will be an empty list.

    content: str
        Section content as string.

    split: bool
        If True, the created section will split the variables
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
        self.components = None
        self.subscripts = None

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
        Parse section object with parsimonious using the grammar given in
        'parsing_grammars/section_elements.peg' and the class
        SectionElementsVisitor to visit the parsed expressions.

        This will break the section (__main__ or macro) in VensimElements,
        which are each model expression LHS and RHS with already parsed
        units and description.

        Parameters
        ----------
        parse_all: bool (optional)
            If True, the created VensimElement objects will be
            automatically parsed. Otherwise, these objects will only be
            added to self.elements but not parsed. Default is True.

        """
        # parse the section to get the elements
        tree = vu.Grammar.get("section_elements").parse(self.content)
        self.elements = SectionElementsVisitor(tree).entries

        if parse_all:
            # parse all elements
            self.elements = [element.parse() for element in self.elements]

            # split subscripts and reality checks from other components
            self.subscripts = [
                element for element in self.elements
                if isinstance(element, SubscriptRange)
            ]
            self.components = [
                element for element in self.elements
                if isinstance(element, Component)
            ]
            self.constraints = [
                element for element in self.elements
                if isinstance(element, Constraint)
            ]
            self.test_inputs = [
                element for element in self.elements
                if isinstance(element, TestInput)
            ]

            # reorder element list for better printing
            self.elements = self.subscripts + self.components\
                + self.constraints + self.test_inputs

            [component.parse() for component in self.components]
            [component.parse() for component in self.constraints]
            [component.parse() for component in self.test_inputs]

    def get_abstract_section(self) -> AbstractSection:
        """
        Instantiate an object of the AbstractSection class that will be used
        during the building process. This method should be called after
        parsing the section (self.parse). This method is automatically called
        by Model's get_abstract_model method, and automatically generates the
        AbstractSubscript ranges and merges the components in elements. It also
        calls the get_abstract_components method from each model component.

        Returns
        -------
        AbstractSection: AbstractSection
          AbstractSection object that can be used for building the model
          in another programming language.

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
            constraints=[
                constraint.get_abstract_component()
                for constraint in self.constraints],
            test_inputs=[
                test_input.get_abstract_component()
                for test_input in self.test_inputs],
            split=self.split,
            views_dict=self.views_dict
        )

    def _merge_components(self) -> List[AbstractElement]:
        """Merge model components by their name."""
        control_vars = ["initial_time", "final_time", "time_step", "saveper"]
        merged = {}
        for component in self.components:
            # get a safe name to merge (case and white/underscore sensitivity)
            name = component.name.lower().replace(" ", "_")
            if name not in merged:
                # create new element if it is the first component
                if name in control_vars:
                    merged[name] = AbstractControlElement(
                        name=component.name,
                        components=[])
                else:
                    merged[name] = AbstractElement(
                        name=component.name,
                        components=[])

            if component.units:
                # add units to element data
                merged[name].units = component.units
            if component.limits:
                # add limits to element data
                merged[name].limits = component.limits
            if component.documentation:
                # add documentation to element data
                merged[name].documentation = component.documentation

            # add AbstractComponent to the list of components
            merged[name].components.append(component.get_abstract_component())

        return list(merged.values())


class SectionElementsVisitor(parsimonious.NodeVisitor):
    """
    Visit section elements to get their equation, units and documentation.
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
