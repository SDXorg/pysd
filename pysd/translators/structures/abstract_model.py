"""
The main Abstract dataclasses provide the structure for the information
from the Component level to the Model level. This classes are hierarchical
An AbstractComponent will be inside an AbstractElement, which is inside an
AbstractSection, which is a part of an AbstractModel.

"""
from dataclasses import dataclass
from typing import Tuple, List, Union
from pathlib import Path


@dataclass
class AbstractComponent:
    """
    Dataclass for a regular component.

    Parameters
    ----------
    subscripts: tuple
        Tuple of length two with first argument the list of subscripts
        in the variable definition and the second argument the list of
        subscripts list that must be ignored (EXCEPT).
    ast: object
        The AbstractSyntaxTree of the component expression
    type: str (optional)
        The type of component. 'Auxiliary' by default.
    subtype: str (optional)
        The subtype of component. 'Normal' by default.

    """
    subscripts: Tuple[List[str], List[List[str]]]
    ast: object
    type: str = "Auxiliary"
    subtype: str = "Normal"

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractComponent %s\n" % (
            "%s" % repr(list(self.subscripts)) if self.subscripts else "")

    def dump(self, depth=None, indent="") -> str:  # pragma: no cover
        """
        Dump the component to a printable version.

        Parameters
        ----------
        depth: int (optional)
            The number of depht levels to show in the dumped output.
            Default is None which will dump everything.

        indent: str (optional)
            The indent to use for a lower level object. Default is ''.

        """
        if depth == 0:
            return self.__str__()

        return self.__str__() + "\n" + self._str_child(depth, indent)

    def _str_child(self, depth, indent) -> str:  # pragma: no cover
        return str(self.ast).replace("\t", indent).replace("\n", "\n" + indent)


@dataclass
class AbstractUnchangeableConstant(AbstractComponent):
    """
    Dataclass for an unchangeable constant component. This class is a child
    of AbstractComponent.

    Parameters
    ----------
    subscripts: tuple
        Tuple of length two with first argument the list of subscripts
        in the variable definition and the second argument the list of
        subscripts list that must be ignored (EXCEPT).
    ast: object
        The AbstractSyntaxTree of the component expression
    type: str (optional)
        The type of component. 'Constant' by default.
    subtype: str (optional)
        The subtype of component. 'Unchangeable' by default.

    """
    subscripts: Tuple[List[str], List[List[str]]]
    ast: object
    type: str = "Constant"
    subtype: str = "Unchangeable"

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractLookup %s\n" % (
            "%s" % repr(list(self.subscripts)) if self.subscripts else "")


@dataclass
class AbstractLookup(AbstractComponent):
    """
    Dataclass for a lookup component. This class is a child of
    AbstractComponent.

    Parameters
    ----------
    subscripts: tuple
        Tuple of length two with first argument the list of subscripts
        in the variable definition and the second argument the list of
        subscripts list that must be ignored (EXCEPT).
    ast: object
        The AbstractSyntaxTree of the component expression
    arguments: str (optional)
        The name of the argument to use. 'x' by default.
    type: str (optional)
        The type of component. 'Lookup' by default.
    subtype: str (optional)
        The subtype of component. 'Hardcoded' by default.

    """
    subscripts: Tuple[List[str], List[List[str]]]
    ast: object
    arguments: str = "x"
    type: str = "Lookup"
    subtype: str = "Hardcoded"

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractLookup %s\n" % (
            "%s" % repr(list(self.subscripts)) if self.subscripts else "")


@dataclass
class AbstractData(AbstractComponent):
    """
    Dataclass for a data component. This class is a child
    of AbstractComponent.

    Parameters
    ----------
    subscripts: tuple
        Tuple of length two with first argument the list of subscripts
        in the variable definition and the second argument the list of
        subscripts list that must be ignored (EXCEPT).
    ast: object
        The AbstractSyntaxTree of the component expression
    keyword: str or None (optional)
        The data object keyword ('interpolate', 'hold_backward',
        'look_forward', 'raw'). Default is None.
    type: str (optional)
        The type of component. 'Data' by default.
    subtype: str (optional)
        The subtype of component. 'Normal' by default.

    """
    subscripts: Tuple[List[str], List[List[str]]]
    ast: object
    keyword: Union[str, None] = None
    type: str = "Data"
    subtype: str = "Normal"

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractData (%s)  %s\n" % (
            self.keyword,
            "%s" % repr(list(self.subscripts)) if self.subscripts else "")

    def dump(self, depth=None, indent="") -> str:  # pragma: no cover
        """
        Dump the component to a printable version.

        Parameters
        ----------
        depth: int (optional)
            The number of depht levels to show in the dumped output.
            Default is None which will dump everything.

        indent: str (optional)
            The indent to use for a lower level object. Default is ''.

        """
        if depth == 0:
            return self.__str__()

        return self.__str__() + "\n" + self._str_child(depth, indent)

    def _str_child(self, depth, indent) -> str:  # pragma: no cover
        return str(self.ast).replace("\n", "\n" + indent)


@dataclass
class AbstractElement:
    """
    Dataclass for an element.

    Parameters
    ----------
    name: str
        The name of the element.
    components: list
        The list of AbstractComponents that define this element.
    units: str (optional)
        The units of the element. '' by default.
    limits: tuple (optional)
        The limits of the element. (None, None) by default.
    units: str (optional)
        The documentation of the element. '' by default.

    """
    name: str
    components: List[AbstractComponent]
    units: str = ""
    limits: tuple = (None, None)
    documentation: str = ""

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractElement:\t%s (%s, %s)\n%s\n" % (
            self.name, self.units, self.limits, self.documentation)

    def dump(self, depth=None, indent="") -> str:  # pragma: no cover
        """
        Dump the element to a printable version.

        Parameters
        ----------
        depth: int (optional)
            The number of depht levels to show in the dumped output.
            Default is None which will dump everything.

        indent: str (optional)
            The indent to use for a lower level object. Default is ''.

        """
        if depth == 0:
            return self.__str__()
        elif depth is not None:
            depth -= 1

        return self.__str__() + "\n" + self._str_child(depth, indent)

    def _str_child(self, depth, indent) -> str:  # pragma: no cover
        return "\n".join([
            component.dump(depth, indent) for component in self.components
            ]).replace("\n", "\n" + indent)


@dataclass
class AbstractControlElement(AbstractElement):
    """
    Dataclass for a control element. This class is a child of
    AbstractElement and has the same attributes.

    Parameters
    ----------
    name: str
        The name of the element.
    components: list
        The list of AbstractComponents that define this element.
    units: str (optional)
        The units of the element. '' by default.
    limits: tuple (optional)
        The limits of the element. (None, None) by default.
    units: str (optional)
        The documentation of the element. '' by default.

    """
    name: str
    components: List[AbstractComponent]
    units: str = ""
    limits: tuple = (None, None)
    documentation: str = ""

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractControlElement:\t%s (%s, %s)\n%s\n" % (
            self.name, self.units, self.limits, self.documentation)


@dataclass
class AbstractSubscriptRange:
    """
    Dataclass for a subscript range.

    Parameters
    ----------
    name: str
        The name of the element.
    subscripts: list or str or dict
        The subscripts as a list of strings for a regular definition,
        str for a copy definition and as a dict for a GET XLS/DIRECT
        definition.
    mapping: list
        The list of subscript range that can be mapped to.

    """
    name: str
    subscripts: Union[list, str, dict]
    mapping: list

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractSubscriptRange:\t%s\n\t%s\n" % (
            self.name,
            "%s <- %s" % (self.subscripts, self.mapping)
            if self.mapping else self.subscripts)

    def dump(self, depth=None, indent="") -> str:  # pragma: no cover
        """
        Dump the subscript range to a printable version.

        Parameters
        ----------
        depth: int (optional)
            The number of depht levels to show in the dumped output.
            Default is None which will dump everything.

        indent: str (optional)
            The indent to use for a lower level object. Default is ''.

        """
        return self.__str__()


@dataclass
class AbstractSection:
    """
    Dataclass for an element.

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
    subscripts: tuple
        Tuple of AbstractSubscriptRanges that are defined in the section.
    elements: tuple
        Tuple of AbstractElements that are defined in the section.
    split: bool
        If split is True the created section will split the variables
        depending on the views_dict.
    views_dict: dict
        The dictionary of the views. Giving the variables classified at
        any level in order to split them by files.

    """
    name: str
    path: Path
    type: str  # main, macro or module
    params: List[str]
    returns: List[str]
    subscripts: Tuple[AbstractSubscriptRange]
    elements: Tuple[AbstractElement]
    split: bool
    views_dict: Union[dict, None]

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractSection (%s):\t%s (%s)\n" % (
            self.type, self.name, self.path)

    def dump(self, depth=None, indent="") -> str:  # pragma: no cover
        """
        Dump the section to a printable version.

        Parameters
        ----------
        depth: int (optional)
            The number of depht levels to show in the dumped output.
            Default is None which will dump everything.

        indent: str (optional)
            The indent to use for a lower level object. Default is ''.

        """
        if depth == 0:
            return self.__str__()
        elif depth is not None:
            depth -= 1

        return self.__str__() + "\n" + self._str_child(depth, indent)

    def _str_child(self, depth, indent) -> str:  # pragma: no cover
        return "\n".join([
            element.dump(depth, indent) for element in self.subscripts
            ] + [
            element.dump(depth, indent) for element in self.elements
            ]).replace("\n", "\n" + indent)


@dataclass
class AbstractModel:
    """
    Dataclass for an element.

    Parameters
    ----------
    original_path: pathlib.Path
        The path to the original file.
    sections: tuple
        Tuple of AbstractSectionss that are defined in the model.

    """
    original_path: Path
    sections: Tuple[AbstractSection]

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractModel:\t%s\n" % self.original_path

    def dump(self, depth=None, indent="") -> str:  # pragma: no cover
        """
        Dump the model to a printable version.

        Parameters
        ----------
        depth: int (optional)
            The number of depht levels to show in the dumped output.
            Default is None which will dump everything.

        indent: str (optional)
            The indent to use for a lower level object. Default is ''.

        """
        if depth == 0:
            return self.__str__()
        elif depth is not None:
            depth -= 1

        return self.__str__() + "\n" + self._str_child(depth, indent)

    def _str_child(self, depth, indent) -> str:  # pragma: no cover
        return "\n".join([
            section.dump(depth, indent) for section in self.sections
            ]).replace("\n", "\n" + indent)
