from dataclasses import dataclass
from typing import Tuple, List, Union
from pathlib import Path


@dataclass
class AbstractComponent:
    subscripts: Tuple[str]
    ast: object
    type: str = "Auxiliary"
    subtype: str = "Normal"

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractComponent %s\n" % (
            "%s" % repr(list(self.subscripts)) if self.subscripts else "")

    def dump(self, depth=None, indent="") -> str:  # pragma: no cover
        if depth == 0:
            return self.__str__()

        return self.__str__() + "\n" + self._str_child(depth, indent)

    def _str_child(self, depth, indent) -> str:  # pragma: no cover
        return str(self.ast).replace("\t", indent).replace("\n", "\n" + indent)


@dataclass
class AbstractUnchangeableConstant(AbstractComponent):
    subscripts: Tuple[str]
    ast: object
    type: str = "Constant"
    subtype: str = "Unchangeable"

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractLookup %s\n" % (
            "%s" % repr(list(self.subscripts)) if self.subscripts else "")


@dataclass
class AbstractLookup(AbstractComponent):
    subscripts: Tuple[str]
    ast: object
    arguments: str = "x"
    type: str = "Lookup"
    subtype: str = "Hardcoded"

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractLookup %s\n" % (
            "%s" % repr(list(self.subscripts)) if self.subscripts else "")


@dataclass
class AbstractData(AbstractComponent):
    subscripts: Tuple[str]
    ast: object
    keyword: Union[str, None] = None
    type: str = "Data"
    subtype: str = "Normal"

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractData (%s)  %s\n" % (
            self.keyword,
            "%s" % repr(list(self.subscripts)) if self.subscripts else "")

    def dump(self, depth=None, indent="") -> str:  # pragma: no cover
        if depth == 0:
            return self.__str__()

        return self.__str__() + "\n" + self._str_child(depth, indent)

    def _str_child(self, depth, indent) -> str:  # pragma: no cover
        return str(self.ast).replace("\n", "\n" + indent)


@dataclass
class AbstractElement:
    name: str
    components: List[AbstractComponent]
    units: str = ""
    range: tuple = (None, None)
    documentation: str = ""

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractElement:\t%s (%s, %s)\n%s\n" % (
            self.name, self.units, self.range, self.documentation)

    def dump(self, depth=None, indent="") -> str:  # pragma: no cover
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
class AbstractSubscriptRange:
    name: str
    subscripts: Tuple[str]
    mapping: Tuple[str]

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractSubscriptRange:\t%s\n\t%s\n" % (
            self.name,
            "%s <- %s" % (self.subscripts, self.mapping)
            if self.mapping else self.subscripts)

    def dump(self, depth=None, indent="") -> str:  # pragma: no cover
        return self.__str__()


@dataclass
class AbstractSection:
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
    original_path: Path
    sections: Tuple[AbstractSection]

    def __str__(self) -> str:  # pragma: no cover
        return "AbstractModel:\t%s\n" % self.original_path

    def dump(self, depth=None, indent="") -> str:  # pragma: no cover
        if depth == 0:
            return self.__str__()
        elif depth is not None:
            depth -= 1

        return self.__str__() + "\n" + self._str_child(depth, indent)

    def _str_child(self, depth, indent) -> str:  # pragma: no cover
        return "\n".join([
            section.dump(depth, indent) for section in self.sections
            ]).replace("\n", "\n" + indent)
