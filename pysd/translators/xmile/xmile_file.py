"""
The XmileFile class allows reading the original Xmile model file,
parsing it into Section elements. The final result can be exported to an
AbstractModel class in order to build a model in other language.
"""
from typing import Union
from pathlib import Path
from lxml import etree

from ..structures.abstract_model import AbstractModel

from .xmile_section import Section
from .xmile_utils import supported_extensions


class XmileFile():
    """
    Create a XmileFile object which allows parsing a xmile file.
    When the object is created the model file is automatically opened
    and parsed with lxml.etree.

    Parameters
    ----------
    xmile_path: str or pathlib.Path
        Path to the Xmile model.

    """
    def __init__(self, xmile_path: Union[str, Path]):
        self.xmile_path = Path(xmile_path)
        self.root_path = self.xmile_path.parent
        self.xmile_root = self._get_root()
        self.ns = self.xmile_root.nsmap[None]  # namespace of the xmile
        self.view_elements = None

    def __str__(self):  # pragma: no cover
        return "\nXmile model file, loaded from:\n\t%s\n" % self.xmile_path

    @property
    def _verbose(self) -> str:  # pragma: no cover
        """Get model information."""
        text = self.__str__()
        for section in self.sections:
            text += section._verbose

        return text

    @property
    def verbose(self):  # pragma: no cover
        """Print model information to standard output."""
        print(self._verbose)

    def _get_root(self) -> etree._Element:
        """
        Read a Xmile file and assign its content to self.model_text

        Returns
        -------
        lxml.etree._Element: parsed xml object

        """
        # check for model extension
        if self.xmile_path.suffix.lower() not in supported_extensions:
            raise ValueError(
                "The file to translate, '%s' " % self.xmile_path
                + "is not a Xmile model. It must end with any of "
                + "%s extensions." % ', '.join(supported_extensions)
            )

        return etree.parse(
            str(self.xmile_path),
            parser=etree.XMLParser(encoding="utf-8", recover=True)
        ).getroot()

    def parse(self, parse_all: bool = True) -> None:
        """
        Create a XmileSection object from the model content and parse it.
        As currently the macros are not supported all the models will
        have only one section. This functionshould split the macros in
        independent sections in the future.

        Parameters
        ----------
        parse_all: bool (optional)
            If True then the created XmileSection objects will be
            automatically parsed. Otherwise, this objects will only be
            added to self.sections but not parser. Default is True.

        """
        # TODO: in order to make macros work we need to split them here
        # in several sections
        # We keep everything in a single section
        self.sections = [Section(
                name="__main__",
                path=self.xmile_path.with_suffix(".py"),
                section_type="main",
                params=[],
                returns=[],
                content_root=self.xmile_root,
                namespace=self.ns,
                split=False,
                views_dict=None)]

        if parse_all:
            for section in self.sections:
                section.parse()

    def get_abstract_model(self) -> AbstractModel:
        """
        Get Abstract Model used for building. This, method should be
        called after parsing the model (self.parse). This automatically
        calls the get_abstract_section method from the model sections.

        Returns
        -------
        AbstractModel: AbstractModel
          Abstract Model object that can be used for building the model
          in another language.

        """
        return AbstractModel(
            original_path=self.xmile_path,
            sections=tuple(section.get_abstract_section()
                           for section in self.sections))
