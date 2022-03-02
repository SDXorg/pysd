import re
from pathlib import Path
import warnings
import parsimonious
from collections.abc import Mapping

from ..structures.abstract_model import AbstractModel

from . import vensim_utils as vu
from .vensim_section import FileSection


class VensimFile():
    """
    Create a VensimFile object which allows parsing a mdl file.

    Parameters
    ----------
    mdl_path: str or pathlib.Path
        Path to the Vensim model.

    encoding: str or None (optional)
        Encoding of the source model file. If None, the encoding will be
        read from the model, if the encoding is not defined in the model
        file it will be set to 'UTF-8'. Default is None.

    """
    def __init__(self, mdl_path, encoding=None):
        self.mdl_path = Path(mdl_path)
        self.root_path = self.mdl_path.parent
        self.model_text = self._read(encoding)
        self.sketch = ""
        self.view_elements = None
        self._split_sketch()

    def __str__(self):
        return "\nVensim model file, loaded from:\n\t%s\n" % self.mdl_path

    @property
    def _verbose(self):
        text = self.__str__()
        for section in self.sections:
            text += section._verbose

        return text

    @property
    def verbose(self):
        print(self._verbose)

    def _read(self, encoding):
        """Read a Vensim file and assign its content to self.model_text"""
        # check for model extension
        if self.mdl_path.suffix.lower() != ".mdl":
            raise ValueError(
                "The file to translate, '%s' " % self.mdl_path
                + "is not a vensim model. It must end with mdl extension."
            )

        if encoding is None:
            encoding = vu._detect_encoding_from_file(self.mdl_path)

        with self.mdl_path.open("r", encoding=encoding,
                                errors="ignore") as in_file:
            model_text = in_file.read()

        return model_text

    def _split_sketch(self):
        """Split model from the sketch"""
        try:
            split_model = self.model_text.split("\\\\\\---///", 1)
            self.model_text = self._clean(split_model[0])
            # remove plots section, if it exists
            self.sketch = split_model[1].split("///---\\\\\\")[0]
        except LookupError:
            pass

    def _clean(self, text):
        return re.sub(r"[\n\t\s]+", " ", re.sub(r"\\\n\t", " ", text))

    def parse(self):
        tree = vu.Grammar.get("file_sections").parse(self.model_text)
        self.sections = FileSectionsParser(tree).entries
        self.sections[0].path = self.mdl_path.with_suffix(".py")
        for section in self.sections[1:]:
            section.path = self.mdl_path.parent.joinpath(
                self.clean_file_names(section.name)[0]
                ).with_suffix(".py")
        # TODO modify names and paths of macros
        for section in self.sections:
            section._parse()

    def parse_sketch(self, subview_sep):
        if self.sketch:
            sketch = list(map(
                lambda x: x.strip(),
                self.sketch.split("\\\\\\---/// ")
            ))
        else:
            warnings.warn(
                "No sketch detected. The model will be built in a "
                "single file.")
            return None

        grammar = vu.Grammar.get("sketch")
        view_elements = {}
        for module in sketch:
            for sketch_line in module.split("\n"):
                # parsed line could have information about new view name
                # or of a variable inside a view
                parsed = SketchParser(grammar.parse(sketch_line))

                if parsed.view_name:
                    view_name = parsed.view_name
                    view_elements[view_name] = set()

                elif parsed.variable_name:
                    view_elements[view_name].add(parsed.variable_name)

        # removes views that do not include any variable in them
        non_empty_views = {
            key: value for key, value in view_elements.items() if value
        }

        # split into subviews, if subview_sep is provided
        views_dict = {}

        if len(non_empty_views) == 1:
            warnings.warn(
                "Only a single view with no subviews was detected. The model"
                " will be built in a single file.")
            return
        elif subview_sep and any(
          sep in view for sep in subview_sep for view in non_empty_views):
            escaped_separators = list(map(lambda x: re.escape(x), subview_sep))
            for full_name, values in non_empty_views.items():
                # split the full view name using the separator and make the
                # individual parts safe file or directory names
                clean_view_parts = self.clean_file_names(
                    *re.split("|".join(escaped_separators), full_name))
                # creating a nested dict for each view.subview
                # (e.g. {view_name: {subview_name: [values]}})
                nested_dict = values

                for item in reversed(clean_view_parts):
                    nested_dict = {item: nested_dict}
                # merging the new nested_dict into the views_dict, preserving
                # repeated keys
                self.merge_nested_dicts(views_dict, nested_dict)
        else:
            # view names do not have separators or separator characters
            # not provided

            if subview_sep and not any(
              sep in view for sep in subview_sep for view in non_empty_views):
                warnings.warn(
                    "The given subview separators were not matched in "
                    "any view name.")

            for view_name, elements in non_empty_views.items():
                views_dict[self.clean_file_names(view_name)[0]] = elements

        self.sections[0].split = True
        self.sections[0].views_dict = views_dict

    def get_abstract_model(self):
        return AbstractModel(
            original_path=self.mdl_path,
            sections=tuple(section.get_abstract_section()
                           for section in self.sections))

    @staticmethod
    def clean_file_names(*args):
        """
        Removes special characters and makes clean file names.

        Parameters
        ----------
        *args: tuple
            Any number of strings to to clean.

        Returns
        -------
        clean: list
            List containing the clean strings.

        """
        return [
            re.sub(
                r"[\W]+", "",
                name.replace(" ", "_")
            ).lstrip("0123456789")
            for name in args]

    def merge_nested_dicts(self, original_dict, dict_to_merge):
        """
        Merge dictionaries recursively, preserving common keys.

        Parameters
        ----------
        original_dict: dict
            Dictionary onto which the merge is executed.

        dict_to_merge: dict
            Dictionary to be merged to the original_dict.

        Returns
        -------
        None

        """
        for key, value in dict_to_merge.items():
            if (key in original_dict and isinstance(original_dict[key], dict)
                    and isinstance(value, Mapping)):
                self.merge_nested_dicts(original_dict[key], value)
            else:
                original_dict[key] = value


class FileSectionsParser(parsimonious.NodeVisitor):
    """Parse file sections"""
    def __init__(self, ast):
        self.entries = [None]
        self.visit(ast)

    def visit_main(self, n, vc):
        # main will be always stored as the first entry
        if self.entries[0] is None:
            self.entries[0] = FileSection(
                name="__main__",
                path=Path("."),
                type="main",
                params=[],
                returns=[],
                content=n.text.strip(),
                split=False,
                views_dict=None
            )
        else:
            # this is needed when macro parts are in the middle of the file
            self.entries[0].content += n.text.strip()

    def visit_macro(self, n, vc):
        self.entries.append(
            FileSection(
                name=vc[2].strip().lower().replace(" ", "_"),
                path=Path("."),
                type="macro",
                params=[x.strip() for x in vc[6].split(",")] if vc[6] else [],
                returns=[x.strip() for x in vc[10].split(",")] if vc[10] else [],
                content=vc[13].strip(),
                split=False,
                views_dict=None
            )
        )

    def generic_visit(self, n, vc):
        return "".join(filter(None, vc)) or n.text or ""


class SketchParser(parsimonious.NodeVisitor):
    def __init__(self, ast):
        self.variable_name = None
        self.view_name = None
        self.visit(ast)

    def visit_view_name(self, n, vc):
        self.view_name = n.text.lower()

    def visit_var_definition(self, n, vc):
        if int(vc[10]) % 2 != 0:  # not a shadow variable
            self.variable_name = vc[4].replace(" ", "_").lower()

    def generic_visit(self, n, vc):
        return "".join(filter(None, vc)) or n.text.strip() or ""
