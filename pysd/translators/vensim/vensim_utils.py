import re
import warnings
import uuid

import parsimonious
from typing import Dict
from pathlib import Path
from chardet import detect


supported_extensions = [".mdl"]


class Grammar():
    _common_grammar = None
    _grammar_path: Path = Path(__file__).parent.joinpath("parsing_grammars")
    _grammar: Dict = {}

    @classmethod
    def get(cls, grammar: str, subs: dict = {}) -> parsimonious.Grammar:
        """Get parsimonious grammar for parsing"""
        if grammar not in cls._grammar:
            # include grammar in the class singleton
            cls._grammar[grammar] = parsimonious.Grammar(
                cls._read_grammar(grammar) % subs
            )

        return cls._grammar[grammar]

    @classmethod
    def _read_grammar(cls, grammar: str) -> str:
        """Read grammar from a file and include common grammar"""
        with cls._gpath(grammar).open(encoding="ascii") as gfile:
            source_grammar: str = gfile.read()

        return cls._include_common_grammar(source_grammar)

    @classmethod
    def _include_common_grammar(cls, source_grammar: str) -> str:
        """Include common grammar"""
        if not cls._common_grammar:
            with cls._gpath("common_grammar").open(encoding="ascii") as gfile:
                cls._common_grammar: str = gfile.read()

        return r"{source_grammar}{common_grammar}".format(
            source_grammar=source_grammar, common_grammar=cls._common_grammar
        )

    @classmethod
    def _gpath(cls, grammar: str) -> Path:
        """Get the grammar file path"""
        return cls._grammar_path.joinpath(grammar).with_suffix(".peg")


def split_arithmetic(structure: object, parsing_ops: dict,
                     expression: str, elements: dict,
                     negatives: set = set()) -> object:
    """
    Split arithmetic pattern and return the corresponding object.

    Parameters
    ----------
    structure: callable
       Callable that generates the arithmetic object to return.
    parsing_ops: dict
       The parsing operators dictionary.
    expression: str
       Original expression with the operator and the hex code to the objects.
    elements: dict
       Dictionary of the hex identifiers and the objects that represent.
    negative: set
       Set of element hex values that must change their sign.

    Returns
    -------
    object: structure
        Final object of the arithmetic operation or initial object if
        no operations are performed.

    """
    pattern = re.compile(parsing_ops)
    parts = pattern.split(expression)  # list of elements ids
    ops = pattern.findall(expression)  # operators list
    if not ops:
        # no operators return original object
        if parts[0] in negatives:
            # make original object negative
            negatives.remove(parts[0])
            return add_element(
                elements,
                structure(["negative"], (elements[parts[0]],)))
        else:
            return expression
    else:
        if not negatives:
            # create arithmetic object
            return add_element(
                elements,
                structure(
                    ops,
                    tuple([elements[id] for id in parts])))
        else:
            # manage negative expressions
            current_id = parts.pop()
            current = elements[current_id]
            if current_id in negatives:
                negatives.remove(current_id)
                current = structure(["negative"], (current,))
            while ops:
                current_id = parts.pop()
                current = structure(
                    [ops.pop()],
                    (elements[current_id], current))
                if current_id in negatives:
                    negatives.remove(current_id)
                    current = structure(["negative"], (current,))

            return add_element(elements, current)


def add_element(elements: dict, element: object) -> str:
    """
    Add element to elements dict using an unique hex identifier

    Parameters
    ----------
    elements: dict
      Dictionary of all elements.

    element: object
      Element to add.

    Returns
    -------
    id: str (hex)
      The name of the key where element is saved in elements.

    """
    id = uuid.uuid4().hex
    elements[id] = element
    return id


def _detect_encoding_from_file(mdl_file: Path) -> str:
    """Detect and return the encoding from a Vensim file"""
    try:
        with mdl_file.open("rb") as in_file:
            f_line: bytes = in_file.readline()
        f_line: str = f_line.decode(detect(f_line)['encoding'])
        return re.search(r"(?<={)(.*)(?=})", f_line).group()
    except (AttributeError, UnicodeDecodeError):
        warnings.warn(
            "No encoding specified or detected to translate the model "
            "file. 'UTF-8' encoding will be used.")
        return "UTF-8"
