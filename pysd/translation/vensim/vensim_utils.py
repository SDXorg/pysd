import re
import warnings
import uuid

import parsimonious
from typing import Dict
from pathlib import Path
from chardet import detect


class Grammar():
    _common_grammar = None
    _grammar_path: Path = Path(__file__).parent.joinpath("parsing_expr")
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

    @classmethod
    def clean(cls) -> None:
        """Clean the saved grammars (used for debugging)"""
        cls._common_grammar = None
        cls._grammar: Dict = {}


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


def split_arithmetic(structure: object, parsing_ops: dict,
                     expression: str, elements: dict,
                     negatives: set = set()) -> object:
    pattern = re.compile(parsing_ops)
    parts = pattern.split(expression)
    ops = pattern.findall(expression)
    if not ops:
        if parts[0] in negatives:
            negatives.remove(parts[0])
            return add_element(
                elements,
                structure(["negative"], (elements[parts[0]],)))
        else:
            return expression
    else:
        if not negatives:
            return add_element(
                elements,
                structure(
                    ops,
                    tuple([elements[id] if id in elements
                           else eval(id) for id in parts])))
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
    id = uuid.uuid4().hex
    elements[id] = element
    return id
