import re
import warnings
import uuid

import parsimonious
from typing import Dict
from pathlib import Path
from chardet import detect


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

        return source_grammar

    @classmethod
    def _gpath(cls, grammar: str) -> Path:
        """Get the grammar file path"""
        return cls._grammar_path.joinpath(grammar).with_suffix(".peg")

    @classmethod
    def clean(cls) -> None:
        """Clean the saved grammars (used for debugging)"""
        cls._common_grammar = None
        cls._grammar: Dict = {}


def split_arithmetic(structure: object, parsing_ops: dict,
                     expression: str, elements: dict,
                     negatives: set = set()) -> object:
    pattern = re.compile(parsing_ops)
    parts = pattern.split(expression)
    ops = pattern.findall(expression)
    ops = list(map(
        lambda x: x.replace('and', ':AND:').replace('or', ':OR:'), ops))
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
    id = uuid.uuid4().hex
    elements[id] = element
    return id
