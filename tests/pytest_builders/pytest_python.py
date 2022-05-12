import pytest
from pathlib import Path

from pysd.builders.python.subscripts import SubscriptManager
from pysd.builders.python.python_model_builder import\
    ComponentBuilder, ElementBuilder, SectionBuilder
from pysd.builders.python.python_expressions_builder import\
    StructureBuilder, BuildAST
from pysd.translators.structures.abstract_model import\
    AbstractComponent, AbstractElement, AbstractSection, AbstractSubscriptRange


class TestStructureBuilder:
    """
    Test for StructureBuilder
    """

    @pytest.fixture
    def section(self):
        return SectionBuilder(
            AbstractSection(
                "main", Path("here"), "__main__",
                [], [], tuple(), tuple(), False, None
            ))

    @pytest.fixture
    def abstract_component(self):
        return AbstractComponent([[], []], "")

    @pytest.fixture
    def element(self, section, abstract_component):
        return ElementBuilder(
            AbstractElement("element", [abstract_component], "", None, ""),
            section
        )

    @pytest.fixture
    def component(self, element, section, abstract_component):
        component_obj = ComponentBuilder(abstract_component, element, section)
        component_obj.subscripts_dict = {}
        return component_obj

    @pytest.fixture
    def structure_builder(self, component):
        return StructureBuilder(None, component)

    @pytest.mark.parametrize(
        "arguments,expected",
        [
            (  # 0
                {},
                {}
            ),
            (  # 1
                {"0": BuildAST("", {"a": 1}, {}, 0)},
                {"a": 1}
            ),
            (  # 2
                {"0": BuildAST("", {"a": 1}, {}, 0),
                 "1": BuildAST("", {"a": 1, "b": 3}, {}, 0)},
                {"a": 2, "b": 3}
            ),
            (  # 3
                {"0": BuildAST("", {"a": 1}, {}, 0),
                 "1": BuildAST("", {"a": 1, "b": 3, "c": 2}, {}, 0),
                 "2": BuildAST("", {"b": 5}, {}, 0)},
                {"a": 2, "b": 8, "c": 2}
            ),
        ],
        ids=["0", "1", "2", "3"]
    )
    def test_join_calls(self, structure_builder, arguments, expected):
        assert structure_builder.join_calls(arguments) == expected

    @pytest.mark.parametrize(
        "arguments,expected",
        [
            (  # 0
                {},
                {}
            ),
            (  # 1
                {"0": BuildAST("", {}, {"a": [1]}, 0)},
                {"a": [1]}
            ),
            (  # 2a
                {"0": BuildAST("", {}, {"a": [1]}, 0),
                 "1": BuildAST("", {}, {}, 0)},
                {"a": [1]}
            ),
            (  # 2b
                {"0": BuildAST("", {}, {"a": [1]}, 0),
                 "1": BuildAST("", {}, {"b": [2, 3]}, 0)},
                {"a": [1], "b": [2, 3]}
            ),
            (  # 2c
                {"0": BuildAST("", {}, {"a": [1]}, 0),
                 "1": BuildAST("", {}, {"b": [2, 3], "a": [1]}, 0)},
                {"a": [1], "b": [2, 3]}
            ),
            (  # 3
                {"0": BuildAST("", {}, {"a": [1]}, 0),
                 "1": BuildAST("", {}, {"b": [2, 3], "a": [1]}, 0),
                 "2": BuildAST("", {}, {"b": [2, 3], "c": [10]}, 0)},
                {"a": [1], "b": [2, 3], "c": [10]}
            ),
        ],
        ids=["0", "1", "2a", "2b", "2c", "3"]
    )
    def test__get_final_subscripts(self, structure_builder,
                                   arguments, expected):
        assert structure_builder.get_final_subscripts(arguments) == expected


class TestSubscriptManager:
    @pytest.mark.parametrize(
        "arguments,raise_type,error_message",
        [
            (  # invalid definition
                [[AbstractSubscriptRange("my subs", 5, [])], Path("here")],
                ValueError,
                "Invalid definition of subscript 'my subs':\n\t5"
            ),
        ],
        ids=["invalid definition"]
    )
    def test_invalid_subscripts(self, arguments, raise_type, error_message):
        with pytest.raises(raise_type, match=error_message):
            SubscriptManager(*arguments)
