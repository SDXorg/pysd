import pytest
from pathlib import Path

from pysd.builders.python.namespace import NamespaceManager
from pysd.builders.python.subscripts import SubscriptManager
from pysd.builders.python.python_model_builder import\
    ComponentBuilder, ElementBuilder, SectionBuilder
from pysd.builders.python.python_expressions_builder import\
    ReferenceBuilder, StructureBuilder, BuildAST
from pysd.translators.structures.abstract_expressions import\
    ReferenceStructure, SubscriptsReferenceStructure
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
                [], [], tuple(), tuple(), [], [], False, None
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

    @pytest.mark.parametrize(
        "reference_str,origin_name,namespace,subscripts",
        [
            (
                ReferenceStructure(
                    "abc_jki",
                    SubscriptsReferenceStructure(("dim1", "dim1", "dim3"))
                ),
                "Abc Jki",
                {"hgfhd": "hgfhd", "Abc Jki": "abc_jki"},
                {"dim1": ["A", "B", "C"],
                 "dim2": ["A", "B", "C"], "dim3": ["A", "B", "C"]}
            ),
            (
                ReferenceStructure(
                    "abc_jki",
                    SubscriptsReferenceStructure(("dim1", "dim1"))
                ),
                "abc_jki",
                {"hgfhd": "hgfhd"},
                {"dim1": ["A", "B", "C"], "dim2": ["A", "B", "C"]}
            ),
        ]
    )
    def test_referencebuilder_subscripts_warning(self, component,
                                                 reference_str,
                                                 origin_name,
                                                 namespace, subscripts):
        component.element.name = "My Var"
        component.section.subscripts._subscripts = subscripts
        component.section.subscripts.mapping = {
            dim: [] for dim in subscripts}
        component.section.namespace.namespace = namespace
        warning_message =\
            f"The reference to '{origin_name}' in variable 'My Var' has "\
            r"duplicated subscript ranges\. If mapping is used in one "\
            r"of them, please, rewrite reference subscripts to avoid "\
            r"duplicates\. Otherwise, the final model may crash\.\.\."\

        with pytest.warns(UserWarning, match=warning_message):
            ReferenceBuilder(reference_str, component)

    @pytest.mark.parametrize(
        "reference_str,subscripts",
        [
            (
                ReferenceStructure(
                    "abc_jki",
                    SubscriptsReferenceStructure(["dim1", "A", "dim3"])
                ),
                {"dim1": ["A", "B", "C"],
                 "dim2": ["A", "B", "C"], "dim3": ["A", "B", "C"]}
            ),
            (
                ReferenceStructure(
                    "abc_jki",
                    SubscriptsReferenceStructure(["A", "A"])
                ),
                {"dim1": ["A", "B", "C"], "dim2": ["A", "B", "C"]}
            ),
        ]
    )
    def test_referencebuilder_subscripts_nowarning(self, component,
                                                   reference_str,
                                                   subscripts):
        component.element.name = "My Var"
        component.section.subscripts._subscripts = subscripts
        component.section.subscripts.mapping = {
            dim: [] for dim in subscripts}

        ReferenceBuilder(reference_str, component)


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


class TestNamespaceManager:

    @pytest.fixture
    def namespace(self, elements):
        ns = NamespaceManager()
        [ns.add_to_namespace(element) for element in elements]
        return ns

    @pytest.mark.parametrize(
        "identifier,elements,raise_type,error_message",
        [
            (  # invalid definition
                'my_value',
                ["your_value"],
                ValueError,
                r"'my_value' not found in the namespace\."
            ),
            (  # invalid definition
                'my_value',
                ["your_value", "my_value2"],
                ValueError,
                r"'my_value' not found in the namespace\."
            ),
        ]
    )
    def test_get_original_name_invalid_identifier(self, identifier, namespace,
                                                  raise_type, error_message):
        with pytest.raises(raise_type, match=error_message):
            namespace.get_original_name(identifier)

    @pytest.mark.parametrize(
        "identifier,elements,expected",
        [
            (  # invalid definition
                'my_value',
                ["my_value"],
                "my_value"
            ),
            (  # invalid definition
                'my_value',
                ["your_value", "My Value"],
                "My Value"
            ),
        ]
    )
    def test_get_original_name(self, identifier, namespace, expected):
        assert namespace.get_original_name(identifier) == expected
