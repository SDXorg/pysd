import pytest
from pathlib import Path
from parsimonious import VisitationError

from pysd.translators.vensim.vensim_file import VensimFile
from pysd.translators.vensim.vensim_element import Element


@pytest.mark.parametrize(
    "path",
    [
        (  # teacup
            "samples/teacup/teacup.mdl"
        ),
        (  # macros
            "tests/macro_multi_expression/test_macro_multi_expression.mdl"
        ),
        (  # mapping
            "tests/subscript_mapping_vensim/test_subscript_mapping_vensim.mdl"
        ),
        (  # data
            "tests/data_from_other_model/test_data_from_other_model.mdl"
        ),
        (  # except
            "tests/except/test_except.mdl"
        )
    ],
    ids=["teacup", "macros", "mapping", "data", "except"]
)
class TestVensimFile:
    """
    Test for splitting Vensim views in modules and submodules
    """
    @pytest.fixture
    def model_path(self, _root, path):
        return _root.joinpath("test-models").joinpath(path)

    def test_read_vensim_file(self, model_path):
        # assert that the files don't exist in the temporary directory
        ven_file = VensimFile(model_path)

        assert hasattr(ven_file, "mdl_path")
        assert hasattr(ven_file, "root_path")
        assert hasattr(ven_file, "model_text")

        assert isinstance(getattr(ven_file, "mdl_path"), Path)
        assert isinstance(getattr(ven_file, "root_path"), Path)
        assert isinstance(getattr(ven_file, "model_text"), str)

    def test_file_split_file_sections(self, model_path):
        ven_file = VensimFile(model_path)
        ven_file.parse()


class TestElements:
    """
    Test for splitting Vensim views in modules and submodules
    """
    @pytest.fixture
    def element(self, equation):
        return Element(equation, "", "")

    @pytest.mark.parametrize(
        "equation,error_message",
        [
            (  # no-letter
                "dim: (1-12)",
                "A numeric range must contain at least one letter."
            ),
            (  # greater
                "dim: (a12-a10)",
                "The number of the first subscript value must be lower "
                "than the second subscript value in a subscript numeric"
                " range."
            ),
            (  # different-leading
                "dim: (aba1-abc12)",
                "Only matching names ending in numbers are valid."
            ),
        ],
        ids=["no-letter", "greater", "different-leading"]
    )
    def test_subscript_range_error(self, element, error_message):
        # assert that the files don't exist in the temporary directory
        with pytest.raises(VisitationError, match=error_message):
            element.parse()

    @pytest.mark.parametrize(
        "equation,mapping",
        [
            (  # single
                "subcon : subcon1,subcon2->(del: del con1, del con2)",
                ["del"]
            ),
            (  # single2
                "subcon : subcon1,subcon2 -> (del:del con1,del con2)",
                ["del"]
            ),
            (  # multiple
                "class: class1,class2->(metal:class1 metal,class2 metal),"
                "(our metal:ourC1,ourC2)",
                ["metal", "our metal"]
            ),
            (  # multiple2
                "class: class1,class2-> (metal:class1 metal,class2 metal) ,"
                " (our metal:ourC1,ourC2)",
                ["metal", "our metal"]
            ),
        ],
        ids=["single", "single2", "multiple", "multiple2"]
    )
    def test_complex_mapping(self, element, mapping):
        # parse the mapping
        warning_message = r"Subscript mapping detected\. "\
            r"This feature works only for simple cases\."
        with pytest.warns(UserWarning, match=warning_message):
            out = element.parse()

        assert out.mapping == mapping

    @pytest.mark.parametrize(
        "equation,mapping",
        [
            (  # single
                "subcon : subcon1,subcon2 -> del",
                ["del"]
            ),
            (  # single2
                "subcon : subcon1,subcon2->del",
                ["del"]
            ),
            (  # multiple
                "class: class1,class2->metal,our metal",
                ["metal", "our metal"]
            ),
            (  # multiple2
                "class: class1,class2->metal , our metal",
                ["metal", "our metal"]
            ),
        ],
        ids=["single", "single2", "multiple", "multiple2"]
    )
    def test_simple_mapping(self, element, mapping):
        # parse the mapping
        out = element.parse()
        assert out.mapping == mapping
