
import pytest
from pathlib import Path

from pysd.translation.vensim.vensim_file import VensimFile


@pytest.mark.parametrize(
    "path",
    [
        (  # teacup
            "test-models/samples/teacup/teacup.mdl"
        ),
        (  # macros
            "test-models/tests/macro_multi_expression/test_macro_multi_expression.mdl"
        ),
        (  # mapping
            "test-models/tests/subscript_mapping_vensim/test_subscript_mapping_vensim.mdl"
        ),
        (  # data
            "test-models/tests/data_from_other_model/test_data_from_other_model.mdl"
        ),
        (  # except
            "test-models/tests/except/test_except.mdl"
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
        return _root.joinpath(path)

    @pytest.mark.dependency(name="read_vensim_file")
    def test_read_vensim_file(self, model_path):
        # assert that the files don't exist in the temporary directory
        ven_file = VensimFile(model_path)

        assert hasattr(ven_file, "mdl_path")
        assert hasattr(ven_file, "root_path")
        assert hasattr(ven_file, "model_text")

        assert isinstance(getattr(ven_file, "mdl_path"), Path)
        assert isinstance(getattr(ven_file, "root_path"), Path)
        assert isinstance(getattr(ven_file, "model_text"), str)

    @pytest.mark.dependency(depends=["read_vensim_file"])
    def test_file_split_file_sections(self, model_path):
        ven_file = VensimFile(model_path)
        ven_file.parse()
