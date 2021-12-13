import pytest
import itertools

from pysd.py_backend.data import Columns


class TestColumns:
    @pytest.fixture(scope="class")
    def out_teacup(self, _root):
        return _root.joinpath("data/out_teacup.csv")

    @pytest.fixture(scope="class")
    def out_teacup_transposed(self, _root):
        return _root.joinpath("data/out_teacup_transposed.csv")

    def test_clean_columns(self, out_teacup):
        # test the singleton works well for laizy loading
        Columns.clean()
        assert Columns._files == {}
        Columns.read(out_teacup)
        assert Columns._files != {}
        assert out_teacup in Columns._files
        Columns.clean()
        assert Columns._files == {}

    def test_transposed_frame(self, out_teacup, out_teacup_transposed):
        # test loading transposed frames
        cols1, trans1 = Columns.get_columns(out_teacup)
        cols2, trans2 = Columns.get_columns(out_teacup_transposed)
        Columns.clean()

        assert cols1 == cols2
        assert not trans1
        assert trans2

    def test_get_columns(self, out_teacup, out_teacup_transposed):
        # test getting specific columns by name
        cols0, trans0 = Columns.get_columns(out_teacup)

        cols1, trans1 = Columns.get_columns(
            out_teacup,
            vars=["Room Temperature", "Teacup Temperature"])

        cols2, trans2 = Columns.get_columns(
            out_teacup_transposed,
            vars=["Heat Loss to Room"])

        cols3 = Columns.get_columns(
            out_teacup_transposed,
            vars=["No column"])[0]

        Columns.clean()

        assert cols1.issubset(cols0)
        assert cols1 == set(["Room Temperature", "Teacup Temperature"])

        assert cols2.issubset(cols0)
        assert cols2 == set(["Heat Loss to Room"])

        assert cols3 == set()

        assert not trans0
        assert not trans1
        assert trans2

    def test_get_columns_subscripted(self, _root):
        # test get subscripted columns
        data_file = _root.joinpath(
            "test-models/tests/subscript_3d_arrays_widthwise/output.tab"
        )

        data_file2 = _root.joinpath(
            "test-models/tests/subscript_2d_arrays/output.tab"
        )

        subsd = {
            "d3": ["Depth 1", "Depth 2"],
            "d2": ["Column 1", "Column 2"],
            "d1": ["Entry 1", "Entry 2", "Entry 3"]
        }

        cols1 = Columns.get_columns(
            data_file,
            vars=["Three Dimensional Constant"])[0]

        expected = {
            "Three Dimensional Constant[" + ",".join(el) + "]"
            for el in itertools.product(subsd["d1"], subsd["d2"], subsd["d3"])
        }

        assert cols1 == expected

        cols2 = Columns.get_columns(
            data_file2,
            vars=["Rate A", "Stock A"])[0]

        subs = list(itertools.product(subsd["d1"], subsd["d2"]))
        expected = {
            "Rate A[" + ",".join(el) + "]"
            for el in subs
        }

        expected.update({
            "Stock A[" + ",".join(el) + "]"
            for el in subs
        })

        assert cols2 == expected


@pytest.mark.parametrize(
    "file,raise_type,error_message",
    [
        (  # invalid_file_type
            "more-tests/not_vensim/test_not_vensim.txt",
            ValueError,
            "Not able to read '%s'"
        ),
        (  # invalid_file_format
            "data/out_teacup_no_head.csv",
            ValueError,
            "Invalid file format '%s'... varible names should appear"
            + " in the first row or in the first column..."
        )
    ],
    ids=["invalid_file_type", "invalid_file_format"]
)
class TestColumnsErrors:
    # Test errors associated with Columns class

    @pytest.fixture
    def file_path(self, _root, file):
        return _root.joinpath(file)

    def test_columns_errors(self, file_path, raise_type, error_message):
        with pytest.raises(raise_type, match=error_message % str(file_path)):
            Columns.read_file(file_path)
