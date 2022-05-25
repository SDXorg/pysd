import pytest

import pandas as pd

from pysd.tools.benchmarking import runner, load_outputs, assert_frames_close

# most of the features of this script are already tested indirectly when
# running vensim and xmile integration tests


class TestErrors():

    def test_canonical_file_not_found(self, _root):
        error_message = 'Canonical output file not found.'
        with pytest.raises(FileNotFoundError, match=error_message):
            runner(_root.joinpath("more-tests/not_existent.mdl"))

    def test_non_valid_model(self, _root):
        error_message = r"The model file name must be a Vensim \(\.mdl\),"\
            r" a Xmile \(\.xmile, \.xml, \.stmx\) or a PySD \(\.py\) "\
            r"model file\.\.\.$"
        with pytest.raises(ValueError, match=error_message):
            runner(_root.joinpath("more-tests/not_vensim/test_not_vensim.txt"))

    def test_different_frames_error(self, _root):

        error_message = r"\nFollowing columns are not close:\n\t"\
            r"Teacup Temperature\n\nFirst false values \(30\.0\):\n\t"\
            r"Teacup Temperature$"

        with pytest.raises(AssertionError, match=error_message):
            assert_frames_close(
                load_outputs(_root.joinpath("data/out_teacup.csv")),
                load_outputs(_root.joinpath("data/out_teacup_modified.csv")))

        error_message = r"\nFollowing columns are not close:\n\tTeacup "\
            r"Temperature\n\nFirst false values \(30\.0\):\n\tTeacup "\
            r"Temperature\n\nColumn 'Teacup Temperature' is not close\."\
            r"\n\nExpected values:\n\t\[[\-0-9\s\n,\.]*\]"\
            r"\n\nActual values:\n\t\[[\-0-9\s\n,\.]*\]"\
            r"\n\nDifference:\n\t\[[\-0-9\s\n,\.]*\]$"

        with pytest.raises(AssertionError, match=error_message):
            assert_frames_close(
                load_outputs(_root.joinpath("data/out_teacup.csv")),
                load_outputs(_root.joinpath("data/out_teacup_modified.csv")),
                verbose=True)

    def test_different_frames_warning(self, _root):

        warn_message = r"\nFollowing columns are not close:\n\t"\
            r"Teacup Temperature\n\nFirst false values \(30\.0\):\n\t"\
            r"Teacup Temperature$"
        with pytest.warns(UserWarning, match=warn_message):
            assert_frames_close(
                load_outputs(_root.joinpath("data/out_teacup.csv")),
                load_outputs(_root.joinpath("data/out_teacup_modified.csv")),
                assertion="warn")

        warn_message = r"\nFollowing columns are not close:\n\tTeacup "\
            r"Temperature\n\nFirst false values \(30\.0\):\n\tTeacup "\
            r"Temperature\n\nColumn 'Teacup Temperature' is not close\."\
            r"\n\nExpected values:\n\t\[[\-0-9\s\n,\.]*\]"\
            r"\n\nActual values:\n\t\[[\-0-9\s\n,\.]*\]"\
            r"\n\nDifference:\n\t\[[\-0-9\s\n,\.]*\]$"

        with pytest.warns(UserWarning, match=warn_message):
            assert_frames_close(
                load_outputs(_root.joinpath("data/out_teacup.csv")),
                load_outputs(_root.joinpath("data/out_teacup_modified.csv")),
                assertion="warn", verbose=True)

    def test_different_frames_return(self, _root):

        cols, first_false_time, first_false_cols = assert_frames_close(
            load_outputs(_root.joinpath("data/out_teacup.csv")),
            load_outputs(_root.joinpath("data/out_teacup_modified.csv")),
            assertion="return")

        assert cols == {"Teacup Temperature"}
        assert first_false_time == 30.
        assert first_false_cols == {"Teacup Temperature"}

    def test_different_cols(self):
        d1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'd': [6, 7]})
        d2 = pd.DataFrame({'a': [1, 2]})
        d3 = pd.DataFrame({'a': [1, 2], 'c': [3, 4]})

        error_message = r"Columns from actual and expected values must "\
            r"be equal\.\nColumns 'b', 'd' from actual values not found"\
            r" in expected values\.$"
        with pytest.raises(ValueError, match=error_message):
            assert_frames_close(
                actual=d1,
                expected=d2)

        warn_message = r"Columns 'b', 'd' from actual values not found "\
            r"in expected values\.$"
        with pytest.warns(UserWarning, match=warn_message):
            assert_frames_close(
                actual=d1,
                expected=d2,
                assertion="warn")

        warn_message = r"Columns 'b', 'd' from expected values not found "\
            r"in actual values\.$"
        with pytest.warns(UserWarning, match=warn_message):
            assert_frames_close(
                expected=d1,
                actual=d2,
                assertion="warn")

        warn_message = r"Columns 'b', 'd' from actual values not found "\
            r"in expected values\.\nColumns 'c' from expected values not"\
            r" found in actual values\.$"
        with pytest.warns(UserWarning, match=warn_message):
            assert_frames_close(
                actual=d1,
                expected=d3,
                assertion="warn")

    def test_invalid_input(self):

        error_message = r"Inputs must both be pandas DataFrames\."
        with pytest.raises(TypeError, match=error_message):
            assert_frames_close(
                actual=[1, 2],
                expected=[1, 2])

    def test_run_python(self, _root):
        test_model = _root.joinpath("test-models/samples/teacup/teacup.mdl")
        assert (
            runner(str(test_model))[0]
            == runner(test_model.with_suffix(".py"))[0]
        ).all().all()
