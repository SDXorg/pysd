import os
from unittest import TestCase

# most of the features of this script are already tested indirectly when
# running vensim and xmile integration tests

_root = os.path.dirname(__file__)


class TestErrors(TestCase):

    def test_canonical_file_not_found(self):
        from pysd.tools.benchmarking import runner

        with self.assertRaises(FileNotFoundError) as err:
            runner(os.path.join(_root, "more-tests/not_existent.mdl"))

        self.assertIn(
            'Canonical output file not found.',
            str(err.exception))

    def test_non_valid_model(self):
        from pysd.tools.benchmarking import runner

        with self.assertRaises(ValueError) as err:
            runner(os.path.join(
                _root,
                "more-tests/not_vensim/test_not_vensim.txt"))

        self.assertIn(
            'Modelfile should be *.mdl or *.xmile',
            str(err.exception))

    def test_non_valid_outputs(self):
        from pysd.tools.benchmarking import load_outputs

        with self.assertRaises(ValueError) as err:
            load_outputs(
                os.path.join(
                    _root,
                    "more-tests/not_vensim/test_not_vensim.txt"))

        self.assertIn(
            "Not able to read '",
            str(err.exception))
        self.assertIn(
            "more-tests/not_vensim/test_not_vensim.txt'.",
            str(err.exception))

    def test_different_frames_error(self):
        from pysd.tools.benchmarking import load_outputs, assert_frames_close

        with self.assertRaises(AssertionError) as err:
            assert_frames_close(
                load_outputs(os.path.join(_root, "data/out_teacup.csv")),
                load_outputs(
                    os.path.join(_root, "data/out_teacup_modified.csv")))

        self.assertIn(
            "Column 'Teacup Temperature' is not close.",
            str(err.exception))

        self.assertIn(
            "Actual values:\n\t",
            str(err.exception))

        self.assertIn(
            "Expected values:\n\t",
            str(err.exception))

    def test_different_frames_warning(self):
        from warnings import catch_warnings
        from pysd.tools.benchmarking import load_outputs, assert_frames_close

        with catch_warnings(record=True) as ws:
            assert_frames_close(
                load_outputs(os.path.join(_root, "data/out_teacup.csv")),
                load_outputs(
                    os.path.join(_root, "data/out_teacup_modified.csv")),
                assertion="warn")

            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 1)

            self.assertIn(
                "Column 'Teacup Temperature' is not close.",
                str(wu[0].message))

            self.assertIn(
                "Actual values:\n\t",
                str(wu[0].message))

            self.assertIn(
                "Expected values:\n\t",
                str(wu[0].message))

    def test_transposed_frame(self):
        from pysd.tools.benchmarking import load_outputs, assert_frames_close

        assert_frames_close(
            load_outputs(os.path.join(_root, "data/out_teacup.csv")),
            load_outputs(os.path.join(_root, "data/out_teacup_transposed.csv"),
                         transpose=True))
