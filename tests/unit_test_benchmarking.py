from unittest import TestCase

# most of the features of this script are already tested indirectly when
# running vensim and xmile integration tests


class TestErrors(TestCase):

    def test_canonical_file_not_found(self):
        from pysd.tools.benchmarking import runner

        with self.assertRaises(FileNotFoundError) as err:
            runner('more-tests/not_existent.mdl')

        self.assertIn(
            'Canonical output file not found.',
            str(err.exception))

    def test_non_valid_model(self):
        from pysd.tools.benchmarking import runner

        with self.assertRaises(ValueError) as err:
            runner('more-tests/not_vensim/test_not_vensim.txt')

        self.assertIn(
            'Modelfile should be *.mdl or *.xmile',
            str(err.exception))

    def test_non_valid_outputs(self):
        from pysd.tools.benchmarking import load_outputs

        with self.assertRaises(ValueError) as err:
            load_outputs('more-tests/not_vensim/test_not_vensim.txt')

        self.assertIn(
            "Not able to read 'more-tests/not_vensim/test_not_vensim.txt'.",
            str(err.exception))

    def test_different_frames_error(self):
        from pysd.tools.benchmarking import load_outputs, assert_frames_close

        with self.assertRaises(AssertionError) as err:
            assert_frames_close(
                load_outputs('data/out_teacup.csv'),
                load_outputs('data/out_teacup_modified.csv'))

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
                load_outputs('data/out_teacup.csv'),
                load_outputs('data/out_teacup_modified.csv'),
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
            load_outputs('data/out_teacup.csv'),
            load_outputs('data/out_teacup_transposed.csv', transpose=True))

    def test_load_columns(self):
        from pysd.tools.benchmarking import load_outputs

        out0 = load_outputs(
            'data/out_teacup.csv')

        out1 = load_outputs(
            'data/out_teacup.csv',
            columns=["Room Temperature", "Teacup Temperature"])

        out2 = load_outputs(
            'data/out_teacup_transposed.csv',
            transpose=True,
            columns=["Heat Loss to Room"])

        self.assertEqual(
            set(out1.columns),
            set(["Room Temperature", "Teacup Temperature"]))

        self.assertEqual(
            set(out2.columns),
            set(["Heat Loss to Room"]))

        self.assertTrue((out0.index == out1.index).all())
        self.assertTrue((out0.index == out2.index).all())

    def test_different_cols(self):
        from warnings import catch_warnings
        from pysd.tools.benchmarking import assert_frames_close
        import pandas as pd

        d1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'd': [6, 7]})
        d2 = pd.DataFrame({'a': [1, 2]})
        d3 = pd.DataFrame({'a': [1, 2], 'c': [3, 4]})

        with self.assertRaises(ValueError) as err:
            assert_frames_close(
                actual=d1,
                expected=d2)

        self.assertIn(
            "Columns from actual and expected values must be equal.",
            str(err.exception))

        with catch_warnings(record=True) as ws:
            assert_frames_close(
                actual=d1,
                expected=d2,
                assertion="warn")

            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 1)

            self.assertIn("'b'", str(wu[0].message))
            self.assertIn("'d'", str(wu[0].message))
            self.assertIn(
                "from actual values not found in expected values.",
                str(wu[0].message))

        with catch_warnings(record=True) as ws:
            assert_frames_close(
                expected=d1,
                actual=d2,
                assertion="warn")

            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 1)

            self.assertIn("'b'", str(wu[0].message))
            self.assertIn("'d'", str(wu[0].message))
            self.assertIn(
                "from expected values not found in actual values.",
                str(wu[0].message))

        with catch_warnings(record=True) as ws:
            assert_frames_close(
                actual=d1,
                expected=d3,
                assertion="warn")

            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 1)

            self.assertIn("'b'", str(wu[0].message))
            self.assertIn("'d'", str(wu[0].message))
            self.assertIn(
                "from actual values not found in expected values.",
                str(wu[0].message))

            self.assertIn(
                "Columns 'c' from expected values not found in actual "
                "values.", str(wu[0].message))

    def test_invalid_input(self):
        from pysd.tools.benchmarking import assert_frames_close

        with self.assertRaises(TypeError) as err:
            assert_frames_close(
                actual=[1, 2],
                expected=[1, 2])

        self.assertIn(
            "Inputs must both be pandas DataFrames.",
            str(err.exception))
