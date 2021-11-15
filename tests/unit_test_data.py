import os
import itertools
import unittest

import xarray as xr

_root = os.path.dirname(__file__)


class TestColumns(unittest.TestCase):
    def test_clean_columns(self):
        from pysd.py_backend.data import Columns
        Columns.clean()
        self.assertEqual(Columns._files, {})
        Columns.read(
            os.path.join(_root, "data/out_teacup.csv"))
        self.assertNotEqual(Columns._files, {})
        self.assertIn(os.path.join(_root, "data/out_teacup.csv"),
                      Columns._files)
        Columns.clean()
        self.assertEqual(Columns._files, {})

    def test_non_valid_outputs(self):
        from pysd.py_backend.data import Columns

        with self.assertRaises(ValueError) as err:
            Columns.read_file(
                os.path.join(
                    _root,
                    "more-tests/not_vensim/test_not_vensim.txt"))

        self.assertIn(
            "Not able to read '",
            str(err.exception))
        self.assertIn(
            "more-tests/not_vensim/test_not_vensim.txt'.",
            str(err.exception))

    def test_non_valid_file_format(self):
        from pysd.py_backend.data import Columns

        file_name = os.path.join(_root, "data/out_teacup_no_head.csv")
        with self.assertRaises(ValueError) as err:
            Columns.read_file(file_name)

        self.assertIn(
            f"Invalid file format '{file_name}'... varible names "
            + "should appear in the first row or in the first column...",
            str(err.exception))

    def test_transposed_frame(self):
        from pysd.py_backend.data import Columns

        cols1, trans1 = Columns.get_columns(
            os.path.join(_root, "data/out_teacup.csv"))
        cols2, trans2 = Columns.get_columns(
            os.path.join(_root, "data/out_teacup_transposed.csv"))
        Columns.clean()

        self.assertEqual(cols1, cols2)
        self.assertFalse(trans1)
        self.assertTrue(trans2)

    def test_get_columns(self):
        from pysd.py_backend.data import Columns

        cols0, trans0 = Columns.get_columns(
            os.path.join(_root, "data/out_teacup.csv"))

        cols1, trans1 = Columns.get_columns(
            os.path.join(_root, "data/out_teacup.csv"),
            vars=["Room Temperature", "Teacup Temperature"])

        cols2, trans2 = Columns.get_columns(
            os.path.join(_root, "data/out_teacup_transposed.csv"),
            vars=["Heat Loss to Room"])

        cols3 = Columns.get_columns(
            os.path.join(_root, "data/out_teacup_transposed.csv"),
            vars=["No column"])[0]

        Columns.clean()

        self.assertTrue(cols1.issubset(cols0))
        self.assertEqual(
            cols1,
            set(["Room Temperature", "Teacup Temperature"]))

        self.assertTrue(cols2.issubset(cols0))
        self.assertEqual(
            cols2,
            set(["Heat Loss to Room"]))

        self.assertEqual(cols3, set())

        self.assertFalse(trans0)
        self.assertFalse(trans1)
        self.assertTrue(trans2)

    def test_get_columns_subscripted(self):
        from pysd.py_backend.data import Columns

        data_file = os.path.join(
            _root,
            "test-models/tests/subscript_3d_arrays_widthwise/output.tab"
        )

        data_file2 = os.path.join(
            _root,
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
            "\"Three Dimensional Constant[" + ",".join(el) + "]\""
            for el in itertools.product(subsd["d1"], subsd["d2"], subsd["d3"])
        }

        self.assertEqual(cols1, expected)

        cols2 = Columns.get_columns(
            data_file2,
            vars=["Rate A", "Stock A"])[0]

        subs = list(itertools.product(subsd["d1"], subsd["d2"]))
        expected = {
            "\"Rate A[" + ",".join(el) + "]\""
            for el in subs
        }

        expected.update({
            "\"Stock A[" + ",".join(el) + "]\""
            for el in subs
        })

        self.assertEqual(cols2, expected)


class TestData(unittest.TestCase):
    # Several Data cases are tested in unit_test_external while some other
    # are tested indirectly in unit_test_pysd and integration_test_vensim

    def test_no_data_error(self):
        from pysd.py_backend.data import Data
        obj = Data()
        obj.data = None
        obj.interp = "interpolate"
        obj.py_name = "data"
        with self.assertRaises(ValueError) as err:
            obj(1.5)

        self.assertIn(
            "Trying to interpolate data variable before loading the data...",
            str(err.exception))

    def test_invalid_data_regular_error(self):
        # test that try/except block on call doesn't catch errors differents
        # than data = None
        from pysd.py_backend.data import Data

        obj = Data()

        obj.data = 3

        with self.assertRaises(TypeError) as err:
            obj(1.5)

        self.assertIn(
            "'int' object is not subscriptable",
            str(err.exception))

        obj.data = xr.DataArray([10, 20], {'dim1': [0, 1]}, ['dim1'])

        with self.assertRaises(KeyError) as err:
            obj(1.5)

        self.assertIn(
            "'time'",
            str(err.exception))

        obj.data = xr.DataArray([10, 20], {'time': [0, 1]}, ['time'])

        with self.assertRaises(AttributeError) as err:
            obj(1.5)

        self.assertIn(
            "'Data' object has no attribute 'interp'",
            str(err.exception))
