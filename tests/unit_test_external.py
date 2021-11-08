import os
import unittest
import warnings

from importlib.machinery import SourceFileLoader
import numpy as np
import xarray as xr

_root = os.path.dirname(__file__)
_exp = SourceFileLoader(
    'expected_data',
    os.path.join(_root, 'data/expected_data.py')
    ).load_module()


class TestExcels(unittest.TestCase):
    """
    Tests for Excels class
    """
    def test_read_clean(self):
        """
        Test for reading files with pandas
        """
        from pysd.py_backend.external import Excels

        file_name = os.path.join(_root, "data/input.xlsx")
        sheet_name = "Vertical"
        sheet_name2 = "Horizontal"

        # reading a file
        excel = Excels.read(file_name, sheet_name)
        self.assertTrue(isinstance(excel, np.ndarray))

        # check if it is in the dictionary
        self.assertTrue(file_name+sheet_name in
                        list(Excels._Excels))

        Excels.read(file_name, sheet_name2)
        self.assertTrue(file_name+sheet_name2 in
                        list(Excels._Excels))

        # clean
        Excels.clean()
        self.assertEqual(list(Excels._Excels),
                         [])

    def test_read_clean_opyxl(self):
        """
        Test for reading files with openpyxl
        """
        from pysd.py_backend.external import Excels
        from openpyxl import Workbook

        file_name = os.path.join(_root, "data/input.xlsx")

        # reading a file
        excel = Excels.read_opyxl(file_name)
        self.assertTrue(isinstance(excel, Workbook))

        # check if it is in the dictionary
        self.assertEqual(list(Excels._Excels_opyxl),
                         [file_name])

        Excels.read_opyxl(file_name)
        self.assertEqual(list(Excels._Excels_opyxl),
                         [file_name])

        # clean
        Excels.clean()
        self.assertEqual(list(Excels._Excels_opyxl),
                         [])

    def test_close_file(self):
        """
        Test for checking if excel files were closed
        """
        from pysd.py_backend.external import Excels
        import psutil

        p = psutil.Process()

        # number of files already open
        n_files = len(p.open_files())

        file_name = os.path.join(_root, "data/input.xlsx")
        sheet_name = "Vertical"
        sheet_name2 = "Horizontal"

        # reading files
        Excels.read(file_name, sheet_name)
        Excels.read(file_name, sheet_name2)
        Excels.read_opyxl(file_name)

        self.assertGreater(len(p.open_files()), n_files)

        # clean
        Excels.clean()
        self.assertEqual(len(p.open_files()), n_files)


class TestExternalMethods(unittest.TestCase):
    """
    Test for simple methods of External
    """

    def test_col_to_num(self):
        """
        External._num_to_col and External._col_to_num test
        """
        from pysd.py_backend.external import External

        col_to_num = External._col_to_num

        # Check col_to_num
        self.assertEqual(col_to_num("A"), 0)
        self.assertEqual(col_to_num("Z"), 25)
        self.assertEqual(col_to_num("a"), col_to_num("B")-1)
        self.assertEqual(col_to_num("Z"), col_to_num("aa")-1)
        self.assertEqual(col_to_num("Zz"), col_to_num("AaA")-1)

    def test_split_excel_cell(self):
        """
        External._split_excel_cell test
        """
        from pysd.py_backend.external import External

        ext = External('external')

        # No cells, function must return nothing
        nocells = ["A2A", "H0", "0", "5A", "A_1", "ZZZZ1", "A"]

        for nocell in nocells:
            self.assertFalse(ext._split_excel_cell(nocell))

        # Cells
        cells = [(1, 0, "A2"), (573, 7, "h574"),
                 (1, 572, "Va2"), (1, 728, "ABA2")]

        for row, col, cell in cells:
            self.assertEqual((row, col), ext._split_excel_cell(cell))

    def test_reshape(self):
        """
        External._reshape test
        """
        from pysd.py_backend.external import External
        import pandas as pd

        reshape = External._reshape

        data1d = np.array([2, 3, 5, 6])
        data2d = np.array([[2, 3, 5, 6],
                           [1, 7, 5, 8]])

        series1d = pd.Series(data1d)
        df2d = pd.DataFrame(data2d)

        shapes1d = [(4,), (4, 1, 1), (1, 1, 4), (1, 4, 1)]
        shapes2d = [(2, 4), (2, 4, 1), (1, 2, 4), (2, 1, 4)]

        for shape_i in shapes1d:
            self.assertEqual(reshape(data1d, shape_i).shape, shape_i)
            self.assertEqual(reshape(series1d, shape_i).shape, shape_i)

        for shape_i in shapes2d:
            self.assertEqual(reshape(data2d, shape_i).shape, shape_i)
            self.assertEqual(reshape(df2d, shape_i).shape, shape_i)

    def test_series_selector(self):
        """
        External._series_selector test
        """
        from pysd.py_backend.external import External

        ext = External('external')

        # row selector
        self.assertEqual(ext._series_selector("12", "A5"), "row")

        # column selector
        self.assertEqual(ext._series_selector("A", "a44"), "column")
        self.assertEqual(ext._series_selector("A", "AC44"), "column")
        self.assertEqual(ext._series_selector("A", "Bae2"), "column")

        # name selector
        self.assertEqual(ext._series_selector("Att", "a44b"), "name")
        self.assertEqual(ext._series_selector("Adfs", "a0"), "name")
        self.assertEqual(ext._series_selector("Ae_23", "aa_44"), "name")
        self.assertEqual(ext._series_selector("Aeee3", "3a"), "name")
        self.assertEqual(ext._series_selector("Aeee", "aajh2"), "name")

    def test_fill_missing(self):
        from pysd.py_backend.external import External

        # simple casses are tested with 1 dimensional data
        # 1 and 2 dimensional data is tested with test-models
        ext = External("external")
        series = np.arange(12)
        data = np.array([np.nan, np.nan, 1., 3., np.nan, 4.,
                         np.nan, np.nan, 7., 8., np.nan, np.nan])
        hold_back = np.array([1., 1., 1., 3., 3., 4.,
                              4., 4., 7., 8., 8., 8.])
        look_for = np.array([1., 1., 1., 3., 4., 4.,
                             7., 7., 7., 8., 8., 8.])
        interp = np.array([1., 1., 1., 3., 3.5, 4.,
                           5., 6., 7., 8., 8., 8.])

        ext.interp = "hold backward"
        datac = data.copy()
        ext._fill_missing(series, datac)
        self.assertTrue(np.all(hold_back == datac))

        ext.interp = "look forward"
        datac = data.copy()
        ext._fill_missing(series, datac)
        self.assertTrue(np.all(look_for == datac))

        ext.interp = "interpolate"
        datac = data.copy()
        ext._fill_missing(series, datac)
        self.assertTrue(np.all(interp == datac))

    def test_resolve_file(self):
        """
        External._resolve_file
        """
        from pysd.py_backend.external import External

        root = os.path.dirname(__file__)
        ext = External('external')
        ext.file = 'data/input.xlsx'
        ext._resolve_file(root=root)

        self.assertEqual(ext.file, os.path.join(root, 'data/input.xlsx'))

        root = os.path.join(root, 'data')
        ext.file = 'input.xlsx'
        ext._resolve_file(root=root)

        self.assertEqual(ext.file, os.path.join(root, 'input.xlsx'))

        ext.file = 'input2.xlsx'

        with self.assertRaises(FileNotFoundError) as err:
            ext._resolve_file(root=root)

        self.assertIn(
            f"File '{os.path.join(root, 'input2.xlsx')}' not found.",
            str(err.exception))

        # TODO in the future we may add an option to include indirect
        # references with ?. By the moment an error is raised
        ext.file = '?input.xlsx'
        with self.assertRaises(ValueError) as err:
            ext._resolve_file(root=root)

        self.assertIn(
            "Indirect reference to file: ?input.xlsx",
            str(err.exception))


class TestData(unittest.TestCase):
    """
    Test for the full working procedure of ExtData
    class when the data is properly given in the Excel file
    For 1D data all cases are computed.
    For 2D, 3D only some cases are computed as the complete set of
    test will cover all the possibilities.
    """

    # The first two test are for length 0 series and only the retrieved data is
    # calculated as the interpolation result will be constant
    def test_data_interp_h1d_1(self):
        """
        ExtData test for 1d horizontal series interpolation with len 1
        """
        import pysd

        # test as well no file extension
        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "16"
        cell = "B17"
        coords = {}
        interp = None
        py_name = "test_data_interp_h1d_1"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        # test the __str__ method
        print(data)

        expected = xr.DataArray([5], {'time': [4]}, ['time'])

        self.assertTrue(data.data.equals(expected))

    def test_data_interp_hn1d_1(self):
        """
        ExtData test for 1d horizontal series interpolation with len 1
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "time_1"
        cell = "data_1"
        coords = {}
        interp = None
        py_name = "test_data_interp_h1d_1"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        expected = xr.DataArray([5], {'time': [4]}, ['time'])

        self.assertTrue(data.data.equals(expected))

    def test_data_interp_h1d(self):
        """
        ExtData test for 1d horizontal series interpolation
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        interp = None
        py_name = "test_data_interp_h1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_interp_v1d(self):
        """
        ExtData test for 1d vertical series interpolation
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        time_row_or_col = "B"
        cell = "C5"
        coords = {}
        interp = None
        py_name = "test_data_interp_v1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_interp_hn1d(self):
        """
        ExtData test for 1d horizontal series interpolation by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        interp = None
        py_name = "test_data_interp_h1nd"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_interp_vn1d(self):
        """
        ExtData test for 1d vertical series interpolation by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        interp = None
        py_name = "test_data_interp_vn1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_forward_h1d(self):
        """
        ExtData test for 1d horizontal series look forward
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        interp = "look forward"
        py_name = "test_data_forward_h1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.forward_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_forward_v1d(self):
        """
        ExtData test for 1d vertical series look forward
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        time_row_or_col = "B"
        cell = "C5"
        coords = {}
        interp = "look forward"
        py_name = "test_data_forward_v1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.forward_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_forward_hn1d(self):
        """
        ExtData test for 1d horizontal series look forward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        interp = "look forward"
        py_name = "test_data_forward_hn1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.forward_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_forward_vn1d(self):
        """
        ExtData test for 1d vertical series look forward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        interp = "look forward"
        py_name = "test_data_forward_vn1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.forward_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_backward_h1d(self):
        """
        ExtData test for 1d horizontal series hold backward
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        interp = "hold backward"
        py_name = "test_data_backward_h1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.backward_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_backward_v1d(self):
        """
        ExtData test for 1d vertical series hold backward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        time_row_or_col = "B"
        cell = "C5"
        coords = {}
        interp = "hold backward"
        py_name = "test_data_backward_v1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.backward_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_backward_hn1d(self):
        """
        ExtData test for 1d horizontal series hold backward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        interp = "hold backward"
        py_name = "test_data_backward_hn1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.backward_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_backward_vn1d(self):
        """
        ExtData test for 1d vertical series hold backward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        interp = "hold backward"
        py_name = "test_data_backward_vn1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.backward_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_interp_vn2d(self):
        """
        ExtData test for 2d vertical series interpolation by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        time_row_or_col = "time"
        cell = "data_2d"
        coords = {'ABC': ['A', 'B', 'C']}
        interp = None
        py_name = "test_data_interp_vn2d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.interp_2d):
                self.assertTrue(y.equals(data(x)),
                                "Wrong result at X=" + str(x))

    def test_data_forward_hn2d(self):
        """
        ExtData test for 2d vertical series look forward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "time"
        cell = "data_2d"
        coords = {'ABC': ['A', 'B', 'C']}
        interp = "look forward"
        py_name = "test_data_forward_hn2d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.forward_2d):
                self.assertTrue(y.equals(data(x)),
                                "Wrong result at X=" + str(x))

    def test_data_backward_v2d(self):
        """
        ExtData test for 2d vertical series hold backward
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        time_row_or_col = "B"
        cell = "C5"
        coords = {'ABC': ['A', 'B', 'C']}
        interp = "hold backward"
        py_name = "test_data_backward_v2d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.backward_2d):
                self.assertTrue(y.equals(data(x)),
                                "Wrong result at X=" + str(x))

    def test_data_interp_h3d(self):
        """
        ExtData test for 3d horizontal series interpolation
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "4"
        cell_1 = "C5"
        cell_2 = "C8"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C']}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C']}
        interp = None
        py_name = "test_data_interp_h3d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     coords=coords_1,
                                     interp=interp,
                                     py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 time_row_or_col=time_row_or_col,
                 cell=cell_2,
                 coords=coords_2,
                 interp=interp)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.interp_3d):
                self.assertTrue(y.equals(data(x)),
                                "Wrong result at X=" + str(x))

    def test_data_forward_v3d(self):
        """
        ExtData test for 3d vertical series look forward
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        time_row_or_col = "B"
        cell_1 = "C5"
        cell_2 = "F5"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C']}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C']}
        interp = "look forward"
        py_name = "test_data_forward_v3d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     coords=coords_1,
                                     interp=interp,
                                     py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 time_row_or_col=time_row_or_col,
                 cell=cell_2,
                 coords=coords_2,
                 interp=interp)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.forward_3d):
                self.assertTrue(y.equals(data(x)),
                                "Wrong result at X=" + str(x))

    def test_data_backward_hn3d(self):
        """
        ExtData test for 3d horizontal series hold backward by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "time"
        cell_1 = "data_2d"
        cell_2 = "data_2db"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C']}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C']}
        interp = "hold backward"
        py_name = "test_data_backward_hn3d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     coords=coords_1,
                                     interp=interp,
                                     py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 time_row_or_col=time_row_or_col,
                 cell=cell_2,
                 coords=coords_2,
                 interp=interp)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.backward_3d):
                self.assertTrue(y.equals(data(x)),
                                "Wrong result at X=" + str(x))

    def test_data_raw_h1d(self):
        """
        ExtData test for 1d horizontal series raw
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        interp = "raw"
        py_name = "test_data_forward_h1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.raw_1d):
            data_x = data(x)
            if np.isnan(y):
                equal = np.isnan(data_x)
            else:
                equal = y == data_x
            self.assertTrue(equal, "Wrong result at X=" + str(x))


class TestLookup(unittest.TestCase):
    """
    Test for the full working procedure of ExtLookup
    class when the data is properly given in the Excel file
    For 1D data for all cases are computed.
    For 2D, 3D only some cases are computed as the complete set of
    test will cover all the possibilities.
    """

    def test_lookup_h1d(self):
        """
        ExtLookup test for 1d horizontal series
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        x_row_or_col = "4"
        cell = "C5"
        coords = {}
        py_name = "test_lookup_h1d"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell,
                                       coords=coords,
                                       py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_lookup_v1d(self):
        """
        ExtLookup test for 1d vertical series
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        x_row_or_col = "B"
        cell = "C5"
        coords = {}
        py_name = "test_lookup_v1d"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell,
                                       coords=coords,
                                       py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_lookup_hn1d(self):
        """
        ExtLookup test for 1d horizontal series by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        x_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        py_name = "test_lookup_h1nd"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell,
                                       coords=coords,
                                       py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_lookup_vn1d(self):
        """
        ExtLookup test for 1d vertical series by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        x_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        py_name = "test_lookup_vn1d"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell,
                                       coords=coords,
                                       py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_lookup_h2d(self):
        """
        ExtLookup test for 2d horizontal series
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        x_row_or_col = "4"
        cell = "C5"
        coords = {'ABC': ['A', 'B', 'C']}
        py_name = "test_lookup_h2d"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell,
                                       coords=coords,
                                       py_name=py_name)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.interp_2d):
                self.assertTrue(y.equals(data(x)),
                                "Wrong result at X=" + str(x))

    def test_lookup_vn3d(self):
        """
        ExtLookup test for 3d vertical series by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        x_row_or_col = "time"
        cell_1 = "data_2d"
        cell_2 = "data_2db"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C']}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C']}
        py_name = "test_lookup_vn3d"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 x_row_or_col=x_row_or_col,
                 cell=cell_2,
                 coords=coords_2)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.interp_3d):
                self.assertTrue(y.equals(data(x)),
                                "Wrong result at X=" + str(x))

    def test_lookup_vn3d_shape0(self):
        """
        ExtLookup test for 3d vertical series by cellrange names
        passing shape 0 xarray as argument
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        x_row_or_col = "time"
        cell_1 = "data_2d"
        cell_2 = "data_2db"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C']}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C']}
        py_name = "test_lookup_vn3d_shape0"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 x_row_or_col=x_row_or_col,
                 cell=cell_2,
                 coords=coords_2)

        data.initialize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x, y in zip(_exp.xpts, _exp.interp_3d):
                self.assertTrue(y.equals(data(xr.DataArray(x))),
                                "Wrong result at X=" + str(x))

    def test_lookup_vn2d_xarray(self):
        """
        ExtLookup test for 2d vertical series by cellrange names
        using xarray for interpolation
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        x_row_or_col = "time"
        cell_1 = "data_2d"
        coords_1 = {'ABC': ['A', 'B', 'C']}
        py_name = "test_lookup_vn2d_xarray"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       py_name=py_name)

        data.initialize()

        all_smaller = xr.DataArray([-1, -10], {'XY': ['X', 'Y']}, ['XY'])
        all_bigger = xr.DataArray([9, 20, 30], {'ABC': ['A', 'B', 'C']},
                                  ['ABC'])
        all_inside = xr.DataArray([3.5, 5.5], {'XY': ['X', 'Y']}, ['XY'])
        mixed = xr.DataArray([1.5, 20, -30], {'ABC': ['A', 'B', 'C']}, ['ABC'])
        full = xr.DataArray([[1.5, -30], [-10, 2.5], [4., 5.]],
                            {'ABC': ['A', 'B', 'C'], 'XY': ['X', 'Y']},
                            ['ABC', 'XY'])

        all_smaller_out = data.data[0].reset_coords('lookup_dim', drop=True)\
            + 0*all_smaller
        all_bigger_out = data.data[-1].reset_coords('lookup_dim', drop=True)
        all_inside_out = xr.DataArray([[0.5, -1],
                                       [-1, -0.5],
                                       [-0.75, 0]],
                                      {'ABC': ['A', 'B', 'C'],
                                       'XY': ['X', 'Y']},
                                      ['ABC', 'XY'])
        mixed_out = xr.DataArray([0.5, 0, 1],
                                 {'ABC': ['A', 'B', 'C']},
                                 ['ABC'])
        full_out = xr.DataArray([[0.5, 0],
                                 [0, 0],
                                 [-0.5, 0]],
                                {'ABC': ['A', 'B', 'C'], 'XY': ['X', 'Y']},
                                ['ABC', 'XY'])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertTrue(data(all_smaller).equals(all_smaller_out))
            self.assertTrue(data(all_bigger).equals(all_bigger_out))
            self.assertTrue(data(all_inside).equals(all_inside_out))
            self.assertTrue(data(mixed).equals(mixed_out))
            self.assertTrue(data(full).equals(full_out))

    def test_lookup_vn3d_xarray(self):
        """
        ExtLookup test for 3d vertical series by cellrange names
        using xarray for interpolation
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        x_row_or_col = "time"
        cell_1 = "data_2d"
        cell_2 = "data_2db"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C']}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C']}
        py_name = "test_lookup_vn3d_xarray"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 x_row_or_col=x_row_or_col,
                 cell=cell_2,
                 coords=coords_2)

        data.initialize()

        all_smaller = xr.DataArray([-1, -10], {'XY': ['X', 'Y']}, ['XY'])
        all_bigger = xr.DataArray([9, 20, 30], {'ABC': ['A', 'B', 'C']},
                                  ['ABC'])
        all_inside = xr.DataArray([3.5, 7.5], {'XY': ['X', 'Y']}, ['XY'])
        mixed = xr.DataArray([1.5, 20, -30], {'ABC': ['A', 'B', 'C']}, ['ABC'])
        full = xr.DataArray([[1.5, -30], [-10, 2.5], [4., 5.]],
                            {'ABC': ['A', 'B', 'C'], 'XY': ['X', 'Y']},
                            ['ABC', 'XY'])

        all_smaller_out = data.data[0].reset_coords('lookup_dim', drop=True)
        all_bigger_out = data.data[-1].reset_coords('lookup_dim', drop=True)
        all_inside_out = xr.DataArray([[0.5, -1, -0.75],
                                       [0.5,  1,  0]],
                                      {'XY': ['X', 'Y'],
                                       'ABC': ['A', 'B', 'C']},
                                      ['XY', 'ABC'])
        mixed_out = xr.DataArray([[0.5, 0, 1],
                                  [-1, 1,  -1]],
                                 {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']},
                                 ['XY', 'ABC'])

        full_out = xr.DataArray([[0.5, 0, -0.5],
                                 [1, 0, 0]],
                                {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']},
                                ['XY', 'ABC'])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertTrue(data(all_smaller).equals(all_smaller_out))
            self.assertTrue(data(all_bigger).equals(all_bigger_out))
            self.assertTrue(data(all_inside).equals(all_inside_out))
            self.assertTrue(data(mixed).equals(mixed_out))
            self.assertTrue(data(full).equals(full_out))


class TestConstant(unittest.TestCase):
    """
    Test for the full working procedure of ExtConstant
    class when the data is properly given in the Excel file
    For 1D, 2D and 3D all cases are computed.
    """

    def test_constant_0d(self):
        """
        ExtConstant test for 0d data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        cell = "F7"
        cell2 = "C5"
        coords = {}
        py_name = "test_constant_0d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)

        data2 = pysd.external.ExtConstant(file_name=file_name,
                                          sheet=sheet,
                                          root=_root,
                                          cell=cell2,
                                          coords=coords,
                                          py_name=py_name)
        data.initialize()
        data2.initialize()

        self.assertEqual(data(), -1)
        self.assertEqual(data2(), 0)

    def test_constant_n0d(self):
        """
        ExtConstant test for 0d data by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        cell = "constant"
        cell2 = "constant2"
        coords = {}
        py_name = "test_constant_0d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)

        data2 = pysd.external.ExtConstant(file_name=file_name,
                                          sheet=sheet,
                                          root=_root,
                                          cell=cell2,
                                          coords=coords,
                                          py_name=py_name)
        data.initialize()
        data2.initialize()

        self.assertEqual(data(), -1)
        self.assertEqual(data2(), 0)

    def test_constant_h1d(self):
        """
        ExtConstant test for horizontal 1d data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        cell = "C5"
        coords = {'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_h1d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data().equals(_exp.constant_1d))

    def test_constant_v1d(self):
        """
        ExtConstant test for vertical 1d data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        cell = "C5*"
        coords = {'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_v1d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data().equals(_exp.constant_1d))

    def test_constant_hn1d(self):
        """
        ExtConstant test for horizontal 1d data by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        cell = "data_1d"
        coords = {'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_hn1d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data().equals(_exp.constant_1d))

    def test_constant_vn1d(self):
        """
        ExtConstant test for vertical 1d data by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        cell = "data_1d*"
        coords = {'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_vn1d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data().equals(_exp.constant_1d))

    def test_constant_h2d(self):
        """
        ExtConstant test for horizontal 2d data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        cell = "C5"
        coords = {'ABC': ['A', 'B', 'C'], 'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_h2d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data().equals(_exp.constant_2d))

    def test_constant_v2d(self):
        """
        ExtConstant test for vertical 2d data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        cell = "C5*"
        coords = {'ABC': ['A', 'B', 'C'], 'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_v2d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data().equals(_exp.constant_2d))

    def test_constant_hn2d(self):
        """
        ExtConstant test for horizontal 2d data by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        cell = "data_2d"
        coords = {'ABC': ['A', 'B', 'C'], 'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_hn2d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data().equals(_exp.constant_2d))

    def test_constant_vn2d(self):
        """
        ExtConstant test for vertical 2d data by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        cell = "data_2d*"
        coords = {'ABC': ['A', 'B', 'C'], 'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_vn2d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data().equals(_exp.constant_2d))

    def test_constant_h3d(self):
        """
        ExtConstant test for horizontal 3d data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        cell = "C5"
        cell2 = "C8"
        coords = {'ABC': ['A', 'B', 'C'],
                  'XY': ['X'],
                  'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        coords2 = {'ABC': ['A', 'B', 'C'],
                   'XY': ['Y'],
                   'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_h3d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell2,
                 coords=coords2)

        data.initialize()

        self.assertTrue(data().equals(_exp.constant_3d))

    def test_constant_v3d(self):
        """
        ExtConstant test for vertical 3d data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        cell = "C5*"
        cell2 = "F5*"
        coords = {'ABC': ['A', 'B', 'C'],
                  'XY': ['X'],
                  'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        coords2 = {'ABC': ['A', 'B', 'C'],
                   'XY': ['Y'],
                   'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_v3d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell2,
                 coords=coords2)

        data.initialize()

        self.assertTrue(data().equals(_exp.constant_3d))

    def test_constant_hn3d(self):
        """
        ExtConstant test for horizontal 3d data by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        cell = "data_2d"
        cell2 = "data_2db"
        coords = {'ABC': ['A', 'B', 'C'],
                  'XY': ['X'],
                  'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        coords2 = {'ABC': ['A', 'B', 'C'],
                   'XY': ['Y'],
                   'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_hn3d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell2,
                 coords=coords2)

        data.initialize()

        self.assertTrue(data().equals(_exp.constant_3d))

    def test_constant_vn3d(self):
        """
        ExtConstant test for vertical 3d data by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        cell = "data_2d*"
        cell2 = "data_2db*"
        coords = {'ABC': ['A', 'B', 'C'],
                  'XY': ['X'],
                  'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        coords2 = {'ABC': ['A', 'B', 'C'],
                   'XY': ['Y'],
                   'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_vn2d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell2,
                 coords=coords2)

        data.initialize()

        self.assertTrue(data().equals(_exp.constant_3d))


class TestSubscript(unittest.TestCase):
    """
    Test for the full working procedure of ExtSubscript
    class when the data is properly given in the Excel file
    """

    def test_subscript_h(self):
        """
        ExtSubscript test for horizontal subscripts
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        firstcell = "C4"
        lastcell = "J4"
        prefix = 'val'
        expected = ['val0', 'val1', 'val2', 'val3',
                    'val5', 'val6', 'val7', 'val8']

        data = pysd.external.ExtSubscript(file_name=file_name,
                                          sheet=sheet,
                                          root=_root,
                                          firstcell=firstcell,
                                          lastcell=lastcell,
                                          prefix=prefix)

        self.assertTrue(data.subscript, expected)

    def test_subscript_v(self):
        """
        ExtSubscript test for vertical subscripts
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        firstcell = "B5"
        lastcell = "B7"
        prefix = ''
        expected = ['A', 'B', 'C']

        data = pysd.external.ExtSubscript(file_name=file_name,
                                          sheet=sheet,
                                          root=_root,
                                          firstcell=firstcell,
                                          lastcell=lastcell,
                                          prefix=prefix)

        self.assertTrue(data.subscript, expected)


class TestWarningsErrors(unittest.TestCase):
    """
    Test for the warnings and errors of External and its subclasses
    """

    def test_not_implemented_file(self):
        """
        Test for not implemented file
        """
        import pysd

        file_name = "data/not_implemented_file.ods"
        sheet = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        interp = None
        py_name = "test_not_implemented_file"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(NotImplementedError):
            data.initialize()

    def test_non_existent_file(self):
        """
        Test for non-existent file
        """
        import pysd

        file_name = "data/non_existent.xls"
        sheet = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        interp = None
        py_name = "test_non_existent_file"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(FileNotFoundError):
            data.initialize()

    def test_non_existent_sheet_pyxl(self):
        """
        Test for non-existent sheet with openpyxl
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Non-Existent"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        interp = None
        py_name = "test_non_existent_sheet_pyxl"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()

    def test_non_existent_cellrange_name_pyxl(self):
        """
        Test for non-existent cellrange name with openpyxl
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "time"
        cell = "non_exixtent"
        coords = {}
        interp = None
        py_name = "est_non_existent_cellrange_name_pyxl"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(AttributeError):
            data.initialize()

    def test_non_existent_cellrange_name_in_sheet_pyxl(self):
        """
        Test for non-existent cellrange name with openpyxl
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        cell = "constant"
        coords = {}
        py_name = "est_non_existent_cellrange_name_in_sheet_pyxl"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)

        with self.assertRaises(AttributeError):
            data.initialize()

    # Following test are for ExtData class only
    # as the initialization of ExtLookup uses the same function
    def test_data_interp_h1dm_row(self):
        """
        Test for warning 1d horizontal series interpolation when series
        has missing or NaN data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        time_row_or_col = "time_missing"
        cell = "len_0"
        coords = {}
        interp = None
        py_name = "test_data_interp_h1dm_row"

        pysd.external.External.missing = "warning"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with warnings.catch_warnings(record=True) as ws:
            data.initialize()
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertTrue("Not able to interpolate" in str(wu[-1].message))

        self.assertTrue(all(np.isnan(data.data.values)))

    def test_data_interp_h1dm_row2(self):
        """
        Test for warning 1d horizontal series interpolation when series
        has missing or NaN data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        time_row_or_col = "4"
        cell = "C9"
        coords = {"dim": ["B", "C", "D"]}
        interp = None
        py_name = "test_data_interp_h1dm_row2"

        pysd.external.External.missing = "warning"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with warnings.catch_warnings(record=True) as ws:
            data.initialize()
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertTrue("Not able to interpolate" in str(wu[-1].message))

        self.assertFalse(any(np.isnan(data.data.loc[:, "B"].values)))
        self.assertFalse(any(np.isnan(data.data.loc[:, "C"].values)))
        self.assertTrue(all(np.isnan(data.data.loc[:, "D"].values)))

    def test_data_interp_h1dm(self):
        """
        Test for warning 1d horizontal series interpolation when series
        has missing or NaN data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        interp = None
        py_name = "test_data_interp_h1dm"

        pysd.external.External.missing = "warning"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with warnings.catch_warnings(record=True) as ws:
            data.initialize()
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 1)
            self.assertIn("missing", str(wu[0].message))

        with warnings.catch_warnings(record=True) as ws:
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 2)
            self.assertTrue("extrapolating data below the minimum value"
                            + " of the time" in str(wu[0].message))
            self.assertTrue("extrapolating data above the maximum value"
                            + " of the time" in str(wu[1].message))

    def test_data_interp_h1dm_ignore(self):
        """
        Test ignore warning 1d horizontal series interpolation when series
        has missing or NaN data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        interp = None
        py_name = "test_data_interp_h1dm_ignore"

        pysd.external.External.missing = "ignore"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with warnings.catch_warnings(record=True) as ws:
            data.initialize()
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 0)

        with warnings.catch_warnings(record=True) as ws:
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 2)
            self.assertTrue("extrapolating data below the minimum value"
                            + " of the time" in str(wu[0].message))
            self.assertTrue("extrapolating data above the maximum value"
                            + " of the time" in str(wu[1].message))

    def test_data_interp_h1dm_raise(self):
        """
        Test error 1d horizontal series interpolation when series
        has missing or NaN data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        interp = None
        py_name = "test_data_interp_h1dm_ignore"

        pysd.external.External.missing = "raise"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()

    def test_data_interp_v1dm(self):
        """
        Test for warning 1d vertical series interpolation when series has
        missing or NaN data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical missing"
        time_row_or_col = "B"
        cell = "C5"
        coords = {}
        interp = None
        py_name = "test_data_interp_v1dm"

        pysd.external.External.missing = "warning"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with warnings.catch_warnings(record=True) as ws:
            data.initialize()
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 1)
            self.assertTrue("missing" in str(wu[0].message))

        with warnings.catch_warnings(record=True) as ws:
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 2)
            self.assertTrue("extrapolating data below the minimum value"
                            + " of the time" in str(wu[0].message))
            self.assertTrue("extrapolating data above the maximum value"
                            + " of the time" in str(wu[1].message))

    def test_data_interp_v1dm_ignore(self):
        """
        Test ignore warning 1d vertical series interpolation when series has
        missing or NaN data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical missing"
        time_row_or_col = "B"
        cell = "C5"
        coords = {}
        interp = None
        py_name = "test_data_interp_v1dm_ignore"

        pysd.external.External.missing = "ignore"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with warnings.catch_warnings(record=True) as ws:
            data.initialize()
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 0)

        with warnings.catch_warnings(record=True) as ws:
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 2)
            self.assertTrue("extrapolating data below the minimum value"
                            + " of the time" in str(wu[0].message))
            self.assertTrue("extrapolating data above the maximum value"
                            + " of the time" in str(wu[1].message))

    def test_data_interp_v1dm_raise(self):
        """
        Test error 1d vertical series interpolation when series has
        missing or NaN data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical missing"
        time_row_or_col = "B"
        cell = "C5"
        coords = {}
        interp = None
        py_name = "test_data_interp_v1dm_ignore"

        pysd.external.External.missing = "raise"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()

    def test_data_interp_hn1dm(self):
        """
        Test for warning 1d horizontal series by cellrange names
        when series has missing or NaN data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        time_row_or_col = "time_missing"
        cell = "data_1d"
        coords = {}
        interp = None
        py_name = "test_data_interp_h1dm"

        pysd.external.External.missing = "warning"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with warnings.catch_warnings(record=True) as ws:
            data.initialize()
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 1)
            self.assertTrue("missing" in str(wu[0].message))

        with warnings.catch_warnings(record=True) as ws:
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 2)
            self.assertTrue("extrapolating data below the minimum value"
                            + " of the time" in str(wu[0].message))
            self.assertTrue("extrapolating data above the maximum value"
                            + " of the time" in str(wu[1].message))

    def test_data_interp_hn1dm_ignore(self):
        """
        Test ignore warning 1d horizontal series by cellrange names
        when series has missing or NaN data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        time_row_or_col = "time_missing"
        cell = "data_1d"
        coords = {}
        interp = None
        py_name = "test_data_interp_h1dm_ignore"

        pysd.external.External.missing = "ignore"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with warnings.catch_warnings(record=True) as ws:
            data.initialize()
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 0)

        with warnings.catch_warnings(record=True) as ws:
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                self.assertEqual(y, data(x), "Wrong result at X=" + str(x))
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 2)
            self.assertTrue("extrapolating data below the minimum value"
                            + " of the time" in str(wu[0].message))
            self.assertTrue("extrapolating data above the maximum value"
                            + " of the time" in str(wu[1].message))

    def test_data_interp_hn1dm_raise(self):
        """
        Test for error 1d horizontal series by cellrange names
        when series has missing or NaN data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        time_row_or_col = "time_missing"
        cell = "data_1d"
        coords = {}
        interp = None
        py_name = "test_data_interp_h1dm_raise"

        pysd.external.External.missing = "raise"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()

    def test_data_interp_hn3dmd(self):
        """
        Test for warning 3d horizontal series interpolation by cellrange names
        with missing data values. More cases are tested with test-models
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        time_row_or_col = "time"
        cell_1 = "data_2d"
        cell_2 = "data_2db"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C']}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C']}
        interp = "interpolate"
        py_name = "test_data_interp_hn3dmd"

        pysd.external.External.missing = "warning"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     interp=interp,
                                     coords=coords_1,
                                     py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 time_row_or_col=time_row_or_col,
                 cell=cell_2,
                 interp=interp,
                 coords=coords_2)

        with warnings.catch_warnings(record=True) as ws:
            data.initialize()
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 2)
            self.assertTrue(np.all(
                ["missing" in str(w.message) for w in wu]
                ))
            self.assertTrue(np.all(
                ["will be filled" in str(w.message) for w in wu]
                ))

        with warnings.catch_warnings(record=True) as ws:
            for x, y in zip(_exp.xpts, _exp.interp_3d):
                self.assertTrue(y.equals(data(x)),
                                "Wrong result at X=" + str(x))
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 2)
            self.assertTrue("extrapolating data below the minimum value"
                            + " of the time" in str(wu[0].message))
            self.assertTrue("extrapolating data above the maximum value"
                            + " of the time" in str(wu[1].message))

    def test_data_interp_hn3dmd_raw(self):
        """
        Test for warning 1d horizontal series interpolation when series
        has missing or NaN data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        time_row_or_col = "time"
        cell_1 = "data_2d"
        cell_2 = "data_2db"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C']}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C']}
        interp = "raw"
        py_name = "test_data_interp_hn3dmd_raw"

        pysd.external.External.missing = "warning"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     interp=interp,
                                     coords=coords_1,
                                     py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 time_row_or_col=time_row_or_col,
                 cell=cell_2,
                 interp=interp,
                 coords=coords_2)

        with warnings.catch_warnings(record=True) as ws:
            data.initialize()
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 2)
            self.assertTrue(np.all(
                ["missing" in str(w.message) for w in wu]
                ))
            self.assertTrue(np.all(
                ["will be filled" not in str(w.message) for w in wu]
                ))

    def test_lookup_hn3dmd_raise(self):
        """
        Test for error 3d horizontal series interpolation with missing data
        values.
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        x_row_or_col = "15"
        cell_1 = "C16"
        cell_2 = "C19"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C']}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C']}
        py_name = "test_lookup_hn3dmd_raise"
        pysd.external.External.missing = "raise"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 x_row_or_col=x_row_or_col,
                 cell=cell_2,
                 coords=coords_2)

        with self.assertRaises(ValueError):
            data.initialize()

    def test_lookup_hn3dmd_ignore(self):
        """
        Test for ignore warnings 3d horizontal series interpolation with
        missing data values.
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        x_row_or_col = "15"
        cell_1 = "C16"
        cell_2 = "C19"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C']}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C']}
        py_name = "test_lookup_hn3dmd_ignore"
        pysd.external.External.missing = "ignore"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 x_row_or_col=x_row_or_col,
                 cell=cell_2,
                 coords=coords_2)

        with warnings.catch_warnings(record=True) as ws:
            data.initialize()
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 0)

        with warnings.catch_warnings(record=True) as ws:
            for x, y in zip(_exp.xpts, _exp.interp_3d):
                self.assertTrue(y.equals(data(x)),
                                "Wrong result at X=" + str(x))
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 2)
            self.assertTrue("extrapolating data below the minimum value"
                            + " of the series" in str(wu[0].message))
            self.assertTrue("extrapolating data above the maximum value"
                            + " of the series" in str(wu[1].message))

    def test_constant_h3dm(self):
        """
        Test for warning in 3d horizontal series interpolation with missing
        values.
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        cell_1 = "C16"
        cell_2 = "C19"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C'],
                    'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C'],
                    'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_h3dm"
        pysd.external.External.missing = "warning"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell_1,
                                         coords=coords_1,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell_2,
                 coords=coords_2)

        with warnings.catch_warnings(record=True) as ws:
            data.initialize()
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 2)
            self.assertTrue(np.all(
                ["missing" in str(w.message) for w in wu]
                ))

    def test_constant_h3dm_ignore(self):
        """
        Test for ignore in 3d horizontal series interpolation with missing
        values.
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        cell_1 = "C16"
        cell_2 = "C19"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C'],
                    'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C'],
                    'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_h3dm_ignore"
        pysd.external.External.missing = "ignore"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell_1,
                                         coords=coords_1,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell_2,
                 coords=coords_2)

        with warnings.catch_warnings(record=True) as ws:
            data.initialize()
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 0)

    def test_constant_h3dm_raise(self):
        """
        Test for error 3d horizontal constants with missing values.
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        cell_1 = "C16"
        cell_2 = "C19"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C'],
                    'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C'],
                    'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_h3dm_raise"
        pysd.external.External.missing = "raise"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell_1,
                                         coords=coords_1,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell_2,
                 coords=coords_2)

        with self.assertRaises(ValueError):
            data.initialize()

    def test_constant_hn3dm_raise(self):
        """
        Test for error 3d horizontal constants with missing values by cellrange
        name.
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        cell_1 = "data_2d"
        cell_2 = "data_2db"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C'],
                    'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C'],
                    'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_hn3dm_raise"
        pysd.external.External.missing = "raise"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell_1,
                                         coords=coords_1,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell_2,
                 coords=coords_2)

        with self.assertRaises(ValueError):
            data.initialize()

    def test_data_interp_h1d0(self):
        """
        Test for error 1d horizontal series for len 0 series
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        time_row_or_col = "3"
        cell = "C5"
        coords = {}
        interp = None
        py_name = "test_data_interp_h1d0"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()

    def test_data_interp_v1d0(self):
        """
        Test for error 1d vertical series for len 0 series
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical missing"
        time_row_or_col = "A"
        cell = "C5"
        coords = {}
        interp = None
        py_name = "test_data_interp_v1d0"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()

    def test_data_interp_hn1d0(self):
        """
        Test for error in series by cellrange names
        when series has length 0
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        time_row_or_col = "len_0"
        cell = "data_1d"
        coords = {}
        interp = None
        py_name = "test_data_interp_h1d0"
        pysd.external.External.missing = "warning"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()

    def test_data_interp_hn1dt(self):
        """
        Test for error in series by cellrange names
        when series is a sheetle
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "data_2d"
        cell = "data_1d"
        coords = {}
        interp = None
        py_name = "test_data_interp_h1dt"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()

    def test_data_interp_hns(self):
        """
        Test for error in data when it doen't have the same
        shape as the given coordinates
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "time"
        cell = "data_2d"
        coords = {'ABC': ['A', 'B']}
        interp = None
        py_name = "test_data_interp_hns"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()

    def test_data_interp_vnss(self):
        """
        Test for error in data when it doen't have the same
        shape in the first dimension as the length of series
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical missing"
        time_row_or_col = "time_short"
        cell = "data_2d_short"
        coords = {'ABC': ['A', 'B', 'C']}
        interp = None
        py_name = "test_data_interp_vnss"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()

    # Following test are independent of the reading option
    def test_data_interp_hnnwd(self):
        """
        Test for error in series when the series is not
        well defined
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "No monotonous"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        interp = None
        py_name = "test_data_interp_hnnwd"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError) as err:
            data.initialize()

        self.assertIn("has repeated values", str(err.exception))

    def test_data_raw_hnnm(self):
        """
        Test for error in series when the series is not monotonous
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "No monotonous"
        time_row_or_col = "10"
        cell = "C12"
        coords = {}
        interp = None
        py_name = "test_data_interp_hnnm"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        expected = {-1: 2, 0: 2, 1: 2, 2: 3,
                    3: -1, 4: -1, 5: 1, 6: 1,
                    7: 0, 8: 0, 9: 0}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(-1, 9):
                self.assertEqual(data(i), expected[i])

        time_row_or_col = "11"
        py_name = "test_data_interp_hnnnm2"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        expected = {-1: 0, 0: 0, 1: 0, 2: 1,
                    3: 2, 4: 3, 5: -1, 6: -1,
                    7: 1, 8: 2, 9: 2}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(-1, 9):
                self.assertEqual(data(i), expected[i])

    def test_data_h3d_interpnv(self):
        """
        ExtData test for error when the interpolation method is not valid
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {'ABC': ['A', 'B', 'C'], 'XY': ['X']}
        interp = "hold forward"
        py_name = "test_data_h3d_interpnv"

        with self.assertRaises(ValueError):
            pysd.external.ExtData(file_name=file_name,
                                  sheet=sheet,
                                  time_row_or_col=time_row_or_col,
                                  root=_root,
                                  cell=cell,
                                  coords=coords,
                                  interp=interp,
                                  py_name=py_name)

    def test_data_h3d_interp(self):
        """
        ExtData test for error when the interpolation method is different
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "4"
        cell_1 = "C5"
        cell_2 = "C8"
        coords_1 = {'ABC': ['A', 'B', 'C'], 'XY': ['X']}
        coords_2 = {'ABC': ['A', 'B', 'C'], 'XY': ['Y']}
        interp = None
        interp2 = "look forward"
        py_name = "test_data_h3d_interp"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     coords=coords_1,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.add(file_name=file_name,
                     sheet=sheet,
                     time_row_or_col=time_row_or_col,
                     cell=cell_2,
                     coords=coords_2,
                     interp=interp2)

    def test_data_h3d_add(self):
        """
        ExtData test for error when add doesn't have the same dim
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "4"
        cell_1 = "C5"
        cell_2 = "C8"
        coords_1 = {'ABC': ['A', 'B', 'C'], 'XY': ['X']}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C']}
        interp = None
        py_name = "test_data_h3d_add"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     coords=coords_1,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.add(file_name=file_name,
                     sheet=sheet,
                     time_row_or_col=time_row_or_col,
                     cell=cell_2,
                     coords=coords_2,
                     interp=interp)

    def test_lookup_h3d_add(self):
        """
        ExtLookup test for error when add doesn't have the same dim
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        x_row_or_col = "4"
        cell_1 = "C5"
        cell_2 = "C8"
        coords_1 = {'ABC': ['A', 'B', 'C'], 'XY': ['X']}
        coords_2 = {'ABC': ['A', 'B', 'C']}
        py_name = "test_lookup_h3d_add"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       py_name=py_name)

        with self.assertRaises(ValueError):
            data.add(file_name=file_name,
                     sheet=sheet,
                     x_row_or_col=x_row_or_col,
                     cell=cell_2,
                     coords=coords_2)

    def test_constant_h3d_add(self):
        """
        ExtConstant test for error when add doesn't have the same dim
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        cell = "C5"
        cell2 = "C8"
        coords = {'XY': ['X'],
                  'ABC': ['A', 'B', 'C'],
                  'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        coords2 = {'XY': ['Y'],
                   'ABC': ['A', 'B', 'C'],
                   'val2': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_h3d_add"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)

        with self.assertRaises(ValueError):
            data.add(file_name=file_name,
                     sheet=sheet,
                     cell=cell2,
                     coords=coords2)

    def test_constant_hns(self):
        """
        Test for error in data when it doen't have the same
        shape as the given coordinates
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        cell = "data_2d"
        coords = {'ABC': ['A', 'B']}
        py_name = "test_constant_hns"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()

    def text_openpyxl_str(self):
        """
        Test for reading data with strings with openpyxl
        """
        import pysd

        pysd.external.External.missing = "keep"

        file_name = "data/input.xlsx"
        sheet = "CASE AND NON V"  # test case insensitivity
        cell = "series"
        x_row_or_col = "unit"
        coords = {}
        py_name = "test_openpyxl_str"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell,
                                       coords=coords,
                                       py_name=py_name)

        expected = xr.DataArray(
            [np.nan, 1, 2, 3, 4, 5],
            {'lookup_dim': [10., 11., 12., 13., 14., 15.]},
            ['lookup_dim'])

        data.initialize()

        self.assertTrue(data.data.equals(expected))

        cell = "no_constant"
        sheet = "caSE anD NON V"  # test case insensitivity

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         py_name=py_name)

        data.initialize()

        self.assertTrue(np.isnan(data.data))


class DownwardCompatibility(unittest.TestCase):
    """
    These tests are defined to make the external objects compatible
    with SDQC library. If any change in PySD breaks these tests it
    should be checked with SDQC library and correct it.
    """
    def test_constant_hn3dm_keep(self):
        """
        Test for keep 3d horizontal constants with missing values by cellrange
        name.
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        cell_1 = "data_2d"
        cell_2 = "data_2db"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C'],
                    'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C'],
                    'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_hn3dm_raise"
        pysd.external.External.missing = "keep"

        expected = xr.DataArray(
            [[[0, 0, 1, 1, -1, -1, 0, np.nan],
              [0, 1, 1, -1, -1, 0, np.nan, np.nan],
              [np.nan, 1, -1, -1, 0, np.nan, np.nan, 0]],
             [[1, -1, -1, 0, 0, 0, 0, 1],
              [-1, -1., 0, np.nan, 0, 0, 1, np.nan],
              [-1, 0, np.nan, np.nan, 0, 1, 1, -1]]],
            {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C'],
             'val': [0, 1, 2, 3, 5, 6, 7, 8]},
            ['XY', 'ABC', 'val'])

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell_1,
                                         coords=coords_1,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell_2,
                 coords=coords_2)

        data.initialize()

        self.assertTrue(data().equals(expected))

    def test_lookup_hn3dmd_keep(self):
        """
        Test for keep 3d horizontal series interpolation with
        missing data values.
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        x_row_or_col = "15"
        cell_1 = "C16"
        cell_2 = "C19"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C']}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C']}
        py_name = "test_lookup_hn3dmd_ignore"
        pysd.external.External.missing = "keep"

        expected = xr.DataArray(
            [[[0, 0, np.nan],
              [1, -1, -1]],
             [[0, 1, 1],
              [-1, -1, 0]],
             [[1, 1, -1],
              [-1, 0, np.nan]],
             [[1, -1, -1],
              [0., np.nan, np.nan]],
             [[-1, -1, 0],
              [0, 0, 0]],
             [[-1, 0, np.nan],
              [0, 0, 1]],
             [[0, np.nan, np.nan],
              [0, 1, 1]],
             [[np.nan, np.nan, 0],
              [1, np.nan, -1]]],
            {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C'],
             'lookup_dim': [0., 1., 2., 3., 5., 6., 7., 8.]},
            ['lookup_dim', 'XY', 'ABC'])

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 x_row_or_col=x_row_or_col,
                 cell=cell_2,
                 coords=coords_2)

        data.initialize()

        self.assertTrue(data.data.equals(expected))

    def test_data_interp_v1dm_keep(self):
        """
        Test keep 1d vertical series interpolation when series has
        missing or NaN data
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical missing"
        time_row_or_col = "B"
        cell = "C5"
        coords = {}
        interp = None
        py_name = "test_data_interp_v1dm_ignore"

        pysd.external.External.missing = "keep"

        expected = xr.DataArray(
            [0, 0, 1, 1, 3, -1, -1, 0, 0],
            {'time': [0., 1., 2., 3., np.nan, 5., 6., 7., 8.]},
            ['time'])

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        self.assertTrue(data.data.equals(expected))

    def test_data_interp_hnnm_keep(self):
        """
        Test for keep in series when the series is not
        strictly monotonous
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "No monotonous"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        interp = None
        py_name = "test_data_interp_hnnm"

        pysd.external.External.missing = "keep"

        expected = xr.DataArray(
            [0, 0, 1, 1, -1, -1, 0, 0],
            {'time': [0., 1., 2., 7., 5., 6., 7., 8.]},
            ['time'])

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        self.assertTrue(data.data.equals(expected))

    def test_lookup_data_attr(self):
        """
        Test for keep in series when the series is not
        strictly monotonous
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "No monotonous"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        interp = None
        py_name = "test_data_interp_hnnm"

        datD = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     py_name=py_name)

        datL = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=time_row_or_col,
                                       root=_root,
                                       cell=cell,
                                       coords=coords,
                                       py_name=py_name)
        datD.initialize()
        datL.initialize()

        self.assertTrue(hasattr(datD, 'time_row_or_cols'))
        self.assertTrue(hasattr(datL, 'x_row_or_cols'))
