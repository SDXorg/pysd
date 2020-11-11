import os
import sys
import imp
import unittest

import numpy as np

_root = os.path.dirname(__file__)
_py_version = sys.version_info[0]*10 + sys.version_info[1]
_exp = imp.load_source('expected_data', 'data/expected_data.py')

# Following test are designed to test the simple methods of External

class TestExternalMethods(unittest.TestCase):

    def test_num_to_col_and_col_to_num(self):
        """
        External._num_to_col and External._col_to_num test
        """
        import pysd

        num_to_col = pysd.external.External._num_to_col
        col_to_num = pysd.external.External._col_to_num

        # Check col_to_num
        self.assertEqual(col_to_num("A"), 0)
        self.assertEqual(col_to_num("Z"), 25)
        self.assertEqual(col_to_num("a"), col_to_num("B")-1)
        self.assertEqual(col_to_num("Z"), col_to_num("aa")-1)
        self.assertEqual(col_to_num("Zz"), col_to_num("AaA")-1)
        
        cols = ["A", "AA", "AAA", "Z", "ZZ", "ZZZ",
                "N", "WB", "ASJ", "K", "HG", "BTF"]
               
        # Check num_to_col inverts col_to_num
        for col in cols:
            self.assertEqual(num_to_col(col_to_num(col)), col)

    def test_split_excel_cell(self):
        """
        External._split_excel_cell test
        """
        import pysd

        split_excel_cell = pysd.external.External._split_excel_cell
        
        # No cells, function must return nothing
        nocells = ["A2A", "H0", "0", "5A", "A_1", "ZZZZ1", "A"]
        
        for nocell in nocells:
            self.assertFalse(split_excel_cell(nocell))
        
        # Cells
        cells = [(2, "A", "A2"), (574, "h", "h574"),
                 (2, "Va", "Va2"), (2, "ABA", "ABA2")]
  
        for row, col, cell in cells:
            self.assertEqual((row, col), split_excel_cell(cell))

    def test_reshape(self):
        """
        External._reshape test
        """
        import pysd
        import numpy as np
        import pandas as pd

        reshape = pysd.external.External._reshape
        
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
        import pysd

        series_selector = pysd.external.External._series_selector

        # row selector
        self.assertEqual(series_selector("12", "A5"), "row")

        # column selector
        self.assertEqual(series_selector("A", "a44"), "column")
        self.assertEqual(series_selector("A", "AC44"), "column")
        self.assertEqual(series_selector("A", "Bae2"), "column")

        # name selector    
        self.assertEqual(series_selector("Att", "a44b"), "name")
        self.assertEqual(series_selector("Adfs", "a0"), "name")
        self.assertEqual(series_selector("Ae_23", "aa_44"), "name")
        self.assertEqual(series_selector("Aeee3", "3a"), "name")
        self.assertEqual(series_selector("Aeee", "aajh2"), "name")

# Following test are designed to test the full working of External subclasses
# For 1D data for each class and for all cases are computed
# For 2D, 3D only some cases are computed as the complete set of test will
# cover all the possibilities
# Data with missing or NaN values are not tested

class TestData(unittest.TestCase):

    def test_data_interp_h1d(self):
        """
        ExtData test for 1d horizontal series interpolation
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        dims = []
        interp = None
        py_name = "test_data_interp_h1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_interp_v1d(self):
        """
        ExtData test for 1d vertical series interpolation
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        time_row_or_col = "B"
        cell = "C5"
        coords = {}
        dims = []
        interp = None
        py_name = "test_data_interp_v1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_data_interp_hn1d(self):
        """
        ExtData test for 1d horizontal series interpolation by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        dims = []
        interp = None
        py_name = "test_data_interp_h1nd"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))
            
    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_data_interp_vn1d(self):
        """
        ExtData test for 1d vertical series interpolation by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        dims = []
        interp = None
        py_name = "test_data_interp_vn1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_forward_h1d(self):
        """
        ExtData test for 1d horizontal series look forward
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        dims = []
        interp = "look forward"
        py_name = "test_data_forward_h1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.forward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 
 
    def test_data_forward_v1d(self):
        """
        ExtData test for 1d vertical series look forward
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        time_row_or_col = "B"
        cell = "C5"
        coords = {}
        dims = []
        interp = "look forward"
        py_name = "test_data_forward_v1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.forward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_data_forward_hn1d(self):
        """
        ExtData test for 1d horizontal series look forward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        dims = []
        interp = "look forward"
        py_name = "test_data_forward_hn1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.forward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_data_forward_vn1d(self):
        """
        ExtData test for 1d vertical series look forward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        dims = []
        interp = "look forward"
        py_name = "test_data_forward_vn1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.forward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 
 
    def test_data_backward_h1d(self):
        """
        ExtData test for 1d horizontal series hold backward
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        dims = []
        interp = "hold backward"
        py_name = "test_data_backward_h1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.backward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 
  
    def test_data_backward_v1d(self):
        """
        ExtData test for 1d vertical series hold backward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        time_row_or_col = "B"
        cell = "C5"
        coords = {}
        dims = []
        interp = "hold backward"
        py_name = "test_data_backward_v1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.backward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_data_backward_hn1d(self):
        """
        ExtData test for 1d horizontal series hold backward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        dims = []
        interp = "hold backward"
        py_name = "test_data_backward_hn1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.backward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_data_backward_vn1d(self):
        """
        ExtData test for 1d vertical series hold backward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        dims = []
        interp = "hold backward"
        py_name = "test_data_backward_vn1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.backward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_data_interp_vn2d(self):
        """
        ExtData test for 2d vertical series interpolation by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        time_row_or_col = "time"
        cell = "data_2d"
        coords = {'ABC': ['A', 'B', 'C']}
        dims = ['ABC']
        interp = None
        py_name = "test_data_interp_vn2d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.interp_2d):
           self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_data_forward_hn2d(self):
        """
        ExtData test for 2d vertical series look forward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "time"
        cell = "data_2d"
        coords = {'ABC': ['A', 'B', 'C']}
        dims = ['ABC']
        interp = "look forward"
        py_name = "test_data_forward_hn2d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.forward_2d):
           self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))

    def test_data_backward_v2d(self):
        """
        ExtData test for 2d vertical series hold backward
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        time_row_or_col = "B"
        cell = "C5"
        coords = {'ABC': ['A', 'B', 'C']}
        dims = ['ABC']
        interp = "hold backward"
        py_name = "test_data_backward_v2d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.backward_2d):
           self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))

    def test_data_interp_h3d(self):
        """
        ExtData test for 3d horizontal series interpolation
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "4"
        cell_1 = "C5"
        cell_2 = "C8"
        coords_1 = {'ABC': ['A', 'B', 'C'], 'XY':['X']}
        coords_2 = {'ABC': ['A', 'B', 'C'], 'XY':['Y']}
        dims = ['XY', 'ABC']
        interp = None
        py_name = "test_data_interp_h3d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     coords=coords_1,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)
        
        data.add(file_name=file_name,
                 tab=tab,
                 time_row_or_col=time_row_or_col,
                 root=_root,
                 cell=cell_2,
                 coords=coords_2,
                 dims=dims,
                 interp=interp)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.interp_3d):
           self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))

    def test_data_forward_v3d(self):
        """
        ExtData test for 3d vertical series look forward
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        time_row_or_col = "B"
        cell_1 = "C5"
        cell_2 = "F5"
        coords_1 = {'ABC': ['A', 'B', 'C'], 'XY':['X']}
        coords_2 = {'ABC': ['A', 'B', 'C'], 'XY':['Y']}
        dims = ['XY', 'ABC']
        interp = "look forward"
        py_name = "test_data_forward_v3d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     coords=coords_1,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)
        
        data.add(file_name=file_name,
                 tab=tab,
                 time_row_or_col=time_row_or_col,
                 root=_root,
                 cell=cell_2,
                 coords=coords_2,
                 dims=dims,
                 interp=interp)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.forward_3d):
           self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_data_backward_hn3d(self):
        """
        ExtData test for 3d horizontal series hold backward by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "time"
        cell_1 = "data_2d"
        cell_2 = "data_2db"
        coords_1 = {'ABC': ['A', 'B', 'C'], 'XY':['X']}
        coords_2 = {'ABC': ['A', 'B', 'C'], 'XY':['Y']}
        dims = ['XY', 'ABC']
        interp = "hold backward"
        py_name = "test_data_backward_hn3d"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     coords=coords_1,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)
        
        data.add(file_name=file_name,
                 tab=tab,
                 time_row_or_col=time_row_or_col,
                 root=_root,
                 cell=cell_2,
                 coords=coords_2,
                 dims=dims,
                 interp=interp)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.backward_3d):
           self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))


class TestLookup(unittest.TestCase):

    def test_lookup_h1d(self):
        """
        ExtLookup test for 1d horizontal series
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        x_row_or_col = "4"
        cell = "C5"
        coords = {}
        dims = []
        py_name = "test_lookup_h1d"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       tab=tab,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell,
                                       coords=coords,
                                       dims=dims,
                                       py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_lookup_v1d(self):
        """
        ExtLookup test for 1d vertical series
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        x_row_or_col = "B"
        cell = "C5"
        coords = {}
        dims = []
        py_name = "test_lookup_v1d"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       tab=tab,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell,
                                       coords=coords,
                                       dims=dims,
                                       py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_lookup_hn1d(self):
        """
        ExtLookup test for 1d horizontal series by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        x_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        dims = []
        py_name = "test_lookup_h1nd"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       tab=tab,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell,
                                       coords=coords,
                                       dims=dims,
                                       py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_lookup_vn1d(self):
        """
        ExtLookup test for 1d vertical series by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        x_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        dims = []
        py_name = "test_lookup_vn1d"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       tab=tab,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell,
                                       coords=coords,
                                       dims=dims,
                                       py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_lookup_h2d(self):
        """
        ExtLookup test for 2d horizontal series
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        x_row_or_col = "4"
        cell = "C5"
        coords = {'ABC': ['A', 'B', 'C']}
        dims = ['ABC']
        py_name = "test_lookup_h2d"

        data = pysd.external.ExtLookup(file_name=file_name,
                                     tab=tab,
                                     x_row_or_col=x_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.interp_2d):
            self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_lookup_vn3d(self):
        """
        ExtLookup test for 3d vertical series by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        x_row_or_col = "time"
        cell_1 = "data_2d"
        cell_2 = "data_2db"
        coords_1 = {'ABC': ['A', 'B', 'C'], 'XY':['X']}
        coords_2 = {'ABC': ['A', 'B', 'C'], 'XY':['Y']}
        dims = ['XY', 'ABC']
        py_name = "test_lookup_vn3d"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       tab=tab,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       dims=dims,
                                       py_name=py_name)
        
        data.add(file_name=file_name,
                 tab=tab,
                 x_row_or_col=x_row_or_col,
                 root=_root,
                 cell=cell_2,
                 coords=coords_2,
                 dims=dims)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.interp_3d):
           self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))


class TestConstant(unittest.TestCase):

    def test_constant_0d(self):
        """
        ExtConstant test for 0d data
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        cell = "F7"
        cell2 = "C5"
        coords = {}
        dims = []
        py_name = "test_constant_0d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)

        data2 = pysd.external.ExtConstant(file_name=file_name,
                                          tab=tab,
                                          root=_root,
                                          cell=cell2,
                                          coords=coords,
                                          dims=dims,
                                          py_name=py_name)
        data.initialize()
        data2.initialize()

        self.assertEqual(data.data, -1)
        self.assertEqual(data2.data, 0)

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_constant_n0d(self):
        """
        ExtConstant test for 0d data by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        cell = "constant"
        cell2 = "constant2"
        coords = {}
        dims = []
        py_name = "test_constant_0d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)

        data2 = pysd.external.ExtConstant(file_name=file_name,
                                          tab=tab,
                                          root=_root,
                                          cell=cell2,
                                          coords=coords,
                                          dims=dims,
                                          py_name=py_name)
        data.initialize()
        data2.initialize()

        self.assertEqual(data.data, -1)
        self.assertEqual(data2.data, 0)
      
    def test_constant_h1d(self):
        """
        ExtConstant test for horizontal 1d data
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        cell = "C5"
        coords = {'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        dims = ['val']
        py_name = "test_constant_h1d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data.data.equals(_exp.constant_1d))

    def test_constant_v1d(self):
        """
        ExtConstant test for vertical 1d data
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        cell = "C5*"
        coords = {'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        dims = ['val']
        py_name = "test_constant_v1d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data.data.equals(_exp.constant_1d))

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_constant_hn1d(self):
        """
        ExtConstant test for horizontal 1d data by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        cell = "data_1d"
        coords = {'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        dims = ['val']
        py_name = "test_constant_hn1d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data.data.equals(_exp.constant_1d))

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_constant_vn1d(self):
        """
        ExtConstant test for vertical 1d data by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        cell = "data_1d*"
        coords = {'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        dims = ['val']
        py_name = "test_constant_vn1d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data.data.equals(_exp.constant_1d))

    def test_constant_h2d(self):
        """
        ExtConstant test for horizontal 2d data
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        cell = "C5"
        coords = {'ABC': ['A', 'B', 'C'], 'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        dims = ['ABC', 'val']
        py_name = "test_constant_h2d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data.data.equals(_exp.constant_2d))

    def test_constant_v2d(self):
        """
        ExtConstant test for vertical 2d data
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        cell = "C5*"
        coords = {'ABC': ['A', 'B', 'C'], 'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        dims = ['ABC', 'val']
        py_name = "test_constant_v2d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data.data.equals(_exp.constant_2d))

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_constant_hn2d(self):
        """
        ExtConstant test for horizontal 2d data by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        cell = "data_2d"
        coords = {'ABC': ['A', 'B', 'C'], 'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        dims = ['ABC', 'val']
        py_name = "test_constant_hn2d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data.data.equals(_exp.constant_2d))

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_constant_vn2d(self):
        """
        ExtConstant test for vertical 2d data by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        cell = "data_2d*"
        coords = {'ABC': ['A', 'B', 'C'], 'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        dims = ['ABC', 'val']
        py_name = "test_constant_vn2d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)
        data.initialize()

        self.assertTrue(data.data.equals(_exp.constant_2d))

    def test_constant_h3d(self):
        """
        ExtConstant test for horizontal 3d data
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        cell = "C5"
        cell2 = "C8"
        coords = {'XY': ['X'],
                  'ABC': ['A', 'B', 'C'],
                  'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        coords2 = {'XY': ['Y'],
                   'ABC': ['A', 'B', 'C'],
                   'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        dims = ['ABC', 'XY', 'val']
        py_name = "test_constant_h3d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)
        
        data.add(file_name=file_name,
                 tab=tab,
                 root=_root,
                 cell=cell2,
                 coords=coords2,
                 dims=dims)

        data.initialize()

        self.assertTrue(data.data.equals(_exp.constant_3d))

    def test_constant_v3d(self):
        """
        ExtConstant test for vertical 3d data
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        cell = "C5*"
        cell2 = "F5*"
        coords = {'XY': ['X'],
                  'ABC': ['A', 'B', 'C'],
                  'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        coords2 = {'XY': ['Y'],
                   'ABC': ['A', 'B', 'C'],
                   'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        dims = ['ABC', 'XY', 'val']
        py_name = "test_constant_v3d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)
        
        data.add(file_name=file_name,
                 tab=tab,
                 root=_root,
                 cell=cell2,
                 coords=coords2,
                 dims=dims)

        data.initialize()

        self.assertTrue(data.data.equals(_exp.constant_3d))

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_constant_hn3d(self):
        """
        ExtConstant test for horizontal 3d data by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        cell = "data_2d"
        cell2 = "data_2db"
        coords = {'XY': ['X'],
                  'ABC': ['A', 'B', 'C'],
                  'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        coords2 = {'XY': ['Y'],
                   'ABC': ['A', 'B', 'C'],
                   'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        dims = ['ABC', 'XY', 'val']
        py_name = "test_constant_hn3d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)
        
        data.add(file_name=file_name,
                 tab=tab,
                 root=_root,
                 cell=cell2,
                 coords=coords2,
                 dims=dims)

        data.initialize()

        self.assertTrue(data.data.equals(_exp.constant_3d))

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_constant_vn3d(self):
        """
        ExtConstant test for vertical 3d data by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical"
        cell = "data_2d*"
        cell2 = "data_2db*"
        coords = {'XY': ['X'],
                  'ABC': ['A', 'B', 'C'],
                  'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        coords2 = {'XY': ['Y'],
                   'ABC': ['A', 'B', 'C'],
                   'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        dims = ['ABC', 'XY', 'val']
        py_name = "test_constant_vn2d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)
        
        data.add(file_name=file_name,
                 tab=tab,
                 root=_root,
                 cell=cell2,
                 coords=coords2,
                 dims=dims)

        data.initialize()

        self.assertTrue(data.data.equals(_exp.constant_3d))


class TestSubscript(unittest.TestCase):

    def test_subscript_h(self):
        """
        ExtSubscript test for horizontal subscripts
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        firstcell = "C4"
        lastcell = "J4"
        prefix = 'val'
        expected = ['val0', 'val1', 'val2', 'val3', 
                    'val5', 'val6', 'val7', 'val8']

        data = pysd.external.ExtSubscript(file_name=file_name,
                                          tab=tab,
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
        tab = "Horizontal"
        firstcell = "B5"
        lastcell = "B7"
        prefix = ''
        expected = ['A', 'B', 'C']

        data = pysd.external.ExtSubscript(file_name=file_name,
                                          tab=tab,
                                          firstcell=firstcell,
                                          lastcell=lastcell,
                                          prefix=prefix)

        self.assertTrue(data.subscript, expected)

# TODO add test as the previous ones but with missing and NaN values

