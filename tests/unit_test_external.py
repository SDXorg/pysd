import os
import sys
import imp
import unittest

import numpy as np

_root = os.path.dirname(__file__)
_py_version = sys.version_info[0]*10 + sys.version_info[1]
_exp = imp.load_source('expected_data', 'data/expected_data.py')

class TestExcels(unittest.TestCase):
    """
    Tests for Excels class
    """
    def test_read_clean(self):
        """
        Test for reading files with pandas
        """
        import pysd
        import pandas as pd

        file_name = "data/input.xlsx"

        # reading a file
        excel = pysd.external.Excels.read(file_name)
        self.assertTrue(isinstance(excel, pd.ExcelFile))

        # check if it is in the dictionary
        self.assertEqual(list(pysd.external.Excels._Excels),
                         [file_name])

        pysd.external.Excels.read(file_name)
        self.assertEqual(list(pysd.external.Excels._Excels),
                         [file_name])

        # clean
        pysd.external.Excels.clean()
        self.assertEqual(list(pysd.external.Excels._Excels),
                         [])

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")       
    def test_read_clean_opyxl(self):
        """
        Test for reading files with openpyxl
        """
        import pysd
        from openpyxl import Workbook

        file_name = "data/input.xlsx"

        # reading a file
        excel = pysd.external.Excels.read_opyxl(file_name)
        self.assertTrue(isinstance(excel, Workbook))

        # check if it is in the dictionary
        self.assertEqual(list(pysd.external.Excels._Excels_opyxl),
                         [file_name])

        pysd.external.Excels.read_opyxl(file_name)
        self.assertEqual(list(pysd.external.Excels._Excels_opyxl),
                         [file_name])

        # clean                
        pysd.external.Excels.clean()
        self.assertEqual(list(pysd.external.Excels._Excels_opyxl),
                         [])     



class TestExternalMethods(unittest.TestCase):
    """
    Test for simple methods of External
    """

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
        import xarray as xr

        # test as well no file extension
        file_name = "data/input"
        tab = "Horizontal"
        time_row_or_col = "16"
        cell = "B17"
        coords = {}
        dims = []
        interp = None
        py_name = "test_data_interp_h1d_1"

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
        
        # test the __str__ method
        print(data)

        self.assertTrue(data.data.equals(xr.DataArray([5],{'time':[4]},['time'])))
        
    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_data_interp_hn1d_1(self):
        """
        ExtData test for 1d horizontal series interpolation with len 1
        """
        import pysd
        import xarray as xr

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "time_1"
        cell = "data_1"
        coords = {}
        dims = []
        interp = None
        py_name = "test_data_interp_h1d_1"

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

        self.assertTrue(data.data.equals(xr.DataArray([5],{'time':[4]},['time'])))
            
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

    def test_data_raw_h1d(self):
        """
        ExtData test for 1d horizontal series raw
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        dims = []
        interp = "raw"
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

        self.assertEqual(data(), -1)
        self.assertEqual(data2(), 0)

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

        self.assertEqual(data(), -1)
        self.assertEqual(data2(), 0)
      
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

        self.assertTrue(data().equals(_exp.constant_1d))

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

        self.assertTrue(data().equals(_exp.constant_1d))

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

        self.assertTrue(data().equals(_exp.constant_1d))

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

        self.assertTrue(data().equals(_exp.constant_1d))

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

        self.assertTrue(data().equals(_exp.constant_2d))

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

        self.assertTrue(data().equals(_exp.constant_2d))

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

        self.assertTrue(data().equals(_exp.constant_2d))

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

        self.assertTrue(data().equals(_exp.constant_2d))

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

        self.assertTrue(data().equals(_exp.constant_3d))

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

        self.assertTrue(data().equals(_exp.constant_3d))

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

        self.assertTrue(data().equals(_exp.constant_3d))

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
        tab = "Horizontal"
        firstcell = "C4"
        lastcell = "J4"
        prefix = 'val'
        expected = ['val0', 'val1', 'val2', 'val3', 
                    'val5', 'val6', 'val7', 'val8']

        data = pysd.external.ExtSubscript(file_name=file_name,
                                          tab=tab,
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
        tab = "Horizontal"
        firstcell = "B5"
        lastcell = "B7"
        prefix = ''
        expected = ['A', 'B', 'C']

        data = pysd.external.ExtSubscript(file_name=file_name,
                                          tab=tab,
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
        tab = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        dims = []
        interp = None
        py_name = "test_not_implemented_file"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
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
        tab = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        dims = []
        interp = None
        py_name = "test_non_existent_file"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)
                                     
        with self.assertRaises(IOError):
            data.initialize()

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_non_existent_sheet_pyxl(self):
        """
        Test for non-existent sheet with openpyxl
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Non-Existent"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        dims = []
        interp = None
        py_name = "test_non_existent_sheet_pyxl"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)
                                     
        with self.assertRaises(ValueError):
            data.initialize()

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_non_existent_cellrange_name_pyxl(self):
        """
        Test for non-existent cellrange name with openpyxl
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "time"
        cell = "non_exixtent"
        coords = {}
        dims = []
        interp = None
        py_name = "est_non_existent_cellrange_name_pyxl"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)
                                     
        with self.assertRaises(AttributeError):
            data.initialize()
  
    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_non_existent_cellrange_name_in_sheet_pyxl(self):
        """
        Test for non-existent cellrange name with openpyxl
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal missing"
        cell = "constant"
        coords = {}
        dims = []
        py_name = "est_non_existent_cellrange_name_in_sheet_pyxl"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)
                                     
        with self.assertRaises(AttributeError):
            data.initialize()
    @unittest.skipIf(_py_version >= 36,
                     "openpyxl only supported for Python >= 3.6")
    def test_not_able_to_import_openpyxl(self):
        """
        Test for the warining raised after trying to import openpyxl in Python < 3.6
        """
        from warnings import catch_warnings
        
        with catch_warnings(record=True) as w:
            from pysd import external
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue("openpyxl" in str(w[-1].message))

    # Following test are for ExtData class only
    # as the initialization of ExtLookup uses the same function
    
    @unittest.skipIf(_py_version < 36,
                     "more warnings are arised for Python < 3.6")
    def test_data_interp_h1dm(self):
        """
        Test for warning 1d horizontal series interpolation with missing data
        """
        import pysd
        from warnings import catch_warnings

        file_name = "data/input.xlsx"
        tab = "Horizontal missing"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        dims = []
        interp = None
        py_name = "test_data_interp_h1dm"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        with catch_warnings(record=True) as w:
            data.initialize()
            wu = [wus for wus in w if issubclass(wus.category, UserWarning)]
            self.assertEqual(len(wu), 1)
            self.assertTrue("missing" in str(wu[-1].message))

        print(data.data)
        for x, y in zip(_exp.xpts, _exp.interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))


    @unittest.skipIf(_py_version < 36,
                     "more warnings are arised for Python < 3.6")
    def test_data_interp_v1dm(self):
        """
        Test for warning 1d vertical series interpolation with missing data
        """
        import pysd
        from warnings import catch_warnings

        file_name = "data/input.xlsx"
        tab = "Vertical missing"
        time_row_or_col = "B"
        cell = "C5"
        coords = {}
        dims = []
        interp = None
        py_name = "test_data_interp_v1dm"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        with catch_warnings(record=True) as w:
            data.initialize()
            self.assertEqual(len(w), 1)       
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue("missing" in str(w[-1].message))

        for x, y in zip(_exp.xpts, _exp.interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))
 
    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")      
    def test_data_interp_hn1dm(self):
        """
        Test for warning 1d horizontal series by cellrange names
        when series has missing or NaN data
        """
        import pysd
        from warnings import catch_warnings

        file_name = "data/input.xlsx"
        tab = "Horizontal missing"
        time_row_or_col = "time_missing"
        cell = "data_1d"
        coords = {}
        dims = []
        interp = None
        py_name = "test_data_interp_h1dm"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        with catch_warnings(record=True) as w:
            data.initialize()
            self.assertEqual(len(w), 1)       
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue("missing" in str(w[-1].message))

        for x, y in zip(_exp.xpts, _exp.interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_interp_h1d0(self):
        """
        Test for error 1d horizontal series for len 0 series
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal missing"
        time_row_or_col = "3"
        cell = "C5"
        coords = {}
        dims = []
        interp = None
        py_name = "test_data_interp_h1d0"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
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
        tab = "Vertical missing"
        time_row_or_col = "A"
        cell = "C5"
        coords = {}
        dims = []
        interp = None
        py_name = "test_data_interp_v1d0"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        
        with self.assertRaises(ValueError):
            data.initialize()
 
 
    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")      
    def test_data_interp_hn1d0(self):
        """
        Test for error in series by cellrange names
        when series has length 0
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal missing"
        time_row_or_col = "len_0"
        cell = "data_1d"
        coords = {}
        dims = []
        interp = None
        py_name = "test_data_interp_h1d0"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()

    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")      
    def test_data_interp_hn1dt(self):
        """
        Test for error in series by cellrange names
        when series is a table
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "data_2d"
        cell = "data_1d"
        coords = {}
        dims = []
        interp = None
        py_name = "test_data_interp_h1dt"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()
            
    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")              
    def test_data_interp_hns(self):
        """
        Test for error in data when it doen't have the shame
        shape as the given coordinates
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "time"
        cell = "data_2d"
        coords = {'ABC': ['A', 'B']}
        dims = ['ABC']
        interp = None
        py_name = "test_data_interp_hns"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()
     
    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")              
    def test_data_interp_vnss(self):
        """
        Test for error in data when it doen't have the shame
        shape in the first dimension as the length of series
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Vertical missing"
        time_row_or_col = "time_short"
        cell = "data_2d_short"
        coords = {'ABC': ['A', 'B', 'C']}
        dims = ['ABC']
        interp = None
        py_name = "test_data_interp_vnss"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()

    # Following test are independent of the reading option
    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")              
    def test_data_interp_hnnm(self):
        """
        Test for error in series when the series is not
        strictly monotonous
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "No monotonous"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        dims = []
        interp = None
        py_name = "test_data_interp_hnnm"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()

    def test_data_h3d_interpnv(self):
        """
        ExtData test for error when the interpolation method is not valid
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords= {'ABC': ['A', 'B', 'C'], 'XY':['X']}
        dims = ['XY', 'ABC']
        interp = "hold forward"
        py_name = "test_data_h3d_interpnv"

        with self.assertRaises(ValueError):
            data = pysd.external.ExtData(file_name=file_name,
                                         tab=tab,
                                         time_row_or_col=time_row_or_col,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         interp=interp,
                                         py_name=py_name)
                
    def test_data_h3d_interp(self):
        """
        ExtData test for error when the interpolation method is different
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
        interp2 ="look forward"
        py_name = "test_data_h3d_interp"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     coords=coords_1,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)
        
        with self.assertRaises(ValueError):
            data.add(file_name=file_name,
                     tab=tab,
                     time_row_or_col=time_row_or_col,
                     root=_root,
                     cell=cell_2,
                     coords=coords_2,
                     dims=dims,
                     interp=interp2)
                     
    def test_data_h3d_add(self):
        """
        ExtData test for error when add doesn't have the same dim
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
        dims2 = ['ABC', 'Xy']
        interp = None
        py_name = "test_data_h3d_add"

        data = pysd.external.ExtData(file_name=file_name,
                                     tab=tab,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     coords=coords_1,
                                     dims=dims,
                                     interp=interp,
                                     py_name=py_name)
        
        with self.assertRaises(ValueError):
            data.add(file_name=file_name,
                     tab=tab,
                     time_row_or_col=time_row_or_col,
                     root=_root,
                     cell=cell_2,
                     coords=coords_2,
                     dims=dims2,
                     interp=interp)


    def test_lookup_h3d_add(self):
        """
        ExtLookup test for error when add doesn't have the same dim
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        x_row_or_col = "4"
        cell_1 = "C5"
        cell_2 = "C8"
        coords_1 = {'ABC': ['A', 'B', 'C'], 'XY':['X']}
        coords_2 = {'ABC': ['A', 'B', 'C'], 'XY':['Y']}
        dims = ['XY', 'ABC']
        dims2 = ['ABC']
        py_name = "test_lookup_h3d_add"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       tab=tab,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       dims=dims,
                                       py_name=py_name)
        
        with self.assertRaises(ValueError):
            data.add(file_name=file_name,
                     tab=tab,
                     x_row_or_col=x_row_or_col,
                     root=_root,
                     cell=cell_2,
                     coords=coords_2,
                     dims=dims2)

    def test_constant_h3d_add(self):
        """
        ExtConstant test for error when add doesn't have the same dim
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
        dims2 = ['ABC', 'XY', 'val2']
        py_name = "test_constant_h3d_add"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)
        
        with self.assertRaises(ValueError):
            data.add(file_name=file_name,
                     tab=tab,
                     root=_root,
                     cell=cell2,
                     coords=coords2,
                     dims=dims2)
        
    def test_lookup_h1d_nf(self):
        """
        Error in ExtLookup when a non 0 dimensional array is passed
        """
        import pysd
        import xarray as xr

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
        
        data(np.array([1]))
        data(xr.DataArray([1]))
        
        with self.assertRaises(TypeError):
            data(np.array([1, 2]))
            data(xr.DataArray([1, 1]))


            
    @unittest.skipIf(_py_version < 36,
                     "openpyxl only supported for Python >= 3.6")     
                              
    def test_constant_hns(self):
        """
        Test for error in data when it doen't have the shame
        shape as the given coordinates
        """
        import pysd

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        cell = "data_2d"
        coords = {'ABC': ['A', 'B']}
        dims = ['ABC']
        interp = None
        py_name = "test_constant_hns"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         tab=tab,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         dims=dims,
                                         py_name=py_name)

        with self.assertRaises(ValueError):
            data.initialize()
  

