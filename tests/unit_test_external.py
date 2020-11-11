import os
import unittest

import numpy as np

_root = os.path.dirname(__file__)

class TestData(unittest.TestCase):

    def test_data_interp_h1d(self):
        """
        ExtData test for 1d horizontal series interpolation
        """
        import pysd
        from data.expected_data import xpts, interp_1d

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

        for x, y in zip(xpts, interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_interp_v1d(self):
        """
        ExtData test for 1d vertical series interpolation
        """
        import pysd
        from data.expected_data import xpts, interp_1d

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

        for x, y in zip(xpts, interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_interp_hn1d(self):
        """
        ExtData test for 1d horizontal series interpolation by cellrange names
        """
        import pysd
        from data.expected_data import xpts, interp_1d

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

        for x, y in zip(xpts, interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_interp_vn1d(self):
        """
        ExtData test for 1d vertical series interpolation by cellrange names
        """
        import pysd
        from data.expected_data import xpts, interp_1d

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

        for x, y in zip(xpts, interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_data_forward_h1d(self):
        """
        ExtData test for 1d horizontal series look forward
        """
        import pysd
        from data.expected_data import xpts, forward_1d

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

        for x, y in zip(xpts, forward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 
 
    def test_data_forward_v1d(self):
        """
        ExtData test for 1d vertical series look forward
        """
        import pysd
        from data.expected_data import xpts, forward_1d

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

        for x, y in zip(xpts, forward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 
 
    def test_data_forward_hn1d(self):
        """
        ExtData test for 1d horizontal series look forward by cell range names
        """
        import pysd
        from data.expected_data import xpts, forward_1d

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

        for x, y in zip(xpts, forward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 
 
    def test_data_forward_vn1d(self):
        """
        ExtData test for 1d vertical series look forward by cell range names
        """
        import pysd
        from data.expected_data import xpts, forward_1d

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

        for x, y in zip(xpts, forward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 
 
    def test_data_backward_h1d(self):
        """
        ExtData test for 1d horizontal series hold backward
        """
        import pysd
        from data.expected_data import xpts, backward_1d

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

        for x, y in zip(xpts, backward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 
  
    def test_data_backward_v1d(self):
        """
        ExtData test for 1d vertical series hold backward by cell range names
        """
        import pysd
        from data.expected_data import xpts, backward_1d

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

        for x, y in zip(xpts, backward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 
 
    def test_data_backward_hn1d(self):
        """
        ExtData test for 1d horizontal series hold backward by cell range names
        """
        import pysd
        from data.expected_data import xpts, backward_1d

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

        for x, y in zip(xpts, backward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 
 
    def test_data_backward_vn1d(self):
        """
        ExtData test for 1d vertical series hold backward by cell range names
        """
        import pysd
        from data.expected_data import xpts, backward_1d

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

        for x, y in zip(xpts, backward_1d):
           self.assertEqual(y, data(x), "Wrong result at X=" + str(x)) 

    def test_data_interp_vn2d(self):
        """
        ExtData test for 2d vertical series interpolation by cell range names
        """
        import pysd
        from data.expected_data import xpts, interp_2d

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

        for x, y in zip(xpts, interp_2d):
           self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))

    def test_data_forward_hn2d(self):
        """
        ExtData test for 2d vertical series look forward by cell range names
        """
        import pysd
        from data.expected_data import xpts, forward_2d

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

        for x, y in zip(xpts, forward_2d):
           self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))

    def test_data_backward_v2d(self):
        """
        ExtData test for 2d vertical series hold backward
        """
        import pysd
        from data.expected_data import xpts, backward_2d

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

        for x, y in zip(xpts, backward_2d):
           self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))

    def test_data_interp_h3d(self):
        """
        ExtData test for 3d horizontal series interpolation
        """
        import pysd
        from data.expected_data import xpts, interp_3d

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

        for x, y in zip(xpts, interp_3d):
           self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))

    def test_data_forward_v3d(self):
        """
        ExtData test for 3d vertical series look forward
        """
        import pysd
        from data.expected_data import xpts, forward_3d

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

        for x, y in zip(xpts, forward_3d):
           self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))

    def test_data_backward_hn3d(self):
        """
        ExtData test for 3d horizontal series hold backward by cellrange names
        """
        import pysd
        from data.expected_data import xpts, backward_3d

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

        for x, y in zip(xpts, backward_3d):
           self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))


class TestLookup(unittest.TestCase):

    def test_lookup_h1d(self):
        """
        ExtLookup test for 1d horizontal series
        """
        import pysd
        from data.expected_data import xpts, interp_1d

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

        for x, y in zip(xpts, interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_lookup_v1d(self):
        """
        ExtLookup test for 1d vertical series
        """
        import pysd
        from data.expected_data import xpts, interp_1d

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

        for x, y in zip(xpts, interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_lookup_hn1d(self):
        """
        ExtLookup test for 1d horizontal series by cellrange names
        """
        import pysd
        from data.expected_data import xpts, interp_1d

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

        for x, y in zip(xpts, interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_lookup_vn1d(self):
        """
        ExtLookup test for 1d vertical series by cellrange names
        """
        import pysd
        from data.expected_data import xpts, interp_1d

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

        for x, y in zip(xpts, interp_1d):
            self.assertEqual(y, data(x), "Wrong result at X=" + str(x))

    def test_lookup_h2d(self):
        """
        ExtLookup test for 2d horizontal series
        """
        import pysd
        from data.expected_data import xpts, interp_2d

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

        for x, y in zip(xpts, interp_2d):
            self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))

    def test_lookup_vn3d(self):
        """
        ExtLookup test for 3d vertical series by cellrange names
        """
        import pysd
        from data.expected_data import xpts, interp_3d

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

        for x, y in zip(xpts, interp_3d):
           self.assertTrue(y.equals(data(x)), "Wrong result at X=" + str(x))


