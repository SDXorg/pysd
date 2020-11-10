import os
import unittest

import numpy as np

_root = os.path.dirname(__file__)

class TestData(unittest.TestCase):

    def test_data_interp_h1d(self):
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
        import pysd
        from data.expected_data import xpts, forward_1d

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        dims = []
        interp = "look forward"
        py_name = "test_forward_h1d"

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
        import pysd
        from data.expected_data import xpts, forward_1d

        file_name = "data/input.xlsx"
        tab = "Vertical"
        time_row_or_col = "B"
        cell = "C5"
        coords = {}
        dims = []
        interp = "look forward"
        py_name = "test_forward_v1d"

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
        import pysd
        from data.expected_data import xpts, forward_1d

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        dims = []
        interp = "look forward"
        py_name = "test_forward_hn1d"

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
        import pysd
        from data.expected_data import xpts, forward_1d

        file_name = "data/input.xlsx"
        tab = "Vertical"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        dims = []
        interp = "look forward"
        py_name = "test_forward_vn1d"

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
        import pysd
        from data.expected_data import xpts, backward_1d

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        dims = []
        interp = "hold backward"
        py_name = "test_backward_h1d"

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
        import pysd
        from data.expected_data import xpts, backward_1d

        file_name = "data/input.xlsx"
        tab = "Vertical"
        time_row_or_col = "B"
        cell = "C5"
        coords = {}
        dims = []
        interp = "hold backward"
        py_name = "test_backward_v1d"

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
        import pysd
        from data.expected_data import xpts, backward_1d

        file_name = "data/input.xlsx"
        tab = "Horizontal"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        dims = []
        interp = "hold backward"
        py_name = "test_backward_hn1d"

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
        import pysd
        from data.expected_data import xpts, backward_1d

        file_name = "data/input.xlsx"
        tab = "Vertical"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        dims = []
        interp = "hold backward"
        py_name = "test_backward_vn1d"

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
 

