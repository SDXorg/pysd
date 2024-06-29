import sys
import re
import importlib.util

import pytest

import numpy as np
import xarray as xr


@pytest.fixture(scope="session")
def _exp(_root):
    spec = importlib.util.spec_from_file_location(
        'expected_data', _root.joinpath('data/expected_data.py'))
    expected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(expected)
    return expected


class TestExcels():
    """
    Tests for Excels class
    """
    def test_read_clean(self, _root):
        """
        Test for reading files with pandas
        """
        from pysd.py_backend.external import Excels

        file_name = _root.joinpath("data/input.xlsx")
        sheet_name = "Vertical"
        sheet_name2 = "Horizontal"

        # reading a file
        excel = Excels.read(file_name, sheet_name)
        assert isinstance(excel, np.ndarray)

        # check if it is in the dictionary
        assert file_name.joinpath(sheet_name) in list(Excels._Excels)

        Excels.read(file_name, sheet_name2)
        assert file_name.joinpath(sheet_name2) in list(Excels._Excels)

        # clean
        Excels.clean()
        assert list(Excels._Excels) == []

    def test_read_clean_opyxl(self, _root):
        """
        Test for reading files with openpyxl
        """
        from pysd.py_backend.external import Excels
        from openpyxl import Workbook

        file_name = _root.joinpath("data/input.xlsx")

        # reading a file
        excel = Excels.read_opyxl(file_name)
        assert isinstance(excel, Workbook)

        # check if it is in the dictionary
        assert list(Excels._Excels_opyxl) == [file_name]

        Excels.read_opyxl(file_name)
        assert list(Excels._Excels_opyxl) == [file_name]

        # clean
        Excels.clean()
        assert list(Excels._Excels_opyxl) == []

    @pytest.mark.skip(
        reason="may fail in parallel running"
    )
    def test_close_file(self, _root):
        """
        Test for checking if excel files were closed
        """
        from pysd.py_backend.external import Excels
        import psutil

        p = psutil.Process()

        # number of files already open
        n_files = len(p.open_files())

        file_name = _root.joinpath("data/input.xlsx")
        sheet_name = "Vertical"
        sheet_name2 = "Horizontal"

        # reading files
        Excels.read(file_name, sheet_name)
        Excels.read(file_name, sheet_name2)
        Excels.read_opyxl(file_name)

        assert len(p.open_files()) > n_files

        # clean
        Excels.clean()
        assert len(p.open_files()) == n_files


class TestExternalMethods():
    """
    Test for simple methods of External
    """

    def test_col_to_num(self, _root):
        """
        External._num_to_col and External._col_to_num test
        """
        from pysd.py_backend.external import External

        col_to_num = External._col_to_num

        # Check col_to_num
        assert col_to_num("A") == 0
        assert col_to_num("Z") == 25
        assert col_to_num("a") == col_to_num("B")-1
        assert col_to_num("Z") == col_to_num("aa")-1
        assert col_to_num("Zz") == col_to_num("AaA")-1

    def test_split_excel_cell(self, _root):
        """
        External._split_excel_cell test
        """
        from pysd.py_backend.external import External

        ext = External('external')

        # No cells, function must return nothing
        nocells = ["A2A", "H0", "0", "5A", "A_1", "ZZZZ1", "A"]

        for nocell in nocells:
            assert not ext._split_excel_cell(nocell)

        # Cells
        cells = [(1, 0, "A2"), (573, 7, "h574"),
                 (1, 572, "Va2"), (1, 728, "ABA2")]

        for row, col, cell in cells:
            assert (row, col) == ext._split_excel_cell(cell)

    def test_reshape(self, _root):
        """
        External._reshape test
        """
        from pysd.py_backend.external import External

        reshape = External._reshape

        data0d = np.array(5)
        data1d = np.array([2, 3, 5, 6])
        data2d = np.array([[2, 3, 5, 6],
                           [1, 7, 5, 8]])

        float0d = float(data0d)
        int0d = int(data0d)
        series1d = xr.DataArray(data1d)
        df2d = xr.DataArray(data2d)

        shapes0d = [(1,), (1, 1)]
        shapes1d = [(4,), (4, 1, 1), (1, 1, 4), (1, 4, 1)]
        shapes2d = [(2, 4), (2, 4, 1), (1, 2, 4), (2, 1, 4)]

        for shape_i in shapes0d:
            assert reshape(data0d, shape_i).shape == shape_i
            assert reshape(float0d, shape_i).shape == shape_i
            assert reshape(int0d, shape_i).shape == shape_i

        for shape_i in shapes1d:
            assert reshape(data1d, shape_i).shape == shape_i
            assert reshape(series1d, shape_i).shape == shape_i

        for shape_i in shapes2d:
            assert reshape(data2d, shape_i).shape == shape_i
            assert reshape(df2d, shape_i).shape == shape_i

    def test_series_selector(self, _root):
        """
        External._series_selector test
        """
        from pysd.py_backend.external import External

        ext = External('external')

        # row selector
        assert ext._series_selector("12", "A5") == "row"

        # column selector
        assert ext._series_selector("A", "a44") == "column"
        assert ext._series_selector("A", "AC44") == "column"
        assert ext._series_selector("A", "Bae2") == "column"

        # name selector
        assert ext._series_selector("Att", "a44b") == "name"
        assert ext._series_selector("Adfs", "a0") == "name"
        assert ext._series_selector("Ae_23", "aa_44") == "name"
        assert ext._series_selector("Aeee3", "3a") == "name"
        assert ext._series_selector("Aeee", "aajh2") == "name"

    def test_fill_missing(self, _root):
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

        ext.interp = "hold_backward"
        datac = data.copy()
        ext._fill_missing(series, datac)
        assert np.all(hold_back == datac)

        ext.interp = "look_forward"
        datac = data.copy()
        ext._fill_missing(series, datac)
        assert np.all(look_for == datac)

        ext.interp = "interpolate"
        datac = data.copy()
        ext._fill_missing(series, datac)
        assert np.all(interp == datac)

    def test_resolve_file(self, _root):
        """
        External._resolve_file
        """
        from pysd.py_backend.external import External

        ext = External('external')
        ext.file = 'data/input.xlsx'
        ext._resolve_file(root=_root)

        assert ext.file == _root.joinpath('data/input.xlsx')

        root = _root.joinpath('data')
        ext.file = 'input.xlsx'
        ext._resolve_file(root=root)

        assert ext.file == root.joinpath('input.xlsx')

        # assert file path is properly 'resolved'
        ext.file = '../data/input.xlsx'
        ext._resolve_file(root=root)

        assert ext.file == root.joinpath('input.xlsx')

        ext.file = 'input2.xlsx'

        error_message = r"File '.*%s' not found." % 'input2.xlsx'
        with pytest.raises(FileNotFoundError, match=error_message):
            ext._resolve_file(root=root)

        # TODO in the future we may add an option to include indirect
        # references with ?. By the moment an error is raised
        ext.file = '?input.xlsx'
        error_message = r"Indirect reference to file: '\?input\.xlsx'"
        with pytest.raises(ValueError, match=error_message):
            ext._resolve_file(root=_root)

    @pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="not working on Windows"
    )
    def test_read_empty_cells_openpyxl(self, _root):
        from pysd.py_backend.external import External

        ext = External("external")
        ext.file = "data/input.xlsx"
        ext.sheet = "Horizontal missing"
        ext._resolve_file(root=_root)

        error_message = r"external\nThe cells are empty\.\n"\
            + ext._file_sheet
        with pytest.raises(ValueError, match=error_message):
            ext._get_data_from_file_opyxl("empty_cell")
        with pytest.raises(ValueError, match=error_message):
            ext._get_data_from_file_opyxl("empty_array")
        with pytest.raises(ValueError, match=error_message):
            ext._get_data_from_file_opyxl("empty_array2")
        with pytest.raises(ValueError, match=error_message):
            ext._get_data_from_file_opyxl("empty_table")

    def test_read_empty_cells_pandas(self, _root):
        from pysd.py_backend.external import External

        ext = External("external")
        ext.file = "data/input.xlsx"
        ext.sheet = "Horizontal missing"
        ext._resolve_file(root=_root)

        error_message = r"external\nThe cells are empty\.\n"
        with pytest.raises(ValueError, match=error_message):
            ext._get_data_from_file((100, 101), (100, 101))
        with pytest.raises(ValueError, match=error_message):
            ext._get_data_from_file((100, 101), (100, 140))
        with pytest.raises(ValueError, match=error_message):
            ext._get_data_from_file((100, 140), (100, 101))
        with pytest.raises(ValueError, match=error_message):
            ext._get_data_from_file((100, 140), (100, 150))
        with pytest.raises(ValueError, match=error_message):
            ext._get_data_from_file((2, 20), (100, 150))
        with pytest.raises(ValueError, match=error_message):
            ext._get_data_from_file((100, 140), (2, 20))


class TestData():
    """
    Test for the full working procedure of ExtData
    class when the data is properly given in the Excel file
    For 1D data all cases are computed.
    For 2D, 3D only some cases are computed as the complete set of
    test will cover all the possibilities.
    """

    # The first two test are for length 0 series and only the retrieved data is
    # calculated as the interpolation result will be constant
    def test_data_interp_h1d_1(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        # test the __str__ method
        print(data)

        expected = xr.DataArray([5], {'time': [4]}, ['time'])

        assert data.data.equals(expected)

    def test_data_interp_hn1d_1(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        expected = xr.DataArray([5], {'time': [4]}, ['time'])

        assert data.data.equals(expected)

    def test_data_interp_h1d(self, _root, _exp):
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
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_interp_v1d(self, _root, _exp):
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
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                y == data(x), "Wrong result at X=" + str(x)

    def test_data_interp_hn1d(self, _root, _exp):
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
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                y == data(x), "Wrong result at X=" + str(x)

    def test_data_interp_vn1d(self, _root, _exp):
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
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_forward_h1d(self, _root, _exp):
        """
        ExtData test for 1d horizontal series look_forward
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        interp = "look_forward"
        py_name = "test_data_forward_h1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.forward_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_forward_v1d(self, _root, _exp):
        """
        ExtData test for 1d vertical series look_forward
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        time_row_or_col = "B"
        cell = "C5"
        coords = {}
        interp = "look_forward"
        py_name = "test_data_forward_v1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.forward_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_forward_hn1d(self, _root, _exp):
        """
        ExtData test for 1d horizontal series look_forward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        interp = "look_forward"
        py_name = "test_data_forward_hn1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.forward_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_forward_vn1d(self, _root, _exp):
        """
        ExtData test for 1d vertical series look_forward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        interp = "look_forward"
        py_name = "test_data_forward_vn1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.forward_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_backward_h1d(self, _root, _exp):
        """
        ExtData test for 1d horizontal series hold_backward
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "4"
        cell = "C5"
        coords = {}
        interp = "hold_backward"
        py_name = "test_data_backward_h1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.backward_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_backward_v1d(self, _root, _exp):
        """
        ExtData test for 1d vertical series hold_backward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        time_row_or_col = "B"
        cell = "C5"
        coords = {}
        interp = "hold_backward"
        py_name = "test_data_backward_v1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.backward_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_backward_hn1d(self, _root, _exp):
        """
        ExtData test for 1d horizontal series hold_backward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        interp = "hold_backward"
        py_name = "test_data_backward_hn1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.backward_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_backward_vn1d(self, _root, _exp):
        """
        ExtData test for 1d vertical series hold_backward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        time_row_or_col = "time"
        cell = "data_1d"
        coords = {}
        interp = "hold_backward"
        py_name = "test_data_backward_vn1d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.backward_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_interp_vn2d(self, _root, _exp):
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
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()
        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.interp_2d):
                assert y.equals(data(x)), "Wrong result at X=" + str(x)

    def test_data_forward_hn2d(self, _root, _exp):
        """
        ExtData test for 2d vertical series look_forward by cell range names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "time"
        cell = "data_2d"
        coords = {'ABC': ['A', 'B', 'C']}
        interp = "look_forward"
        py_name = "test_data_forward_hn2d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()
        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.forward_2d):
                assert y.equals(data(x)), "Wrong result at X=" + str(x)

    def test_data_backward_v2d(self, _root, _exp):
        """
        ExtData test for 2d vertical series hold_backward
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        time_row_or_col = "B"
        cell = "C5"
        coords = {'ABC': ['A', 'B', 'C']}
        interp = "hold_backward"
        py_name = "test_data_backward_v2d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.backward_2d):
                assert y.equals(data(x)), "Wrong result at X=" + str(x)

    def test_data_interp_h3d(self, _root, _exp):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']}
        interp = None
        py_name = "test_data_interp_h3d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     coords=coords_1,
                                     interp=interp,
                                     final_coords=final_coords,
                                     py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 time_row_or_col=time_row_or_col,
                 cell=cell_2,
                 coords=coords_2,
                 interp=interp)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.interp_3d):
                assert y.equals(data(x)), "Wrong result at X=" + str(x)

    def test_data_forward_v3d(self, _root, _exp):
        """
        ExtData test for 3d vertical series look_forward
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Vertical"
        time_row_or_col = "B"
        cell_1 = "C5"
        cell_2 = "F5"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C']}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C']}
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']}
        interp = "look_forward"
        py_name = "test_data_forward_v3d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     coords=coords_1,
                                     interp=interp,
                                     final_coords=final_coords,
                                     py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 time_row_or_col=time_row_or_col,
                 cell=cell_2,
                 coords=coords_2,
                 interp=interp)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.forward_3d):
                assert y.equals(data(x)), "Wrong result at X=" + str(x)

    def test_data_backward_hn3d(self, _root, _exp):
        """
        ExtData test for 3d horizontal series hold_backward by cellrange names
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        time_row_or_col = "time"
        cell_1 = "data_2d"
        cell_2 = "data_2db"
        coords_1 = {'XY': ['X'], 'ABC': ['A', 'B', 'C']}
        coords_2 = {'XY': ['Y'], 'ABC': ['A', 'B', 'C']}
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']}
        interp = "hold_backward"
        py_name = "test_data_backward_hn3d"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     coords=coords_1,
                                     interp=interp,
                                     final_coords=final_coords,
                                     py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 time_row_or_col=time_row_or_col,
                 cell=cell_2,
                 coords=coords_2,
                 interp=interp)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.backward_3d):
                assert y.equals(data(x)), "Wrong result at X=" + str(x)

    def test_data_raw_h1d(self, _root, _exp):
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
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        for x, y in zip(_exp.xpts, _exp.raw_1d):
            data_x = data(x)
            if np.isnan(y):
                equal = np.isnan(data_x)
            else:
                equal = y == data_x
            assert equal, "Wrong result at X=" + str(x)


class TestLookup():
    """
    Test for the full working procedure of ExtLookup
    class when the data is properly given in the Excel file
    For 1D data for all cases are computed.
    For 2D, 3D only some cases are computed as the complete set of
    test will cover all the possibilities.
    """

    def test_lookup_h1d(self, _root, _exp):
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
                                       final_coords=coords,
                                       py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_lookup_v1d(self, _root, _exp):
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
                                       final_coords=coords,
                                       py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_lookup_hn1d(self, _root, _exp):
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
                                       final_coords=coords,
                                       py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_lookup_vn1d(self, _root, _exp):
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
                                       final_coords=coords,
                                       py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_lookup_h2d(self, _root, _exp):
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
                                       final_coords=coords,
                                       py_name=py_name)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.interp_2d):
                assert y.equals(data(x)), "Wrong result at X=" + str(x)

    def test_lookup_vn3d(self, _root, _exp):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']}
        py_name = "test_lookup_vn3d"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       final_coords=final_coords,
                                       py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 x_row_or_col=x_row_or_col,
                 cell=cell_2,
                 coords=coords_2)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.interp_3d):
                assert y.equals(data(x)), "Wrong result at X=" + str(x)

    def test_lookup_vn3d_shape0(self, _root, _exp):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']}
        py_name = "test_lookup_vn3d_shape0"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       final_coords=final_coords,
                                       py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 x_row_or_col=x_row_or_col,
                 cell=cell_2,
                 coords=coords_2)

        data.initialize()

        with pytest.warns(UserWarning):
            for x, y in zip(_exp.xpts, _exp.interp_3d):
                assert y.equals(data(xr.DataArray(x))), \
                    "Wrong result at X=" + str(x)

    def test_lookup_vn2d_xarray(self, _root):
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
                                       final_coords=coords_1,
                                       py_name=py_name)

        data.initialize()

        coords_2 = {'XY': ['X', 'Y']}

        all_smaller = xr.DataArray([-1, -10], coords_2, ['XY'])
        all_bigger = xr.DataArray([9, 20, 30], coords_1, ['ABC'])
        all_inside = xr.DataArray([3.5, 5.5], coords_2, ['XY'])
        mixed = xr.DataArray([1.5, 20, -30], coords_1, ['ABC'])
        full = xr.DataArray([[1.5, -30], [-10, 2.5], [4., 5.]],
                            {**coords_1, **coords_2},
                            ['ABC', 'XY'])

        all_smaller_out = data.data[0].reset_coords('lookup_dim', drop=True)\
            + 0*all_smaller
        all_bigger_out = data.data[-1].reset_coords('lookup_dim', drop=True)
        all_inside_out = xr.DataArray([[0.5, -1],
                                       [-1, -0.5],
                                       [-0.75, 0]],
                                      {**coords_1, **coords_2},
                                      ['ABC', 'XY'])
        mixed_out = xr.DataArray([0.5, 0, 1], coords_1, ['ABC'])
        full_out = xr.DataArray([[0.5, 0, -0.5],
                                 [0, 0, 0]],
                                {**coords_2, **coords_1},
                                ['XY', 'ABC'])

        with pytest.warns(UserWarning):
            assert data(
                all_smaller, {**coords_1, **coords_2}
            ).equals(all_smaller_out)
            assert data(all_bigger, coords_1).equals(all_bigger_out)
            assert data(
                all_inside, {**coords_1, **coords_2}
            ).equals(all_inside_out)
            assert data(mixed, coords_1).equals(mixed_out)
            assert data(full, {**coords_2, **coords_1}).equals(full_out)

    def test_lookup_vn3d_xarray(self, _root):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']}
        py_name = "test_lookup_vn3d_xarray"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       final_coords=final_coords,
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

        with pytest.warns(UserWarning):
            assert data(all_smaller, final_coords).equals(all_smaller_out)
            assert data(all_bigger, final_coords).equals(all_bigger_out)
            assert data(all_inside, final_coords).equals(all_inside_out)
            assert data(mixed, final_coords).equals(mixed_out)
            assert data(full, final_coords).equals(full_out)


class TestConstant():
    """
    Test for the full working procedure of ExtConstant
    class when the data is properly given in the Excel file
    For 1D, 2D and 3D all cases are computed.
    """

    def test_constant_0d(self, _root):
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
                                         final_coords=coords,
                                         py_name=py_name)

        data2 = pysd.external.ExtConstant(file_name=file_name,
                                          sheet=sheet,
                                          root=_root,
                                          cell=cell2,
                                          coords=coords,
                                          final_coords=coords,
                                          py_name=py_name)
        data.initialize()
        data2.initialize()

        assert data() == -1
        assert data2() == 0

    def test_constant_n0d(self, _root):
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
                                         final_coords=coords,
                                         py_name=py_name)

        data2 = pysd.external.ExtConstant(file_name=file_name,
                                          sheet=sheet,
                                          root=_root,
                                          cell=cell2,
                                          coords=coords,
                                          final_coords=coords,
                                          py_name=py_name)
        data.initialize()
        data2.initialize()

        assert data() == -1
        assert data2() == 0

    def test_constant_h1d(self, _root, _exp):
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
                                         final_coords=coords,
                                         py_name=py_name)
        data.initialize()

        assert data().equals(_exp.constant_1d)

    def test_constant_v1d(self, _root, _exp):
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
                                         final_coords=coords,
                                         py_name=py_name)
        data.initialize()

        assert data().equals(_exp.constant_1d)

    def test_constant_hn1d(self, _root, _exp):
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
                                         final_coords=coords,
                                         py_name=py_name)
        data.initialize()

        assert data().equals(_exp.constant_1d)

    def test_constant_vn1d(self, _root, _exp):
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
                                         final_coords=coords,
                                         py_name=py_name)
        data.initialize()

        assert data().equals(_exp.constant_1d)

    def test_constant_h2d(self, _root, _exp):
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
                                         final_coords=coords,
                                         py_name=py_name)
        data.initialize()

        assert data().equals(_exp.constant_2d)

    def test_constant_v2d(self, _root, _exp):
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
                                         final_coords=coords,
                                         py_name=py_name)
        data.initialize()

        assert data().equals(_exp.constant_2d)

    def test_constant_hn2d(self, _root, _exp):
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
                                         final_coords=coords,
                                         py_name=py_name)
        data.initialize()

        assert data().equals(_exp.constant_2d)

    def test_constant_vn2d(self, _root, _exp):
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
                                         final_coords=coords,
                                         py_name=py_name)
        data.initialize()

        assert data().equals(_exp.constant_2d)

    def test_constant_h3d(self, _root, _exp):
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
        final_coords = {'ABC': ['A', 'B', 'C'],
                        'XY': ['X', 'Y'],
                        'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_h3d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         final_coords=final_coords,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell2,
                 coords=coords2)

        data.initialize()

        assert data().equals(_exp.constant_3d)

    def test_constant_v3d(self, _root, _exp):
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
        final_coords = {'ABC': ['A', 'B', 'C'],
                        'XY': ['X', 'Y'],
                        'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_v3d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         final_coords=final_coords,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell2,
                 coords=coords2)

        data.initialize()

        assert data().equals(_exp.constant_3d)

    def test_constant_hn3d(self, _root, _exp):
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
        final_coords = {'ABC': ['A', 'B', 'C'],
                        'XY': ['X', 'Y'],
                        'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_hn3d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         final_coords=final_coords,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell2,
                 coords=coords2)

        data.initialize()

        assert data().equals(_exp.constant_3d)

    def test_constant_vn3d(self, _root, _exp):
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
        final_coords = {'ABC': ['A', 'B', 'C'],
                        'XY': ['X', 'Y'],
                        'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_vn2d"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         final_coords=final_coords,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell2,
                 coords=coords2)

        data.initialize()

        assert data().equals(_exp.constant_3d)


class TestSubscript():
    """
    Test for the full working procedure of ExtSubscript
    class when the data is properly given in the Excel file
    """

    def test_subscript_h(self, _root):
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

        assert data.subscript == expected

    def test_subscript_h2(self, _root):
        """
        ExtSubscript test for horizontal subscripts
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        firstcell = "C4"
        lastcell = "4"
        prefix = 'v'
        expected = ['v0', 'v1', 'v2', 'v3',
                    'v5', 'v6', 'v7', 'v8']

        data = pysd.external.ExtSubscript(file_name=file_name,
                                          sheet=sheet,
                                          root=_root,
                                          firstcell=firstcell,
                                          lastcell=lastcell,
                                          prefix=prefix)

        assert data.subscript == expected

    def test_subscript_v(self, _root):
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

        assert data.subscript == expected

    def test_subscript_v2(self, _root):
        """
        ExtSubscript test for vertical subscripts
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        firstcell = "B8"
        lastcell = "B"
        prefix = ''
        expected = ['A', 'B', 'C', '4', '5']

        data = pysd.external.ExtSubscript(file_name=file_name,
                                          sheet=sheet,
                                          root=_root,
                                          firstcell=firstcell,
                                          lastcell=lastcell,
                                          prefix=prefix)

        assert data.subscript == expected

    def test_subscript_d(self, _root):
        """
        ExtSubscript test for diagonal subscripts
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        firstcell = "A5"
        lastcell = "B10"
        prefix = ''
        expected = ['X', 'A', 'B', 'C', 'Y', 'A', 'B', 'C']

        data = pysd.external.ExtSubscript(file_name=file_name,
                                          sheet=sheet,
                                          root=_root,
                                          firstcell=firstcell,
                                          lastcell=lastcell,
                                          prefix=prefix)

        assert data.subscript == expected

    def test_subscript_d2(self, _root):
        """
        ExtSubscript test for diagonal subscripts
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "No monotonous"
        firstcell = "H10"
        lastcell = ""
        prefix = "j"
        expected = ['j3', 'j2', 'j1', 'j6', 'j4', 'j8', 'j-1', 'j3', 'j2']

        data = pysd.external.ExtSubscript(file_name=file_name,
                                          sheet=sheet,
                                          root=_root,
                                          firstcell=firstcell,
                                          lastcell=lastcell,
                                          prefix=prefix)

        assert data.subscript == expected

    def test_subscript_name_h(self, _root):
        """
        ExtSubscript test for horizontal subscripts
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        firstcell = "time_missing"
        lastcell = "B7"
        prefix = "l"
        expected = ['l0', 'l1', 'l2', 'l3',
                    'l5', 'l6', 'l7', 'l8']

        data = pysd.external.ExtSubscript(file_name=file_name,
                                          sheet=sheet,
                                          root=_root,
                                          firstcell=firstcell,
                                          lastcell=lastcell,
                                          prefix=prefix)

        assert data.subscript == expected

    def test_subscript_name_v(self, _root):
        """
        ExtSubscript test for vertical subscripts
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        firstcell = "vertical_index"
        lastcell = ""
        prefix = "p"
        expected = ['pA', 'pB', 'pC']

        data = pysd.external.ExtSubscript(file_name=file_name,
                                          sheet=sheet,
                                          root=_root,
                                          firstcell=firstcell,
                                          lastcell=lastcell,
                                          prefix=prefix)

        assert data.subscript == expected

    def test_subscript_name_d(self, _root):
        """
        ExtSubscript test for diagonal subscripts
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal missing"
        firstcell = "d_names"
        lastcell = None
        prefix = ""
        expected = ['X', 'A', '0', 'B', '0', 'C', '1']

        data = pysd.external.ExtSubscript(file_name=file_name,
                                          sheet=sheet,
                                          root=_root,
                                          firstcell=firstcell,
                                          lastcell=lastcell,
                                          prefix=prefix)

        assert data.subscript == expected


class TestWarningsErrors():
    """
    Test for the warnings and errors of External and its subclasses
    """

    def test_not_implemented_file(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        error_message = r"The files with extension .ods are not implemented"
        with pytest.raises(NotImplementedError, match=error_message):
            data.initialize()

    def test_non_existent_file(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)
        error_message = r"File '.*non_existent\.xls' not found\."
        with pytest.raises(FileNotFoundError, match=error_message):
            data.initialize()

    def test_non_existent_sheet_pyxl(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        error_message = r"The sheet doesn't exist\.\.\."
        with pytest.raises(ValueError, match=error_message):
            data.initialize()

    def test_non_existent_cellrange_name_pyxl(self, _root):
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
        py_name = "test_non_existent_cellrange_name_pyxl"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     final_coords=coords,
                                     py_name=py_name)

        error_message = "The cellrange name 'non_exixtent'\nDoesn't exist in"
        with pytest.raises(AttributeError, match=error_message):
            data.initialize()

    def test_non_existent_cellrange_name_in_sheet_pyxl(self, _root):
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
                                         final_coords=coords,
                                         py_name=py_name)

        error_message = "The cellrange name 'constant'\nDoesn't exist in"
        with pytest.raises(AttributeError, match=error_message):
            data.initialize()

    # Following test are for ExtData class only
    # as the initialization of ExtLookup uses the same function
    def test_data_interp_h1dm_row(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        with pytest.warns(UserWarning) as record:
            data.initialize()

        warn_message = "Not able to interpolate"
        assert any([
            re.match(warn_message, str(warn.message))
            for warn in record
        ]), f"Couldn't match warning:\n{warn_message}"

        assert all(np.isnan(data.data.values))

    def test_data_interp_h1dm_row2(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        with pytest.warns(UserWarning) as record:
            data.initialize()

        warn_message = "Not able to interpolate"
        assert any([
            re.match(warn_message, str(warn.message))
            for warn in record
        ]), f"Couldn't match warning:\n{warn_message}"

        assert not any(np.isnan(data.data.loc[:, "B"].values))
        assert not any(np.isnan(data.data.loc[:, "C"].values))
        assert all(np.isnan(data.data.loc[:, "D"].values))

    def test_data_interp_h1dm(self, _root, _exp):
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
                                     final_coords=coords,
                                     py_name=py_name)

        with pytest.warns(UserWarning, match='missing'):
            data.initialize()

        warn_message = r"extrapolating data (above|below) the "\
            r"(maximum|minimum) value of the time"
        with pytest.warns(UserWarning, match=warn_message):
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_interp_h1dm_ignore(self, _root, _exp):
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
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        warn_message = r"extrapolating data (above|below) the "\
            r"(maximum|minimum) value of the time"
        with pytest.warns(UserWarning, match=warn_message):
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_interp_h1dm_raise(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        error_message = "Dimension value missing or non-valid"
        with pytest.raises(ValueError, match=error_message):
            data.initialize()

    def test_data_interp_v1dm(self, _root, _exp):
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
                                     final_coords=coords,
                                     py_name=py_name)

        with pytest.warns(UserWarning, match='missing'):
            data.initialize()

        warn_message = r"extrapolating data (above|below) the "\
            r"(maximum|minimum) value of the time"
        with pytest.warns(UserWarning, match=warn_message):
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_interp_v1dm_ignore(self, _root, _exp):
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
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        warn_message = r"extrapolating data (above|below) the "\
            r"(maximum|minimum) value of the time"
        with pytest.warns(UserWarning, match=warn_message):
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_interp_v1dm_raise(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        error_message = "Dimension value missing or non-valid"
        with pytest.raises(ValueError, match=error_message):
            data.initialize()

    def test_data_interp_hn1dm(self, _root, _exp):
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
                                     final_coords=coords,
                                     py_name=py_name)

        with pytest.warns(UserWarning, match='missing'):
            data.initialize()

        warn_message = r"extrapolating data (above|below) the "\
            r"(maximum|minimum) value of the time"
        with pytest.warns(UserWarning, match=warn_message):
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_interp_hn1dm_ignore(self, _root, _exp):
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
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        warn_message = r"extrapolating data (above|below) the "\
            r"(maximum|minimum) value of the time"
        with pytest.warns(UserWarning, match=warn_message):
            for x, y in zip(_exp.xpts, _exp.interp_1d):
                assert y == data(x), "Wrong result at X=" + str(x)

    def test_data_interp_hn1dm_raise(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        error_message = "Dimension value missing or non-valid"
        with pytest.raises(ValueError, match=error_message):
            data.initialize()

    def test_data_interp_hn3dmd(self, _root, _exp):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']}
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
                                     final_coords=final_coords,
                                     py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 time_row_or_col=time_row_or_col,
                 cell=cell_2,
                 interp=interp,
                 coords=coords_2)

        with pytest.warns(UserWarning, match='missing'):
            data.initialize()

        warn_message = r"extrapolating data (above|below) the "\
            r"(maximum|minimum) value of the time"
        with pytest.warns(UserWarning, match=warn_message):
            for x, y in zip(_exp.xpts, _exp.interp_3d):
                assert y.equals(data(x)), "Wrong result at X=" + str(x)

    def test_data_interp_hn3dmd_raw(self, _root):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']}
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
                                     final_coords=final_coords,
                                     py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 time_row_or_col=time_row_or_col,
                 cell=cell_2,
                 interp=interp,
                 coords=coords_2)

        with pytest.warns(UserWarning, match='missing'):
            data.initialize()

    def test_lookup_hn3dmd_raise(self, _root):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']}
        py_name = "test_lookup_hn3dmd_raise"
        pysd.external.External.missing = "raise"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       final_coords=final_coords,
                                       py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 x_row_or_col=x_row_or_col,
                 cell=cell_2,
                 coords=coords_2)

        error_message = "Data value missing or non-valid"
        with pytest.raises(ValueError, match=error_message):
            data.initialize()

    def test_lookup_hn3dmd_ignore(self, _root, _exp):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']}
        py_name = "test_lookup_hn3dmd_ignore"
        pysd.external.External.missing = "ignore"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       final_coords=final_coords,
                                       py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 x_row_or_col=x_row_or_col,
                 cell=cell_2,
                 coords=coords_2)

        data.initialize()

        warn_message = r"extrapolating data (above|below) the "\
            r"(maximum|minimum) value of the series"
        with pytest.warns(UserWarning, match=warn_message):
            for x, y in zip(_exp.xpts, _exp.interp_3d):
                assert y.equals(data(x)), "Wrong result at X=" + str(x)

    def test_constant_h3dm(self, _root):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C'],
                        'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_h3dm"
        pysd.external.External.missing = "warning"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell_1,
                                         coords=coords_1,
                                         final_coords=final_coords,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell_2,
                 coords=coords_2)

        with pytest.warns(UserWarning, match='missing'):
            data.initialize()

    def test_constant_h3dm_ignore(self, _root):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C'],
                        'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_h3dm_ignore"
        pysd.external.External.missing = "ignore"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell_1,
                                         coords=coords_1,
                                         final_coords=final_coords,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell_2,
                 coords=coords_2)

        data.initialize()

    def test_constant_h3dm_raise(self, _root):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C'],
                        'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_h3dm_raise"
        pysd.external.External.missing = "raise"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell_1,
                                         coords=coords_1,
                                         final_coords=final_coords,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell_2,
                 coords=coords_2)

        error_message = "Constant value missing or non-valid"
        with pytest.raises(ValueError, match=error_message):
            data.initialize()

    def test_constant_hn3dm_raise(self, _root):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C'],
                        'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_hn3dm_raise"
        pysd.external.External.missing = "raise"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell_1,
                                         coords=coords_1,
                                         final_coords=final_coords,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell_2,
                 coords=coords_2)

        error_message = "Constant value missing or non-valid"
        with pytest.raises(ValueError, match=error_message):
            data.initialize()

    def test_data_interp_h1d0(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        error_message = "has length 0"
        with pytest.raises(ValueError, match=error_message):
            data.initialize()

    def test_data_interp_v1d0(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        error_message = "has length 0"
        with pytest.raises(ValueError, match=error_message):
            data.initialize()

    def test_data_interp_hn1d0(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        error_message = "has length 0"
        with pytest.raises(ValueError, match=error_message):
            data.initialize()

    def test_data_interp_hn1dt(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        error_message = "is a table and not a vector"
        with pytest.raises(ValueError, match=error_message):
            data.initialize()

    def test_data_interp_hns(self, _root):
        """
        Test for error in data when it doesn't have the same
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
                                     final_coords=coords,
                                     py_name=py_name)

        error_message = "has not the same size as the given coordinates"
        with pytest.raises(ValueError, match=error_message):
            data.initialize()

    def test_data_interp_vnss(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        error_message = "don't have the same length in the 1st dimension"
        with pytest.raises(ValueError, match=error_message):
            data.initialize()

    # Following test are independent of the reading option
    def test_data_interp_hnnwd(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        with pytest.raises(ValueError, match="has repeated values"):
            data.initialize()

    def test_data_raw_hnnm(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        expected = {-1: 2, 0: 2, 1: 2, 2: 3,
                    3: -1, 4: -1, 5: 1, 6: 1,
                    7: 0, 8: 0, 9: 0}

        with pytest.warns(UserWarning):
            for i in range(-1, 9):
                assert data(i) == expected[i]

        time_row_or_col = "11"
        py_name = "test_data_interp_hnnnm2"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell,
                                     coords=coords,
                                     interp=interp,
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        expected = {-1: 0, 0: 0, 1: 0, 2: 1,
                    3: 2, 4: 3, 5: -1, 6: -1,
                    7: 1, 8: 2, 9: 2}

        with pytest.warns(UserWarning):
            for i in range(-1, 9):
                data(i) == expected[i]

    def test_data_h3d_interpnv(self, _root):
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

        error_message = r"The interpolation method \(interp\) must be "\
            r"'raw', 'interpolate', 'look_forward' or 'hold_backward'"
        with pytest.raises(ValueError, match=error_message):
            pysd.external.ExtData(file_name=file_name,
                                  sheet=sheet,
                                  time_row_or_col=time_row_or_col,
                                  root=_root,
                                  cell=cell,
                                  coords=coords,
                                  interp=interp,
                                  final_coords=coords,
                                  py_name=py_name)

    def test_data_h3d_interp(self, _root):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']}
        interp = None
        interp2 = "look_forward"
        py_name = "test_data_h3d_interp"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     coords=coords_1,
                                     interp=interp,
                                     final_coords=final_coords,
                                     py_name=py_name)

        error_message = "Error matching interpolation method with "\
            "previously defined one"
        with pytest.raises(ValueError, match=error_message):
            data.add(file_name=file_name,
                     sheet=sheet,
                     time_row_or_col=time_row_or_col,
                     cell=cell_2,
                     coords=coords_2,
                     interp=interp2)

    def test_data_h3d_add(self, _root):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']}
        interp = None
        py_name = "test_data_h3d_add"

        data = pysd.external.ExtData(file_name=file_name,
                                     sheet=sheet,
                                     time_row_or_col=time_row_or_col,
                                     root=_root,
                                     cell=cell_1,
                                     coords=coords_1,
                                     interp=interp,
                                     final_coords=final_coords,
                                     py_name=py_name)

        error_message = "Error matching dimensions with previous data"
        with pytest.raises(ValueError, match=error_message):
            data.add(file_name=file_name,
                     sheet=sheet,
                     time_row_or_col=time_row_or_col,
                     cell=cell_2,
                     coords=coords_2,
                     interp=interp)

    def test_lookup_h3d_add(self, _root):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']}
        py_name = "test_lookup_h3d_add"

        data = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=x_row_or_col,
                                       root=_root,
                                       cell=cell_1,
                                       coords=coords_1,
                                       final_coords=final_coords,
                                       py_name=py_name)

        error_message = "Error matching dimensions with previous data"
        with pytest.raises(ValueError, match=error_message):
            data.add(file_name=file_name,
                     sheet=sheet,
                     x_row_or_col=x_row_or_col,
                     cell=cell_2,
                     coords=coords_2)

    def test_constant_h3d_add(self, _root):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C'],
                        'val': [0, 1, 2, 3, 5, 6, 7, 8]}
        py_name = "test_constant_h3d_add"

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         final_coords=final_coords,
                                         py_name=py_name)

        error_message = "Error matching dimensions with previous data"
        with pytest.raises(ValueError, match=error_message):
            data.add(file_name=file_name,
                     sheet=sheet,
                     cell=cell2,
                     coords=coords2)

    def test_constant_hns(self, _root):
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
                                         final_coords=coords,
                                         py_name=py_name)

        error_message = 'has not the same shape as the given coordinates'
        with pytest.raises(ValueError, match=error_message):
            data.initialize()

    def text_openpyxl_str(self, _root):
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
                                       final_coords=coords,
                                       py_name=py_name)

        expected = xr.DataArray(
            [np.nan, 1, 2, 3, 4, 5],
            {'lookup_dim': [10., 11., 12., 13., 14., 15.]},
            ['lookup_dim'])

        data.initialize()

        assert data.data.equals(expected)

        cell = "no_constant"
        sheet = "caSE anD NON V"  # test case insensitivity

        data = pysd.external.ExtConstant(file_name=file_name,
                                         sheet=sheet,
                                         root=_root,
                                         cell=cell,
                                         coords=coords,
                                         final_coords=coords,
                                         py_name=py_name)

        data.initialize()

        assert np.isnan(data.data)

    def test_subscript_name_non_existent_sheet(self, _root):
        """
        ExtSubscript test for diagonal subscripts
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "No exit"
        firstcell = "d_names"
        lastcell = None
        prefix = ""

        error_message = r"The sheet doesn't exist\.\.\."
        with pytest.raises(ValueError, match=error_message):
            pysd.external.ExtSubscript(file_name=file_name,
                                       sheet=sheet,
                                       root=_root,
                                       firstcell=firstcell,
                                       lastcell=lastcell,
                                       prefix=prefix)

    def test_subscript_name_non_existent_cellrange_name(self, _root):
        """
        Test for non-existent cellrange name with openpyxl
        """
        import pysd

        file_name = "data/input.xlsx"
        sheet = "Horizontal"
        firstcell = "fake-cell"
        lastcell = None
        prefix = ""

        error_message = "The cellrange name 'fake-cell'\nDoesn't exist in"
        with pytest.raises(AttributeError, match=error_message):
            pysd.external.ExtSubscript(file_name=file_name,
                                       sheet=sheet,
                                       root=_root,
                                       firstcell=firstcell,
                                       lastcell=lastcell,
                                       prefix=prefix)


class DownwardCompatibility():
    """
    These tests are defined to make the external objects compatible
    with SDQC library. If any change in PySD breaks these tests it
    should be checked with SDQC library and correct it.
    """
    def test_constant_hn3dm_keep(self, _root):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C'],
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
                                         final_coords=final_coords,
                                         py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 cell=cell_2,
                 coords=coords_2)

        data.initialize()

        assert data().equals(expected)

    def test_lookup_hn3dmd_keep(self, _root):
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
        final_coords = {'XY': ['X', 'Y'], 'ABC': ['A', 'B', 'C']}
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
                                       final_coords=final_coords,
                                       py_name=py_name)

        data.add(file_name=file_name,
                 sheet=sheet,
                 x_row_or_col=x_row_or_col,
                 cell=cell_2,
                 coords=coords_2)

        data.initialize()

        assert data.data.equals(expected)

    def test_data_interp_v1dm_keep(self, _root):
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
        py_name = "test_data_interp_v1dm_keep"

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
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        assert data.data.equals(expected)

    def test_data_interp_hnnm_keep(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        data.initialize()

        assert data.data.equals(expected)

    def test_lookup_data_attr(self, _root):
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
                                     final_coords=coords,
                                     py_name=py_name)

        datL = pysd.external.ExtLookup(file_name=file_name,
                                       sheet=sheet,
                                       x_row_or_col=time_row_or_col,
                                       root=_root,
                                       cell=cell,
                                       coords=coords,
                                       final_coords=coords,
                                       py_name=py_name)
        datD.initialize()
        datL.initialize()

        assert hasattr(datD, 'time_row_or_cols')
        assert hasattr(datL, 'x_row_or_cols')
