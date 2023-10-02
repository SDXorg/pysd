import pytest

import xarray as xr
import pandas as pd

from pysd.py_backend.data import Data, TabData


@pytest.mark.filterwarnings("ignore")
class TestDataErrors():
    # Test errors associated with Data class
    # Several Data cases are tested in pytest_external while some other
    # are tested indirectly in pytest_pysd and integration_test_vensim

    @pytest.fixture
    def data(self, value, interp):
        obj = Data()
        obj.data = value
        obj.interp = interp
        obj.py_name = "data"
        return obj

    @pytest.mark.parametrize(
        "value,interp,raise_type,error_message",
        [
            (  # not_loaded_data
                None,
                "interpolate",
                ValueError,
                "Trying to interpolate data variable before loading "
                "the data..."
            ),
            # test that try/except block on call doesn't catch errors
            # differents than data = None
            (  # try_except
                xr.DataArray([10, 20], {'time': [0, 1]}, ['time']),
                None,
                AttributeError,
                "'Data' object has no attribute 'is_float'"
            )
        ],
        ids=["not_loaded_data", "try_except"]
    )
    def test_data_errors(self, data, raise_type, error_message):
        with pytest.raises(raise_type, match=error_message):
            data(1.5)

    def test_invalid_interp_method(self):
        error_message = r"\nThe interpolation method \(interp\) must be"\
            r" 'raw', 'interpolate', 'look_forward' or 'hold_backward'"
        with pytest.raises(ValueError, match=error_message):
            TabData("", "", {}, interp="invalid")


@pytest.mark.parametrize(
    "value,new_value,expected",
    [
        (  # float-constant
            xr.DataArray([10, 20], {'time': [0, 1]}, ['time']),
            26,
            26
        ),
        (  # float-series
            xr.DataArray([10, 20], {'time': [0, 1]}, ['time']),
            pd.Series(index=[1, 20, 40], data=[2, 10, 2]),
            xr.DataArray([2, 10, 2], {"time": [1, 20, 40]}, ["time"])
        ),
        (  # array-constantfloat
            xr.DataArray(
                [[10, 20], [30, 40]],
                {"time": [0, 1], "dim":["A", "B"]},
                ["time", "dim"]),
            26,
            xr.DataArray(26, {"dim": ["A", "B"]}, ["dim"]),
        ),
        (  # array-seriesfloat
            xr.DataArray(
                [[10, 20], [30, 40]],
                {"time": [0, 1], "dim":["A", "B"]},
                ["time", "dim"]),
            pd.Series(index=[1, 20, 40], data=[2, 10, 2]),
            xr.DataArray(
                [[2, 2], [10, 10], [2, 2]],
                {"time": [1, 20, 40], "dim":["A", "B"]},
                ["time", "dim"])
        ),
        (  # array-constantarray
            xr.DataArray(
                [[[10, 20], [30, 40]], [[15, 25], [35, 45]]],
                {"time": [0, 1], "dim":["A", "B"], "dim2": ["C", "D"]},
                ["time", "dim", "dim2"]),
            xr.DataArray(
                [1, 2],
                {"dim": ["A", "B"]},
                ["dim"]),
            xr.DataArray(
                [[1, 2], [1, 2]],
                {"dim": ["A", "B"], "dim2": ["C", "D"]},
                ["dim", "dim2"])
        ),
        (  # array-seriesarray
            xr.DataArray(
                [[[10, 20], [30, 40]], [[15, 25], [35, 45]]],
                {"time": [0, 1], "dim":["A", "B"], "dim2": ["C", "D"]},
                ["time", "dim", "dim2"]),
            pd.Series(index=[1, 20, 40], data=[
                xr.DataArray([1, 2], {"dim": ["A", "B"]}, ["dim"]),
                xr.DataArray([10, 20], {"dim": ["A", "B"]}, ["dim"]),
                xr.DataArray([1, 2], {"dim": ["A", "B"]}, ["dim"])
            ]),
            xr.DataArray(
                [[[1, 1], [2, 2]], [[10, 10], [20, 20]], [[1, 1], [2, 2]]],
                {"time": [1, 20, 40], "dim":["A", "B"], "dim2": ["C", "D"]},
                ["time", "dim", "dim2"])
        )
    ],
    ids=[
        "float-constant", "float-series",
        "array-constantfloat", "array-seriesfloat",
        "array-constantarray", "array-seriesarray"
    ]
)
class TestDataSetValues():

    @pytest.fixture
    def data(self, value):
        obj = Data()
        obj.data = value
        obj.interp = "interp"
        obj.is_float = len(value.shape) < 2
        obj.final_coords = {
            dim: value.coords[dim] for dim in value.dims if dim != "time"
        }
        obj.py_name = "data"
        return obj

    def test_data_set_value(self, data, new_value, expected):
        data.set_values(new_value)
        if isinstance(expected, (float, int)):
            assert data.data == expected
        else:
            assert data.data.equals(expected)
