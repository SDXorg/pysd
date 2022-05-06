import pytest

import xarray as xr
import pandas as pd

from pysd.py_backend.lookups import Lookups


@pytest.mark.parametrize(
    "value,new_value,expected",
    [
        (  # float-constant
            xr.DataArray([10, 20], {'lookup_dim': [0, 1]}, ['lookup_dim']),
            26,
            26
        ),
        (  # float-series
            xr.DataArray([10, 20], {'lookup_dim': [0, 1]}, ['lookup_dim']),
            pd.Series(index=[1, 20, 40], data=[2, 10, 2]),
            xr.DataArray(
                [2, 10, 2],
                {"lookup_dim": [1, 20, 40]},
                ["lookup_dim"]
            )

        ),
        (  # array-constantfloat
            xr.DataArray(
                [[10, 20], [30, 40]],
                {"lookup_dim": [0, 1], "dim":["A", "B"]},
                ["lookup_dim", "dim"]),
            26,
            xr.DataArray(26, {"dim": ["A", "B"]}, ["dim"]),
        ),
        (  # array-seriesfloat
            xr.DataArray(
                [[10, 20], [30, 40]],
                {"lookup_dim": [0, 1], "dim":["A", "B"]},
                ["lookup_dim", "dim"]),
            pd.Series(index=[1, 20, 40], data=[2, 10, 2]),
            xr.DataArray(
                [[2, 2], [10, 10], [2, 2]],
                {"lookup_dim": [1, 20, 40], "dim":["A", "B"]},
                ["lookup_dim", "dim"])
        ),
        (  # array-constantarray
            xr.DataArray(
                [[[10, 20], [30, 40]], [[15, 25], [35, 45]]],
                {"lookup_dim": [0, 1], "dim":["A", "B"], "dim2": ["C", "D"]},
                ["lookup_dim", "dim", "dim2"]),
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
                {"lookup_dim": [0, 1], "dim":["A", "B"], "dim2": ["C", "D"]},
                ["lookup_dim", "dim", "dim2"]),
            pd.Series(index=[1, 20, 40], data=[
                xr.DataArray([1, 2], {"dim": ["A", "B"]}, ["dim"]),
                xr.DataArray([10, 20], {"dim": ["A", "B"]}, ["dim"]),
                xr.DataArray([1, 2], {"dim": ["A", "B"]}, ["dim"])
            ]),
            xr.DataArray(
                [[[1, 2], [1, 2]], [[10, 20], [10, 20]], [[1, 2], [1, 2]]],
                {
                    "lookup_dim": [1, 20, 40],
                    "dim":["A", "B"],
                    "dim2": ["C", "D"]
                },
                ["lookup_dim", "dim", "dim2"])
        )
    ],
    ids=[
        "float-constant", "float-series",
        "array-constantfloat", "array-seriesfloat",
        "array-constantarray", "array-seriesarray"
    ]
)
class TestLookupsSetValues():

    @pytest.fixture
    def lookups(self, value):
        obj = Lookups()
        obj.data = value
        obj.interp = "interp"
        obj.is_float = len(value.shape) < 2
        obj.final_coords = {
            dim: value.coords[dim] for dim in value.dims if dim != "lookup_dim"
        }
        obj.py_name = "lookup"
        return obj

    def test_lookups_set_value(self, lookups, new_value, expected):
        lookups.set_values(new_value)
        if isinstance(expected, (float, int)):
            assert lookups.data == expected
        else:
            assert lookups.data.equals(expected)
