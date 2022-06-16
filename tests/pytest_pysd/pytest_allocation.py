import pytest
import numpy as np
import xarray as xr

from pysd.py_backend.allocation import\
    allocate_by_priority, _allocate_by_priority_1d


class TestAllocateByPriority():
    @pytest.mark.parametrize(
        "requests,priority,width,supply,expected",
        [
            (
                xr.DataArray([6, 3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray([10, 1, 0], {'dim': ["A", "B", "C"]}),
                3,
                15,
                xr.DataArray([6, 3, 3], {'dim': ["A", "B", "C"]}),
            ),
            (
                xr.DataArray([6, 3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray([10, 1, 0], {'dim': ["A", "B", "C"]}),
                3,
                5,
                xr.DataArray([5, 0, 0], {'dim': ["A", "B", "C"]}),
            ),
            (
                xr.DataArray([6, 3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray([10, 5, 0], {'dim': ["A", "B", "C"]}),
                3,
                7.5,
                xr.DataArray([6, 1.5, 0], {'dim': ["A", "B", "C"]}),
            ),
        ],
    )
    def test_allocate_by_priority(self, requests, priority, width,
                                  supply, expected):
        # Test some simple cases, the complicate cases are tested with
        # a full integration test
        assert allocate_by_priority(
            requests, priority, width, supply).equals(expected)

    @pytest.mark.parametrize(
        "requests,priority,width,supply,expected",
        [
            (
                np.array([6, 3, 3]),
                np.array([10, 1, 0]),
                3,
                15,
                np.array([6, 3, 3]),
            ),
            (
                np.array([6, 3, 3]),
                np.array([10, 1, 0]),
                3,
                5,
                np.array([5, 0, 0]),
            ),
            (
                np.array([6, 3, 3]),
                np.array([10, 5, 0]),
                3,
                7.5,
                np.array([6, 1.5, 0]),
            ),
        ],
    )
    def test__allocate_by_priority_1d(self, requests, priority, width,
                                      supply, expected):
        # Test some simple cases, the complicate cases are tested with
        # a full integration test
        assert np.all(_allocate_by_priority_1d(
            requests, priority, width, supply) == expected)

    @pytest.mark.parametrize(
        "requests,priority,width,supply,raise_type,error_message",
        [
            (  # negative-request
                xr.DataArray([6, -3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray([10, 1, 0], {'dim': ["A", "B", "C"]}),
                3,
                15,
                ValueError,
                r"There are some negative request values\. Ensure that "
                r"your request is always non-negative\. Allocation requires "
                r"all quantities to be positive or 0\.\n.*"
            ),
            (  # 0 width
                xr.DataArray([6, 3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray([10, 1, 0], {'dim': ["A", "B", "C"]}),
                0,
                5,
                ValueError,
                r"width=0 is not allowed\. width should be greater than 0\."
            ),
            (  # negative width
                xr.DataArray([6, 3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray([10, 5, 0], {'dim': ["A", "B", "C"]}),
                -3,
                7.5,
                ValueError,
                r"width=-3 is not allowed\. width should be greater than 0\."
            ),
            (  # negative supply
                xr.DataArray([6, 3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray([10, 5, 0], {'dim': ["A", "B", "C"]}),
                3,
                -7.5,
                ValueError,
                r"supply=-7\.5 is not allowed\. "
                r"supply should be non-negative\."
            ),
        ],
    )
    def test_allocate_by_priority_errors(self, requests, priority,
                                         width, supply, raise_type,
                                         error_message):
        # Test some simple cases, the complicate cases are tested with
        # a full integration test
        with pytest.raises(raise_type, match=error_message):
            allocate_by_priority(requests, priority, width, supply)
