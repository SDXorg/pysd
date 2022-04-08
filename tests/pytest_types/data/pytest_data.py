import pytest

import xarray as xr

from pysd.py_backend.data import Data


@pytest.mark.parametrize(
    "value,interp,raise_type,error_message",
    [
        (  # not_loaded_data
            None,
            "interpolate",
            ValueError,
            "Trying to interpolate data variable before loading the data..."
        ),
        # test that try/except block on call doesn't catch errors differents
        # than data = None
        (  # try_except
            xr.DataArray([10, 20], {'time': [0, 1]}, ['time']),
            None,
            AttributeError,
            "'Data' object has no attribute 'is_float'"
        )
    ],
    ids=["not_loaded_data", "try_except"]
)
@pytest.mark.filterwarnings("ignore")
class TestDataErrors():
    # Test errors associated with Data class
    # Several Data cases are tested in unit_test_external while some other
    # are tested indirectly in unit_test_pysd and integration_test_vensim

    @pytest.fixture
    def data(self, value, interp):
        obj = Data()
        obj.data = value
        obj.interp = interp
        obj.py_name = "data"
        return obj

    def test_data_errors(self, data, raise_type, error_message):
        with pytest.raises(raise_type, match=error_message):
            data(1.5)
