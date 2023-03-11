
import pytest
import shutil

import numpy as np
import xarray as xr

import pysd


class TestRandomModel:
    """Submodel selecting class"""
    # messages for selecting submodules
    @pytest.fixture(scope="class")
    def model_path(self, shared_tmpdir, _root):
        """
        Copy test folder to a temporary folder therefore we avoid creating
        PySD model files in the original folder
        """
        new_file = shared_tmpdir.joinpath("test_random.mdl")
        shutil.copy(
            _root.joinpath("more-tests/random/test_random.mdl"),
            new_file
        )
        return new_file

    def test_translate(self, model_path):
        """
        Translate the model or read a translated version.
        This way each file is only translated once.
        """
        # expected file
        model = pysd.read_vensim(model_path)
        random_vars = [
            "A B uniform matrix",
            "A B uniform matrix 1",
            "A B uniform matrix 1 0",
            "A B uniform scalar",
            "A B uniform vec",
            "A B uniform vec 1",
            "normal A B uniform matrix",
            "normal A B uniform matrix 1",
            "normal A B uniform matrix 1 0",
            "normal scalar",
            "normal vec",
            "normal vec 1",
            "uniform matrix",
            "uniform scalar",
            "uniform vec"
        ]
        out = model.run(return_columns=random_vars, flatten_output=False)
        for var in out.columns:
            if isinstance(out[var].values[0], xr.DataArray):
                values = np.array([a.values for a in out[var].values])
            else:
                values = out[var].values
            # assert all values are different in each dimension and time step
            assert len(np.unique(values)) == np.prod(values.shape)
