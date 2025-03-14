import pytest
import shutil

import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats

import pysd
from pysd.py_backend.functions import xidz
from pysd.translators.vensim.vensim_element import Component
from pysd.builders.python.python_expressions_builder import\
    CallBuilder, NumericBuilder


tolerance = {
    'data': {
        'mean': 1e-2,
        'variance': 1e-2,
        'skewness': 5e-2,
        'kurtosis': 10e-2
    },
    'expc': {
        'mean': 5e-3,
        'variance': 5e-3,
        'skewness': 2.5e-2,
        'kurtosis': 5e-2
    }
}


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


class TestRandomVensim():
    @pytest.fixture(scope="function")
    def data_raw(self, input_file, _test_random):
        file = _test_random.joinpath(input_file)
        data = pd.read_table(file, sep='\t', index_col=0)
        return data

    @pytest.fixture(scope="function")
    def data_python(self, data_raw, fake_component, random_size):
        out = {}
        for col in data_raw.columns:
            component = Component('', ([], []), col)
            component.parse()
            builder = CallBuilder(component.ast, fake_component)
            # TODO get minmax from here based on definition
            args = {
                i: NumericBuilder(arg, fake_component).build({})
                for i, arg in builder.arguments.items()
            }
            expr = builder.build(args).expression
            expr = expr.replace('()', str(random_size))
            if expr.startswith('float'):
                # remove float conversion as size is set bigger than 1
                expr = expr.replace('float(', '')[:-1]
            out[col] = eval(expr)

        return pd.DataFrame(out)

    @pytest.mark.parametrize(
        "input_file",
        [
            (  # uniform distribution
                'random_uniform/uniform_vensim.tab'
            ),
            (  # truncated normal distribution
                'random_normal/normal_vensim.tab'
            ),
            (  # truncated exponential distribution
                'random_exponential/exponential_vensim.tab'
            ),
        ],
        ids=["uniform", "normal", "exponential"]
    )
    def test_statistics_vensim(self, data_raw, data_python):
        raw_desc = stats.describe(data_raw, axis=0, nan_policy='omit')
        py_desc = stats.describe(data_python, axis=0, nan_policy='omit')
        for stat in ('mean', 'variance', 'skewness', 'kurtosis'):
            assert np.allclose(
                getattr(py_desc, stat),
                getattr(raw_desc, stat),
                atol=tolerance['data'][stat], rtol=tolerance['data'][stat]
                ), 'Failed when comparing %s:\n\t%s\n\t%s' % (
                    stat, getattr(raw_desc, stat), getattr(py_desc, stat))

    @pytest.mark.parametrize(
        "input_file",
        [
            (  # uniform distribution
                'random_uniform/uniform_expected.tab'
            ),
            (  # truncated normal distribution
                'random_normal/normal_expected.tab'
            ),
            (  # truncated exponential distribution
                'random_exponential/exponential_expected.tab'
            ),
        ],
        ids=["uniform", "normal", "exponential"]
    )
    def test_statistics_expected(self, data_raw, data_python):
        py_desc = stats.describe(data_python, axis=0, nan_policy='omit')
        for stat in ('mean', 'variance', 'skewness', 'kurtosis'):
            assert np.allclose(
                getattr(py_desc, stat),
                data_raw.loc[stat].values,
                atol=tolerance['expc'][stat], rtol=tolerance['expc'][stat]
                ), 'Failed when comparing %s:\n\t%s\n\t%s' % (
                    stat, data_raw.loc[stat], getattr(py_desc, stat))

        assert np.all(data_raw.loc['min'] < py_desc.minmax[0])
        assert np.all(data_raw.loc['max'] > py_desc.minmax[1])
