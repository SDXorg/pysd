import pytest

from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd

from pysd.tools.ncfiles import NCFile

@pytest.fixture(scope="session")
def sample_dataset():
    # generation of data (slight modification from xarray.Dataset example)
    np.random.seed(0)
    temperature = 15 + 8 * np.random.randn(3, 2, 2)
    precipitation = 10 * np.random.rand(3, 2, 2)
    altitude = 1000 * np.random.rand(2, 2)
    lon = [[-99.83, -99.32], [-99.79, -99.23]]
    lat = [[42.25, 42.21], [42.63, 42.59]]
    time = pd.date_range("2014-09-06", periods=3)

    ds = xr.Dataset(
        data_vars=dict(
            temperature=(
                ["time", "x", "y"], temperature),
            precipitation=(
                ["time", "x", "y"], precipitation),
            altitude=(["x", "y"], altitude),
            non_dimensional_var=([], np.array(5)),
            time=(["time"], time)
        ),
        coords=dict(
            lon=(["x", "y"], lon),
            lat=(["x", "y"], lat),
        ),
        attrs=dict(description="Weather related data."),
    )

    return ds

@pytest.fixture(scope="session")
def nc_file(sample_dataset, tmp_path_factory):

    path = tmp_path_factory.mktemp("data") / "dataset.nc"

    sample_dataset.to_netcdf(
        path, mode="w", format="NETCDF4", engine="netcdf4")

    return path


class TestNCFile():

    @pytest.mark.parametrize(
        "file_path,raises,expect",
        [(1, True, TypeError),
        ("unexisting_file.nc", True, FileNotFoundError),
        (Path(__file__), True, ValueError),
        ("nc_file", False, "nc_file")
        ],
        ids=["Wrong type", "Not found", "Wrong extension", "Correct path"]
        )
    def test__validate_nc_path_errors(
         self, request, file_path, raises, expect):
        if not raises:
            nc_file = request.getfixturevalue(file_path)
            assert NCFile._validate_nc_path(nc_file) == nc_file
        else:
            with pytest.raises(expect):
                NCFile._validate_nc_path(file_path)

    @pytest.mark.parametrize(
        "subset,raises,expected",
        [
            ([], False, [
                            "temperature",
                            "precipitation",
                            "altitude",
                            "non_dimensional_var",
                            "time"
                        ]
            ),
            (["temperature"], False, ["temperature", "time"]),
            (["non_existing"], True, ValueError),
            ("I am not a list", True, TypeError)
        ],
        ids=["empty", "one var", "non existing single var", "wrong type"]
        )
    @pytest.mark.filterwarnings("ignore")
    def test__validate_ds_subset(
         self, sample_dataset, subset, raises, expected):

        if raises:
            with pytest.raises(expected):
                NCFile._validate_ds_subset(sample_dataset, subset)
        else:
            assert NCFile._validate_ds_subset(
                sample_dataset, subset) == expected

    @pytest.mark.parametrize(
        "data_var,dims,coords,expected",
        [
            ("temperature", ["x", "y"], (0,0), ("temperature[0,0]", np.array(
                [29.11241876774131 ,29.94046392119974, 14.174249185651536]))
            ),
            ("altitude", ["x", "y"], (1,1), ("altitude[1,1]", np.array(
                944.66891705))
            ),
        ],
        ids = ["time-dependent data var", # returns vector
                "constant data var"] # returns scalar
        )
    def test__index_da_by_coord_labels(
        self, sample_dataset, data_var, dims, coords, expected):

        da = sample_dataset.data_vars[data_var]

        result = NCFile._index_da_by_coord_labels(
            da, dims, coords)

        assert result[0] == expected[0]
        assert np.allclose(result[1], expected[1])

    @pytest.mark.parametrize(
        "data_var,expected_keys",
        [
        ("temperature",
        [
            'temperature[0,0]',
            'temperature[0,1]',
            'temperature[1,0]',
            'temperature[1,1]'
        ]),
        ("altitude",
        [
            'altitude[0,0]',
            'altitude[0,1]',
            'altitude[1,0]',
            'altitude[1,1]'
        ]),
        ],
        ids = ["time-dependent data var", "constant data var"]
        )
    def test_da_to_dict(self, sample_dataset, data_var, expected_keys):
        idx = "time"
        da = sample_dataset.data_vars[data_var]

        serial = NCFile.da_to_dict(da, index_dim=idx)
        assert all(map(lambda x: x in serial.keys(), expected_keys))

        delayed = NCFile.da_to_dict_delayed(da, index_dim=idx)

        assert all(
            map(lambda key: np.allclose(serial[key], delayed[key]),
                expected_keys
            )
        )

    @pytest.mark.parametrize(
        "d,raises,expected",
        [
        ({"time": [1, 2, 3], "col1": 4, "col2": [2, 3, 4]}, False,
        [[1, 2, 3], [4]*3]
        ),
        ({"col1": 4, "col2": [2, 3, 4]}, True, "irrelevant")
        ],
        ids=["ok", "missing time"])
    def test_dict_to_df(self, d, raises, expected):

        if raises:
            with pytest.raises(KeyError):
                NCFile.dict_to_df(d)
        else:
            df = NCFile.dict_to_df(d)

            assert all(df.index == expected[0])
            assert all(df["col1"] == expected[1])

    def test_df_to_text_file_errors(self, nc_file):

        with pytest.raises(TypeError):
            NCFile.df_to_text_file(None, str(nc_file), time_in_row=False)

        csv_path = nc_file.parent / (nc_file.name + '.csv')
        with pytest.raises(ValueError):
            NCFile.df_to_text_file(None, csv_path, time_in_row="False")

    @pytest.mark.parametrize(
        "outfmt,time_in_row,parallel", [
            (".csv", True, True),
            (".csv", True, False),
            (".csv", False, True),
            (".csv", False, False),
            (".tab", True, True),
            (".tab", True, False),
            (".tab", False, True),
            (".tab", False, False)
            ])
    def test_to_text_file(
         self, shared_tmpdir, nc_file, outfmt, time_in_row, parallel):
        obj = NCFile(nc_file, parallel=parallel)
        assert obj.ncfile == nc_file
        assert isinstance(obj.ds, xr.Dataset)

        outfile = shared_tmpdir / f"data_from_nc{outfmt}"
        obj.to_text_file(outfile, time_in_row=time_in_row)
        assert outfile.is_file()

    def test_ds_to_df(self, sample_dataset):
        df = NCFile.ds_to_df(sample_dataset, subset=["temperature"])
        assert all(map(lambda x: x.startswith("temperature"),df.columns))
        assert df.shape == (3, 4)
