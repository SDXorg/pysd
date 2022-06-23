import sys
import pytest
import shutil

import numpy as np
import pandas as pd

from pysd.tools.benchmarking import assert_frames_close
from pysd import read_vensim, load


@pytest.fixture(scope="module")
def data_folder(_root):
    return _root.joinpath("more-tests/data_model/")


@pytest.fixture(scope="module")
def data_model(data_folder):
    return data_folder.joinpath("test_data_model.mdl")


@pytest.fixture
def data_files(data_files_short, data_folder):
    if isinstance(data_files_short, str):
        return data_folder.joinpath(data_files_short)
    elif isinstance(data_files_short, list):
        return [data_folder.joinpath(df) for df in data_files_short]
    else:
        return {
            data_folder.joinpath(df): value
            for df, value in data_files_short.items()
            }


times = np.arange(11)


@pytest.mark.parametrize(
    "data_files_short,expected",
    [
        (  # one_file
            "data1.tab",
            pd.DataFrame(
                index=times,
                data={'var1': times, "var2": 2*times, "var3": 3*times}
            )
        ),
        (  # two_files
            ["data3.tab",
             "data1.tab"],
            pd.DataFrame(
                index=times,
                data={'var1': -times, "var2": -2*times, "var3": 3*times}
            )

        ),
        (  # transposed_file
            ["data2.tab"],
            pd.DataFrame(
                index=times,
                data={'var1': times-5, "var2": 2*times-5, "var3": 3*times-5}
            )

        ),
        (  # dict_file
            {"data2.tab": ["\"data-3\""],
             "data1.tab": ["data_1", "Data 2"]},
            pd.DataFrame(
                index=times,
                data={'var1': times, "var2": 2*times, "var3": 3*times-5}
            )
        )

    ],
    ids=["one_file", "two_files", "transposed_file", "dict_file"]
)
class TestPySDData:

    @pytest.fixture
    def model(self, data_model, data_files, shared_tmpdir):
        # translated file
        file = shared_tmpdir.joinpath(data_model.with_suffix(".py").name)
        if file.is_file():
            # load already translated file
            return load(file, data_files)
        else:
            # copy mdl file to tmp_dir and translate it
            file = shared_tmpdir.joinpath(data_model.name)
            shutil.copy(data_model, file)
            return read_vensim(file, data_files)

    def test_get_data_and_run(self, model, expected):
        assert_frames_close(
            model.run(return_columns=["var1", "var2", "var3"]),
            expected)

    def test_modify_data(self, model, expected):
        out = model.run(params={
            "var1": pd.Series(index=[1, 3, 7], data=[10, 20, 30]),
            "var2": 10
        })

        assert (out["var2"] == 10).all()
        assert (
            out["var1"] == [10, 10, 15, 20, 22.5, 25, 27.5, 30, 30, 30, 30]
        ).all()


class TestPySDDataErrors:
    def model(self, data_model, data_files, shared_tmpdir):
        # translated file
        file = shared_tmpdir.joinpath(data_model.with_suffix(".py").name)
        if file.is_file():
            # load already translated file
            return load(file, data_files)
        else:
            # copy mdl file to tmp_dir and translate it
            file = shared_tmpdir.joinpath(data_model.name)
            shutil.copy(data_model, file)
            return read_vensim(file, data_files)

    def test_run_error(self, data_model,  shared_tmpdir):
        model = self.model(data_model, [], shared_tmpdir)
        error_message = "Trying to interpolate data variable before loading"\
            + " the data..."

        with pytest.raises(ValueError, match=error_message):
            model.run(return_columns=["var1", "var2", "var3"])

    @pytest.mark.parametrize(
        "data_files_short,raise_type,error_message",
        [
            (  # missing_data
                "data3.tab",
                ValueError,
                "Data for \"data-3\" not found in %s"
            ),
            (  # data_variable_not_found_from_dict_file
                {"data1.tab": ["non-existing-var"]},
                ValueError,
                "'non-existing-var' not found as model data variable"
            ),
        ],
        ids=["missing_data", "data_variable_not_found_from_dict_file"]
    )
    @pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason=r"bad scape \e")
    def test_loading_error(self, data_model, data_files, raise_type,
                           error_message, shared_tmpdir):
        with pytest.raises(raise_type, match=error_message % (data_files)):
            self.model(
                data_model, data_files, shared_tmpdir)
