import sys
import re
import os
from pathlib import Path
import shutil
import subprocess

import pytest

import pandas as pd
import numpy as np

import pysd
from pysd.tools.benchmarking import load_outputs, assert_frames_close

test_model_ven = "test-models/samples/teacup/teacup.mdl"

encoding_stdout = sys.stdout.encoding or "utf-8"
encoding_stderr = sys.stderr.encoding or "utf-8"

call = "python -m pysd"


@pytest.fixture
def out_tab(tmp_path):
    return tmp_path.joinpath("cli_output.tab")


@pytest.fixture
def out_csv(tmp_path):
    return tmp_path.joinpath("cli_output.csv")


@pytest.fixture
def test_model(model, _root):
    return _root.joinpath(model)


@pytest.fixture
def test_copy(tmp_path, test_model):
    """
    Copy test folder to a temporary folder therefore we avoid creating
    PySD model files in the original folder
    """
    test_folder = tmp_path.joinpath(test_model.parent.name)
    shutil.copytree(test_model.parent, test_folder)
    return test_folder.joinpath(test_model.name)


def split_bash(string):
    """
    Function to split the bash command as bash does

    "ABC '1, 2, 3' CBD" -> ["ABC", "1, 2, 3", "CBD"]
    """
    open = False
    s = ""
    spl = []
    for c in string:
        if c in ["'", '"']:
            # open or close ''
            open = not open
        elif c == " " and not open:
            s and spl.append(s)
            s = ""
        else:
            s += c
    s and spl.append(s)
    return spl


class TestPySD():
    """ These tests are similar to pytest_pysd but adapted for cli """
    def test_read_not_model(self, _root):

        model = _root.joinpath("more-tests/not_vensim/test_not_vensim.txt")
        command = f"{call} {model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)
        assert out.returncode != 0
        assert f"PySD: error: when parsing {model}" in stderr
        assert "The model file name must be a Vensim (.mdl), a Xmile "\
            "(.xmile, .xml, .stmx) or a PySD (.py) model file..." in stderr

    def test_read_model_not_exists(self, _root):

        model = _root.joinpath("more-tests/not_vensim/test_not_vensim.mdl")
        command = f"{call} {model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)
        assert out.returncode != 0
        assert f"PySD: error: when parsing {model}" in stderr
        assert "The model file does not exist..." in stderr

    def test_read_not_valid_output(self, _root):

        out_xls_file = _root.joinpath("cli_output.xls")
        command = f"{call} -o {out_xls_file} {test_model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)
        assert out.returncode != 0
        assert f"PySD: error: when parsing {out_xls_file}" in stderr
        assert "The output file name must be .tab or .csv..." in stderr

    def test_read_not_valid_time_stamps(self):

        time_stamps = "1, 3, 4, a"
        command = f"{call} -R '{time_stamps}' {test_model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)

        assert out.returncode != 0
        assert f"PySD: error: when parsing {time_stamps}" in stderr
        assert "The return time stamps must be separated by commas..."\
            in stderr

        time_stamps = "1 3 4"
        command = f"{call} --return-timestamps='{time_stamps}' {test_model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)
        assert out.returncode != 0
        assert f"PySD: error: when parsing {time_stamps}" in stderr
        assert "The return time stamps must be separated by commas..."\
            in stderr

    @pytest.mark.parametrize("model", [test_model_ven])
    def test_read_not_valid_new_value(self, test_model):

        new_value = "foo=[1,2,3]"
        command = f"{call} {test_model} {new_value}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)
        assert out.returncode != 0
        assert f"PySD: error: when parsing {new_value}" in stderr
        assert "You must use variable=new_value to redefine values" in stderr

        new_value = "[1,2,3]"
        command = f"{call} {test_model} {new_value}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)
        assert out.returncode != 0
        assert f"PySD: error: when parsing {new_value}" in stderr
        assert "You must use variable=new_value to redefine values" in stderr

        new_value = "foo:[[1,2,3],[4,5,6]]"
        command = f"{call} {test_model} {new_value}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)
        assert out.returncode != 0
        assert f"PySD: error: when parsing {new_value}" in stderr
        assert "You must use variable=new_value to redefine values" in stderr

    def test_print_version(self):

        command = f"{call} -v"
        out = subprocess.run(split_bash(command), capture_output=True)
        stdout = out.stdout.decode(encoding_stdout)
        assert out.returncode == 0
        assert f"PySD {pysd.__version__}" in stdout

        command = f"{call} --version"
        out = subprocess.run(split_bash(command), capture_output=True)
        stdout = out.stdout.decode(encoding_stdout)
        assert out.returncode == 0
        assert f"PySD {pysd.__version__}" in stdout

    def test_print_help(self):

        command = f"{call}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stdout = out.stdout.decode(encoding_stdout)
        assert out.returncode == 0
        assert "usage: python -m pysd [" in stdout

        command = f"{call} -h"
        out = subprocess.run(split_bash(command), capture_output=True)
        stdout = out.stdout.decode(encoding_stdout)
        assert out.returncode == 0
        assert "usage: python -m pysd [" in stdout

        command = f"{call} --help"
        out = subprocess.run(split_bash(command), capture_output=True)
        stdout = out.stdout.decode(encoding_stdout)
        assert out.returncode == 0
        assert "usage: python -m pysd [" in stdout

    @pytest.mark.parametrize("model", [test_model_ven])
    def test_translate_file(self, test_copy, out_tab):
        command = f"{call} --translate {test_copy}"
        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        assert not out_tab.exists()
        assert test_copy.with_suffix(".py").exists()

    @pytest.mark.parametrize(
        "model", ["more-tests/split_model/test_split_model.mdl"]
    )
    def test_read_vensim_split_model(self, test_copy):

        model_name = test_copy.with_suffix("").name
        folder = test_copy.parent
        modules_dirname = folder / ("modules_" + model_name)

        command = f"{call} --translate --split-views {test_copy}"
        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0

        # check that _subscript_dict json file was created
        assert (folder / ("_subscripts_" + model_name + ".json")).exists()

        # check that the main model file was created
        assert test_copy.with_suffix(".py").exists()

        # check that the modules folder was created
        assert modules_dirname.exists()
        assert modules_dirname.is_dir()
        assert (modules_dirname / "_modules.json").exists()

        # check creation of module files
        assert (modules_dirname / "view_1.py").exists()
        assert (modules_dirname / "view2.py").exists()
        assert (modules_dirname / "view_3.py").exists()

    @pytest.mark.parametrize(
        "model", ["more-tests/split_model/test_split_model_subviews.mdl"]
    )
    def test_read_vensim_split_model_subviews(self, test_copy):

        model_name = test_copy.with_suffix("").name
        folder = test_copy.parent
        modules_dirname = folder / ("modules_" + model_name)

        separator = "."
        command = f"{call} --translate --split-views "\
                  f"--subview-sep={separator} {test_copy}"
        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0

        # check that _subscript_dict json file was created
        assert (folder / ("_subscripts_" + model_name + ".json")).exists()

        # check that the main model file was created
        assert test_copy.with_suffix(".py").exists()

        # check that the modules folder was created
        assert modules_dirname.exists()
        assert modules_dirname.is_dir()
        assert (modules_dirname / "_modules.json").exists()

        # check creation of module files
        assert (modules_dirname / "view_1").exists()
        assert (modules_dirname / "view_1").is_dir()
        assert (modules_dirname / "view_1" / "submodule_1.py").exists()
        assert (modules_dirname / "view_1" / "submodule_2.py").exists()
        assert (modules_dirname / "view_2.py").exists()

    @pytest.mark.parametrize("model", [test_model_ven])
    def test_run_return_timestamps(self, out_csv, out_tab, test_copy):

        timestamps =\
            np.random.randint(1, 5, 5).cumsum().astype(float).astype(str)
        command = f"{call} -o {out_csv} -R {','.join(timestamps)} "\
                  f" {test_copy}"
        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks = load_outputs(out_csv)
        assert (stocks.index.values.astype(str) == timestamps).all()

        command = f"{call} -o {out_tab} -R 5 {test_copy}"
        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks = load_outputs(out_tab)
        assert (stocks.index.values == [5])

    @pytest.mark.parametrize("model", [test_model_ven])
    def test_run_return_columns(self, out_csv, test_copy, tmp_path):
        return_columns = ["Room Temperature", "Teacup Temperature"]
        command = f"{call} -o {out_csv} -r '{', '.join(return_columns)}' "\
                  f" {test_copy}"

        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks = load_outputs(out_csv)
        assert set(stocks.columns) == set(return_columns)

        out_csv = out_csv.parent / "new_out.csv"

        # from txt
        txt_file = tmp_path / "return_columns.txt"
        return_columns = ["Room Temperature", "Teacup Temperature"]
        with open(txt_file, "w") as file:
            file.write("\n".join(return_columns))

        command = f"{call} -o {out_csv} -r {txt_file} {test_copy}"

        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks = load_outputs(out_csv)
        assert set(stocks.columns) == set(return_columns)

        out_csv = out_csv.parent / "new_out2.csv"

        return_columns = ["room_temperature", "teacup_temperature"]
        command = f"{call} -o {out_csv} -r "\
                  f"'{', '.join(return_columns)}' "\
                  f" {test_copy}"

        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks = load_outputs(out_csv)
        assert set(stocks.columns) == set(return_columns)

    @pytest.mark.parametrize("model", [test_model_ven])
    def test_model_arguments(self, out_tab, test_copy):
        # check initial time
        initial_time = 10
        command = f"{call} -o {out_tab} -I {initial_time} {test_copy}"
        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks = load_outputs(out_tab)
        assert stocks.index.values[0] == initial_time

        out_tab = out_tab.parent / "new_out.tab"

        # check final time
        final_time = 20
        command = f"{call} -o {out_tab} -F {final_time} {test_copy}"
        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks = load_outputs(out_tab)
        assert stocks.index.values[-1] == final_time

        out_tab = out_tab.parent / "new_out2.tab"

        # check time step
        time_step = 10
        command = f"{call} -o {out_tab} -T {time_step} {test_copy}"
        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks = load_outputs(out_tab)
        assert (np.diff(stocks.index.values) == time_step).all()
        assert (stocks["SAVEPER"] == time_step).all().all()
        assert (stocks["TIME STEP"] == time_step).all().all()

        out_tab = out_tab.parent / "new_out3.tab"

        # check saveper
        time_step = 5
        saveper = 10
        command = f"{call} -o {out_tab} -T {time_step} "\
                  f"-S {saveper} {test_copy}"
        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks = load_outputs(out_tab)
        assert (np.diff(stocks.index.values) == saveper).all()
        assert (stocks["SAVEPER"] == saveper).all().all()
        assert (stocks["TIME STEP"] == time_step).all().all()

        out_tab = out_tab.parent / "new_out4.tab"

        # check all
        initial_time = 15
        time_step = 5
        saveper = 10
        final_time = 45
        command = f"{call} -o {out_tab} --time-step={time_step} "\
                  f"--saveper={saveper} --initial-time={initial_time} "\
                  f"--final-time={final_time} {test_copy}"
        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks = load_outputs(out_tab)
        assert (np.diff(stocks.index.values) == saveper).all()
        assert stocks.index.values[0] == initial_time
        assert stocks.index.values[-1] == final_time

    @pytest.mark.parametrize("model", [test_model_ven])
    def test_initial_conditions_tuple_pysafe_names(self, out_tab, test_copy):
        import pysd
        model = pysd.read_vensim(test_copy)
        initial_time = 3000
        return_timestamps = np.arange(initial_time, initial_time+10)
        stocks = model.run(
            initial_condition=(initial_time, {"teacup_temperature": 33}),
            return_timestamps=return_timestamps)

        command = f"{call} -o {out_tab} -I {initial_time} -R "\
                  f"'{', '.join(return_timestamps.astype(str))}'"\
                  f" {test_copy.with_suffix('.py')}"\
                  f" teacup_temperature:33"

        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks2 = load_outputs(out_tab)
        assert_frames_close(stocks2, stocks)

    @pytest.mark.parametrize(
        "model", ["test-models/samples/teacup/teacup.xmile"]
    )
    def test_set_constant_parameter(self, out_tab, test_copy):

        value = 20
        command = f"{call} -o {out_tab} -r room_temperature "\
                  f" {test_copy} room_temperature={value}"

        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks = load_outputs(out_tab)
        assert (stocks["room_temperature"] == value).all()

    @pytest.mark.parametrize(
        "model",
        [
            "test-models/tests/get_lookups_subscripted_args/"
            "test_get_lookups_subscripted_args.mdl"
        ]
    )
    def test_set_timeseries_parameter_lookup(self, out_tab, test_copy):

        timeseries = np.arange(30)
        data = np.round(50 + np.random.rand(len(timeseries)).cumsum(), 4)

        temp_timeseries = pd.Series(
            index=timeseries,
            data=data)

        timeseries_bash = "[[" + ",".join(timeseries.astype(str)) + "],["\
                          + ",".join(data.astype(str)) + "]]"

        command = f"{call} -o {out_tab} -r lookup_1d_time "\
                  f"-R {','.join(timeseries.astype(str))} "\
                  f" {test_copy}"\
                  f" lookup_1d_time={timeseries_bash}"
        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks = load_outputs(out_tab)
        assert (stocks["lookup_1d_time"] == temp_timeseries).all()

        out_tab = out_tab.parent / "new_out.tab"

        command = f"{call} -o {out_tab} -r lookup_2d_time "\
                  f"-R {','.join(timeseries.astype(str))}"\
                  f" {test_copy}"\
                  f" lookup_2d_time={timeseries_bash}"
        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks = load_outputs(out_tab)
        assert (stocks["lookup_2d_time[Row1]"] == temp_timeseries).all()
        assert (stocks["lookup_2d_time[Row2]"] == temp_timeseries).all()

    @pytest.mark.parametrize("model", [test_model_ven])
    def test_export_import(self, out_tab, test_copy, tmp_path):
        import pysd
        from pysd.tools.benchmarking import assert_frames_close

        exp_file = tmp_path / "teacup15.pic"
        model = pysd.read_vensim(test_copy)
        stocks = model.run(return_timestamps=[0, 10, 20, 30])
        assert (stocks["INITIAL TIME"] == 0).all().all()

        command = f"{call} -o {out_tab} -e {exp_file} -F 15 -R 0,10"\
                  f" {test_copy}"

        command2 = f"{call} -o {out_tab} -i {exp_file} -R 20,30"\
                   f" {test_copy}"

        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks1 = load_outputs(out_tab)

        out = subprocess.run(split_bash(command2), capture_output=True)
        assert out.returncode == 0
        stocks2 = load_outputs(out_tab)

        assert (stocks1["INITIAL TIME"] == 0).all().all()
        assert (stocks1["FINAL TIME"] == 15).all().all()
        assert (stocks2["INITIAL TIME"] == 15).all().all()
        stocks.drop("INITIAL TIME", axis=1, inplace=True)
        stocks1.drop("INITIAL TIME", axis=1, inplace=True)
        stocks2.drop("INITIAL TIME", axis=1, inplace=True)
        stocks.drop("FINAL TIME", axis=1, inplace=True)
        stocks1.drop("FINAL TIME", axis=1, inplace=True)
        stocks2.drop("FINAL TIME", axis=1, inplace=True)

        assert_frames_close(stocks1, stocks.loc[[0, 10]])
        assert_frames_close(stocks2, stocks.loc[[20, 30]])

    @pytest.mark.parametrize(
        "model",
        [
            "test-models/tests/data_from_other_model/"
            "test_data_from_other_model.mdl"
        ]
    )
    def test_run_model_with_data(self, test_copy, out_tab):
        data_file = test_copy.parent / "data.tab"

        command = f"{call} -o {out_tab} -D {data_file}"\
                  f" {test_copy}"

        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode == 0
        stocks = load_outputs(out_tab)
        canon = load_outputs(test_copy.parent / "output.tab")

        assert_frames_close(stocks[canon.columns], canon)

        # invalid data file
        command = f"{call} -o {out_tab} -D my_file.txt"\
                  f" {test_copy}"

        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode != 0
        stderr = out.stderr.decode(encoding_stderr)
        assert "PySD: error: when parsing my_file.txt" in stderr
        assert "The data file name must be .tab or .csv..." in stderr

        # not found data file
        command = f"{call} -o {out_tab} -D my_file.tab"\
                  f" {test_copy}"

        out = subprocess.run(split_bash(command), capture_output=True)
        assert out.returncode != 0
        stderr = out.stderr.decode(encoding_stderr)
        assert "PySD: error: when parsing my_file.tab" in stderr
        assert "The data file does not exist..." in stderr

    @pytest.mark.parametrize("model", [test_model_ven])
    def test_save_without_name(self, out_tab, test_copy, tmp_path):
        prev_cwd = Path.cwd()
        os.chdir(tmp_path)
        command = f"{call} {test_copy}"
        out = subprocess.run(split_bash(command), capture_output=True)
        os.chdir(prev_cwd)
        assert out.returncode == 0
        stdout = out.stdout.decode(encoding_stdout)
        outputs = re.findall("(?<=Data saved in ').*(?=')", stdout)[0]

        out = load_outputs(tmp_path / outputs)
        out2 = pysd.read_vensim(test_copy).run()
        assert_frames_close(out, out2)
