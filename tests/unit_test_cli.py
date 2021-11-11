import sys
import os
import shutil
import unittest
import subprocess

import pandas as pd
import numpy as np

from pysd.tools.benchmarking import load_outputs, assert_frames_close
from pysd import __version__

_root = os.path.dirname(__file__)

test_model = os.path.join(_root, "test-models/samples/teacup/teacup.mdl")
test_model_xmile = os.path.join(
    _root, "test-models/samples/teacup/teacup.xmile")
test_model_subs = os.path.join(
    _root,
    "test-models/tests/subscript_2d_arrays/test_subscript_2d_arrays.mdl")
test_model_look = os.path.join(
    _root,
    "test-models/tests/get_lookups_subscripted_args/"
    + "test_get_lookups_subscripted_args.mdl")

out_tab_file = os.path.join(_root, "cli_output.tab")
out_csv_file = os.path.join(_root, "cli_output.csv")

encoding_stdout = sys.stdout.encoding or "utf-8"
encoding_stderr = sys.stderr.encoding or "utf-8"

call = "python -m pysd"


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


class TestPySD(unittest.TestCase):
    """ These tests are similar to unit_test_pysd but adapted for cli """
    def test_read_not_model(self):

        model = os.path.join(
            _root, "more-tests/not_vensim/test_not_vensim.txt")
        command = f"{call} {model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn(f"PySD: error: when parsing {model}", stderr)
        self.assertIn(
            "The model file name must be Vensim (.mdl), Xmile (.xmile) "
            "or PySD (.py) model file...", stderr)

    def test_read_model_not_exists(self):

        model = os.path.join(
            _root, "more-tests/not_vensim/test_not_vensim.mdl")
        command = f"{call} {model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn(f"PySD: error: when parsing {model}", stderr)
        self.assertIn(
            "The model file does not exist...", stderr)

    def test_read_not_valid_output(self):

        out_xls_file = os.path.join(_root, "cli_output.xls")
        command = f"{call} -o {out_xls_file} {test_model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn(f"PySD: error: when parsing {out_xls_file}", stderr)
        self.assertIn(
            "The output file name must be .tab or .csv...", stderr)

    def test_read_not_valid_time_stamps(self):

        time_stamps = "1, 3, 4, a"
        command = f"{call} -R '{time_stamps}' {test_model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn(f"PySD: error: when parsing {time_stamps}", stderr)
        self.assertIn(
            "The return time stamps much be separated by commas...\n", stderr)

        time_stamps = "1 3 4"
        command = f"{call} --return-timestamps='{time_stamps}' {test_model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn(f"PySD: error: when parsing {time_stamps}", stderr)
        self.assertIn(
            "The return time stamps much be separated by commas...\n", stderr)

    def test_read_not_valid_new_value(self):

        new_value = "foo=[1,2,3]"
        command = f"{call} {test_model} {new_value}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn(f"PySD: error: when parsing {new_value}", stderr)
        self.assertIn(
            "You must use variable=new_value to redefine values", stderr)

        new_value = "[1,2,3]"
        command = f"{call} {test_model} {new_value}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn(f"PySD: error: when parsing {new_value}", stderr)
        self.assertIn(
            "You must use variable=new_value to redefine values", stderr)

        new_value = "foo:[[1,2,3],[4,5,6]]"
        command = f"{call} {test_model} {new_value}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stderr = out.stderr.decode(encoding_stderr)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn(f"PySD: error: when parsing {new_value}", stderr)
        self.assertIn(
            "You must use variable=new_value to redefine values", stderr)

    def test_print_version(self):

        command = f"{call} -v"
        out = subprocess.run(split_bash(command), capture_output=True)
        stdout = out.stdout.decode(encoding_stdout)
        self.assertEqual(out.returncode, 0)
        self.assertIn(f"PySD {__version__}", stdout)

        command = f"{call} --version"
        out = subprocess.run(split_bash(command), capture_output=True)
        stdout = out.stdout.decode(encoding_stdout)
        self.assertEqual(out.returncode, 0)
        self.assertIn(f"PySD {__version__}", stdout)

    def test_print_help(self):

        command = f"{call}"
        out = subprocess.run(split_bash(command), capture_output=True)
        stdout = out.stdout.decode(encoding_stdout)
        self.assertEqual(out.returncode, 0)
        self.assertIn("usage: python -m pysd [", stdout)

        command = f"{call} -h"
        out = subprocess.run(split_bash(command), capture_output=True)
        stdout = out.stdout.decode(encoding_stdout)
        self.assertEqual(out.returncode, 0)
        self.assertIn("usage: python -m pysd [", stdout)

        command = f"{call} --help"
        out = subprocess.run(split_bash(command), capture_output=True)
        stdout = out.stdout.decode(encoding_stdout)
        self.assertEqual(out.returncode, 0)
        self.assertIn(
            "usage: python -m pysd [", stdout)

    def test_translate_file(self):

        model_py = test_model.replace(".mdl", ".py")

        if os.path.isfile(model_py):
            os.remove(model_py)
        if os.path.isfile(out_tab_file):
            os.remove(out_tab_file)

        command = f"{call} --translate {test_model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        self.assertFalse(os.path.isfile(out_tab_file))
        self.assertTrue(os.path.isfile(model_py))
        os.remove(model_py)

    def test_read_vensim_split_model(self):

        root_dir = os.path.join(_root, "more-tests/split_model") + "/"

        model_name = "test_split_model"
        namespace_filename = "_namespace_" + model_name + ".json"
        subscript_dict_filename = "_subscripts_" + model_name + ".json"
        modules_filename = "_modules.json"
        modules_dirname = "modules_" + model_name
        model_name_mdl = root_dir + model_name + ".mdl"

        command = f"{call} --translate --split-views {model_name_mdl}"
        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)

        # check that _namespace and _subscript_dict json files where created
        self.assertTrue(os.path.isfile(root_dir + namespace_filename))
        self.assertTrue(os.path.isfile(root_dir + subscript_dict_filename))

        # check that the main model file was created
        self.assertTrue(os.path.isfile(root_dir + model_name + ".py"))

        # check that the modules folder was created
        self.assertTrue(os.path.isdir(root_dir + modules_dirname))
        self.assertTrue(
            os.path.isfile(root_dir + modules_dirname + "/" + modules_filename)
        )

        # check creation of module files
        self.assertTrue(
            os.path.isfile(root_dir + modules_dirname + "/" + "view_1.py"))
        self.assertTrue(
            os.path.isfile(root_dir + modules_dirname + "/" + "view2.py"))
        self.assertTrue(
            os.path.isfile(root_dir + modules_dirname + "/" + "view_3.py"))

        # remove newly created files
        os.remove(root_dir + model_name + ".py")
        os.remove(root_dir + namespace_filename)
        os.remove(root_dir + subscript_dict_filename)

        # remove newly created modules folder
        shutil.rmtree(root_dir + modules_dirname)

    def test_read_vensim_split_model_subviews(self):
        import pysd
        from pysd.tools.benchmarking import assert_frames_close

        root_dir = os.path.join(_root, "more-tests/split_model/")

        model_name = "test_split_model_subviews"
        model_name_mdl = root_dir + model_name + ".mdl"

        model_split = pysd.read_vensim(
            root_dir + model_name + ".mdl", split_views=True,
            subview_sep=["."]
        )

        namespace_filename = "_namespace_" + model_name + ".json"
        subscript_dict_filename = "_subscripts_" + model_name + ".json"
        modules_dirname = "modules_" + model_name

        separator = "."
        command = f"{call} --translate --split-views "\
                  f"--subview-sep={separator} {model_name_mdl}"
        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)

        # check that the modules folders were created
        self.assertTrue(os.path.isdir(root_dir + modules_dirname + "/view_1"))

        # check creation of module files
        self.assertTrue(
            os.path.isfile(root_dir + modules_dirname + "/view_1/" +
                           "submodule_1.py"))
        self.assertTrue(
            os.path.isfile(root_dir + modules_dirname + "/view_1/" +
                           "submodule_2.py"))
        self.assertTrue(
            os.path.isfile(root_dir + modules_dirname + "/view_2.py"))

        # check that the results of the split model are the same than those
        # without splitting
        model_non_split = pysd.read_vensim(
            root_dir + model_name + ".mdl", split_views=False
        )

        result_split = model_split.run()
        result_non_split = model_non_split.run()

        # results of a split model are the same that those of the regular
        # model (un-split)
        assert_frames_close(result_split, result_non_split, atol=0, rtol=0)

        # remove newly created files
        os.remove(root_dir + model_name + ".py")
        os.remove(root_dir + namespace_filename)
        os.remove(root_dir + subscript_dict_filename)

        # remove newly created modules folder
        shutil.rmtree(root_dir + modules_dirname)

    def test_run_return_timestamps(self):

        timestamps =\
            np.random.randint(1, 5, 5).cumsum().astype(float).astype(str)
        command = f"{call} -o {out_csv_file} -R {','.join(timestamps)} "\
                  f" {test_model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks = load_outputs(out_csv_file)
        self.assertTrue((stocks.index.values.astype(str) == timestamps).all())
        os.remove(out_csv_file)

        command = f"{call} -o {out_csv_file} -R 5 {test_model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks = load_outputs(out_csv_file)
        self.assertTrue((stocks.index.values == [5]))
        os.remove(out_csv_file)

    def test_run_return_columns(self):
        return_columns = ["Room Temperature", "Teacup Temperature"]
        command = f"{call} -o {out_csv_file} -r "\
                  f"'{', '.join(return_columns)}' "\
                  f" {test_model}"

        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks = load_outputs(out_csv_file)
        self.assertEqual(set(stocks.columns), set(return_columns))
        os.remove(out_csv_file)

        # from txt
        txt_file = os.path.join(_root, "return_columns.txt")
        return_columns = ["Room Temperature", "Teacup Temperature"]
        with open(txt_file, "w") as file:
            file.write("\n".join(return_columns))

        command = f"{call} -o {out_csv_file} -r {txt_file} {test_model}"

        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks = load_outputs(out_csv_file)
        self.assertEqual(set(stocks.columns), set(return_columns))

        os.remove(txt_file)
        os.remove(out_csv_file)

        return_columns = ["room_temperature", "teacup_temperature"]
        command = f"{call} -o {out_csv_file} -r "\
                  f"'{', '.join(return_columns)}' "\
                  f" {test_model}"

        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks = load_outputs(out_csv_file)
        self.assertEqual(set(stocks.columns), set(return_columns))
        os.remove(out_csv_file)

    def test_model_arguments(self):
        # check initial time
        initial_time = 10
        command = f"{call} -o {out_tab_file} -I {initial_time} {test_model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks = load_outputs(out_tab_file)
        self.assertTrue(stocks.index.values[0] == initial_time)
        os.remove(out_tab_file)

        # check final time
        final_time = 20
        command = f"{call} -o {out_tab_file} -F {final_time} {test_model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks = load_outputs(out_tab_file)
        self.assertTrue(stocks.index.values[-1] == final_time)
        os.remove(out_tab_file)

        # check time step
        time_step = 10
        command = f"{call} -o {out_tab_file} -T {time_step} {test_model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks = load_outputs(out_tab_file)
        self.assertTrue((np.diff(stocks.index.values) == time_step).all())
        self.assertTrue((stocks["SAVEPER"] == time_step).all().all())
        self.assertTrue((stocks["TIME STEP"] == time_step).all().all())
        os.remove(out_tab_file)

        # check saveper
        time_step = 5
        saveper = 10
        command = f"{call} -o {out_tab_file} -T {time_step} "\
                  f"-S {saveper} {test_model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks = load_outputs(out_tab_file)
        self.assertTrue((np.diff(stocks.index.values) == saveper).all())
        self.assertTrue((stocks["SAVEPER"] == saveper).all().all())
        self.assertTrue((stocks["TIME STEP"] == time_step).all().all())
        os.remove(out_tab_file)

        # check all
        initial_time = 15
        time_step = 5
        saveper = 10
        final_time = 45
        command = f"{call} -o {out_tab_file} --time-step={time_step} "\
                  f"--saveper={saveper} --initial-time={initial_time} "\
                  f"--final-time={final_time} {test_model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks = load_outputs(out_tab_file)
        self.assertTrue((np.diff(stocks.index.values) == saveper).all())
        self.assertTrue(stocks.index.values[0] == initial_time)
        self.assertTrue(stocks.index.values[-1] == final_time)
        os.remove(out_tab_file)

    def test_initial_conditions_tuple_pysafe_names(self):
        import pysd
        model = pysd.read_vensim(test_model)
        initial_time = 3000
        return_timestamps = np.arange(initial_time, initial_time+10)
        stocks = model.run(
            initial_condition=(initial_time, {"teacup_temperature": 33}),
            return_timestamps=return_timestamps)

        command = f"{call} -o {out_tab_file} -I {initial_time} -R "\
                  f"'{', '.join(return_timestamps.astype(str))}'"\
                  f" {test_model.replace('.mdl', '.py')}"\
                  f" teacup_temperature:33"

        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks2 = load_outputs(out_tab_file)
        assert_frames_close(stocks2, stocks)
        os.remove(out_tab_file)

    def test_set_constant_parameter(self):

        value = 20
        command = f"{call} -o {out_tab_file} -r room_temperature "\
                  f" {test_model_xmile}"\
                  f" room_temperature={value}"

        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks = load_outputs(out_tab_file)
        self.assertTrue((stocks["room_temperature"] == value).all())
        os.remove(out_tab_file)

    def test_set_timeseries_parameter_lookup(self):

        timeseries = np.arange(30)
        data = np.round(50 + np.random.rand(len(timeseries)).cumsum(), 4)

        temp_timeseries = pd.Series(
            index=timeseries,
            data=data)

        timeseries_bash = "[[" + ",".join(timeseries.astype(str)) + "],["\
                          + ",".join(data.astype(str)) + "]]"

        command = f"{call} -o {out_tab_file} -r lookup_1d_time "\
                  f"-R {','.join(timeseries.astype(str))} "\
                  f" {test_model_look}"\
                  f" lookup_1d_time={timeseries_bash}"
        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks = load_outputs(out_tab_file)
        self.assertTrue((stocks["lookup_1d_time"] == temp_timeseries).all())
        os.remove(out_tab_file)

        command = f"{call} -o {out_tab_file} -r lookup_2d_time "\
                  f"-R {','.join(timeseries.astype(str))}"\
                  f" {test_model_look}"\
                  f" lookup_2d_time={timeseries_bash}"
        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks = load_outputs(out_tab_file)
        self.assertTrue(
            (stocks["lookup_2d_time[Row1]"] == temp_timeseries).all())
        self.assertTrue(
            (stocks["lookup_2d_time[Row2]"] == temp_timeseries).all())
        os.remove(out_tab_file)

    def test_export_import(self):
        import pysd
        from pysd.tools.benchmarking import assert_frames_close

        exp_file = "teacup15.pic"
        model = pysd.read_vensim(test_model)
        stocks = model.run(return_timestamps=[0, 10, 20, 30])
        self.assertTrue((stocks["INITIAL TIME"] == 0).all().all())

        command = f"{call} -o {out_tab_file} -e {exp_file} -F 15 -R 0,10"\
                  f" {test_model}"

        command2 = f"{call} -o {out_tab_file} -i {exp_file} -R 20,30"\
                   f" {test_model}"

        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks1 = load_outputs(out_tab_file)

        out = subprocess.run(split_bash(command2), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks2 = load_outputs(out_tab_file)

        os.remove(exp_file)
        os.remove(out_tab_file)

        self.assertTrue((stocks1["INITIAL TIME"] == 0).all().all())
        self.assertTrue((stocks1["FINAL TIME"] == 15).all().all())
        self.assertTrue((stocks2["INITIAL TIME"] == 15).all().all())
        stocks.drop("INITIAL TIME", axis=1, inplace=True)
        stocks1.drop("INITIAL TIME", axis=1, inplace=True)
        stocks2.drop("INITIAL TIME", axis=1, inplace=True)
        stocks.drop("FINAL TIME", axis=1, inplace=True)
        stocks1.drop("FINAL TIME", axis=1, inplace=True)
        stocks2.drop("FINAL TIME", axis=1, inplace=True)

        assert_frames_close(stocks1, stocks.loc[[0, 10]])
        assert_frames_close(stocks2, stocks.loc[[20, 30]])

    def test_run_model_with_data(self):
        data_file = os.path.join(
            _root, "test-models/tests/data_from_other_model/data.tab")
        model_file = os.path.join(
            _root,
            "test-models/tests/data_from_other_model/"
            + "test_data_from_other_model.mdl")

        command = f"{call} -o {out_tab_file} -D {data_file}"\
                  f" {model_file}"

        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stocks = load_outputs(out_tab_file)
        canon = load_outputs(os.path.join(
            _root,
            "test-models/tests/data_from_other_model/output.tab"))

        assert_frames_close(stocks[canon.columns], canon)

        # invalid data file
        command = f"{call} -o {out_tab_file} -D my_file.txt"\
                  f" {model_file}"

        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertNotEqual(out.returncode, 0)
        stderr = out.stderr.decode(encoding_stderr)
        self.assertIn("PySD: error: when parsing my_file.txt", stderr)
        self.assertIn(
            "The data file name must be .tab or .csv...", stderr)

        # not found data file
        command = f"{call} -o {out_tab_file} -D my_file.tab"\
                  f" {model_file}"

        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertNotEqual(out.returncode, 0)
        stderr = out.stderr.decode(encoding_stderr)
        self.assertIn("PySD: error: when parsing my_file.tab", stderr)
        self.assertIn(
            "The data file does not exist...", stderr)

    def test_save_without_name(self):
        import re

        command = f"{call} {test_model}"
        command2 = f"{call} -o {out_tab_file} {test_model}"
        out = subprocess.run(split_bash(command), capture_output=True)
        self.assertEqual(out.returncode, 0)
        stdout = out.stdout.decode(encoding_stdout)
        outputs = re.findall("(?<=Data saved in ').*(?=')", stdout)[0]
        out2 = subprocess.run(split_bash(command2), capture_output=True)
        self.assertEqual(out2.returncode, 0)

        out, out2 = load_outputs(outputs), load_outputs(out_tab_file)
        os.remove(outputs)
        os.remove(out_tab_file)

        self.assertTrue((out - out2 == 0).all().all())
