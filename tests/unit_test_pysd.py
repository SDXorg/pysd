import unittest
import os
import shutil
from warnings import simplefilter, catch_warnings
import pandas as pd
import numpy as np
import xarray as xr

_root = os.path.dirname(__file__)

test_model = os.path.join(_root, "test-models/samples/teacup/teacup.mdl")
test_model_subs = os.path.join(
    _root,
    "test-models/tests/subscript_2d_arrays/test_subscript_2d_arrays.mdl")
test_model_look = os.path.join(
    _root,
    "test-models/tests/get_lookups_subscripted_args/"
    + "test_get_lookups_subscripted_args.mdl")

more_tests = os.path.join(_root, "more-tests")


class TestPySD(unittest.TestCase):
    def test_load_different_version_error(self):
        import pysd

        # old PySD major version
        with self.assertRaises(ImportError):
            pysd.load(more_tests + "/version/test_old_version.py")

        # current PySD major version
        pysd.load(more_tests + "/version/test_current_version.py")

    def test_load_type_error(self):
        import pysd

        with self.assertRaises(ImportError):
            pysd.load(more_tests + "/type_error/test_type_error.py")

    def test_read_not_model_vensim(self):
        import pysd

        with self.assertRaises(ValueError):
            pysd.read_vensim(more_tests + "/not_vensim/test_not_vensim.txt")

    def test_run(self):
        import pysd

        model = pysd.read_vensim(test_model)
        stocks = model.run()
        self.assertTrue(isinstance(stocks, pd.DataFrame))  # return a dataframe
        self.assertTrue(
            "Teacup Temperature" in stocks.columns.values
        )  # contains correct column
        self.assertGreater(len(stocks), 3)  # has multiple rows
        self.assertTrue(
            stocks.notnull().all().all()
        )  # there are no null values in the set

    def test_run_ignore_missing(self):
        import pysd

        model_mdl = os.path.join(
            _root,
            'test-models/tests/get_with_missing_values_xlsx/'
            + 'test_get_with_missing_values_xlsx.mdl')
        model_py = os.path.join(
            _root,
            'test-models/tests/get_with_missing_values_xlsx/'
            + 'test_get_with_missing_values_xlsx.py')

        with catch_warnings(record=True) as ws:
            # warnings for missing values
            model = pysd.read_vensim(model_mdl, missing_values="ignore")
            self.assertTrue(all(["missing" not in str(w.message) for w in ws]))

        with catch_warnings(record=True) as ws:
            # warnings for missing values
            model.run()
            self.assertTrue(all(["missing" not in str(w.message) for w in ws]))

        with catch_warnings(record=True) as ws:
            # ignore warnings for missing values
            model = pysd.load(model_py)
            self.assertTrue(any(["missing" in str(w.message) for w in ws]))

        with catch_warnings(record=True) as ws:
            # ignore warnings for missing values
            model.run()
            self.assertTrue(any(["missing" in str(w.message) for w in ws]))

        with self.assertRaises(ValueError):
            # errors for missing values
            pysd.load(model_py, missing_values="raise")

    def test_read_vensim_split_model(self):
        import pysd
        from pysd.tools.benchmarking import assert_frames_close

        root_dir = more_tests + "/split_model/"

        model_name = "test_split_model"
        model_split = pysd.read_vensim(
            root_dir + model_name + ".mdl", split_modules=True
        )

        namespace_filename = "_namespace_" + model_name + ".json"
        subscript_dict_filename = "_subscripts_" + model_name + ".json"
        modules_filename = "_modules.json"
        modules_dirname = "modules_" + model_name

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

        # check dictionaries
        self.assertIn("Stock", model_split.components._namespace.keys())
        self.assertIn("view2", model_split.components._modules.keys())
        self.assertIsInstance(model_split.components._subscript_dict, dict)

        # check that the results of the split model are the same than those
        # without splitting
        model_non_split = pysd.read_vensim(
            root_dir + model_name + ".mdl", split_modules=False
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

    def test_read_vensim_split_model_with_macro(self):
        import pysd
        from pysd.tools.benchmarking import assert_frames_close

        root_dir = more_tests + "/split_model_with_macro/"

        model_name = "test_split_model_with_macro"
        model_split = pysd.read_vensim(
            root_dir + model_name + ".mdl", split_modules=True
        )

        namespace_filename = "_namespace_" + model_name + ".json"
        subscript_dict_filename = "_subscripts_" + model_name + ".json"
        modules_dirname = "modules_" + model_name

        # check that the results of the split model are the same
        # than those without splitting
        model_non_split = pysd.read_vensim(
            root_dir + model_name + ".mdl", split_modules=False
        )

        result_split = model_split.run()
        result_non_split = model_non_split.run()

        # results of a split model are the same that those of the regular model
        assert_frames_close(result_split, result_non_split, atol=0, rtol=0)

        # remove newly created files
        os.remove(root_dir + model_name + ".py")
        os.remove(root_dir + "expression_macro.py")
        os.remove(root_dir + namespace_filename)
        os.remove(root_dir + subscript_dict_filename)

        # remove newly created modules folder
        shutil.rmtree(root_dir + modules_dirname)

    def test_read_vensim_split_model_warning(self):
        import pysd
        # setting the split_modules=True when the model has a single
        # view should generate a warning
        with catch_warnings(record=True) as ws:
            pysd.read_vensim(
                test_model, split_modules=True
            )  # set stock value using params

        wu = [w for w in ws if issubclass(w.category, UserWarning)]

        self.assertEqual(len(wu), 1)
        self.assertTrue(
            "Only one module was detected" in str(wu[0].message)
        )  # check that warning references the stock

    def test_run_includes_last_value(self):
        import pysd

        model = pysd.read_vensim(test_model)
        res = model.run()
        self.assertEqual(res.index[-1], model.components.final_time())

    def test_run_build_timeseries(self):
        import pysd

        model = pysd.read_vensim(test_model)

        model.components.initial_time = lambda: 3
        model.components.final_time = lambda: 7
        model.components.time_step = lambda: 1
        model.initialize()

        res = model.run()

        actual = list(res.index)
        expected = [3.0, 4.0, 5.0, 6.0, 7.0]
        self.assertSequenceEqual(actual, expected)

    def test_run_progress(self):
        import pysd

        # same as test_run but with progressbar
        model = pysd.read_vensim(test_model)
        stocks = model.run(progress=True)
        self.assertTrue(isinstance(stocks, pd.DataFrame))
        self.assertTrue("Teacup Temperature" in stocks.columns.values)
        self.assertGreater(len(stocks), 3)
        self.assertTrue(stocks.notnull().all().all())

    def test_run_return_timestamps(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/17"""
        import pysd

        model = pysd.read_vensim(test_model)
        timestamps = np.random.rand(5).cumsum()
        stocks = model.run(return_timestamps=timestamps)
        self.assertTrue((stocks.index.values == timestamps).all())

        stocks = model.run(return_timestamps=5)
        self.assertEqual(stocks.index[0], 5)

        timestamps = ['A', 'B']
        with self.assertRaises(TypeError):
            model.run(return_timestamps=timestamps)

    def test_run_return_timestamps_past_final_time(self):
        """ If the user enters a timestamp that is longer than the euler
        timeseries that is defined by the normal model file, should
        extend the euler series to the largest timestamp"""
        import pysd

        model = pysd.read_vensim(test_model)
        return_timestamps = list(range(0, 100, 10))
        stocks = model.run(return_timestamps=return_timestamps)
        self.assertSequenceEqual(return_timestamps, list(stocks.index))

    def test_return_timestamps_with_range(self):
        """
        Tests that return timestamps may receive a 'range'.
        It will be cast to a numpy array in the end...
        """
        import pysd

        model = pysd.read_vensim(test_model)
        return_timestamps = range(0, 100, 10)
        stocks = model.run(return_timestamps=return_timestamps)
        self.assertSequenceEqual(return_timestamps, list(stocks.index))

    def test_run_return_columns_original_names(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/26
        - Also checks that columns are returned in the correct order"""
        import pysd

        model = pysd.read_vensim(test_model)
        return_columns = ["Room Temperature", "Teacup Temperature"]
        result = model.run(return_columns=return_columns)
        self.assertEqual(set(result.columns), set(return_columns))

    def test_run_return_columns_step(self):
        """
        Return only cache 'step' variables
        """
        import pysd
        model = pysd.read_vensim(test_model)
        result = model.run(return_columns='step')
        self.assertEqual(
            set(result.columns),
            {'Teacup Temperature', 'SAVEPER', 'Heat Loss to Room'})

    def test_run_reload(self):
        """ Addresses https://github.com/JamesPHoughton/pysd/issues/99"""
        import pysd

        model = pysd.read_vensim(test_model)
        result0 = model.run()
        result1 = model.run(params={"Room Temperature": 1000})
        result2 = model.run()
        result3 = model.run(reload=True)

        self.assertTrue((result0 == result3).all().all())
        self.assertFalse((result0 == result1).all().all())
        self.assertTrue((result1 == result2).all().all())

    def test_run_return_columns_pysafe_names(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/26"""
        import pysd

        model = pysd.read_vensim(test_model)
        return_columns = ["room_temperature", "teacup_temperature"]
        result = model.run(return_columns=return_columns)
        self.assertEqual(set(result.columns), set(return_columns))

    def test_run_export_import(self):
        import pysd
        from pysd.tools.benchmarking import assert_frames_close

        with catch_warnings():
            simplefilter("ignore")
            model = pysd.read_vensim(test_model)
            stocks = model.run(return_timestamps=[0, 10, 20, 30])
            self.assertTrue((stocks['INITIAL TIME'] == 0).all().all())
            self.assertTrue((stocks['FINAL TIME'] == 30).all().all())

            model.initialize()
            stocks1 = model.run(return_timestamps=[0, 10], final_time=12)
            self.assertTrue((stocks1['INITIAL TIME'] == 0).all().all())
            self.assertTrue((stocks1['FINAL TIME'] == 12).all().all())
            model.export('teacup12.pic')
            model.initialize()
            stocks2 = model.run(initial_condition='teacup12.pic',
                                return_timestamps=[20, 30])
            self.assertTrue((stocks2['INITIAL TIME'] == 12).all().all())
            self.assertTrue((stocks2['FINAL TIME'] == 30).all().all())
            stocks.drop('INITIAL TIME', axis=1, inplace=True)
            stocks1.drop('INITIAL TIME', axis=1, inplace=True)
            stocks2.drop('INITIAL TIME', axis=1, inplace=True)
            stocks.drop('FINAL TIME', axis=1, inplace=True)
            stocks1.drop('FINAL TIME', axis=1, inplace=True)
            stocks2.drop('FINAL TIME', axis=1, inplace=True)
            os.remove('teacup12.pic')

            assert_frames_close(stocks1, stocks.loc[[0, 10]])
            assert_frames_close(stocks2, stocks.loc[[20, 30]])

            # delays
            test_delays = os.path.join(
                _root,
                'test-models/tests/delays/test_delays.mdl')
            model = pysd.read_vensim(test_delays)
            stocks = model.run(return_timestamps=20)
            model.initialize()
            model.run(return_timestamps=[], final_time=7)
            model.export('delays7.pic')
            stocks2 = model.run(initial_condition='delays7.pic',
                                return_timestamps=20)
            self.assertTrue((stocks['INITIAL TIME'] == 0).all().all())
            self.assertTrue((stocks2['INITIAL TIME'] == 7).all().all())
            stocks.drop('INITIAL TIME', axis=1, inplace=True)
            stocks2.drop('INITIAL TIME', axis=1, inplace=True)
            stocks.drop('FINAL TIME', axis=1, inplace=True)
            stocks2.drop('FINAL TIME', axis=1, inplace=True)
            os.remove('delays7.pic')

            assert_frames_close(stocks2, stocks)

            # delay fixed
            test_delayf = os.path.join(
                _root,
                'test-models/tests/delay_fixed/test_delay_fixed.mdl')
            model = pysd.read_vensim(test_delayf)
            stocks = model.run(return_timestamps=20)
            model.initialize()
            model.run(return_timestamps=7)
            model.export('delayf7.pic')
            stocks2 = model.run(initial_condition='delayf7.pic',
                                return_timestamps=20)
            self.assertTrue((stocks['INITIAL TIME'] == 0).all().all())
            self.assertTrue((stocks2['INITIAL TIME'] == 7).all().all())
            stocks.drop('INITIAL TIME', axis=1, inplace=True)
            stocks2.drop('INITIAL TIME', axis=1, inplace=True)
            stocks.drop('FINAL TIME', axis=1, inplace=True)
            stocks2.drop('FINAL TIME', axis=1, inplace=True)
            os.remove('delayf7.pic')

            assert_frames_close(stocks2, stocks)

            # smooth
            test_smooth = os.path.join(
                _root,
                'test-models/tests/subscripted_smooth/'
                + 'test_subscripted_smooth.mdl')
            model = pysd.read_vensim(test_smooth)
            stocks = model.run(return_timestamps=20, flatten_output=True)
            model.initialize()
            model.run(return_timestamps=7)
            model.export('smooth7.pic')
            stocks2 = model.run(initial_condition='smooth7.pic',
                                return_timestamps=20,
                                flatten_output=True)
            self.assertTrue((stocks['INITIAL TIME'] == 0).all().all())
            self.assertTrue((stocks2['INITIAL TIME'] == 7).all().all())
            stocks.drop('INITIAL TIME', axis=1, inplace=True)
            stocks2.drop('INITIAL TIME', axis=1, inplace=True)
            stocks.drop('FINAL TIME', axis=1, inplace=True)
            stocks2.drop('FINAL TIME', axis=1, inplace=True)
            os.remove('smooth7.pic')

            assert_frames_close(stocks2, stocks)

            # trend
            test_trend = os.path.join(
                _root,
                'test-models/tests/subscripted_trend/'
                + 'test_subscripted_trend.mdl')
            model = pysd.read_vensim(test_trend)
            stocks = model.run(return_timestamps=20, flatten_output=True)
            model.initialize()
            model.run(return_timestamps=7)
            model.export('trend7.pic')
            stocks2 = model.run(initial_condition='trend7.pic',
                                return_timestamps=20,
                                flatten_output=True)
            self.assertTrue((stocks['INITIAL TIME'] == 0).all().all())
            self.assertTrue((stocks2['INITIAL TIME'] == 7).all().all())
            stocks.drop('INITIAL TIME', axis=1, inplace=True)
            stocks2.drop('INITIAL TIME', axis=1, inplace=True)
            stocks.drop('FINAL TIME', axis=1, inplace=True)
            stocks2.drop('FINAL TIME', axis=1, inplace=True)
            os.remove('trend7.pic')

            assert_frames_close(stocks2, stocks)

            # initial
            test_initial = os.path.join(
                _root, 'test-models/tests/initial_function/test_initial.mdl')
            model = pysd.read_vensim(test_initial)
            stocks = model.run(return_timestamps=20)
            model.initialize()
            model.run(return_timestamps=7)
            model.export('initial7.pic')
            stocks2 = model.run(initial_condition='initial7.pic',
                                return_timestamps=20)
            self.assertTrue((stocks['INITIAL TIME'] == 0).all().all())
            self.assertTrue((stocks2['INITIAL TIME'] == 7).all().all())
            stocks.drop('INITIAL TIME', axis=1, inplace=True)
            stocks2.drop('INITIAL TIME', axis=1, inplace=True)
            stocks.drop('FINAL TIME', axis=1, inplace=True)
            stocks2.drop('FINAL TIME', axis=1, inplace=True)
            os.remove('initial7.pic')

            assert_frames_close(stocks2, stocks)

            # sample if true
            test_sample_if_true = os.path.join(
                _root,
                'test-models/tests/sample_if_true/test_sample_if_true.mdl')
            model = pysd.read_vensim(test_sample_if_true)
            stocks = model.run(return_timestamps=20, flatten_output=True)
            model.initialize()
            model.run(return_timestamps=7)
            model.export('sample_if_true7.pic')
            stocks2 = model.run(initial_condition='sample_if_true7.pic',
                                return_timestamps=20,
                                flatten_output=True)
            self.assertTrue((stocks['INITIAL TIME'] == 0).all().all())
            self.assertTrue((stocks2['INITIAL TIME'] == 7).all().all())
            stocks.drop('INITIAL TIME', axis=1, inplace=True)
            stocks2.drop('INITIAL TIME', axis=1, inplace=True)
            stocks.drop('FINAL TIME', axis=1, inplace=True)
            stocks2.drop('FINAL TIME', axis=1, inplace=True)
            os.remove('sample_if_true7.pic')

            assert_frames_close(stocks2, stocks)

    def test_initial_conditions_tuple_pysafe_names(self):
        import pysd

        model = pysd.read_vensim(test_model)
        stocks = model.run(
            initial_condition=(3000, {"teacup_temperature": 33}),
            return_timestamps=list(range(3000, 3010))
        )

        self.assertEqual(stocks["Teacup Temperature"].iloc[0], 33)

    def test_initial_conditions_tuple_original_names(self):
        """ Responds to https://github.com/JamesPHoughton/pysd/issues/77"""
        import pysd

        model = pysd.read_vensim(test_model)
        stocks = model.run(
            initial_condition=(3000, {"Teacup Temperature": 33}),
            return_timestamps=list(range(3000, 3010)),
        )
        self.assertEqual(stocks.index[0], 3000)
        self.assertEqual(stocks["Teacup Temperature"].iloc[0], 33)

    def test_initial_conditions_current(self):
        import pysd

        model = pysd.read_vensim(test_model)
        stocks1 = model.run(return_timestamps=list(range(0, 31)))
        stocks2 = model.run(
            initial_condition="current", return_timestamps=list(range(30, 45))
        )
        self.assertEqual(
            stocks1["Teacup Temperature"].iloc[-1],
            stocks2["Teacup Temperature"].iloc[0],
        )

    def test_initial_condition_bad_value(self):
        import pysd

        model = pysd.read_vensim(test_model)

        with self.assertRaises(FileNotFoundError):
            model.run(initial_condition="bad value")

    def test_initial_conditions_subscripted_value_with_constant(self):
        import pysd

        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output = xr.DataArray([[5, 5], [5, 5], [5, 5]], coords, dims)

        model = pysd.read_vensim(test_model_subs)

        with catch_warnings(record=True) as ws:
            res = model.run(initial_condition=(5, {'initial_values': 5}),
                            return_columns=['Initial Values'],
                            return_timestamps=list(range(5, 10)))
            # use only future warnings
            wf = [w for w in ws if issubclass(w.category, FutureWarning)]
            self.assertEqual(len(wf), 1)
            self.assertIn(
                "a constant value with initial_conditions will be deprecated",
                str(wf[0].message))

        self.assertTrue(output.equals(res['Initial Values'].iloc[0]))
        self.assertEqual(res.index[0], 5)

    def test_initial_conditions_subscripted_value_with_partial_xarray(self):
        import pysd

        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output = xr.DataArray([[5, 3], [5, 3], [5, 3]], coords, dims)
        input_val = xr.DataArray(
            [5, 3],
            {'Second Dimension Subscript': ['Column 1', 'Column 2']},
            ['Second Dimension Subscript'])

        model = pysd.read_vensim(test_model_subs)
        with catch_warnings(record=True) as ws:
            res = model.run(initial_condition=(5,
                                               {'Initial Values': input_val}),
                            return_columns=['Initial Values'],
                            return_timestamps=list(range(5, 10)))
            # use only future warnings
            wf = [w for w in ws if issubclass(w.category, FutureWarning)]
            self.assertEqual(len(wf), 1)
            self.assertIn(
                "a constant value with initial_conditions will be deprecated",
                str(wf[0].message))

        self.assertTrue(output.equals(res['Initial Values'].iloc[0]))
        self.assertEqual(res.index[0], 5)

    def test_initial_conditions_subscripted_value_with_xarray(self):
        import pysd

        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output = xr.DataArray([[5, 3], [4, 8], [9, 3]], coords, dims)

        model = pysd.read_vensim(test_model_subs)

        with catch_warnings(record=True) as ws:
            res = model.run(initial_condition=(5, {'initial_values': output}),
                            return_columns=['Initial Values'],
                            return_timestamps=list(range(5, 10)))
            # use only future warnings
            wf = [w for w in ws if issubclass(w.category, FutureWarning)]
            self.assertEqual(len(wf), 1)
            self.assertIn(
                "a constant value with initial_conditions will be deprecated",
                str(wf[0].message))

        self.assertTrue(output.equals(res['Initial Values'].iloc[0]))
        self.assertEqual(res.index[0], 5)

    def test_initial_conditions_subscripted_value_with_numpy_error(self):
        import pysd

        input_ = np.array([[5, 3], [4, 8], [9, 3]])

        model = pysd.read_vensim(test_model_subs)

        with self.assertRaises(TypeError):
            model.run(initial_condition=(5, {'initial_values': input_}),
                      return_columns=['Initial Values'],
                      return_timestamps=list(range(5, 10)))

    def test_set_constant_parameter(self):
        """ In response to:
        re: https://github.com/JamesPHoughton/pysd/issues/5"""
        import pysd

        model = pysd.read_vensim(test_model)
        model.set_components({"room_temperature": 20})
        self.assertEqual(model.components.room_temperature(), 20)

        model.run(params={"room_temperature": 70})
        self.assertEqual(model.components.room_temperature(), 70)

        with self.assertRaises(NameError):
            model.set_components({'not_a_var': 20})

    def test_set_timeseries_parameter(self):
        import pysd

        model = pysd.read_vensim(test_model)
        timeseries = list(range(30))
        temp_timeseries = pd.Series(
            index=timeseries,
            data=(50 + np.random.rand(len(timeseries)).cumsum())
        )
        res = model.run(
            params={"room_temperature": temp_timeseries},
            return_columns=["room_temperature"],
            return_timestamps=timeseries,
        )
        self.assertTrue((res["room_temperature"] == temp_timeseries).all())

    def test_set_component_with_real_name(self):
        import pysd

        model = pysd.read_vensim(test_model)
        model.set_components({"Room Temperature": 20})
        self.assertEqual(model.components.room_temperature(), 20)

        model.run(params={"Room Temperature": 70})
        self.assertEqual(model.components.room_temperature(), 70)

    def test_set_components_warnings(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/80"""
        import pysd

        model = pysd.read_vensim(test_model)
        with catch_warnings(record=True) as w:
            simplefilter("always")
            model.set_components(
                {"Teacup Temperature": 20, "Characteristic Time": 15}
            )  # set stock value using params
        self.assertEqual(len(w), 1)
        self.assertTrue(
            "Teacup Temperature" in str(w[0].message)
        )  # check that warning references the stock

    def test_set_components_with_function(self):
        def test_func():
            return 5

        import pysd

        model = pysd.read_vensim(test_model)
        model.set_components({"Room Temperature": test_func})
        res = model.run(return_columns=["Room Temperature"])
        self.assertEqual(test_func(), res["Room Temperature"].iloc[0])

    def test_set_subscripted_value_with_constant(self):
        import pysd

        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output = xr.DataArray([[5, 5], [5, 5], [5, 5]], coords, dims)

        model = pysd.read_vensim(test_model_subs)
        model.set_components({"initial_values": 5, "final_time": 10})
        res = model.run(return_columns=["Initial Values"])
        self.assertTrue(output.equals(res["Initial Values"].iloc[0]))

    def test_set_subscripted_value_with_partial_xarray(self):
        import pysd

        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output = xr.DataArray([[5, 3], [5, 3], [5, 3]], coords, dims)
        input_val = xr.DataArray(
            [5, 3],
            {"Second Dimension Subscript": ["Column 1", "Column 2"]},
            ["Second Dimension Subscript"],
        )

        model = pysd.read_vensim(test_model_subs)
        model.set_components({"Initial Values": input_val, "final_time": 10})
        res = model.run(return_columns=["Initial Values"])
        self.assertTrue(output.equals(res["Initial Values"].iloc[0]))

    def test_set_subscripted_value_with_xarray(self):
        import pysd

        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output = xr.DataArray([[5, 3], [4, 8], [9, 3]], coords, dims)

        model = pysd.read_vensim(test_model_subs)
        model.set_components({"initial_values": output, "final_time": 10})
        res = model.run(return_columns=["Initial Values"])
        self.assertTrue(output.equals(res["Initial Values"].iloc[0]))

    def test_set_constant_parameter_lookup(self):
        import pysd

        model = pysd.read_vensim(test_model_look)

        with catch_warnings():
            # avoid warnings related to extrapolation
            simplefilter("ignore")
            model.set_components({"lookup_1d": 20})
            for i in range(100):
                self.assertEqual(model.components.lookup_1d(i), 20)

            model.run(params={"lookup_1d": 70})
            for i in range(100):
                self.assertEqual(model.components.lookup_1d(i), 70)

            model.set_components({"lookup_2d": 20})
            for i in range(100):
                self.assertTrue(
                    model.components.lookup_2d(i).equals(
                        xr.DataArray(20, {"Rows": ["Row1", "Row2"]}, ["Rows"])
                    )
                )

            model.run(params={"lookup_2d": 70})
            for i in range(100):
                self.assertTrue(
                    model.components.lookup_2d(i).equals(
                        xr.DataArray(70, {"Rows": ["Row1", "Row2"]}, ["Rows"])
                    )
                )

            xr1 = xr.DataArray([-10, 50], {"Rows": ["Row1", "Row2"]}, ["Rows"])
            model.set_components({"lookup_2d": xr1})
            for i in range(100):
                self.assertTrue(model.components.lookup_2d(i).equals(xr1))

            xr2 = xr.DataArray([-100, 500], {"Rows": ["Row1", "Row2"]},
                               ["Rows"])
            model.run(params={"lookup_2d": xr2})
            for i in range(100):
                self.assertTrue(model.components.lookup_2d(i).equals(xr2))

    def test_set_timeseries_parameter_lookup(self):
        import pysd

        model = pysd.read_vensim(test_model_look)
        timeseries = list(range(30))

        with catch_warnings():
            # avoid warnings related to extrapolation
            simplefilter("ignore")
            temp_timeseries = pd.Series(
                index=timeseries, data=(50 +
                                        np.random.rand(len(timeseries)
                                                       ).cumsum())
            )

            res = model.run(
                params={"lookup_1d": temp_timeseries},
                return_columns=["lookup_1d_time"],
                return_timestamps=timeseries,
            )

            self.assertTrue((res["lookup_1d_time"] == temp_timeseries).all())

            res = model.run(
                params={"lookup_2d": temp_timeseries},
                return_columns=["lookup_2d_time"],
                return_timestamps=timeseries,
            )

            self.assertTrue(
                all(
                    [
                        a.equals(xr.DataArray(b, {"Rows": ["Row1", "Row2"]},
                                              ["Rows"]))
                        for a, b in zip(res["lookup_2d_time"].values,
                                        temp_timeseries)
                    ]
                )
            )

            temp_timeseries2 = pd.Series(
                index=timeseries,
                data=[
                    xr.DataArray([50 + x, 20 - y], {"Rows": ["Row1", "Row2"]},
                                 ["Rows"])
                    for x, y in zip(
                        np.random.rand(len(timeseries)).cumsum(),
                        np.random.rand(len(timeseries)).cumsum(),
                    )
                ],
            )

            res = model.run(
                params={"lookup_2d": temp_timeseries2},
                return_columns=["lookup_2d_time"],
                return_timestamps=timeseries,
            )

            self.assertTrue(
                all(
                    [
                        a.equals(b)
                        for a, b in zip(res["lookup_2d_time"].values,
                                        temp_timeseries2)
                    ]
                )
            )

    def test_set_subscripted_value_with_numpy_error(self):
        import pysd

        input_ = np.array([[5, 3], [4, 8], [9, 3]])

        model = pysd.read_vensim(test_model_subs)
        with self.assertRaises(TypeError):
            model.set_components({"initial_values": input_, "final_time": 10})

    def test_set_subscripted_timeseries_parameter_with_constant(self):
        import pysd

        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]

        model = pysd.read_vensim(test_model_subs)
        timeseries = list(range(10))
        val_series = [50 + rd for rd in np.random.rand(len(timeseries)
                                                       ).cumsum()]
        xr_series = [xr.DataArray(val, coords, dims) for val in val_series]

        temp_timeseries = pd.Series(index=timeseries, data=val_series)
        res = model.run(
            params={"initial_values": temp_timeseries, "final_time": 10},
            return_columns=["initial_values"],
            return_timestamps=timeseries,
        )

        self.assertTrue(
            np.all(
                [r.equals(t) for r, t in zip(res["initial_values"].values,
                                             xr_series)]
            )
        )

    def test_set_subscripted_timeseries_parameter_with_partial_xarray(self):
        import pysd

        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        out_b = xr.DataArray([[0, 0], [0, 0], [0, 0]], coords, dims)
        input_val = xr.DataArray(
            [5, 3],
            {"Second Dimension Subscript": ["Column 1", "Column 2"]},
            ["Second Dimension Subscript"],
        )

        model = pysd.read_vensim(test_model_subs)
        timeseries = list(range(10))
        val_series = [input_val + rd for rd in np.random.rand(len(timeseries)
                                                              ).cumsum()]
        temp_timeseries = pd.Series(index=timeseries, data=val_series)
        out_series = [out_b + val for val in val_series]
        model.set_components({"initial_values": temp_timeseries,
                              "final_time": 10})
        res = model.run(return_columns=["initial_values"])
        self.assertTrue(
            np.all(
                [r.equals(t) for r, t in zip(res["initial_values"].values,
                                             out_series)]
            )
        )

    def test_set_subscripted_timeseries_parameter_with_xarray(self):
        import pysd

        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]

        init_val = xr.DataArray([[5, 3], [4, 8], [9, 3]], coords, dims)

        model = pysd.read_vensim(test_model_subs)
        timeseries = list(range(10))
        temp_timeseries = pd.Series(
            index=timeseries,
            data=[init_val + rd for rd in np.random.rand(len(timeseries)
                                                         ).cumsum()],
        )
        res = model.run(
            params={"initial_values": temp_timeseries, "final_time": 10},
            return_columns=["initial_values"],
            return_timestamps=timeseries,
        )

        self.assertTrue(
            np.all(
                [
                    r.equals(t)
                    for r, t in zip(
                        res["initial_values"].values, temp_timeseries.values
                    )
                ]
            )
        )

    def test_docs(self):
        """ Test that the model prints some documentation """
        import pysd

        model = pysd.read_vensim(test_model)
        self.assertIsInstance(str(model), str)  # tests string conversion of
        # model

        doc = model.doc()
        self.assertIsInstance(doc, pd.DataFrame)
        self.assertSetEqual(
            {
                "Characteristic Time",
                "Teacup Temperature",
                "FINAL TIME",
                "Heat Loss to Room",
                "INITIAL TIME",
                "Room Temperature",
                "SAVEPER",
                "TIME STEP",
            },
            set(doc["Real Name"].values),
        )

        self.assertEqual(
            doc[doc["Real Name"] == "Heat Loss to Room"]["Unit"].values[0],
            "Degrees Fahrenheit/Minute",
        )
        self.assertEqual(
            doc[doc["Real Name"] == "Teacup Temperature"]["Py Name"].values[0],
            "teacup_temperature",
        )
        self.assertEqual(
            doc[doc["Real Name"] == "INITIAL TIME"]["Comment"].values[0],
            "The initial time for the simulation.",
        )
        self.assertEqual(
            doc[doc["Real Name"] == "Characteristic Time"]["Type"].values[0],
            "constant"
        )
        self.assertEqual(
            doc[doc["Real Name"] == "Teacup Temperature"]["Lims"].values[0],
            "(32.0, 212.0)",
        )

    def test_docs_multiline_eqn(self):
        """ Test that the model prints some documentation """
        import pysd

        path2model = os.path.join(
            _root,
            "test-models/tests/multiple_lines_def/" +
            "test_multiple_lines_def.mdl")
        model = pysd.read_vensim(path2model)

        doc = model.doc()

        self.assertEqual(doc[doc["Real Name"] == "price"]["Unit"].values[0],
                         "euros/kg")
        self.assertEqual(doc[doc["Real Name"] == "price"]["Py Name"].values[0],
                         "price")
        self.assertEqual(
            doc[doc["Real Name"] == "price"]["Subs"].values[0], "['fruits']"
        )
        self.assertEqual(doc[doc["Real Name"] == "price"]["Eqn"].values[0],
                         "1.2; .; .; .; 1.4")

    def test_stepwise_cache(self):
        run_history = []
        result_history = []

        global time
        time = lambda: 0  # for testing cache function
        from pysd import cache

        cache.time = time()

        @cache.step
        def upstream(run_hist, res_hist):
            run_hist.append("U")
            return "up"

        def downstream(run_hist, res_hist):
            run_hist.append("D")
            result_history.append(upstream(run_hist, res_hist))
            return "down"

        # initially neither function has a chache value
        self.assertFalse("upstream" in cache.data["step"])
        self.assertFalse("downstream" in cache.data["step"])

        # when the functions are called,
        # the cache is instantiated in the upstream (cached) function
        result_history.append(downstream(run_history, result_history))
        self.assertTrue("upstream" in cache.data["step"])
        self.assertFalse("upstream" in cache.data["run"])
        self.assertFalse("downstream" in cache.data["step"])
        self.assertEqual(cache.time, 0)
        self.assertListEqual(run_history, ["D", "U"])
        self.assertListEqual(result_history, ["up", "down"])

        # cleaning only run cache shouldn't affect the step cache
        cache.clean("run")
        self.assertTrue("upstream" in cache.data["step"])

        # at the second call, the uncached function is run,
        # but the cached upstream function returns its prior value
        result_history.append(downstream(run_history, result_history))
        self.assertEqual(cache.time, 0)
        self.assertListEqual(run_history, ["D", "U", "D"])
        self.assertListEqual(result_history, ["up", "down", "up", "down"])

        # when the time is reset, both functions are run again.
        time = lambda: 2
        cache.reset(time())

        result_history.append(downstream(run_history, result_history))
        self.assertEqual(cache.time, 2)
        self.assertListEqual(run_history, ["D", "U", "D", "D", "U"])
        self.assertListEqual(result_history, ["up", "down", "up", "down",
                                              "up", "down"])

    def test_runwise_cache(self):
        # Checks backward compatibility, must be changed to @cache.run when
        # deprecated
        run_history = []
        result_history = []

        global time
        time = lambda: 0  # for testing cache function
        from pysd import cache

        cache.time = time()

        @cache.run
        def upstream(run_hist, res_hist):
            run_hist.append("U")
            return "up"

        def downstream(run_hist, res_hist):
            run_hist.append("D")
            result_history.append(upstream(run_hist, res_hist))
            return "down"

        # initially neither function has a chache value
        self.assertFalse("upstream" in cache.data["run"])
        self.assertFalse("downstream" in cache.data["run"])

        # when the functions are called,
        # the cache is instantiated in the upstream (cached) function
        result_history.append(downstream(run_history, result_history))
        self.assertEqual(cache.time, 0)
        self.assertTrue("upstream" in cache.data["run"])
        self.assertFalse("upstream" in cache.data["step"])
        self.assertFalse("downstream" in cache.data["run"])
        self.assertListEqual(run_history, ["D", "U"])
        self.assertListEqual(result_history, ["up", "down"])

        # cleaning only step cache shouldn't affect the step cache
        cache.clean("step")
        self.assertTrue("upstream" in cache.data["run"])

        # at the second call, the uncached function is run,
        # but the cached upstream function returns its prior value
        result_history.append(downstream(run_history, result_history))
        self.assertEqual(cache.time, 0)
        self.assertListEqual(run_history, ["D", "U", "D"])
        self.assertListEqual(result_history, ["up", "down", "up", "down"])

        # when the time is reset, this has no impact on the upstream cache.
        time = lambda: 2
        cache.reset(time())

        result_history.append(downstream(run_history, result_history))
        self.assertEqual(cache.time, 2)
        self.assertListEqual(run_history, ["D", "U", "D", "D"])
        self.assertListEqual(result_history, ["up", "down", "up", "down",
                                              "up", "down"])

    def test_initialize(self):
        import pysd

        model = pysd.read_vensim(test_model)
        initial_temp = model.components.teacup_temperature()
        model.run()
        final_temp = model.components.teacup_temperature()
        model.initialize()
        reset_temp = model.components.teacup_temperature()
        self.assertNotEqual(initial_temp, final_temp)
        self.assertEqual(initial_temp, reset_temp)

    def test_initialize_order(self):
        import pysd
        model = pysd.load(more_tests + "/initialization_order/"
                          "test_initialization_order.py")

        if model._stateful_elements[0].py_name.endswith('stock_a'):
            # we want to have stock b first always
            model._stateful_elements.reverse()

        self.assertEqual(model.components.stock_b(), 42)
        self.assertEqual(model.components.stock_a(), 42)
        model.components.initial_parameter = lambda: 1
        model.initialize()
        self.assertEqual(model.components.stock_b(), 1)
        self.assertEqual(model.components.stock_a(), 1)

    def test_set_state(self):
        import pysd

        model = pysd.read_vensim(test_model)

        initial_temp = model.components.teacup_temperature()

        new_time = np.random.rand()

        with catch_warnings(record=True) as ws:
            # Test that we can set with real names
            model.set_state(new_time, {'Teacup Temperature': 500})
            self.assertNotEqual(initial_temp, 500)
            self.assertEqual(model.components.teacup_temperature(), 500)
            self.assertEqual(model.components.time(), new_time)
            # use only future warnings
            wf = [w for w in ws if issubclass(w.category, FutureWarning)]
            self.assertEqual(len(wf), 1)
            self.assertIn(
                "set_state will be deprecated, use set_initial_value instead.",
                str(wf[0].message))

        with catch_warnings(record=True) as ws:
            # Test setting with pysafe names
            model.set_state(new_time + 1, {'teacup_temperature': 202})
            self.assertEqual(model.components.teacup_temperature(), 202)
            self.assertEqual(model.components.time(), new_time + 1)
            # use only future warnings
            wf = [w for w in ws if issubclass(w.category, FutureWarning)]
            self.assertEqual(len(wf), 1)
            self.assertIn(
                "set_state will be deprecated, use set_initial_value instead.",
                str(wf[0].message))

        with catch_warnings(record=True) as ws:
            # Test setting with stateful object name
            model.set_state(new_time + 2, {'_integ_teacup_temperature': 302})
            self.assertEqual(model.components.teacup_temperature(), 302)
            self.assertEqual(model.components.time(), new_time + 2)
            # use only future warnings
            wf = [w for w in ws if issubclass(w.category, FutureWarning)]
            self.assertEqual(len(wf), 1)
            self.assertIn(
                "set_state will be deprecated, use set_initial_value instead.",
                str(wf[0].message))

    def test_set_initial_value(self):
        import pysd
        model = pysd.read_vensim(test_model)

        initial_temp = model.components.teacup_temperature()

        new_time = np.random.rand()

        # Test that we can set with real names
        model.set_initial_value(new_time, {'Teacup Temperature': 500})
        self.assertNotEqual(initial_temp, 500)
        self.assertEqual(model.components.teacup_temperature(), 500)
        self.assertEqual(model.components.time(), new_time)

        # Test setting with pysafe names
        model.set_initial_value(new_time + 1, {'teacup_temperature': 202})
        self.assertEqual(model.components.teacup_temperature(), 202)
        self.assertEqual(model.components.time(), new_time + 1)

        # Test setting with stateful object name
        model.set_initial_value(new_time + 2,
                                {'_integ_teacup_temperature': 302})
        self.assertEqual(model.components.teacup_temperature(), 302)
        self.assertEqual(model.components.time(), new_time + 2)

        with self.assertRaises(NameError):
            model.set_initial_value(new_time, {'not_a_var': 500})

    def test_set_initial_value_lookup(self):
        import pysd

        model = pysd.read_vensim(test_model_look)

        new_time = np.random.rand()

        # Test that we can set with real names
        with catch_warnings(record=True) as ws:
            model.set_initial_value(new_time, {'lookup 1d': 500})
            # use only future warnings
            wf = [w for w in ws if issubclass(w.category, FutureWarning)]
            self.assertEqual(len(wf), 1)
            self.assertIn(
                "a constant value with initial_conditions will be deprecated",
                str(wf[0].message))

        self.assertEqual(model.components.lookup_1d(0), 500)
        self.assertEqual(model.components.lookup_1d(100), 500)

        with catch_warnings(record=True) as ws:
            model.set_initial_value(new_time, {'lookup 2d': 520})
            # use only future warnings
            wf = [w for w in ws if issubclass(w.category, FutureWarning)]
            self.assertEqual(len(wf), 1)
            self.assertIn(
                "a constant value with initial_conditions will be deprecated",
                str(wf[0].message))

        expected = xr.DataArray(520, {"Rows": ["Row1", "Row2"]}, ["Rows"])
        self.assertTrue(model.components.lookup_2d(0).equals(expected))
        self.assertTrue(model.components.lookup_2d(100).equals(expected))

        with catch_warnings():
            # avoid warnings related to extrapolation
            simplefilter("ignore")
            model.run()

    def test_set_initial_value_subscripted_value_with_constant(self):
        import pysd

        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output_b = xr.DataArray([[0, 0], [0, 0], [0, 0]], coords, dims)

        new_time = np.random.rand()

        model = pysd.read_vensim(test_model_subs)
        initial_stock = model.components.stock_a()

        # Test that we can set with real names
        model.set_initial_value(new_time, {'Stock A': 500})
        self.assertFalse(initial_stock.equals(output_b + 500))
        self.assertTrue(model.components.stock_a().equals(output_b + 500))

        # Test setting with pysafe names
        model.set_initial_value(new_time + 1, {'stock_a': 202})
        self.assertTrue(model.components.stock_a().equals(output_b + 202))

        # Test setting with stateful object name
        model.set_initial_value(new_time + 2, {'_integ_stock_a': 302})
        self.assertTrue(model.components.stock_a().equals(output_b + 302))

        # Test error when coords are not a subset
        with self.assertRaises(ValueError):
            model.set_initial_value(
                new_time + 2,
                {'_integ_stock_a': xr.DataArray(302, {'D': ['A', 'B']}, ['D'])}
            )

    def test_set_initial_value_subscripted_value_with_partial_xarray(self):
        import pysd

        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output1 = xr.DataArray([[5, 3], [5, 3], [5, 3]], coords, dims)
        input_val1 = xr.DataArray(
            [5, 3],
            {"Second Dimension Subscript": ["Column 1", "Column 2"]},
            ["Second Dimension Subscript"],
        )

        output2 = xr.DataArray([[55, 33], [55, 33], [55, 33]], coords, dims)
        input_val2 = xr.DataArray(
            [55, 33],
            {"Second Dimension Subscript": ["Column 1", "Column 2"]},
            ["Second Dimension Subscript"],
        )

        output3 = xr.DataArray([[40, 20], [40, 20], [40, 20]], coords, dims)
        input_val3 = xr.DataArray(
            [40, 20],
            {"Second Dimension Subscript": ["Column 1", "Column 2"]},
            ["Second Dimension Subscript"],
        )

        new_time = np.random.rand()

        model = pysd.read_vensim(test_model_subs)
        initial_stock = model.components.stock_a()

        # Test that we can set with real names
        model.set_initial_value(new_time, {'Stock A': input_val1})
        self.assertFalse(initial_stock.equals(output1))
        self.assertTrue(model.components.stock_a().equals(output1))

        # Test setting with pysafe names
        model.set_initial_value(new_time + 1, {'stock_a': input_val2})
        self.assertTrue(model.components.stock_a().equals(output2))

        # Test setting with stateful object name
        model.set_initial_value(new_time + 2, {'_integ_stock_a': input_val3})
        self.assertTrue(model.components.stock_a().equals(output3))

    def test_set_initial_value_subscripted_value_with_xarray(self):
        import pysd

        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output1 = xr.DataArray([[5, 3], [4, 8], [9, 3]], coords, dims)
        output2 = xr.DataArray([[53, 43], [84, 80], [29, 63]], coords, dims)
        output3 = xr.DataArray([[54, 32], [40, 87], [93, 93]], coords, dims)

        new_time = np.random.rand()

        model = pysd.read_vensim(test_model_subs)
        initial_stock = model.components.stock_a()

        # Test that we can set with real names
        model.set_initial_value(new_time, {'Stock A': output1})
        self.assertFalse(initial_stock.equals(output1))
        self.assertTrue(model.components.stock_a().equals(output1))

        # Test setting with pysafe names
        model.set_initial_value(new_time + 1, {'stock_a': output2})
        self.assertTrue(model.components.stock_a().equals(output2))

        # Test setting with stateful object name
        model.set_initial_value(new_time + 2, {'_integ_stock_a': output3})
        self.assertTrue(model.components.stock_a().equals(output3))

    def test_set_initial_value_subscripted_value_with_numpy_error(self):
        import pysd

        input1 = np.array([[5, 3], [4, 8], [9, 3]])
        input2 = np.array([[53, 43], [84, 80], [29, 63]])
        input3 = np.array([[54, 32], [40, 87], [93, 93]])

        new_time = np.random.rand()

        model = pysd.read_vensim(test_model_subs)

        # Test that we can set with real names
        with self.assertRaises(TypeError):
            model.set_initial_value(new_time, {'Stock A': input1})

        # Test setting with pysafe names
        with self.assertRaises(TypeError):
            model.set_initial_value(new_time + 1, {'stock_a': input2})

        # Test setting with stateful object name
        with self.assertRaises(TypeError):
            model.set_initial_value(new_time + 2, {'_integ_stock_a': input3})

    def test_replace_element(self):
        import pysd

        model = pysd.read_vensim(test_model)
        stocks1 = model.run()
        model.components.characteristic_time = lambda: 3
        stocks2 = model.run()
        self.assertGreater(
            stocks1["Teacup Temperature"].loc[10],
            stocks2["Teacup Temperature"].loc[10]
        )

    def test_set_initial_condition_origin_full(self):
        import pysd

        model = pysd.read_vensim(test_model)
        initial_temp = model.components.teacup_temperature()
        initial_time = model.components.time()

        new_state = {"Teacup Temperature": 500}
        new_time = 10

        model.set_initial_condition((new_time, new_state))
        set_temp = model.components.teacup_temperature()
        set_time = model.components.time()

        self.assertNotEqual(
            set_temp,
            initial_temp,
            "Test definition is wrong, please change configuration",
        )
        self.assertEqual(set_temp, 500)

        self.assertNotEqual(
            initial_time,
            new_time,
            "Test definition is wrong, please change configuration",
        )
        self.assertEqual(new_time, set_time)

        model.set_initial_condition("original")
        set_temp = model.components.teacup_temperature()
        set_time = model.components.time()

        self.assertEqual(initial_temp, set_temp)
        self.assertEqual(initial_time, set_time)

    def test_set_initial_condition_origin_short(self):
        import pysd

        model = pysd.read_vensim(test_model)
        initial_temp = model.components.teacup_temperature()
        initial_time = model.components.time()

        new_state = {"Teacup Temperature": 500}
        new_time = 10

        model.set_initial_condition((new_time, new_state))
        set_temp = model.components.teacup_temperature()
        set_time = model.components.time()

        self.assertNotEqual(
            set_temp,
            initial_temp,
            "Test definition is wrong, please change configuration",
        )
        self.assertEqual(set_temp, 500)

        self.assertNotEqual(
            initial_time,
            new_time,
            "Test definition is wrong, please change configuration",
        )
        self.assertEqual(new_time, set_time)

        model.set_initial_condition("o")
        set_temp = model.components.teacup_temperature()
        set_time = model.components.time()

        self.assertEqual(initial_temp, set_temp)
        self.assertEqual(initial_time, set_time)

    def test_set_initial_condition_for_stock_component(self):
        import pysd

        model = pysd.read_vensim(test_model)
        initial_temp = model.components.teacup_temperature()
        initial_time = model.components.time()

        new_state = {"Teacup Temperature": 500}
        new_time = 10

        model.set_initial_condition((new_time, new_state))
        set_temp = model.components.teacup_temperature()
        set_time = model.components.time()

        self.assertNotEqual(
            set_temp,
            initial_temp,
            "Test definition is wrong, please change configuration",
        )
        self.assertEqual(set_temp, 500)

        self.assertNotEqual(
            initial_time, 10, "Test definition is wrong, please change" +
            " configuration"
        )
        self.assertEqual(set_time, 10)

    def test_set_initial_condition_for_constant_component(self):
        import pysd

        model = pysd.read_vensim(test_model)
        initial_temp = model.components.teacup_temperature()
        initial_time = model.components.time()

        new_state = {"Room Temperature": 100}
        new_time = 10

        with catch_warnings(record=True) as ws:
            model.set_initial_condition((new_time, new_state))
            # use only future warnings
            wf = [w for w in ws if issubclass(w.category, FutureWarning)]
            self.assertEqual(len(wf), 1)
            self.assertIn(
                "a constant value with initial_conditions will be deprecated",
                str(wf[0].message))

        set_temp = model.components.room_temperature()
        set_time = model.components.time()

        self.assertNotEqual(
            set_temp,
            initial_temp,
            "Test definition is wrong, please change configuration",
        )
        self.assertEqual(set_temp, 100)

        self.assertNotEqual(
            initial_time, 10, "Test definition is wrong, please change " +
            "configuration"
        )
        self.assertEqual(set_time, 10)

    def test_get_args(self):
        import pysd

        model = pysd.read_vensim(test_model)
        model2 = pysd.read_vensim(test_model_look)

        self.assertEqual(model.get_args('Room Temperature'), [])
        self.assertEqual(model.get_args('room_temperature'), [])
        self.assertEqual(model.get_args('teacup_temperature'), [])
        self.assertEqual(model.get_args('_integ_teacup_temperature'), [])

        self.assertEqual(model2.get_args('lookup 1d'), ['x'])
        self.assertEqual(model2.get_args('lookup_1d'), ['x'])
        self.assertEqual(model2.get_args('lookup 2d'), ['x'])
        self.assertEqual(model2.get_args('lookup_2d'), ['x'])

        with self.assertRaises(NameError):
            model.get_args('not_a_var')

    def test_get_coords(self):
        import pysd

        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]

        coords_dims = (coords, dims)

        model = pysd.read_vensim(test_model)
        model2 = pysd.read_vensim(test_model_subs)

        self.assertIsNone(model.get_coords("Room Temperature"))
        self.assertIsNone(model.get_coords("room_temperature"))
        self.assertIsNone(model.get_coords("teacup_temperature"))
        self.assertIsNone(model.get_coords("_integ_teacup_temperature"))

        self.assertEqual(model2.get_coords("Initial Values"), coords_dims)
        self.assertEqual(model2.get_coords("initial_values"), coords_dims)
        self.assertEqual(model2.get_coords("Stock A"), coords_dims)
        self.assertEqual(model2.get_coords("stock_a"), coords_dims)
        self.assertEqual(model2.get_coords("_integ_stock_a"), coords_dims)

        with self.assertRaises(NameError):
            model.get_coords('not_a_var')

    def test__build_euler_timeseries(self):
        import pysd

        model = pysd.read_vensim(test_model)
        model.components.initial_time = lambda: 3
        model.components.final_time = lambda: 50
        model.components.time_step = lambda: 1
        model.initialize()

        actual = list(model._build_euler_timeseries(return_timestamps=[10]))
        expected = range(3, 11, 1)
        self.assertSequenceEqual(actual, expected)

        actual = list(model._build_euler_timeseries(return_timestamps=[10],
                                                    final_time=50))
        expected = range(3, 51, 1)
        self.assertSequenceEqual(actual, expected)

    def test__integrate(self):
        import pysd

        # Todo: think through a stronger test here...
        model = pysd.read_vensim(test_model)
        model.progress = False
        res = model._integrate(time_steps=list(range(5)),
                               capture_elements=['teacup_temperature'],
                               return_timestamps=list(range(0, 5, 2)))
        self.assertIsInstance(res, pd.DataFrame)
        self.assertIn('teacup_temperature', res)
        self.assertTrue(all(res.index.values == list(range(0, 5, 2))))

    def test_default_returns_with_construction_functions(self):
        """
        If the run function is called with no arguments, should still be able
        to get default return functions.

        """
        import pysd

        model = pysd.read_vensim(os.path.join(
            _root, "test-models/tests/delays/test_delays.mdl"))
        ret = model.run()
        self.assertTrue(
            {
                "Initial Value",
                "Input",
                "Order Variable",
                "Output Delay1",
                "Output Delay1I",
                "Output Delay3",
            }
            <= set(ret.columns.values)
        )

    def test_default_returns_with_lookups(self):
        """
        Addresses https://github.com/JamesPHoughton/pysd/issues/114
        The default settings should skip model elements with no particular
        return value
        """
        import pysd

        model = pysd.read_vensim(os.path.join(
                _root, "test-models/tests/lookups/test_lookups.mdl"))
        ret = model.run()
        self.assertTrue(
            {"accumulation", "rate", "lookup function call"} <=
            set(ret.columns.values)
        )

    def test_py_model_file(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/86"""
        import pysd

        model = pysd.read_vensim(test_model)
        self.assertEqual(model.py_model_file,
                         test_model.replace(".mdl", ".py"))

    def test_mdl_file(self):
        """Relates to https://github.com/JamesPHoughton/pysd/issues/86"""
        import pysd

        model = pysd.read_vensim(test_model)
        self.assertEqual(model.mdl_file, test_model)

    @unittest.skip("infinite loop")
    def test_incomplete_model(self):
        import pysd

        with catch_warnings(record=True) as w:
            simplefilter("always")
            model = pysd.read_vensim(os.path.join(
                _root,
                "test-models/tests/incomplete_equations/"
                + "test_incomplete_model.mdl"
            ))
        self.assertTrue(any([warn.category == SyntaxWarning for warn in w]))

        with catch_warnings(record=True) as w:
            model.run()
        self.assertEqual(len(w), 1)


class TestModelInteraction(unittest.TestCase):
    """ The tests in this class test pysd's interaction with itself
        and other modules. """

    def test_multiple_load(self):
        """
        Test that we can load and run multiple models at the same time,
        and that the models don't interact with each other. This can
        happen if we arent careful about class attributes vs instance
        attributes

        This test responds to issue:
        https://github.com/JamesPHoughton/pysd/issues/23

        """
        import pysd

        model_1 = pysd.read_vensim(os.path.join(
                _root, "test-models/samples/teacup/teacup.mdl"))
        model_2 = pysd.read_vensim(os.path.join(
                _root, "test-models/samples/SIR/SIR.mdl"))

        self.assertNotIn("teacup_temperature", dir(model_2.components))
        self.assertIn("susceptible", dir(model_2.components))

        self.assertNotIn("susceptible", dir(model_1.components))
        self.assertIn("teacup_temperature", dir(model_1.components))

    def test_no_crosstalk(self):
        """
        Need to check that if we instantiate two copies of the same model,
        changes to one copy do not influence the other copy.

        Checks for issue: https://github.com/JamesPHoughton/pysd/issues/108
        that time is not shared between the two models

        """
        # Todo: this test could be made more comprehensive
        import pysd

        model_1 = pysd.read_vensim(os.path.join(
                _root, "test-models/samples/teacup/teacup.mdl"))
        model_2 = pysd.read_vensim(os.path.join(
                _root, "test-models/samples/SIR/SIR.mdl"))

        model_1.components.initial_time = lambda: 10
        self.assertNotEqual(model_2.components.initial_time, 10)

        # check that the model time is not shared between the two objects
        model_1.run()
        self.assertNotEqual(model_1.time(), model_2.time())

    def test_restart_cache(self):
        """
        Test that when we cache a model variable at the 'run' time,
         if the variable is changed and the model re-run, the cache updates
         to the new variable, instead of maintaining the old one.
        """
        import pysd

        model = pysd.read_vensim(test_model)
        model.run()
        old = model.components.room_temperature()
        model.set_components({"Room Temperature": 345})
        new = model.components.room_temperature()
        model.run()
        self.assertEqual(new, 345)
        self.assertNotEqual(old, new)

    def test_circular_reference(self):
        import pysd

        with self.assertRaises(ValueError) as err:
            pysd.load(
                more_tests
                + "/circular_reference/test_circular_reference.py")

        self.assertIn("_integ_integ", str(err.exception))
        self.assertIn("_delay_delay", str(err.exception))
        self.assertIn(
            "Unresolvable Reference: "
            + "Probable circular initialization...\n"
            + "Not able to initialize the "
            + "following objects:",
            str(err.exception),
        )

    def test_not_able_to_update_stateful_object(self):
        import pysd

        integ = pysd.functions.Integ(
            lambda: xr.DataArray([1, 2], {"Dim": ["A", "B"]}, ["Dim"]),
            lambda: xr.DataArray(0, {"Dim": ["A", "B"]}, ["Dim"]),
            "my_integ_object",
        )

        integ.initialize()

        with self.assertRaises(ValueError) as err:
            integ.update(np.array([[1, 2], [3, 4]]))

        self.assertIn(
            "Could not update the value of my_integ_object", str(err.exception)
        )


class TestMultiRun(unittest.TestCase):
    def test_delay_reinitializes(self):
        import pysd

        model = pysd.read_vensim(os.path.join(
            _root,
            "test-models/tests/delays/test_delays.mdl"))
        res1 = model.run()
        res2 = model.run()
        self.assertTrue(all(res1 == res2))
