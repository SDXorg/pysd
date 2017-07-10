import unittest
import pandas as pd
import numpy as np
from pysd import utils

test_model = 'test-models/samples/teacup/teacup.mdl'


class TestPySD(unittest.TestCase):
    def test_run(self):
        import pysd
        model = pysd.read_vensim(test_model)
        stocks = model.run()
        self.assertTrue(isinstance(stocks, pd.DataFrame))  # return a dataframe
        self.assertTrue('Teacup Temperature' in stocks.columns.values)  # contains correct column
        self.assertGreater(len(stocks), 3)  # has multiple rows
        self.assertTrue(stocks.notnull().all().all())  # there are no null values in the set

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
        expected = [3., 4., 5., 6., 7.]
        self.assertSequenceEqual(actual, expected)

    def test_run_return_timestamps(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/17"""
        import pysd
        model = pysd.read_vensim(test_model)
        timestamps = np.random.rand(5).cumsum()
        stocks = model.run(return_timestamps=timestamps)
        self.assertTrue((stocks.index.values == timestamps).all())

        stocks = model.run(return_timestamps=5)
        self.assertEqual(stocks.index[0], 5)

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
        return_columns = ['Room Temperature', 'Teacup Temperature']
        result = model.run(return_columns=return_columns)
        self.assertEqual(set(result.columns), set(return_columns))

    def test_run_return_columns_pysafe_names(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/26"""
        import pysd
        model = pysd.read_vensim(test_model)
        return_columns = ['room_temperature', 'teacup_temperature']
        result = model.run(return_columns=return_columns)
        self.assertEqual(set(result.columns), set(return_columns))

    def test_initial_conditions_tuple_pysafe_names(self):
        import pysd
        model = pysd.read_vensim(test_model)
        stocks = model.run(initial_condition=(3000, {'teacup_temperature': 33}),
                           return_timestamps=list(range(3000, 3010)))
        self.assertEqual(stocks.index[0], 3000)
        self.assertEqual(stocks['Teacup Temperature'].iloc[0], 33)

    def test_initial_conditions_tuple_original_names(self):
        """ Responds to https://github.com/JamesPHoughton/pysd/issues/77"""
        import pysd
        model = pysd.read_vensim(test_model)
        stocks = model.run(initial_condition=(3000, {'Teacup Temperature': 33}),
                           return_timestamps=list(range(3000, 3010)))
        self.assertEqual(stocks.index[0], 3000)
        self.assertEqual(stocks['Teacup Temperature'].iloc[0], 33)

    def test_initial_conditions_current(self):
        import pysd
        model = pysd.read_vensim(test_model)
        stocks1 = model.run(return_timestamps=list(range(0, 31)))
        stocks2 = model.run(initial_condition='current',
                            return_timestamps=list(range(30, 45)))
        self.assertEqual(stocks1['Teacup Temperature'].iloc[-1],
                         stocks2['Teacup Temperature'].iloc[0])

    def test_initial_condition_bad_value(self):
        import pysd
        model = pysd.read_vensim(test_model)
        with self.assertRaises(ValueError):
            model.run(initial_condition='bad value')

    def test_set_constant_parameter(self):
        """ In response to: re: https://github.com/JamesPHoughton/pysd/issues/5"""
        import pysd
        model = pysd.read_vensim(test_model)
        model.set_components({'room_temperature': 20})
        self.assertEqual(model.components.room_temperature(), 20)

        model.run(params={'room_temperature': 70})
        self.assertEqual(model.components.room_temperature(), 70)

    def test_set_timeseries_parameter(self):
        import pysd
        model = pysd.read_vensim(test_model)
        timeseries = list(range(30))
        temp_timeseries = pd.Series(index=timeseries,
                                    data=(50 + np.random.rand(len(timeseries)).cumsum()))
        res = model.run(params={'room_temperature': temp_timeseries},
                        return_columns=['room_temperature'],
                        return_timestamps=timeseries)
        self.assertTrue((res['room_temperature'] == temp_timeseries).all())

    def test_set_component_with_real_name(self):
        import pysd
        model = pysd.read_vensim(test_model)
        model.set_components({'Room Temperature': 20})
        self.assertEqual(model.components.room_temperature(), 20)

        model.run(params={'Room Temperature': 70})
        self.assertEqual(model.components.room_temperature(), 70)

    def test_set_components_warnings(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/80"""
        import pysd
        import warnings
        model = pysd.read_vensim(test_model)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.set_components({'Teacup Temperature': 20,
                                  'Characteristic Time': 15})  # set stock value using params
        self.assertEqual(len(w), 1)
        self.assertTrue(
            'Teacup Temperature' in str(w[0].message))  # check that warning references the stock

    def test_set_components_with_function(self):
        def test_func():
            return 5

        import pysd
        model = pysd.read_vensim(test_model)
        model.set_components({'Room Temperature': test_func})
        res = model.run(return_columns=['Room Temperature'])
        self.assertEqual(test_func(), res['Room Temperature'].iloc[0])

    def test_docs(self):
        """ Test that the model prints some documentation """
        import pysd
        model = pysd.read_vensim(test_model)
        self.assertIsInstance(str(model), str)  # tests string conversion of model

        doc = model.doc()
        self.assertIsInstance(doc, pd.DataFrame)
        self.assertIn('Teacup Temperature', doc['Real Name'].values)
        self.assertIn('teacup_temperature', doc['Py Name'].values)

    def test_cache(self):
        # Todo: test stepwise and runwise caching
        import pysd
        model = pysd.read_vensim(test_model)
        model.run()
        self.assertIsNotNone(model.components.room_temperature.cache_val)

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

    def test_reset_state(self):
        import pysd
        import warnings
        model = pysd.read_vensim(test_model)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.reset_state()
        self.assertEqual(len(w), 1)

    def test_set_state(self):
        import pysd
        model = pysd.read_vensim(test_model)

        initial_temp = model.components.teacup_temperature()
        initial_time = model.components.time()

        new_time = np.random.rand()

        # Test that we can set with real names
        model.set_state(new_time, {'Teacup Temperature': 500})
        self.assertNotEqual(initial_temp, 500)
        self.assertEqual(model.components.teacup_temperature(), 500)

        # Test setting with pysafe names
        model.set_state(new_time + 1, {'teacup_temperature': 202})
        self.assertEqual(model.components.teacup_temperature(), 202)

        # Test setting with stateful object name
        model.set_state(new_time + 2, {'integ_teacup_temperature': 302})
        self.assertEqual(model.components.teacup_temperature(), 302)

    def test_replace_element(self):
        import pysd
        model = pysd.read_vensim(test_model)
        stocks1 = model.run()
        model.components.characteristic_time = lambda: 3
        stocks2 = model.run()
        self.assertGreater(stocks1['Teacup Temperature'].loc[10],
                           stocks2['Teacup Temperature'].loc[10])

    def test_set_initial_condition(self):
        import pysd
        model = pysd.read_vensim(test_model)
        initial_temp = model.components.teacup_temperature()
        initial_time = model.components.time()

        new_state = {'Teacup Temperature': 500}
        new_time = np.random.rand()

        model.set_initial_condition((new_time, new_state))
        set_temp = model.components.teacup_temperature()
        set_time = model.components.time()

        self.assertNotEqual(set_temp, initial_temp)
        self.assertEqual(set_temp, 500)

        self.assertNotEqual(initial_time, new_time)
        self.assertEqual(new_time, set_time)

        model.set_initial_condition('original')
        set_temp = model.components.teacup_temperature()
        set_time = model.components.time()

        self.assertEqual(initial_temp, set_temp)
        self.assertEqual(initial_time, set_time)

    def test__build_euler_timeseries(self):
        import pysd
        model = pysd.read_vensim(test_model)
        model.components.initial_time = lambda: 3
        model.components.final_time = lambda: 10
        model.components.time_step = lambda: 1
        model.initialize()

        actual = list(model._build_euler_timeseries(return_timestamps=[10]))
        expected = range(3, 11, 1)
        self.assertSequenceEqual(actual, expected)

    def test__integrate(self):
        import pysd
        # Todo: think through a stronger test here...
        model = pysd.read_vensim(test_model)
        res = model._integrate(time_steps=list(range(5)),
                               capture_elements=['teacup_temperature'],
                               return_timestamps=list(range(0, 5, 2)))
        self.assertIsInstance(res, list)
        self.assertIsInstance(res[0], dict)

    def test_default_returns_with_construction_functions(self):
        """
        If the run function is called with no arguments

        """
        import pysd
        model = pysd.read_vensim('test-models/tests/delays/test_delays.mdl')
        ret = model.run()
        self.assertTrue({'Stock Delay1I',
                         'Stock Delay3I',
                         'Stock Delay1',
                         'Stock DelayN',
                         'Stock Delay3'} <=
                        set(ret.columns.values))

    def test_default_returns_with_lookups(self):
        """
        Addresses https://github.com/JamesPHoughton/pysd/issues/114
        The default settings should skip model elements with no particular
        return value
        """
        import pysd
        model = pysd.read_vensim('test-models/tests/lookups/test_lookups.mdl')
        ret = model.run()
        self.assertTrue({'accumulation',
                         'rate',
                         'lookup function call'} <=
                        set(ret.columns.values))

    def test_py_model_file(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/86"""
        import pysd
        model = pysd.read_vensim(test_model)
        self.assertEqual(model.py_model_file, test_model.replace('.mdl', '.py'))

    def test_mdl_file(self):
        """Relates to https://github.com/JamesPHoughton/pysd/issues/86"""
        import pysd
        model = pysd.read_vensim(test_model)
        self.assertEqual(model.mdl_file, test_model)


class TestModelInteraction(unittest.TestCase):
    """ The tests in this class test pysd's interaction with itself
        and other modules. """

    def test_multiple_load(self):
        """
        Test that we can load and run multiple models at the same time,
        and that the models don't interact with each other. This can
        happen if we arent careful about class attributes vs instance attributes

        This test responds to issue:
        https://github.com/JamesPHoughton/pysd/issues/23

        """
        import pysd

        model_1 = pysd.read_vensim('test-models/samples/teacup/teacup.mdl')
        model_2 = pysd.read_vensim('test-models/samples/SIR/SIR.mdl')

        self.assertNotIn('teacup_temperature', dir(model_2.components))
        self.assertIn('susceptible', dir(model_2.components))

        self.assertNotIn('susceptible', dir(model_1.components))
        self.assertIn('teacup_temperature', dir(model_1.components))

    def test_no_crosstalk(self):
        """
        Need to check that if we instantiate two copies of the same model,
        changes to one copy do not influence the other copy.

        Checks for issue: https://github.com/JamesPHoughton/pysd/issues/108
        that time is not shared between the two models

        """
        # Todo: this test could be made more comprehensive
        import pysd

        model_1 = pysd.read_vensim('test-models/samples/teacup/teacup.mdl')
        model_2 = pysd.read_vensim('test-models/samples/SIR/SIR.mdl')

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
        model.set_components({'Room Temperature': 345})
        new = model.components.room_temperature()
        model.run()
        self.assertEqual(new, 345)
        self.assertNotEqual(old, new)
