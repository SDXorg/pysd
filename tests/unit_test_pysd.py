import unittest
import pandas as pd
import numpy as np

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
        return_timestamps = range(0, 100, 10)
        stocks = model.run(return_timestamps=return_timestamps)
        self.assertSequenceEqual(return_timestamps, list(stocks.index))

    def test_run_return_columns_fullnames(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/26"""
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

    def test_initial_conditions(self):
        import pysd
        model = pysd.read_vensim(test_model)
        stocks = model.run(initial_condition=(0, {'teacup_temperature': 33}))
        actual = stocks['Teacup Temperature'].loc[0]
        expected = 33
        self.assertEqual(actual, expected)

        stocks = model.run(initial_condition='current',
                           return_timestamps=range(31, 45))
        self.assertGreater(stocks['Teacup Temperature'].loc[44], 0)

        with self.assertRaises(TypeError):
            self.run(initial_condition='bad value')

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
        timeseries = range(30)
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

    @unittest.skip('@SimonStrong working')
    def test_docs(self):
        """ Test that the model prints some documentation """
        import pysd
        model = pysd.read_vensim(test_model)
        self.assertIsInstance(model.__str__, str)  # tests model.__str__
        self.assertIsInstance(model.doc(), str)  # tests the function we wrote
        self.assertIsInstance(model.doc(short=True), str)

    def test_cache(self):
        # Todo: test stepwise and runwise caching
        import pysd
        model = pysd.read_vensim(test_model)
        model.run()
        self.assertIsNotNone(model.components.room_temperature.cache)

    def test_reset_state(self):
        import pysd
        model = pysd.read_vensim(test_model)
        initial_state = model.components._state.copy()
        model.run()
        final_state = model.components._state.copy()
        model.reset_state()
        reset_state = model.components._state.copy()
        self.assertNotEqual(initial_state, final_state)
        self.assertEqual(initial_state, reset_state)

    def test_set_state(self):
        import pysd
        model = pysd.read_vensim(test_model)

        initial_state = model.components._state.copy()
        initial_time = model.components.time()

        new_state = {key: np.random.rand() for key in initial_state.iterkeys()}
        new_time = np.random.rand()

        model.set_state(new_time, new_state)

        set_state = model.components._state.copy()
        set_time = model.components.time()

        self.assertNotEqual(initial_state, new_state)
        self.assertEqual(set_state, new_state)

        self.assertNotEqual(initial_time, new_time)
        self.assertEqual(new_time, set_time)

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
        initial_state = model.components._state.copy()
        initial_time = model.components.time()

        new_state = {key: np.random.rand() for key in initial_state.iterkeys()}
        new_time = np.random.rand()

        model.set_initial_condition((new_time, new_state))
        set_state = model.components._state.copy()
        set_time = model.components.time()

        self.assertNotEqual(initial_state, new_state)
        self.assertEqual(set_state, new_state)

        self.assertNotEqual(initial_time, new_time)
        self.assertEqual(new_time, set_time)

        model.set_initial_condition('original')
        set_state = model.components._state.copy()
        set_time = model.components.time()

        self.assertEqual(initial_state, set_state)
        self.assertEqual(initial_time, set_time)

    def test__build_euler_timeseries(self):
        import pysd
        model = pysd.read_vensim(test_model)
        model.components.initial_time = lambda: 3
        model.components.final_time = lambda: 10
        model.components.time_step = lambda: 1

        actual = list(model._build_euler_timeseries())
        expected = range(3, 11, 1)
        self.assertSequenceEqual(actual, expected)

    def test_build_euler_timeseries_with_timestamps(self):
        import pysd
        model = pysd.read_vensim(test_model)
        model.components.initial_time = lambda: 3
        model.components.final_time = lambda: 7
        model.components.time_step = lambda: 1

        actual = list(model._build_euler_timeseries([3.14, 5.7]))
        expected = [3, 3.14, 4, 5, 5.7, 6, 7]
        self.assertSequenceEqual(actual, expected)

    def test__timeseries_component(self):
        import pysd
        model = pysd.read_vensim(test_model)
        temp_timeseries = pd.Series(index=range(0, 30, 1),
                                    data=range(0, 60, 2))
        func = model._timeseries_component(temp_timeseries)
        model.components._t = 0
        self.assertEqual(func(), 0)

        model.components._t = 2.5
        self.assertEqual(func(), 5)

        model.components._t = 3.1
        self.assertEqual(func(), 6.2)

    def test__constant_component(self):
        import pysd
        model = pysd.read_vensim(test_model)
        val = 12.3
        func = model._constant_component(val)
        model.components._t = 0
        self.assertEqual(func(), val)

        model.components._t = 2.5
        self.assertEqual(func(), val)

    def test__euler_step(self):
        import pysd
        model = pysd.read_vensim(test_model)
        state = model.components._state.copy()
        next_step = model._euler_step(model.components._dfuncs,
                                      model.components._state,
                                      1)
        self.assertIsInstance(next_step, dict)
        self.assertNotEqual(next_step, state)
        double_step = model._euler_step(model.components._dfuncs,
                                        model.components._state,
                                        2)
        self.assertEqual(double_step['teacup_temperature'] - next_step['teacup_temperature'],
                         next_step['teacup_temperature'] - state['teacup_temperature'])

    def test__integrate(self):
        import pysd
        # Todo: think through a stronger test here...
        model = pysd.read_vensim(test_model)
        res = model._integrate(derivative_functions=model.components._dfuncs,
                               timesteps=range(5),
                               capture_elements=['teacup_temperature'],
                               return_timestamps=range(0, 5, 2))
        self.assertIsInstance(res, list)
        self.assertIsInstance(res[0], dict)


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

    def test_no_crosstalk(self):
        """
        Need to check that if we instantiate two copies of the same model,
        changes to one copy do not influence the other copy.
        """
        # Todo: this test could be made more comprehensive
        import pysd

        model_1 = pysd.read_vensim('test-models/samples/teacup/teacup.mdl')
        model_2 = pysd.read_vensim('test-models/samples/SIR/SIR.mdl')

        model_1.components.initial_time = lambda: 10
        self.assertNotEqual(model_2.components.initial_time, 10)

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



