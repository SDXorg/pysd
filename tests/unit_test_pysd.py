from unittest import TestCase
import pandas as pd
import numpy as np

test_model = 'test-models/samples/teacup/teacup.mdl'


class TestPySD(TestCase):
    def test_run(self):
        import pysd
        model = pysd.read_vensim(test_model)
        stocks = model.run()
        self.assertTrue(isinstance(stocks, pd.DataFrame))  # return a dataframe
        self.assertTrue('teacup_temperature' in stocks.columns.values)  # contains correct column
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

    def test_run_return_columns(self):
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
        self.assertEqual(stocks['teacup_temperature'].loc[0], 33)

        stocks = model.run(initial_condition='current', return_timestamps=range(31, 45))
        self.assertGreater(stocks['teacup_temperature'].loc[44], 0)

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

    def test_flatten_nonexisting_subscripts(self):
        """ Even when the model has no subscripts, we should be able to set this to either value"""
        import pysd
        model = pysd.read_vensim(test_model)
        model.run(flatten_subscripts=True)
        model.run(flatten_subscripts=False)

    def test_docs(self):
        """ Test that the model prints the documentation """
        # Todo: Test that this prints the docstring from teacup.mdl as we would like it,
        # not just that it prints a string.
        import pysd
        model = pysd.read_vensim(test_model)
        self.assertIsInstance(model.__str__, basestring)  # tests model.__str__
        self.assertIsInstance(model.doc(), basestring)  # tests the function we wrote
        self.assertIsInstance(model.doc(short=True), basestring)

    def test_collection(self):
        import pysd
        model = pysd.read_vensim(test_model)
        model.run(params={'room_temperature': 75},
                  return_timestamps=range(0, 30), collect=True)
        model.run(params={'room_temperature': 25}, initial_condition='current',
                  return_timestamps=range(30, 60), collect=True)
        stocks = model.get_record()
        self.assertTrue(all(stocks.index.values == np.array(range(0, 60))))
        # We may drop this use case, as its relatively low value,
        # and meeting it makes things slower.
        # Todo: test clearing the record

    def test_cache(self):
        import pysd
        model = pysd.read_vensim(test_model)
        model.run()
        self.assertIsNotNone(model.components.room_temperature.cache)

    def test_reset_state(self):
        self.fail()

    def test_get_record(self):
        self.fail()

    def test_clear_record(self):
        self.fail()

    def test_set_components(self):
        self.fail()

    def test_set_state(self):
        self.fail()

    def test_set_initial_condition(self):
        self.fail()

    def test__build_timeseries(self):
        self.fail()

    def test__timeseries_component(self):
        self.fail()

    def test__constant_component(self):
        self.fail()

    def test__euler_step(self):
        self.fail()

    def test__integrate(self):
        self.fail()

    def test__flatten_dataframe(self):
        self.fail()


class TestModelInteraction(TestCase):
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

        model_1 = pysd.read_vensim('test-models/samples/teacup/Teacup.mdl')
        model_2 = pysd.read_vensim('test-models/samples/SIR/SIR.mdl')

        self.assertNotIn('teacup_temperature', dir(model_2.components))

    def test_no_crosstalk(self):
        """
        Need to check that if we instantiate two copies of the same model,
        changes to one copy do not influence the other copy.
        """

        self.fail()

