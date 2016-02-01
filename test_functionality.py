"""
This set of tests is to be used when developing pysd functionality
as it tests the various functions that pysd provides, on a simple model.

To test the full range of sd models, use test_pysd.py
"""

import pysd
import unittest
import pandas as pd
import numpy as np

print '*'*150
print 'Testing module at location:', pysd.__file__


class TestVensimImporter(unittest.TestCase):
    """ Test Import functionality """
    @classmethod
    def setUpClass(cls):
        cls.model = pysd.read_vensim('tests/old_tests/vensim/Teacup.mdl')


class TestSubscriptSpecificFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = pysd.read_vensim('tests/test-models/tests/subscript_aggregation/test_subscript_aggregation.mdl')

    def test_flatten(self):
        # todo: write this test
        pass


class TestBasicFunctionality(unittest.TestCase):
    """ Testing basic execution functionality """
    # Probably should change this, because we modify the model when we
    # run it with new parameters, and best practice would be to make these
    # tests independent"""
    
    @classmethod
    def setUpClass(cls):
        cls.model = pysd.read_vensim('tests/old_tests/vensim/Teacup.mdl')
    
    def test_run(self):
        stocks = self.model.run()
        self.assertTrue(isinstance(stocks, pd.DataFrame))  # return a dataframe
        self.assertTrue('teacup_temperature' in stocks.columns.values)  # contains correct column
        self.assertGreater(len(stocks), 3)  # has multiple rows
        self.assertTrue(stocks.notnull().all().all())  # there are no null values in the set

    def test_run_return_timestamps(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/17"""
        timestamps = np.random.rand(5).cumsum()
        stocks = self.model.run(return_timestamps=timestamps)
        self.assertTrue((stocks.index.values == timestamps).all())
    
        stocks = self.model.run(return_timestamps=5)
        self.assertEqual(stocks.index[0], 5)

    def test_run_return_columns(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/26"""
        return_columns = ['room_temperature', 'teacup_temperature']
        result = self.model.run(return_columns=return_columns)
        self.assertEqual(set(result.columns), set(return_columns))

    def test_initial_conditions(self):
        stocks = self.model.run(initial_condition=(0, {'teacup_temperature': 33}))
        self.assertEqual(stocks['teacup_temperature'].loc[0], 33)
        
        stocks = self.model.run(initial_condition='current', return_timestamps=range(31, 45))
        self.assertGreater(stocks['teacup_temperature'].loc[44], 0)
    
        with self.assertRaises(ValueError):
            self.model.run(initial_condition='bad value')

    def test_set_constant_parameter(self):
        """ In response to: re: https://github.com/JamesPHoughton/pysd/issues/5"""
        self.model.set_components({'room_temperature': 20})
        self.assertEqual(self.model.components.room_temperature(), 20)
        
        self.model.run(params={'room_temperature': 70})
        self.assertEqual(self.model.components.room_temperature(), 70)

    def test_set_timeseries_parameter(self):
        timeseries = range(30)
        temp_timeseries = pd.Series(index=timeseries,
                                    data=(50 + np.random.rand(len(timeseries)).cumsum()))
        print temp_timeseries
        res = self.model.run(params={'room_temperature': temp_timeseries},
                             return_columns=['room_temperature'],
                             return_timestamps=timeseries)
        print res['room_temperature']
        self.assertTrue((res['room_temperature'] == temp_timeseries).all())

    def test_flatten_nonexisting_subscripts(self):
        """ Even when the model has no subscripts, we should be able to set this to either value"""
        self.model.run(flatten_subscripts=True)
        self.model.run(flatten_subscripts=False)

    def test_docs(self):
        """ Test that the model prints the documentation """
        # Todo: Test that this prints the docstring from teacup.mdl as we would like it,
        # not just that it prints a string.
        self.assertIsInstance(self.model.__str__, basestring)  # tests model.__str__
        self.assertIsInstance(self.model.doc(), basestring)  # tests the function we wrote
        self.assertIsInstance(self.model.doc(short=True), basestring)

    def test_collection(self):
        self.model.run(params={'room_temperature': 75},
                       return_timestamps=range(0, 30), collect=True)
        self.model.run(params={'room_temperature': 25}, initial_condition='current',
                       return_timestamps=range(30, 60), collect=True)
        stocks = self.model.get_record()
        self.assertTrue(all(stocks.index.values == np.array(range(0, 60))))
        # We may drop this use case, as its relatively low value,
        # and meeting it makes things slower.
        # Todo: test clearing the record

    def test_set_components(self):
        # Todo: write this test
        pass

    def test_set_state(self):
        # Todo: write this test
        pass

    def test_set_initial_condition(self):
        # Todo: write this test
        pass

    def test_cache(self):
        self.model.run()
        self.assertIsNotNone(self.model.components.room_temperature.cache)


class TestMetaStuff(unittest.TestCase):
    """ The tests in this class test pysd's interaction with itself
        and other modules. """

    def test_multiple_load(self):
        """Test that we can load and run multiple models at the same time,
        and that the models don't interact with each other. This can
        happen if we arent careful about class attributes vs instance attributes"""
        # Todo: Make this a stricter test,
        #  perhaps by checking that the components do not share members they shouldn't

        model_1 = pysd.read_vensim('tests/old_tests/vensim/Teacup.mdl')
        model_1.run()

        model_2 = pysd.read_vensim('tests/old_tests/vensim/test_lookups.mdl')
        model_2.run()


if __name__ == '__main__':
    unittest.main()
