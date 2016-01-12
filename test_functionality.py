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


class Test_Vensim_Importer(unittest.TestCase):
    """ Test Import functionality """
    @classmethod
    def setUpClass(cls):
        cls.model = pysd.read_vensim('tests/old_tests/vensim/Teacup.mdl')

    @unittest.skip("in development")
    def test_documentation(self):
        self.assertIsInstance(self.model.components.doc(), basestring)


class Test_PySD(unittest.TestCase):
    """ Testing basic execution functionality """
        #Probably should change this, because we modify the model when we
        #run it with new parameters, and best practice would be to make these
        #tests independednt"""
    
    @classmethod
    def setUpClass(cls):
        cls.model = pysd.read_vensim('tests/old_tests/vensim/Teacup.mdl')
    
    def test_run(self):
        stocks = self.model.run()
        self.assertTrue(isinstance(stocks, pd.DataFrame)) #should return a dataframe
        self.assertTrue('teacup_temperature' in stocks.columns.values) #should contain the stock column
        self.assertGreater(len(stocks), 3) #should have some number of rows (at least more than 1, but here 3)
        self.assertTrue(stocks.notnull().all().all()) #there are no null values in the set

    def test_run_return_timestamps(self):
        #re: https://github.com/JamesPHoughton/pysd/issues/17

        timestamps = np.random.rand(5).cumsum()
        stocks = self.model.run(return_timestamps=timestamps)
        self.assertTrue((stocks.index.values == timestamps).all())
    
        stocks = self.model.run(return_timestamps=5)
        self.assertEqual(stocks.index[0],5)

    def test_run_return_columns(self):
        #re: https://github.com/JamesPHoughton/pysd/issues/26
        return_columns = ['room_temperature','teacup_temperature']
        result = self.model.run(return_columns=return_columns)
        self.assertEqual(set(result.columns), set(return_columns))


    def test_initial_conditions(self):
        stocks = self.model.run(initial_condition=(0, {'teacup_temperature':33}))
        self.assertEqual(stocks['teacup_temperature'].loc[0], 33)
        
        stocks = self.model.run(initial_condition='current', return_timestamps=range(31,45))
        self.assertGreater(stocks['teacup_temperature'].loc[44], 0)
    
        with self.assertRaises(ValueError):
            self.model.run(initial_condition='bad value')

    def test_set_constant_parameter(self):
        #re: https://github.com/JamesPHoughton/pysd/issues/5
        self.model.set_components({'room_temperature':20})
        self.assertEqual(self.model.components.room_temperature(),20)
        
        self.model.run(params={'room_temperature':70})
        self.assertEqual(self.model.components.room_temperature(),70)

    def test_set_timeseries_parameter(self):
        timeseries = range(30)
        temp_timeseries = pd.Series(index=timeseries,
                                    data=(50 + np.random.rand(len(timeseries)).cumsum()))
        print temp_timeseries
        res = self.model.run(params={'room_temperature':temp_timeseries},
                             return_columns=['room_temperature'],
                             return_timestamps=timeseries)
        print res['room_temperature']
        self.assertTrue((res['room_temperature'] == temp_timeseries).all())

    @unittest.skip("in development")
    def test_docs(self):
        #test that the model prints the documentation
        print model #tests model.__str__
        print model.doc() #tests the function we wrote
        model.doc(short=True) #tests condensed model function printing.

    def test_collection(self):
        self.model.run(params={'room_temperature':75},
                  return_timestamps=range(0,30), collect=True)
        self.model.run(params={'room_temperature':25}, initial_condition='current',
                  return_timestamps=range(30,60), collect=True)
        stocks = self.model.get_record()
        self.assertTrue(all(stocks.index.values == np.array(range(0,60))))
        #We may drop this use case, as its relatively low value, and meeting it makes things slower.


class Test_Meta_Stuff(unittest.TestCase):
    """ The tests in this class test pysd's interaction with itself
        and other modules. """

    @unittest.skip("in development")
    def test_multiple_load(self):
        model_1 = pysd.read_vensim('tests/old_tests/vensim/Teacup.mdl')
        model_1.run()

        model_2 = pysd.read_vensim('tests/old_tests/vensim/test_lookups.mdl')
        model_2.run()


if __name__ == '__main__':
    unittest.main()
