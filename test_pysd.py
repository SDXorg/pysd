#from pysd.translators import vensim2py
import pysd
import unittest
import inspect


print 'Testing module at location:', pysd.__file__


class Test_Vensim_Importer(unittest.TestCase):
    """ Test Import functionality """
    @classmethod
    def setUpClass(cls):
        cls.model = pysd.read_vensim('tests/Teacup.mdl')

    def test_documentation(self):
        self.assertIsInstance(self.model.components.doc(), basestring)


class Test_PySD(unittest.TestCase):
    """ Testing basic execution functionality """
        #Probably should change this, because we modify the model when we
        #run it with new parameters, and best practice would be to make these
        #tests independednt"""
    
    @classmethod
    def setUpClass(cls):
        cls.model = pysd.read_vensim('tests/Teacup.mdl')
    
    def test_run(self):
        self.model.run()
        
    def test_run_return_timestamps(self):
        #re: https://github.com/JamesPHoughton/pysd/issues/17
        self.model.run(return_timestamps=[1,5,7])
        self.model.run(return_timestamps=5)

    @unittest.skip("known bug, to fix")
    def test_run_return_columns(self):
        #re: https://github.com/JamesPHoughton/pysd/issues/26
        self.model.run(return_columns=['room_temperature','teacup_temperature'])
        self.model.run(return_columns='room_temperature')

    def test_initial_conditions(self):
        stocks = self.model.run(initial_condition=(0, {'teacup_temperature':33}))
        self.assertEqual(stocks['teacup_temperature'].loc[0], 33)
        
        stocks = self.model.run(initial_condition='current', return_timestamps=range(31,45))
        self.assertGreater(stocks['teacup_temperature'].loc[44], 0)

    def test_parameter_setting(self):
        #re: https://github.com/JamesPHoughton/pysd/issues/5
        self.model.set_components({'room_temperature':20})
        self.assertEqual(self.model.components.room_temperature(),20)
        
        self.model.run(params={'room_temperature':70})
        self.assertEqual(self.model.components.room_temperature(),70)



class Test_Macro_Stuff(unittest.TestCase):
    """ The tests in this class test pysd's interaction with itself
        and other modules. """

    def test_multiple_load(self):
        model_1 = pysd.read_vensim('tests/Teacup.mdl')
        model_1.run()

        model_2 = pysd.read_vensim('tests/test_lookups.mdl')
        model_2.run()


class Test_Specific_Models(unittest.TestCase):
    """ All of the tests here call upon a specific model file 
        that matches the name of the function """

    def test_lookups(self):
        model = pysd.read_vensim('tests/test_lookups.mdl')
        self.assertEqual(model.components.lookup_function_table(.4),.44)
        model.run()

    def test_long_variable_names(self):
        #re: https://github.com/JamesPHoughton/pysd/issues/19
        model = pysd.read_vensim('tests/test_long_variable_names.mdl')
        self.assertEqual(model.components.particularly_efflusive_auxilliary_descriptor(),2)

    def test_vensim_functions(self):
        model = pysd.read_vensim('tests/test_long_variable_names.mdl')
        self.assertEqual(model.components.particularly_efflusive_auxilliary_descriptor(),2)

    def test_time(self):
        #re: https://github.com/JamesPHoughton/pysd/issues/25
        model = pysd.read_vensim('tests/test_time.mdl')
        model.run()

    def test_multi_views(self):
        #just to make sure having multiple sheets doesn't influence reading
        model = pysd.read_vensim('tests/test_multi_views.mdl')
        model.run()

if __name__ == '__main__':
    unittest.main()
