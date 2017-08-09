import unittest
import os
import pysd
import numpy as np


class TestRangeChecker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_name = 'test_range.xls'
        model = pysd.read_vensim('test-models/tests/variable_ranges/test_variable_ranges.mdl')
        pysd.testing.create_range_test_matrix(model, cls.file_name)
        cls.bounds = pysd.testing.create_range_test_matrix(model, filename=None)
        cls.result = model.run()

    def test_create_range_test_template(self):
        self.assertTrue(os.path.isfile(self.file_name))

    def test_range_test_return_errors(self):
        errors = pysd.testing.range_test(self.result, bounds=self.file_name)
        self.assertEqual(len(errors), 2)

    def test_range_test_raise_errors(self):
        with self.assertRaises(AssertionError):
            pysd.testing.range_test(self.result,
                                    bounds=self.file_name,
                                    errors='raise')

    def test_range_test_from_bounds(self):
        errors = pysd.testing.range_test(self.result, bounds=self.bounds)
        self.assertEqual(len(errors), 2)

    def test_identify_nan(self):
        new_result = self.result.copy()
        new_result.loc[:3, 'Stock'] = np.NaN

        errors = pysd.testing.range_test(new_result, bounds=self.bounds)
        self.assertEqual(len(errors), 3)


    @classmethod
    def tearDownClass(cls):
        os.remove(cls.file_name)
