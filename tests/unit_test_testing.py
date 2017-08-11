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

    def test_no_issues(self):
        new_result = self.result.copy()
        new_result.loc[:, 'Stock'] = 10
        new_result.loc[:, 'Variable'] = 1

        errors = pysd.testing.range_test(new_result, bounds=self.bounds)
        self.assertIsNone(errors)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.file_name)


class TestStaticTestMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_name = 'test_extremes_empty.xls'
        cls.model_file = 'test-models/tests/variable_ranges/test_variable_ranges.mdl'
        cls.model = pysd.read_vensim('test-models/tests/variable_ranges/test_variable_ranges.mdl')
        pysd.testing.create_static_test_matrix(cls.model, cls.file_name)
        cls.filled_file_name = 'test-models/tests/variable_ranges/test_extremes.xls'

    def test_create_range_test_template(self):
        self.assertTrue(os.path.isfile(self.file_name))

    def test_static_test_matrix(self):
        errors = pysd.testing.static_test_matrix(self.model_file,
                                                 excel_file=self.filled_file_name)
        self.assertEqual(len(errors), 1)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.file_name)
