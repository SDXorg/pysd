import os
import unittest
import pandas as pd
import numpy as np
import scipy.stats
import pysd


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
        new_result.loc[:3, 'Lower Bounded Stock'] = np.NaN

        errors = pysd.testing.range_test(new_result, bounds=self.bounds)
        self.assertEqual(len(errors), 3)

    def test_no_issues(self):
        new_result = self.result.copy()
        new_result.loc[:, 'Lower Bounded Stock'] = 10
        new_result.loc[:, 'Broken Upper Bounded Variable'] = 1

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


class TestSamplePSpace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = pysd.read_vensim('test-models/tests/variable_ranges/test_variable_ranges.mdl')
        cls.bounds = pysd.testing.create_range_test_matrix(cls.model, filename=None)
        cls.param_names = ['Unbounded Variable Value Above 0',
                           'Unbounded Variable Value Below 0', 'Both Bounds Above 0',
                           'Both Bounds Below 0', 'Both Bounds Identical',
                           'Unbounded Variable Value 0', 'Lower Bound 0',
                           'Lower Bound Above 0', 'Lower Bound Below 0', 'Upper Bound 0',
                           'Upper Bound Above 0', 'Upper Bound Below 0']
        cls.n_samples = 50001
        cls.samples = pysd.testing.sample_pspace(model=cls.model,
                                                 bounds=cls.bounds,
                                                 param_list=cls.param_names,
                                                 samples=cls.n_samples)

    def test_correct_shape(self):
        self.assertIsInstance(self.samples, pd.DataFrame)
        self.assertEqual(len(self.samples.index), self.n_samples)
        self.assertSetEqual(set(self.samples.columns),
                            set(self.param_names))

    def test_values_within_bounds(self):
        errors = pysd.testing.range_test(self.samples, bounds=self.bounds)
        self.assertIsNone(errors)

    def test_uniform(self):
        self.assertGreater(scipy.stats.kstest(rvs=self.samples['Both Bounds Above 0'],
                                              cdf='uniform', args=(2, 8)).pvalue,
                           .9)

        self.assertGreater(scipy.stats.kstest(rvs=self.samples['Both Bounds Below 0'],
                                              cdf='uniform', args=(-10, 7)).pvalue,
                           .9)

    def test_normal(self):
        self.assertGreater(scipy.stats.kstest(rvs=self.samples['Unbounded Variable Value Above 0'],
                                              cdf='norm', args=(10, 10)).pvalue,
                           .9)

        self.assertGreater(scipy.stats.kstest(rvs=self.samples['Unbounded Variable Value Below 0'],
                                              cdf='norm', args=(-10, 10)).pvalue,
                           .9)

        self.assertGreater(scipy.stats.kstest(rvs=self.samples['Unbounded Variable Value 0'],
                                              cdf='norm', args=(0, 1)).pvalue,
                           .9)

    def test_exponential_lower_bound(self):
        self.assertGreater(scipy.stats.kstest(rvs=self.samples['Lower Bound 0'],
                                              cdf='expon', args=(0, 2)).pvalue,
                           .9)

        self.assertGreater(scipy.stats.kstest(rvs=self.samples['Lower Bound Below 0'],
                                              cdf='expon', args=(-10, 12)).pvalue,
                           .9)

        self.assertGreater(scipy.stats.kstest(rvs=self.samples['Lower Bound Above 0'],
                                              cdf='expon', args=(5, 15)).pvalue,
                           .9)

    def test_exponential_upper_bound(self):
        self.assertGreater(scipy.stats.kstest(rvs=10-self.samples['Upper Bound Above 0'],
                                              cdf='expon', args=(0, 6)).pvalue,
                           .9)

        self.assertGreater(scipy.stats.kstest(rvs=-5-self.samples['Upper Bound Below 0'],
                                              cdf='expon', args=(0, 5)).pvalue,
                           .9)

        self.assertGreater(scipy.stats.kstest(rvs=0-self.samples['Upper Bound 0'],
                                              cdf='expon', args=(0, 2)).pvalue,
                           .9)
