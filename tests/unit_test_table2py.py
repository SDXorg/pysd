import unittest

import xarray as xr


class TestReadTabular(unittest.TestCase):
    def test_read_tab_file(self):
        import pysd
        model = pysd.read_tabular('test-models/samples/teacup/teacup_mdl.tab')
        result = model.run()
        self.assertTrue(isinstance(result, xr.Dataset))  # return a dataset
        self.assertTrue('Teacup Temperature' in result.data_vars)  # contains correct column
        self.assertGreater(len(result), 3)  # has multiple rows
        self.assertTrue(result.notnull().all().all())  # there are no null values in the set
