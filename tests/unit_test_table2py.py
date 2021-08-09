import os
import unittest
import pandas as pd

_root = os.path.dirname(__file__)


class TestReadTabular(unittest.TestCase):
    def test_read_tab_file(self):
        import pysd
        model = pysd.read_tabular(os.path.join(
            _root, 'test-models/samples/teacup/teacup_mdl.tab'))
        result = model.run()
        self.assertTrue(isinstance(result, pd.DataFrame))  # return a dataframe
        self.assertTrue('Teacup Temperature' in result.columns.values)  # contains correct column
        self.assertGreater(len(result), 3)  # has multiple rows
        self.assertTrue(result.notnull().all().all())  # there are no null values in the set
