from unittest import TestCase
import xarray as xr
import pandas as pd
import test_utils

class TestUtils(TestCase):

    def test_get_return_elements(self):
        from utils import get_return_elements

        self.assertEqual(
            get_return_elements(["Inflow A[Entry 1,Column 1]",
                                 "Inflow A[Entry 1,Column 2]"],
                                {'Inflow A': 'inflow_a'},
                                {'Dim1': ['Entry 1', 'Entry 2'],
                                 'Dim2': ['Column 1', 'Column 2']}),
            (['inflow_a'],
             {'Inflow A[Entry 1,Column 1]': ('inflow_a', {'Dim1': ['Entry 1'],
                                                          'Dim2': ['Column 1']}),
              'Inflow A[Entry 1,Column 2]': ('inflow_a', {'Dim1': ['Entry 1'],
                                                          'Dim2': ['Column 2']})}
             )
        )

        self.assertEqual(
            get_return_elements(["Inflow A",
                                 "Inflow B"],
                                {'Dim1': ['Entry 1', 'Entry 2'],
                                 'Dim2': ['Column 1', 'Column 2']}),
                                {'Inflow A': 'inflow_a',
                                 'Inflow B': 'inflow_b'},
            (['inflow_a, inflow_b'],
             {'Inflow A': ('inflow_a', {}),
              'Inflow B': ('inflow_b', {})}
             )
        )

    def test_make_flat_df(self):
        from utils import make_flat_df

        frames = [{'elem1': xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                       {'Dim1': ['A', 'B', 'C'],
                                        'Dim2': ['D', 'E', 'F']}),
                   'elem2': xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                       {'Dim1': ['A', 'B', 'C'],
                                        'Dim2': ['D', 'E', 'F']})},
                  {'elem1': xr.DataArray([[2, 4, 6], [8, 10, 12], [14, 16, 19]],
                                         {'Dim1': ['A', 'B', 'C'],
                                          'Dim2': ['D', 'E', 'F']}),
                   'elem2': xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                         {'Dim1': ['A', 'B', 'C'],
                                          'Dim2': ['D', 'E', 'F']})}]

        return_addresses = {'Elem1[B,F]': ('elem1', {'Dim1': ['B'], 'Dim2': ['F']})}
        df = pd.DataFrame([{'Elem1[B,F]': 8}, {'Elem1[B,F]': 16}])
        resultdf = make_flat_df(frames, return_addresses)

        test_utils.assertFramesClose(resultdf, df, rtol=.01)


    def test_visit_addresses(self):
        from utils import visit_addresses

        frame = {'elem1': xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                       {'Dim1': ['A', 'B', 'C'],
                                        'Dim2': ['D', 'E', 'F']}),
                 'elem2': xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                       {'Dim1': ['A', 'B', 'C'],
                                        'Dim2': ['D', 'E', 'F']})}

        return_addresses = {'Elem1[B,F]': ('elem1', {'Dim1': ['B'], 'Dim2': ['F']})}
        self.assertEqual(visit_addresses(frame, return_addresses),
                         {'Elem1[B,F]': 8})

    def test_visit_addresses_return_array(self):
        """ There could be cases where we want to
        return a whole section of an array - ie, by passing in only part of
        the simulation dictionary. in this case, we can't force to float..."""
        self.fail()
