import doctest
import unittest
from unittest import TestCase

import pandas as pd
import xarray as xr

from . import test_utils


class TestUtils(TestCase):

    def test_get_return_elements_subscirpts(self):
        import pysd

        self.assertEqual(
            pysd.utils.get_return_elements(["Inflow A[Entry 1,Column 1]",
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

    def test_get_return_elements_realnames(self):
        import pysd
        self.assertEqual(
            pysd.utils.get_return_elements(["Inflow A",
                                            "Inflow B"],
                                           subscript_dict={'Dim1': ['Entry 1', 'Entry 2'],
                                                           'Dim2': ['Column 1', 'Column 2']},
                                           namespace={'Inflow A': 'inflow_a',
                                                      'Inflow B': 'inflow_b'}),
            (['inflow_a', 'inflow_b'],
             {'Inflow A': ('inflow_a', {}),
              'Inflow B': ('inflow_b', {})}
             )
        )

    def test_get_return_elements_pysafe_names(self):
        import pysd
        self.assertEqual(
            pysd.utils.get_return_elements(["inflow_a",
                                            "inflow_b"],
                                           subscript_dict={'Dim1': ['Entry 1', 'Entry 2'],
                                                           'Dim2': ['Column 1', 'Column 2']},
                                           namespace={'Inflow A': 'inflow_a',
                                                      'Inflow B': 'inflow_b'}),
            (['inflow_a', 'inflow_b'],
             {'inflow_a': ('inflow_a', {}),
              'inflow_b': ('inflow_b', {})}
             )
        )

    @unittest.skip('obsolete?')
    def test_make_flat_df(self):
        import pysd

        frames = [{'elem1': xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                         {'Dim1': ['A', 'B', 'C'],
                                          'Dim2': ['D', 'E', 'F']},
                                         dims=['Dim1', 'Dim2']),
                   'elem2': xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                         {'Dim1': ['A', 'B', 'C'],
                                          'Dim2': ['D', 'E', 'F']},
                                         dims=['Dim1', 'Dim2'])},
                  {'elem1': xr.DataArray([[2, 4, 6], [8, 10, 12], [14, 16, 19]],
                                         {'Dim1': ['A', 'B', 'C'],
                                          'Dim2': ['D', 'E', 'F']},
                                         dims=['Dim1', 'Dim2']),
                   'elem2': xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                         {'Dim1': ['A', 'B', 'C'],
                                          'Dim2': ['D', 'E', 'F']},
                                         dims=['Dim1', 'Dim2'])}]

        return_addresses = {'Elem1[B,F]': ('elem1', {'Dim1': ['B'], 'Dim2': ['F']})}
        df = pd.DataFrame([{'Elem1[B,F]': 6}, {'Elem1[B,F]': 12}])
        resultdf = pysd.utils.make_flat_df(frames, return_addresses)

        test_utils.assert_frames_close(resultdf, df, rtol=.01)

    def test_visit_addresses(self):
        import pysd

        frame = {'elem1': xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                       {'Dim1': ['A', 'B', 'C'],
                                        'Dim2': ['D', 'E', 'F']},
                                       dims=['Dim1', 'Dim2']),
                 'elem2': xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                       {'Dim1': ['A', 'B', 'C'],
                                        'Dim2': ['D', 'E', 'F']},
                                       dims=['Dim1', 'Dim2'])}

        return_addresses = {'Elem1[B,F]': ('elem1', {'Dim1': ['B'], 'Dim2': ['F']})}
        self.assertEqual(pysd.utils.visit_addresses(frame, return_addresses),
                         {'Elem1[B,F]': 6})

    def test_visit_addresses_nosubs(self):
        import pysd

        frame = {'elem1': 25, 'elem2': 13}
        return_addresses = {'Elem1': ('elem1', {}),
                            'Elem2': ('elem2', {})}

        self.assertEqual(pysd.utils.visit_addresses(frame, return_addresses),
                         {'Elem1': 25, 'Elem2': 13})

    def test_visit_addresses_return_array(self):
        """ There could be cases where we want to
        return a whole section of an array - ie, by passing in only part of
        the simulation dictionary. in this case, we can't force to float..."""
        import pysd

        frame = {'elem1': xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                       {'Dim1': ['A', 'B', 'C'],
                                        'Dim2': ['D', 'E', 'F']},
                                       dims=['Dim1', 'Dim2']),
                 'elem2': xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                       {'Dim1': ['A', 'B', 'C'],
                                        'Dim2': ['D', 'E', 'F']},
                                       dims=['Dim1', 'Dim2'])}
        return_addresses = {'Elem1[A, Dim2]': ('elem1', {'Dim1': ['A'], 'Dim2': ['D', 'E', 'F']})}

        actual = pysd.utils.visit_addresses(frame, return_addresses)
        expected = {'Elem1[A, Dim2]':
                        xr.DataArray([[1, 2, 3]],
                                     {'Dim1': ['A'],
                                      'Dim2': ['D', 'E', 'F']},
                                     dims=['Dim1', 'Dim2']),
                    }
        self.assertIsInstance(list(actual.values())[0], xr.DataArray)
        self.assertEqual(actual['Elem1[A, Dim2]'].shape,
                         expected['Elem1[A, Dim2]'].shape)
        # Todo: test that the values are equal

    def test_make_coord_dict(self):
        import pysd
        self.assertEqual(pysd.utils.make_coord_dict(['Dim1', 'D'],
                                                    {'Dim1': ['A', 'B', 'C'],
                                                     'Dim2': ['D', 'E', 'F']},
                                                    terse=True),
                         {'Dim2': ['D']})
        self.assertEqual(pysd.utils.make_coord_dict(['Dim1', 'D'],
                                                    {'Dim1': ['A', 'B', 'C'],
                                                     'Dim2': ['D', 'E', 'F']},
                                                    terse=False),
                         {'Dim1': ['A', 'B', 'C'], 'Dim2': ['D']})

    def test_find_subscript_name(self):
        import pysd
        self.assertEqual(pysd.utils.find_subscript_name({'Dim1': ['A', 'B'],
                                                         'Dim2': ['C', 'D', 'E'],
                                                         'Dim3': ['F', 'G', 'H', 'I']},
                                                        'D'),
                         'Dim2')

        self.assertEqual(pysd.utils.find_subscript_name({'Dim1': ['A', 'B'],
                                                         'Dim2': ['C', 'D', 'E'],
                                                         'Dim3': ['F', 'G', 'H', 'I']},
                                                        'Dim3'),
                         'Dim3')

    def test_doctests(self):
        import pysd
        doctest.DocTestSuite(pysd.utils)
