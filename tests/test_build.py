import textwrap
from unittest import TestCase


class TestBuild(TestCase):
    def test_build_element(self):  # doctest: +NORMALIZE_WHITESPACE
        from builder import build_element

        self.assertEqual(
                build_element(element={'py_name': 'my_variable',
                                       'py_expr': ['50', '21', '3'],
                                       'subs': [['A'], ['B'], ['C']],
                                       'kind': 'constant',
                                       'doc': 'This is a test.',
                                       'unit': 'kg/s',
                                       'real_name': 'My Variable'},
                              subscript_dict={'Dim1': ['A', 'B', 'C'],
                                              'Dim2': ['D', 'E', 'F', 'G']}),
                textwrap.dedent('''
                @cache('run')
                def my_variable():
                    """
                    My Variable
                    -----------
                    (my_variable)
                    kg/s

                    This is a test.
                    """
                    ret = xr.DataArray(data=np.empty([3])*NaN, coords={'Dim1': ['A', 'B', 'C']})
                    ret.iloc[{'Dim1': ['A']}] = 50
                    ret.iloc[{'Dim1': ['B']}] = 21
                    ret.iloc[{'Dim1': ['C']}] = 3
                    return ret
                '''))

    def test_build(self):
        from builder import build
        self.assertEqual(
                build(elements=[{'kind': 'component',
                                 'subs': [[]],
                                 'doc': '',
                                 'py_name': 'stocka',
                                 'real_name': 'StockA',
                                 'py_expr': ["_state['stocka']"],
                                 'unit': ''},
                                {'kind': 'component',
                                 'subs': [[]],
                                 'doc': 'Provides derivative for stocka function',
                                 'py_name': '_dstocka_dt',
                                 'real_name': 'Implicit',
                                 'py_expr': ['flowa()'],
                                 'unit': 'See docs for stocka'},
                                {'kind': 'setup',
                                 'subs': [[]],
                                 'doc': 'Provides initial conditions for stocka function',
                                 'py_name': 'init_stocka',
                                 'real_name': 'Implicit',
                                 'py_expr': ['-10'],
                                 'unit': 'See docs for stocka'}],
                      subscript_dict={'Dim1': ['A', 'B', 'C']}),
                '''
                @cache('step')
                def stocka():
                """
                StockA
                -----------
                (stocka)
                """
                    return _state['stocka']

                def _dstocka_dt():
                    return flowa()
                ''')
