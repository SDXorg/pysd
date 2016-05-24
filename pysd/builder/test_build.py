from unittest import TestCase


class TestBuild(TestCase):
    def test_build(self):
        from builder import build
        self.assertEqual(
                build(elements=[{'py_name': 'my_variable',
                                 'expr': ['50', '21', '3'],
                                 'subs': [['Dim1', 'A'], ['Dim1', 'B'], ['Dim1', 'C']],
                                 'kind': 'constant',
                                 'doc': 'This is a test.',
                                 'unit': 'kg/s',
                                 'real_name': 'My Variable'}],
                      subscript_dict={'Dim1': ['A', 'B', 'C']}),
                '''
                @cache('run')
                def variable():
                """
                My Variable
                -----------
                (my_variable)
                kg/s

                This is a test.
                """
                    ret = xr.DataArray(data=np.empty(3), coords={'Dim1': ['A', 'B', 'C']})
                    ret.iloc[{'Dim1': ['A']}] = 50
                    ret.iloc[{'Dim2': ['B']}] = 21
                    ret.iloc[{'Dim3': ['C']}] = 3
                    return ret
                ''')

        self.assertEqual(
                build(elements=[{'kind': 'component',
                                 'subs': [[]],
                                 'doc': '',
                                 'py_name': 'stocka',
                                 'real_name': 'StockA',
                                 'py_expr': ["_state['stocka']"],
                                 'unit': ''},
                                {'kind': 'implicit',
                                 'subs': [[]],
                                 'doc': 'Provides derivative for stocka function',
                                 'py_name': '_dstocka_dt',
                                 'real_name': None,
                                 'py_expr': ['flowa()'],
                                 'unit': 'See docs for stocka'},
                                {'kind': 'implicit',
                                 'subs': [[]],
                                 'doc': 'Provides initial conditions for stocka function',
                                 'py_name': 'init_stocka',
                                 'real_name': None,
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
