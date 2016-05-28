import textwrap
from unittest import TestCase


class TestBuild(TestCase):
    def test_build_element(self):
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

        self.assertEqual(
            build_element(element={'kind': 'constant', 'subs': [['Dim1','Dim2']],
                                   'doc': '', 'py_name': 'rate_a', 'real_name': 'Rate A',
                                   'py_expr': ['[0.01,0.02],[0.03,0.04],[0.05,0.06]'], 'unit': ''},
                          subscript_dict={'Dim2': ['D', 'E'],
                                          'Dim1': ['A', 'B', 'C']}),
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
                ret = xr.DataArray(data=np.empty([3])*NaN, coords={'Dim1': ['A', 'B', 'C'],
                                                                   'Dim2': ['D', 'E']})
                ret = [0.01,0.02],[0.03,0.04],[0.05,0.06]
                return ret
            '''))
        )
        )

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

    class TestMergePartialElements(TestCase):
        def test_single_set(self):
            from pysd.builder import merge_partial_elements

            self.assertEqual(
                merge_partial_elements(
                    [{'py_name': 'a', 'py_expr': 'ms', 'subs': ['Name1', 'element1'],
                      'real_name': 'A', 'doc': 'Test', 'unit': None,
                      'kind': 'component'},
                     {'py_name': 'a', 'py_expr': 'njk', 'subs': ['Name1', 'element2'],
                      'real_name': 'A', 'doc': None, 'unit': None,
                      'kind': 'component'},
                     {'py_name': 'a', 'py_expr': 'as', 'subs': ['Name1', 'element3'],
                      'real_name': 'A', 'doc': '', 'unit': None,
                      'kind': 'component'}]),
                [{'py_name': 'a',
                  'py_expr': ['ms', 'njk', 'as'],
                  'subs': [['Name1', 'element1'], ['Name1', 'element2'], ['Name1', 'element3']],
                  'kind': 'component',
                  'doc': 'Test',
                  'real_name': 'A',
                  'unit': None
                  }])

        def test_multiple_sets(self):
            from pysd.builder import merge_partial_elements

            self.assertEqual(
                merge_partial_elements(
                    [{'py_name': 'a', 'py_expr': 'ms', 'subs': ['Name1', 'element1'],
                      'real_name': 'A', 'doc': 'Test', 'unit': None,
                      'kind': 'component'},
                     {'py_name': 'a', 'py_expr': 'njk', 'subs': ['Name1', 'element2'],
                      'real_name': 'A', 'doc': None, 'unit': None,
                      'kind': 'component'},
                     {'py_name': 'a', 'py_expr': 'as', 'subs': ['Name1', 'element3'],
                      'real_name': 'A', 'doc': '', 'unit': None,
                      'kind': 'component'},
                     {'py_name': 'b', 'py_expr': 'bgf', 'subs': ['Name1', 'element1'],
                      'real_name': 'B', 'doc': 'Test', 'unit': None,
                      'kind': 'component'},
                     {'py_name': 'b', 'py_expr': 'r4', 'subs': ['Name1', 'element2'],
                      'real_name': 'B', 'doc': None, 'unit': None,
                      'kind': 'component'},
                     {'py_name': 'b', 'py_expr': 'ymt', 'subs': ['Name1', 'element3'],
                      'real_name': 'B', 'doc': '', 'unit': None,
                      'kind': 'component'}]),
                [{'py_name': 'a',
                  'py_expr': ['ms', 'njk', 'as'],
                  'subs': [['Name1', 'element1'], ['Name1', 'element2'], ['Name1', 'element3']],
                  'kind': 'component',
                  'doc': 'Test',
                  'real_name': 'A',
                  'unit': None
                  },
                 {'py_name': 'b',
                  'py_expr': ['bgf', 'r4', 'ymt'],
                  'subs': [['Name1', 'element1'], ['Name1', 'element2'], ['Name1', 'element3']],
                  'kind': 'component',
                  'doc': 'Test',
                  'real_name': 'B',
                  'unit': None
                  }])

        def test_non_set(self):
            from pysd.builder import merge_partial_elements

            self.assertEqual(
                merge_partial_elements(
                    [{'py_name': 'a', 'py_expr': 'ms', 'subs': ['Name1', 'element1'],
                      'real_name': 'A', 'doc': 'Test', 'unit': None,
                      'kind': 'component'},
                     {'py_name': 'a', 'py_expr': 'njk', 'subs': ['Name1', 'element2'],
                      'real_name': 'A', 'doc': None, 'unit': None,
                      'kind': 'component'},
                     {'py_name': 'c', 'py_expr': 'as', 'subs': ['Name1', 'element3'],
                      'real_name': 'C', 'doc': 'hi', 'unit': None,
                      'kind': 'component'},
                     ]),
                [{'py_name': 'a',
                  'py_expr': ['ms', 'njk'],
                  'subs': [['Name1', 'element1'], ['Name1', 'element2']],
                  'kind': 'component',
                  'doc': 'Test',
                  'real_name': 'A',
                  'unit': None
                  },
                 {'py_name': 'c',
                  'py_expr': ['as'],
                  'subs': [['Name1', 'element3']],
                  'kind': 'component',
                  'doc': 'hi',
                  'real_name': 'C',
                  'unit': None
                  }])
