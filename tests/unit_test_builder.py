import textwrap
from unittest import TestCase


class TestBuildElement(TestCase):
    def test_multiline_subscripted_constant_construction(self):
        from pysd.builder import build_element
        actual = textwrap.dedent(
            build_element(element={'py_name': 'my_variable',
                                   'py_expr': ['50', '21', '3'],
                                   'subs': [['A'], ['B'], ['C']],
                                   'kind': 'constant',
                                   'doc': 'This is a test.',
                                   'unit': 'kg/s',
                                   'real_name': 'My Variable'},
                          subscript_dict={'Dim1': ['A', 'B', 'C'],
                                          'Dim2': ['D', 'E', 'F', 'G']}))

        expected = textwrap.dedent('''
            @cache('run')
            def my_variable():
                """
                My Variable
                -----------
                (my_variable)
                kg/s

                This is a test.
                """
                ret = xr.DataArray(data=np.ones([3])*np.NaN, coords={'Dim1': ['A', 'B', 'C']})
                ret.loc[{'Dim1': ['A']}] = 50
                ret.loc[{'Dim1': ['B']}] = 21
                ret.loc[{'Dim1': ['C']}] = 3
                return ret
                ''')

        self.assertEqual(expected, actual)

    def test_single_line_1d_subscript_constant_construction(self):
        from pysd.builder import build_element
        actual = textwrap.dedent(
            build_element(element={'kind': 'constant',
                                   'subs': [['Dim1']],
                                   'doc': '',
                                   'py_name': 'rate_a',
                                   'real_name': 'Rate A',
                                   'py_expr': ['0.01,0.02,0.03'],
                                   'unit': ''},
                          subscript_dict={'Dim1': ['A', 'B', 'C']})
        )

        expected = textwrap.dedent('''
            @cache('run')
            def rate_a():
                """
                Rate A
                ------
                (rate_a)



                """
                return xr.DataArray(data=[0.01,0.02,0.03], coords={\'Dim1\': [\'A\', \'B\', \'C\']}, dims=[\'Dim1\'] )
            ''')
        self.assertEqual(expected, actual)

    def test_single_line_2d_subscript_constant_construction(self):
        from pysd.builder import build_element
        actual = textwrap.dedent(
            build_element(element={'kind': 'constant',
                                   'subs': [['Dim1', 'Dim2']],
                                   'doc': '',
                                   'py_name': 'rate_a',
                                   'real_name': 'Rate A',
                                   'py_expr': ['[0.01,0.02],[0.03,0.04],[0.05,0.06]'],
                                   'unit': ''},
                          subscript_dict={'Dim2': ['D', 'E'],
                                          'Dim1': ['A', 'B', 'C']})
        )

        expected = textwrap.dedent('''
            @cache('run')
            def rate_a():
                """
                Rate A
                ------
                (rate_a)



                """
                return xr.DataArray(data=[[0.01,0.02],[0.03,0.04],[0.05,0.06]], coords={\'Dim2\': [\'D\', \'E\'], \'Dim1\': [\'A\', \'B\', \'C\']}, dims=[\'Dim1\', \'Dim2\'] )
            ''')
        self.assertEqual(expected, actual)

    def test_setup_with_call(self):
        from pysd.builder import build_element
        actual = textwrap.dedent(
            build_element(element={'kind': 'setup',
                                   'subs': [['Dim1', 'Dim2']],
                                   'doc': '',
                                   'py_name': '_init_stock_a',
                                   'real_name': 'Implicit',
                                   'py_expr': ['initial_values()'],
                                   'unit': ''},
                          subscript_dict={'Dim2': ['D', 'E'],
                                          'Dim1': ['A', 'B', 'C']})
        )

        expected = textwrap.dedent('''

                def _init_stock_a():
                    """
                    Implicit
                    --------
                    (_init_stock_a)



                    """
                    return initial_values()
                ''')
        self.assertEqual(expected, actual)

    def test_instantiate_array_with_float(self):
        """
        When we add a constant or setup variable, the element may be subscripted,
        but defined using only a single number - ie, setting all elements to the same value.
        We need to be able to handle this.
        """
        from pysd.builder import build_element
        actual = textwrap.dedent(
            build_element(element={'kind': 'constant',
                                   'subs': [['Dim1', 'Dim2']],
                                   'doc': '',
                                   'py_name': 'rate_a',
                                   'real_name': 'Rate A',
                                   'py_expr': ['10'],
                                   'unit': ''},
                          subscript_dict={'Dim2': ['D', 'E'],
                                          'Dim1': ['A', 'B', 'C']})
        )

        expected = textwrap.dedent('''
                    @cache('run')
                    def rate_a():
                        """
                        Rate A
                        ------
                        (rate_a)



                        """
                        return xr.DataArray(data=np.ones([2, 3])*10, coords={'Dim2': ['D', 'E'], 'Dim1': ['A', 'B', 'C']})
                    ''')
        self.assertEqual(expected, actual)


class TestBuild(TestCase):
    def test_build(self):
        # Todo: add other builder-specific inclusions to this test
        from pysd.builder import build
        actual = textwrap.dedent(
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
                  namespace={'StockA': 'stocka'},
                  subscript_dict={'Dim1': ['A', 'B', 'C']},
                  outfile_name='return'))

        self.assertIn("_subscript_dict = {'Dim1': ['A', 'B', 'C']}", actual)
        self.assertIn("functions.time = time", actual)
        self.assertIn("_namespace = {'StockA': 'stocka'}", actual)

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
