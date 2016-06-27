import textwrap
from unittest import TestCase
import xarray as xr
import numpy as np
from functions import cache
from numbers import Number


def runner(string, ns=None):
    code = compile(string, '<string>', 'exec')
    if not ns:
        ns = dict()
    ns.update({'cache': cache, 'xr': xr, 'np': np})
    exec (code, ns)
    return ns


class TestBuildElement(TestCase):
    def test_no_subs_constant(self):
        from pysd.builder import build_element
        string = textwrap.dedent(
            build_element(element={'kind': 'constant',
                                   'subs': [[]],
                                   'doc': '',
                                   'py_name': 'my_variable',
                                   'real_name': 'My Variable',
                                   'py_expr': ['0.01'],
                                   'unit': ''},
                          subscript_dict={})
        )
        ns = runner(string)
        a = ns['my_variable']()
        self.assertIsInstance(a, Number)
        self.assertEqual(a, .01)

    def test_no_subs_call(self):
        from pysd.builder import build_element
        string = textwrap.dedent(
            build_element(element={'kind': 'constant',
                                   'subs': [[]],
                                   'doc': '',
                                   'py_name': 'my_variable',
                                   'real_name': 'My Variable',
                                   'py_expr': ['other_variable()'],
                                   'unit': ''},
                          subscript_dict={})
            )
        ns = {'other_variable': lambda: 3}
        ns = runner(string, ns)
        a = ns['my_variable']()
        self.assertIsInstance(a, Number)
        self.assertEqual(a, 3)

    def test_single_line_1d_subscript_constant_construction(self):
        from pysd.builder import build_element
        string = textwrap.dedent(
            build_element(element={'kind': 'constant',
                                   'subs': [['Dim1']],
                                   'doc': '',
                                   'py_name': 'my_variable',
                                   'real_name': 'My Variable',
                                   'py_expr': ['0.01,0.02,0.03'],
                                   'unit': ''},
                          subscript_dict={'Dim1': ['A', 'B', 'C']})
        )

        ns = runner(string)
        a = ns['my_variable']()
        self.assertDictEqual({key: list(val.values) for key, val in a.coords.iteritems()},
                             {'Dim1': ['A', 'B', 'C']})
        self.assertEqual(a.loc[{'Dim1': 'A'}], 0.01)
        self.assertEqual(a.loc[{'Dim1': 'B'}], 0.02)

    def test_single_line_2d_subscript_constant_construction(self):
        from pysd.builder import build_element
        string = textwrap.dedent(
            build_element(element={'kind': 'constant',
                                   'subs': [['Dim1', 'Dim2']],
                                   'doc': '',
                                   'py_name': 'my_variable',
                                   'real_name': 'My Variable',
                                   'py_expr': ['[0.01,0.02],[0.03,0.04],[0.05,0.06]'],
                                   'unit': ''},
                          subscript_dict={'Dim2': ['D', 'E'],
                                          'Dim1': ['A', 'B', 'C']})
        )
        ns = runner(string)
        a = ns['my_variable']()
        self.assertDictEqual({key: list(val.values) for key, val in a.coords.iteritems()},
                             {'Dim2': ['D', 'E'],
                              'Dim1': ['A', 'B', 'C']})
        self.assertEqual(a.loc[{'Dim1': 'A', 'Dim2': 'D'}], 0.01)
        self.assertEqual(a.loc[{'Dim1': 'A', 'Dim2': 'E'}], 0.02)
        self.assertEqual(a.loc[{'Dim1': 'B', 'Dim2': 'E'}], 0.04)

    def test_multiline_subscripted_2d_constant_construction_from_float_constants(self):
        from pysd.builder import build_element
        string = textwrap.dedent(
            build_element(element={'py_name': 'my_variable',
                                   'py_expr': ['50', '21', '3'],   # list of equations
                                   'subs': [['A', 'Dim2'], ['B', 'Dim2'], ['C', 'Dim2']],  # list of subscripts for each equation
                                   'kind': 'constant',
                                   'doc': 'This is a test.',
                                   'unit': 'kg/s',
                                   'real_name': 'My Variable'},
                          subscript_dict={'Dim1': ['A', 'B', 'C'],
                                          'Dim2': ['D', 'E', 'F', 'G']}))

        ns = runner(string)
        a = ns['my_variable']()
        self.assertDictEqual({key: list(val.values) for key, val in a.coords.iteritems()},
                             {'Dim1': ['A', 'B', 'C'], 'Dim2': ['D', 'E', 'F', 'G']})
        self.assertEqual(a.loc[{'Dim1': 'A', 'Dim2': 'D'}], 50)
        self.assertEqual(a.loc[{'Dim1': 'A', 'Dim2': 'E'}], 50)
        self.assertEqual(a.loc[{'Dim1': 'B', 'Dim2': 'E'}], 21)

    def test_multiline_subscripted_2d_constant_construction_from_1d_constants(self):
        from pysd.builder import build_element
        string = textwrap.dedent(
            build_element(element={'py_name': 'my_variable',
                                   'py_expr': ['50, 51, 52, 53',
                                               '21, 22, 23, 24',
                                               '3, 4, 5, 6'],  # list of equations
                                   'subs': [['A', 'Dim2'],
                                            ['B', 'Dim2'],
                                            ['C', 'Dim2']], # list of subscripts for each equation
                                   'kind': 'constant',
                                   'doc': 'This is a test.',
                                   'unit': 'kg/s',
                                   'real_name': 'My Variable'},
                          subscript_dict={'Dim1': ['A', 'B', 'C'],
                                          'Dim2': ['D', 'E', 'F', 'G']}))

        ns = runner(string)
        a = ns['my_variable']()
        self.assertDictEqual({key: list(val.values) for key, val in a.coords.iteritems()},
                             {'Dim1': ['A', 'B', 'C'], 'Dim2': ['D', 'E', 'F', 'G']})
        self.assertEqual(a.loc[{'Dim1': 'A', 'Dim2': 'D'}], 50)
        self.assertEqual(a.loc[{'Dim1': 'A', 'Dim2': 'E'}], 51)
        self.assertEqual(a.loc[{'Dim1': 'B', 'Dim2': 'E'}], 22)

    def test_multiline_subscripted_3d_constant_construction(self):
        from pysd.builder import build_element
        string = textwrap.dedent(
            build_element(element={'py_name': 'my_variable',
                                   'py_expr': ['[1,2],[3,4],[5,6]', '[2,4],[5,3],[1,4]'],   # list of equations
                                   'subs': [['Dim1', 'Dim2', 'F'], ['Dim1', 'Dim2', 'G']],  # list of subscripts for each equation
                                   'kind': 'constant',
                                   'doc': 'This is a test.',
                                   'unit': 'kg/s',
                                   'real_name': 'My Variable'},
                          subscript_dict={'Dim1': ['A', 'B', 'C'],
                                          'Dim2': ['D', 'E'],
                                          'Dim3': ['F', 'G']}))

        ns = runner(string)
        a = ns['my_variable']()
        self.assertDictEqual({key: list(val.values) for key, val in a.coords.iteritems()},
                             {'Dim1': ['A', 'B', 'C'], 'Dim2': ['D', 'E', 'F', 'G']})
        self.assertEqual(a.loc[{'Dim1': 'A', 'Dim2': 'D'}], 50)
        self.assertEqual(a.loc[{'Dim1': 'A', 'Dim2': 'E'}], 50)
        self.assertEqual(a.loc[{'Dim1': 'B', 'Dim2': 'E'}], 21)

    def test_subscript_with_call(self):
        from pysd.builder import build_element
        string = textwrap.dedent(
            build_element(element={'kind': 'setup',
                                   'subs': [['Dim1', 'Dim2']],
                                   'doc': '',
                                   'py_name': 'my_variable',
                                   'real_name': 'My Variable',
                                   'py_expr': ['initial_values()'],
                                   'unit': ''},
                          subscript_dict={'Dim2': ['D', 'E'],
                                          'Dim1': ['A', 'B', 'C']})
        )
        ns = {'initial_values': lambda: xr.DataArray([[1, 2], [3, 4], [5, 6]],
                                                     coords={'Dim2': ['D', 'E'],
                                                             'Dim1': ['A', 'B', 'C']},
                                                     dims=['Dim1', 'Dim2'])}
        ns = runner(string, ns)
        a = ns['my_variable']()
        self.assertDictEqual({key: list(val.values) for key, val in a.coords.iteritems()},
                             {'Dim1': ['A', 'B', 'C'], 'Dim2': ['D', 'E']})
        self.assertEqual(a.loc[{'Dim1': 'A', 'Dim2': 'D'}], 1)
        self.assertEqual(a.loc[{'Dim1': 'B', 'Dim2': 'E'}], 4)

    def test_instantiate_array_with_float(self):
        """
        When we add a constant or setup variable, the element may be subscripted,
        but defined using only a single number - ie, setting all elements to the same value.
        We need to be able to handle this.
        """
        from pysd.builder import build_element
        string = textwrap.dedent(
            build_element(element={'kind': 'constant',
                                   'subs': [['Dim1', 'Dim2']],
                                   'doc': '',
                                   'py_name': 'my_variable',
                                   'real_name': 'My Variable',
                                   'py_expr': ['10'],
                                   'unit': ''},
                          subscript_dict={'Dim2': ['D', 'E'],
                                          'Dim1': ['A', 'B', 'C']})
        )

        ns = runner(string)
        a = ns['my_variable']()
        self.assertDictEqual({key: list(val.values) for key, val in a.coords.iteritems()},
                             {'Dim1': ['A', 'B', 'C'], 'Dim2': ['D', 'E']})
        self.assertEqual(a.loc[{'Dim1': 'A', 'Dim2': 'D'}], 10)
        self.assertEqual(a.loc[{'Dim1': 'B', 'Dim2': 'E'}], 10)



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


class TestMakeXarray(TestCase):
    def test_constant(self):
        from pysd.builder import make_xarray
        actual = make_xarray(subs=['Dim1', 'Dim2', 'F'],
                             expr='5',
                             subscript_dict={'Dim1': ['A', 'B', 'C'],
                                             'Dim2': ['D', 'E'],
                                             'Dim3': ['F', 'G']})

        # Todo: update the expected value with what we decide
        expected = textwrap.dedent("""\
            xr.DataArray(data=5,
                         coords='Dim1': ['A', 'B', 'C'], 'Dim2': ['D', 'E'], 'Dim3': ['F']})""")
        self.assertEqual(actual, expected)

    def test_func_call(self):
        from pysd.builder import make_xarray
        actual = make_xarray(subs=['Dim1', 'Dim2', 'F'],
                             expr='other_function()',
                             subscript_dict={'Dim1': ['A', 'B', 'C'],
                                             'Dim2': ['D', 'E'],
                                             'Dim3': ['F', 'G']})

        # Todo: update the expected value with what we decide
        expected = textwrap.dedent("""\
            xr.DataArray(data=5,
                         coords='Dim1': ['A', 'B', 'C'], 'Dim2': ['D', 'E'], 'Dim3': ['F']})""")
        self.assertEqual(actual, expected)

    def test_array_complete_build(self):
        """
        Variable[Sub1,Sub2]=
            1, 2; 3, 4; 5, 6;
            ~
            ~		|
        """
        from pysd.builder import make_xarray
        string = make_xarray(subs=['Dim1', 'Dim2'],
                             expr='[1, 2], [3, 4], [5, 6]',
                             subscript_dict={'Dim1': ['A', 'B', 'C'],
                                             'Dim2': ['D', 'E']})

        obj = eval(string)
        self.assertSequenceEqual(obj.values, [[1, 2], [3, 4], [5, 6]])

    def test_subscript_updimension(self):
        """
        Test models where we need to broadcast a lower-dimensional
        array to a larger case

        Two Dims[Dim1,Dim2]=
            One Dim[Dim1]
            ~
            ~		|
        """
        def one_dim():
            return xr.DataArray(data=[1, 2, 3],
                                coords={'Dim1': ['A', 'B', 'C']})

        from pysd.builder import make_xarray
        string = make_xarray(subs=['Dim1', 'Dim2'],
                             expr='one_dim()',
                             subscript_dict={'Dim1': ['A', 'B', 'C'],
                                             'Dim2': ['D', 'E']})

        obj = eval(string, locals={'one_dim': one_dim})
        self.assertSequenceEqual(obj.shape, [3, 2])
        self.assertEqual(obj.loc[{'Dim1': 'D'}],
                         obj.loc[{'Dim1': 'E'}])
