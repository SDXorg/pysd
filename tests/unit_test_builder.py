import textwrap

from unittest import TestCase

import numpy as np
from pysd import cache
from numbers import Number
import xarray as xr



def runner(string, ns=None):
    code = compile(string, '<string>', 'exec')
    if not ns:
        ns = dict()
    ns.update({'cache': cache, 'xr': xr, 'np': np})
    exec(code, ns)
    return ns


class TestBuildElement(TestCase):
    def test_no_subs_constant(self):
        from pysd.py_backend.builder import build_element
        string = textwrap.dedent(
            build_element(element={'kind': 'constant',
                                   'subs': [[]],
                                   'doc': '',
                                   'py_name': 'my_variable',
                                   'real_name': 'My Variable',
                                   'py_expr': ['0.01'],
                                   'unit': '',
                                   'eqn': '',
                                   'lims': '',
                                   'arguments': ''},
                          subscript_dict={})
        )
        ns = runner(string)
        a = ns['my_variable']()
        self.assertIsInstance(a, Number)
        self.assertEqual(a, .01)

    def test_no_subs_call(self):
        from pysd.py_backend.builder import build_element
        string = textwrap.dedent(
            build_element(element={'kind': 'constant',
                                   'subs': [[]],
                                   'doc': '',
                                   'py_name': 'my_first_variable',
                                   'real_name': 'My Variable',
                                   'py_expr': ['other_variable()'],
                                   'eqn': '',
                                   'lims': '',
                                   'unit': '',
                                   'arguments': ''},
                          subscript_dict={})
        )
        ns = {'other_variable': lambda: 3}
        ns = runner(string, ns)
        a = ns['my_first_variable']()
        self.assertIsInstance(a, Number)
        self.assertEqual(a, 3)


class TestBuild(TestCase):
    def test_build(self):
        # Todo: add other builder-specific inclusions to this test
        from pysd.py_backend.builder import build
        actual = textwrap.dedent(
            build(elements=[{'kind': 'component',
                             'subs': [],
                             'doc': '',
                             'py_name': 'stocka',
                             'real_name': 'StockA',
                             'py_expr': "_state['stocka']",
                             'eqn': '',
                             'lims': '',
                             'unit': '',
                             'arguments': ''},
                            {'kind': 'component',
                             'subs': '',
                             'doc': 'Provides derivative for stocka function',
                             'py_name': '_dstocka_dt',
                             'real_name': 'Implicit',
                             'py_expr': 'flowa()',
                             'unit': 'See docs for stocka',
                             'eqn': '',
                             'lims': '',
                             'arguments': ''},
                            {'kind': 'setup',
                             'subs': None,
                             'doc': 'Provides initial conditions for stocka function',
                             'py_name': 'init_stocka',
                             'real_name': 'Implicit',
                             'py_expr': '-10',
                             'unit': 'See docs for stocka',
                             'eqn': '',
                             'lims': '',
                             'arguments': ''}],
                  namespace={'StockA': 'stocka'},
                  subscript_dict={'Dim1': ['A', 'B', 'C']},
                  outfile_name='return'))

        self.assertIn('_subscript_dict = {"Dim1": ["A", "B", "C"]}', actual)
        self.assertIn('_namespace = {"StockA": "stocka"}', actual)


class TestMergePartialElements(TestCase):
    def test_single_set(self):
        from pysd.py_backend.builder import merge_partial_elements

        self.assertEqual(
            merge_partial_elements(
                [{'py_name': 'a', 'py_expr': 'ms', 'subs': ['Name1', 'element1'],
                  'real_name': 'A', 'doc': 'Test', 'unit': None, 'eqn': 'eq1', 'lims': '',
                  'kind': 'component', 'arguments': ''},
                 {'py_name': 'a', 'py_expr': 'njk', 'subs': ['Name1', 'element2'],
                  'real_name': 'A', 'doc': None, 'unit': None, 'eqn': 'eq2', 'lims': '',
                  'kind': 'component', 'arguments': ''},
                 {'py_name': 'a', 'py_expr': 'as', 'subs': ['Name1', 'element3'],
                  'real_name': 'A', 'doc': '', 'unit': None, 'eqn': 'eq3', 'lims': '',
                  'kind': 'component', 'arguments': ''}]),
            [{'py_name': 'a',
              'py_expr': ['ms', 'njk', 'as'],
              'subs': [['Name1', 'element1'], ['Name1', 'element2'], ['Name1', 'element3']],
              'kind': 'component',
              'doc': 'Test',
              'real_name': 'A',
              'unit': None,
              'eqn': ['eq1', 'eq2', 'eq3'],
              'lims': '',
              'arguments': ''
              }])

    def test_multiple_sets(self):
        from pysd.py_backend.builder import merge_partial_elements
        actual = merge_partial_elements(
            [{'py_name': 'a', 'py_expr': 'ms', 'subs': ['Name1', 'element1'],
              'real_name': 'A', 'doc': 'Test', 'unit': None, 'eqn': 'eq1', 'lims': '',
              'kind': 'component', 'arguments': ''},
             {'py_name': 'a', 'py_expr': 'njk', 'subs': ['Name1', 'element2'],
              'real_name': 'A', 'doc': None, 'unit': None, 'eqn': 'eq2', 'lims': '',
              'kind': 'component', 'arguments': ''},
             {'py_name': 'a', 'py_expr': 'as', 'subs': ['Name1', 'element3'],
              'real_name': 'A', 'doc': '', 'unit': None, 'eqn': 'eq3', 'lims': '',
              'kind': 'component', 'arguments': ''},
             {'py_name': 'b', 'py_expr': 'bgf', 'subs': ['Name1', 'element1'],
              'real_name': 'B', 'doc': 'Test', 'unit': None, 'eqn': 'eq4', 'lims': '',
              'kind': 'component', 'arguments': ''},
             {'py_name': 'b', 'py_expr': 'r4', 'subs': ['Name1', 'element2'],
              'real_name': 'B', 'doc': None, 'unit': None, 'eqn': 'eq5', 'lims': '',
              'kind': 'component', 'arguments': ''},
             {'py_name': 'b', 'py_expr': 'ymt', 'subs': ['Name1', 'element3'],
              'real_name': 'B', 'doc': '', 'unit': None, 'eqn': 'eq6', 'lims': '',
              'kind': 'component', 'arguments': ''}])

        expected = [{'py_name': 'a',
                     'py_expr': ['ms', 'njk', 'as'],
                     'subs': [['Name1', 'element1'], ['Name1', 'element2'], ['Name1', 'element3']],
                     'kind': 'component',
                     'doc': 'Test',
                     'real_name': 'A',
                     'unit': None,
                     'eqn': ['eq1', 'eq2', 'eq3'],
                     'lims': '',
                     'arguments': ''
                     },
                    {'py_name': 'b',
                     'py_expr': ['bgf', 'r4', 'ymt'],
                     'subs': [['Name1', 'element1'], ['Name1', 'element2'], ['Name1', 'element3']],
                     'kind': 'component',
                     'doc': 'Test',
                     'real_name': 'B',
                     'unit': None,
                     'eqn': ['eq4', 'eq5', 'eq6'],
                     'lims': '',
                     'arguments': ''
                     }]
        self.assertIn(actual[0], expected)
        self.assertIn(actual[1], expected)

    def test_non_set(self):
        from pysd.py_backend.builder import merge_partial_elements
        actual = merge_partial_elements(
            [{'py_name': 'a', 'py_expr': 'ms', 'subs': ['Name1', 'element1'],
              'real_name': 'A', 'doc': 'Test', 'unit': None, 'eqn': 'eq1', 'lims': '',
              'kind': 'component', 'arguments': ''},
             {'py_name': 'a', 'py_expr': 'njk', 'subs': ['Name1', 'element2'],
              'real_name': 'A', 'doc': None, 'unit': None, 'eqn': 'eq2', 'lims': '',
              'kind': 'component', 'arguments': ''},
             {'py_name': 'c', 'py_expr': 'as', 'subs': ['Name1', 'element3'],
              'real_name': 'C', 'doc': 'hi', 'unit': None, 'eqn': 'eq3', 'lims': '',
              'kind': 'component', 'arguments': ''},
             ])

        expected = [{'py_name': 'a',
                     'py_expr': ['ms', 'njk'],
                     'subs': [['Name1', 'element1'], ['Name1', 'element2']],
                     'kind': 'component',
                     'doc': 'Test',
                     'real_name': 'A',
                     'unit': None,
                     'eqn': ['eq1', 'eq2'],
                     'lims': '',
                     'arguments': ''
                     },
                    {'py_name': 'c',
                     'py_expr': ['as'],
                     'subs': [['Name1', 'element3']],
                     'kind': 'component',
                     'doc': 'hi',
                     'real_name': 'C',
                     'unit': None,
                     'eqn': ['eq3'],
                     'lims': '',
                     'arguments': ''
                     }]

        self.assertIn(actual[0], expected)
        self.assertIn(actual[1], expected)
