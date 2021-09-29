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
                                   'merge_subs': [],
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
                                   'merge_subs': [],
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


class TestBuildFunctionCall(TestCase):
    def test_build_function_not_implemented(self):
        from warnings import catch_warnings
        from pysd.py_backend.builder import build_function_call
        args = ['a', 'b']
        nif = {"name": "not_implemented_function",
               "module": "functions",
               "original_name": "NIF"}
        with catch_warnings(record=True) as ws:
            self.assertEqual(build_function_call(nif, args),
                             "not_implemented_function('NIF',a,b)")
            self.assertEqual(len(ws), 1)
            self.assertTrue("Trying to translate NIF which it is "
                            + "not implemented on PySD."
                            in str(ws[0].message))

    def test_build_function_with_time_dependency(self):
        from pysd.py_backend.builder import build_function_call
        args = ['a', 'b']
        pulse = {
            "name": "pulse",
            "parameters": [
                {"name": "time", "type": "time"},
                {"name": "start"},
                {"name": "duration"},
            ],
            "module": "functions",
        }

        dependencies = {'a', 'b'}
        self.assertNotIn('time', dependencies)
        build_function_call(pulse, args, dependencies)
        self.assertIn('time', dependencies)


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
                             'py_expr': ["_state['stocka']"],
                             'eqn': [''],
                             'lims': '',
                             'unit': '',
                             'merge_subs': [],
                             'arguments': ''},
                            {'kind': 'component',
                             'subs': [],
                             'doc': 'Provides derivative for stocka',
                             'py_name': '_dstocka_dt',
                             'real_name': 'Implicit',
                             'py_expr': ['flowa()'],
                             'unit': 'See docs for stocka',
                             'eqn': [''],
                             'lims': '',
                             'merge_subs': [],
                             'arguments': ''},
                            {'kind': 'setup',
                             'subs': [],
                             'doc': 'Provides initial conditions for stocka',
                             'py_name': 'init_stocka',
                             'real_name': 'Implicit',
                             'py_expr': ['-10'],
                             'unit': 'See docs for stocka',
                             'eqn': [''],
                             'lims': '',
                             'merge_subs': [],
                             'arguments': ''}],
                  namespace={'StockA': 'stocka'},
                  subscript_dict={'Dim1': ['A', 'B', 'C']},
                  dependencies={
                      "stocka": {"_integ_stocka"},
                      "_integ_stocka": {
                          "initial": None,
                          "step": {"flowa"}
                      },
                      "flowa": None
                  },
                  outfile_name='return'))
        self.assertIn('_subscript_dict = {"Dim1": ["A", "B", "C"]}', actual)
        self.assertIn('_namespace = {"StockA": "stocka"}', actual)
        self.assertIn(
            '_dependencies = {\n    "stocka": {"_integ_stocka"},'
            + '\n    "_integ_stocka": {"initial": None, "step": {"flowa"}},'
            + '\n    "flowa": None,\n}', actual)


class TestMergePartialElements(TestCase):
    def test_single_set(self):
        from pysd.py_backend.builder import merge_partial_elements

        self.assertEqual(
            merge_partial_elements(
                [{'py_name': 'a', 'py_expr': 'ms',
                  'subs': ['Name1', 'element1'],
                  'merge_subs': ['Name1', 'Elements'],
                  'real_name': 'A', 'doc': 'Test', 'unit': None,
                  'eqn': 'eq1', 'lims': '', 'dependencies': {'b', 'time'},
                  'kind': 'component', 'arguments': ''},
                 {'py_name': 'a', 'py_expr': 'njk',
                  'subs': ['Name1', 'element2'],
                  'merge_subs': ['Name1', 'Elements'],
                  'real_name': 'A', 'doc': None, 'unit': None,
                  'eqn': 'eq2', 'lims': '', 'dependencies': {'c', 'time'},
                  'kind': 'component', 'arguments': ''},
                 {'py_name': 'a', 'py_expr': 'as',
                  'subs': ['Name1', 'element3'],
                  'merge_subs': ['Name1', 'Elements'],
                  'real_name': 'A', 'doc': '', 'unit': None,
                  'eqn': 'eq3', 'lims': '', 'dependencies': {'b'},
                  'kind': 'component', 'arguments': ''}]),
            [{'py_name': 'a',
              'py_expr': ['ms', 'njk', 'as'],
              'subs': [['Name1', 'element1'],
                       ['Name1', 'element2'],
                       ['Name1', 'element3']],
              'merge_subs': ['Name1', 'Elements'],
              'kind': 'component',
              'doc': 'Test',
              'real_name': 'A',
              'unit': None,
              'eqn': ['eq1', 'eq2', 'eq3'],
              'lims': '',
              'dependencies': {'b', 'c', 'time'},
              'arguments': ''
              }])

    def test_multiple_sets(self):
        from pysd.py_backend.builder import merge_partial_elements
        actual = merge_partial_elements(
            [{'py_name': 'a', 'py_expr': 'ms', 'subs': ['Name1', 'element1'],
              'merge_subs': ['Name1', 'Elements'], 'dependencies': {'b'},
              'real_name': 'A', 'doc': 'Test', 'unit': None,
              'eqn': 'eq1', 'lims': '', 'kind': 'component', 'arguments': ''},
             {'py_name': 'a', 'py_expr': 'njk', 'subs': ['Name1', 'element2'],
              'merge_subs': ['Name1', 'Elements'], 'dependencies': {'b'},
              'real_name': 'A', 'doc': None, 'unit': None,
              'eqn': 'eq2', 'lims': '', 'kind': 'component', 'arguments': ''},
             {'py_name': 'a', 'py_expr': 'as', 'subs': ['Name1', 'element3'],
              'merge_subs': ['Name1', 'Elements'], 'dependencies': {'b'},
              'real_name': 'A', 'doc': '', 'unit': None,
              'eqn': 'eq3', 'lims': '', 'kind': 'component', 'arguments': ''},
             {'py_name': 'b', 'py_expr': 'bgf', 'subs': ['Name1', 'element1'],
              'merge_subs': ['Name1', 'Elements'], 'dependencies': {
                  'initial': {'c'}, 'step': set()},
              'real_name': 'B', 'doc': 'Test', 'unit': None,
              'eqn': 'eq4', 'lims': '', 'kind': 'component', 'arguments': ''},
             {'py_name': 'b', 'py_expr': 'r4', 'subs': ['Name1', 'element2'],
              'merge_subs': ['Name1', 'Elements'], 'dependencies': {
                  'initial': {'d'}, 'step': {'time', 'd'}},
              'real_name': 'B', 'doc': None, 'unit': None,
              'eqn': 'eq5', 'lims': '', 'kind': 'component', 'arguments': ''},
             {'py_name': 'b', 'py_expr': 'ymt', 'subs': ['Name1', 'element3'],
              'merge_subs': ['Name1', 'Elements'], 'dependencies': {
                  'initial': set(), 'step': {'time', 'a'}},
              'real_name': 'B', 'doc': '', 'unit': None,
              'eqn': 'eq6', 'lims': '', 'kind': 'component', 'arguments': ''}])

        expected = [{'py_name': 'a',
                     'py_expr': ['ms', 'njk', 'as'],
                     'subs': [['Name1', 'element1'],
                              ['Name1', 'element2'],
                              ['Name1', 'element3']],
                     'merge_subs': ['Name1', 'Elements'],
                     'kind': 'component',
                     'doc': 'Test',
                     'real_name': 'A',
                     'unit': None,
                     'eqn': ['eq1', 'eq2', 'eq3'],
                     'lims': '',
                     'dependencies': {'b'},
                     'arguments': ''
                     },
                    {'py_name': 'b',
                     'py_expr': ['bgf', 'r4', 'ymt'],
                     'subs': [['Name1', 'element1'],
                              ['Name1', 'element2'],
                              ['Name1', 'element3']],
                     'merge_subs': ['Name1', 'Elements'],
                     'kind': 'component',
                     'doc': 'Test',
                     'real_name': 'B',
                     'unit': None,
                     'eqn': ['eq4', 'eq5', 'eq6'],
                     'lims': '',
                     'dependencies': {
                         'initial': {'c', 'd'},
                         'step': {'time', 'a', 'd'}
                     },
                     'arguments': ''
                     }]
        self.assertIn(actual[0], expected)
        self.assertIn(actual[1], expected)

    def test_non_set(self):
        from pysd.py_backend.builder import merge_partial_elements
        actual = merge_partial_elements(
            [{'py_name': 'a', 'py_expr': 'ms', 'subs': ['Name1', 'element1'],
              'merge_subs': ['Name1', 'Elements'], 'dependencies': {'c'},
              'real_name': 'A', 'doc': 'Test', 'unit': None,
              'eqn': 'eq1', 'lims': '', 'kind': 'component', 'arguments': ''},
             {'py_name': 'a', 'py_expr': 'njk', 'subs': ['Name1', 'element2'],
              'merge_subs': ['Name1', 'Elements'], 'dependencies': {'b'},
              'real_name': 'A', 'doc': None, 'unit': None,
              'eqn': 'eq2', 'lims': '', 'kind': 'component', 'arguments': ''},
             {'py_name': 'c', 'py_expr': 'as', 'subs': ['Name1', 'element3'],
              'merge_subs': ['Name1', 'elements3'], 'dependencies': set(),
              'real_name': 'C', 'doc': 'hi', 'unit': None,
              'eqn': 'eq3', 'lims': '', 'kind': 'component', 'arguments': ''},
             ])

        expected = [{'py_name': 'a',
                     'py_expr': ['ms', 'njk'],
                     'subs': [['Name1', 'element1'], ['Name1', 'element2']],
                     'merge_subs': ['Name1', 'Elements'],
                     'kind': 'component',
                     'doc': 'Test',
                     'real_name': 'A',
                     'unit': None,
                     'eqn': ['eq1', 'eq2'],
                     'lims': '',
                     'dependencies': {'b', 'c'},
                     'arguments': ''
                     },
                    {'py_name': 'c',
                     'py_expr': ['as'],
                     'subs': [['Name1', 'element3']],
                     'merge_subs': ['Name1', 'elements3'],
                     'kind': 'component',
                     'doc': 'hi',
                     'real_name': 'C',
                     'unit': None,
                     'eqn': ['eq3'],
                     'lims': '',
                     'dependencies': set(),
                     'arguments': ''
                     }]

        self.assertIn(actual[0], expected)
        self.assertIn(actual[1], expected)

    def test_bad_value_kind(self):
        from pysd.py_backend.builder import build_element

        with self.assertRaises(AttributeError):
            build_element({'kind': 'nonvalid'}, {})

