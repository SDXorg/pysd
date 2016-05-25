from unittest import TestCase


class TestMergePartialElements(TestCase):
    def test_single_set(self):
        from translators.vensim2py2 import merge_partial_elements

        self.assertEqual(
                merge_partial_elements([{'py_name': 'a', 'py_expr': 'ms', 'subs': ['Name1', 'element1'],
                                         'real_name': 'A', 'doc': 'Test', 'unit':None,
                                         'kind':'component'},
                                        {'py_name': 'a', 'py_expr': 'njk', 'subs': ['Name1', 'element2'],
                                         'real_name': 'A', 'doc':None,  'unit':None,
                                         'kind':'component'},
                                        {'py_name': 'a', 'py_expr': 'as', 'subs': ['Name1', 'element3'],
                                         'real_name': 'A', 'doc':'',  'unit':None,
                                         'kind':'component'}]),
                [{'py_name': 'a',
                  'py_expr': ['ms', 'njk', 'as'],
                  'subs':[['Name1', 'element1'], ['Name1', 'element2'], ['Name1', 'element3']],
                  'kind':'component',
                  'doc': 'Test',
                  'real_name': 'A',
                  'unit':None
                  }])

    def test_multiple_sets(self):
        from translators.vensim2py2 import merge_partial_elements

        self.assertEqual(
                merge_partial_elements([{'py_name': 'a', 'py_expr': 'ms', 'subs': ['Name1', 'element1'],
                                         'real_name': 'A', 'doc': 'Test', 'unit':None,
                                         'kind':'component'},
                                        {'py_name': 'a', 'py_expr': 'njk', 'subs': ['Name1', 'element2'],
                                         'real_name': 'A', 'doc':None,  'unit':None,
                                         'kind':'component'},
                                        {'py_name': 'a', 'py_expr': 'as', 'subs': ['Name1', 'element3'],
                                         'real_name': 'A', 'doc':'',  'unit':None,
                                         'kind':'component'},
                                        {'py_name': 'b', 'py_expr': 'bgf', 'subs': ['Name1', 'element1'],
                                         'real_name': 'B', 'doc': 'Test', 'unit':None,
                                         'kind':'component'},
                                        {'py_name': 'b', 'py_expr': 'r4', 'subs': ['Name1', 'element2'],
                                         'real_name': 'B', 'doc':None,  'unit':None,
                                         'kind':'component'},
                                        {'py_name': 'b', 'py_expr': 'ymt', 'subs': ['Name1', 'element3'],
                                         'real_name': 'B', 'doc':'',  'unit':None,
                                         'kind':'component'}]),
                [{'py_name': 'a',
                  'py_expr': ['ms', 'njk', 'as'],
                  'subs':[['Name1', 'element1'], ['Name1', 'element2'], ['Name1', 'element3']],
                  'kind':'component',
                  'doc': 'Test',
                  'real_name': 'A',
                  'unit':None
                  },
                 {'py_name': 'b',
                  'py_expr': ['bgf', 'r4', 'ymt'],
                  'subs':[['Name1', 'element1'], ['Name1', 'element2'], ['Name1', 'element3']],
                  'kind':'component',
                  'doc': 'Test',
                  'real_name': 'B',
                  'unit':None
                  }])

    def test_non_set(self):
        from translators.vensim2py2 import merge_partial_elements

        self.assertEqual(
                merge_partial_elements([{'py_name': 'a', 'py_expr': 'ms', 'subs': ['Name1', 'element1'],
                                         'real_name': 'A', 'doc': 'Test', 'unit':None,
                                         'kind':'component'},
                                        {'py_name': 'a', 'py_expr': 'njk', 'subs': ['Name1', 'element2'],
                                         'real_name': 'A', 'doc':None,  'unit':None,
                                         'kind':'component'},
                                        {'py_name': 'c', 'py_expr': 'as', 'subs': ['Name1', 'element3'],
                                         'real_name': 'C', 'doc':'hi',  'unit':None,
                                         'kind':'component'},
                                        ]),
                [{'py_name': 'a',
                  'py_expr': ['ms', 'njk'],
                  'subs':[['Name1', 'element1'], ['Name1', 'element2']],
                  'kind':'component',
                  'doc': 'Test',
                  'real_name': 'A',
                  'unit':None
                  },
                 {'py_name': 'c',
                  'py_expr': ['as'],
                  'subs':[['Name1', 'element3']],
                  'kind':'component',
                  'doc': 'hi',
                  'real_name': 'C',
                  'unit':None
                  }])
