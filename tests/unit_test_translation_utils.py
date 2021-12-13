from unittest import TestCase


class TestTranslationUtils(TestCase):

    def test_add_entries_underscore(self):
        """"
        Test for add_entries_undescore
        """
        from pysd.translation.utils import add_entries_underscore

        dict1 = {'CD': 10, 'L F': 5}
        dict2 = {'a b': 1, 'C': 2, 'L M H': 4}

        dict1b = dict1.copy()

        add_entries_underscore(dict1b)

        self.assertTrue('L_F' in dict1b)
        self.assertEqual(dict1b['L F'], dict1b['L_F'])

        add_entries_underscore(dict1, dict2)

        self.assertTrue('L_F' in dict1)
        self.assertEqual(dict1['L F'], dict1['L_F'])
        self.assertTrue('a_b' in dict2)
        self.assertEqual(dict2['a b'], dict2['a_b'])
        self.assertTrue('L_M_H' in dict2)
        self.assertEqual(dict2['L M H'], dict2['L_M_H'])

    def test_make_add_identifier(self):
        """
        Test make_add_identifier for the .add methods py_name
        """
        from pysd.translation.utils import make_add_identifier

        build_names = set()

        name = "values"
        build_names.add(name)

        self.assertEqual(make_add_identifier(name, build_names), "valuesADD_1")
        self.assertEqual(make_add_identifier(name, build_names), "valuesADD_2")
        self.assertEqual(make_add_identifier(name, build_names), "valuesADD_3")

        name2 = "bb_a"
        build_names.add(name2)
        self.assertEqual(make_add_identifier(name2, build_names), "bb_aADD_1")
        self.assertEqual(make_add_identifier(name, build_names), "valuesADD_4")
        self.assertEqual(make_add_identifier(name2, build_names), "bb_aADD_2")

    def test_make_python_identifier(self):
        from pysd.translation.utils import make_python_identifier

        self.assertEqual(
            make_python_identifier('Capital'), 'capital')

        self.assertEqual(
            make_python_identifier('multiple words'), 'multiple_words')

        self.assertEqual(
            make_python_identifier('multiple     spaces'), 'multiple_spaces')

        self.assertEqual(
            make_python_identifier('for'), 'for_1')

        self.assertEqual(
            make_python_identifier('  whitespace  '), 'whitespace')

        self.assertEqual(
            make_python_identifier('H@t tr!ck'), 'ht_trck')

        self.assertEqual(
            make_python_identifier('123abc'), 'nvs_123abc')

        self.assertEqual(
            make_python_identifier('Var$', {'Var$': 'var'}),
            'var')

        self.assertEqual(
            make_python_identifier('Var@', {'Var$': 'var'}), 'var_1')

        self.assertEqual(
            make_python_identifier('Var$', {'Var@': 'var', 'Var%': 'var_1'}),
            'var_2')

        my_vars = ["GDP 2010$", "GDP 2010€", "GDP 2010£"]
        namespace = {}
        expected = ["gdp_2010", "gdp_2010_1", "gdp_2010_2"]
        for var, expect in zip(my_vars, expected):
            self.assertEqual(
                make_python_identifier(var, namespace),
                expect)

        self.assertEqual(
            make_python_identifier('1995 value'),
            'nvs_1995_value')

        self.assertEqual(
            make_python_identifier('$ value'),
            'nvs_value')

    def test_make_coord_dict(self):
        from pysd.translation.utils import make_coord_dict
        self.assertEqual(
            make_coord_dict(['Dim1', 'D'],
                            {'Dim1': ['A', 'B', 'C'],
                             'Dim2': ['D', 'E', 'F']},
                            terse=True), {'Dim2': ['D']})
        self.assertEqual(
            make_coord_dict(['Dim1', 'D'],
                            {'Dim1': ['A', 'B', 'C'],
                             'Dim2': ['D', 'E', 'F']},
                            terse=False), {'Dim1': ['A', 'B', 'C'],
                                           'Dim2': ['D']})

    def test_find_subscript_name(self):
        from pysd.translation.utils import find_subscript_name

        self.assertEqual(
            find_subscript_name({'Dim1': ['A', 'B'],
                                 'Dim2': ['C', 'D', 'E'],
                                 'Dim3': ['F', 'G', 'H', 'I']},
                                'D'), 'Dim2')

        self.assertEqual(
            find_subscript_name({'Dim1': ['A', 'B'],
                                 'Dim2': ['C', 'D', 'E'],
                                 'Dim3': ['F', 'G', 'H', 'I']},
                                'Dim3'), 'Dim3')

    def test_make_merge_list(self):
        from warnings import catch_warnings
        from pysd.translation.utils import make_merge_list

        subscript_dict = {
            "layers": ["l1", "l2", "l3"],
            "layers1": ["l1", "l2", "l3"],
            "up": ["l2", "l3"],
            "down": ["l1", "l2"],
            "dim": ["A", "B", "C"],
            "dim1": ["A", "B", "C"]
        }

        self.assertEqual(
            make_merge_list([["l1"], ["up"]],
                            subscript_dict),
            ["layers"])

        self.assertEqual(
            make_merge_list([["l3", "dim1"], ["down", "dim1"]],
                            subscript_dict),
            ["layers", "dim1"])

        self.assertEqual(
            make_merge_list([["l2", "dim1", "dim"], ["l1", "dim1", "dim"]],
                            subscript_dict),
            ["down", "dim1", "dim"])

        self.assertEqual(
            make_merge_list([["layers1", "l2"], ["layers1", "l3"]],
                            subscript_dict),
            ["layers1", "up"])

        # incomplete dimension
        with catch_warnings(record=True) as ws:
            self.assertEqual(
                make_merge_list([["A"], ["B"]],
                                subscript_dict),
                ["dim"])
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertTrue(len(wu), 1)
            self.assertIn("Dimension given by subscripts:"
                          + "\n\t{}\nis incomplete ".format({"A", "B"})
                          + "using {} instead.".format("dim")
                          + "\nSubscript_dict:"
                          + "\n\t{}".format(subscript_dict),
                          str(wu[0].message))

        # invalid dimension
        try:
            make_merge_list([["l1"], ["B"]],
                            subscript_dict)
            self.assertFail()
        except ValueError as err:
            self.assertIn("Impossible to find the dimension that contains:"
                          + "\n\t{}\nFor subscript_dict:".format({"l1", "B"})
                          + "\n\t{}".format(subscript_dict),
                          err.args[0])

        # repeated subscript
        with catch_warnings(record=True) as ws:
            make_merge_list([["dim1", "A", "dim"],
                            ["dim1", "B", "dim"],
                            ["dim1", "C", "dim"]],
                            subscript_dict)
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertTrue(len(wu), 1)
            self.assertIn(
                "Adding new subscript range to subscript_dict:\ndim2: A, B, C",
                str(wu[0].message))

        subscript_dict2 = {
            "dim1": ["A", "B", "C", "D"],
            "dim1n": ["A", "B"],
            "dim1y": ["C", "D"],
            "dim2": ["E", "F", "G", "H"],
            "dim2n": ["E", "F"],
            "dim2y": ["G", "H"]
        }

        # merging two subranges
        self.assertEqual(
            make_merge_list([["dim1y"],
                             ["dim1n"]],
                            subscript_dict2),
            ["dim1"])

        # final subscript in list
        self.assertEqual(
            make_merge_list([["dim1", "dim2n"],
                             ["dim1n", "dim2y"],
                             ["dim1y", "dim2y"]],
                            subscript_dict2),
            ["dim1", "dim2"])

    def test_update_dependency(self):
        from pysd.translation.utils import update_dependency

        deps_dict = {}

        update_dependency("var1", deps_dict)
        self.assertEqual(deps_dict, {"var1": 1})

        update_dependency("var1", deps_dict)
        self.assertEqual(deps_dict, {"var1": 2})

        update_dependency("var2", deps_dict)
        self.assertEqual(deps_dict, {"var1": 2, "var2": 1})

        for i in range(10):
            update_dependency("var1", deps_dict)
        self.assertEqual(deps_dict, {"var1": 12, "var2": 1})
