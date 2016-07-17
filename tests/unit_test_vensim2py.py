import unittest
import xarray as xr


class TestGetFileSections(unittest.TestCase):
    def test_normal_load(self):
        """normal model file with no macros"""
        from pysd.vensim2py import get_file_sections
        actual = get_file_sections(r'a~b~c| d~e~f| g~h~i|')
        expected = [{'returns': [], 'params': [], 'name': 'main', 'string': 'a~b~c| d~e~f| g~h~i|'}]
        self.assertEqual(actual, expected)

    def test_macro_only(self):
        """ Macro Only """
        from pysd.vensim2py import get_file_sections
        actual = get_file_sections(':MACRO: MAC(z) a~b~c| :END OF MACRO:')
        expected = [{'returns': [], 'params': ['z'], 'name': 'MAC', 'string': 'a~b~c|'}]
        self.assertEqual(actual, expected)

    def test_macro_and_model(self):
        """ basic macro and model """
        from pysd.vensim2py import get_file_sections
        actual = get_file_sections(':MACRO: MAC(z) a~b~c| :END OF MACRO: d~e~f| g~h~i|')
        expected = [{'returns': [], 'params': ['z'], 'name': 'MAC', 'string': 'a~b~c|'},
                    {'returns': [], 'params': [], 'name': 'main', 'string': 'd~e~f| g~h~i|'}]
        self.assertEqual(actual, expected)

    def test_macro_multiple_inputs(self):
        """ macro with multiple input parameters """
        from pysd.vensim2py import get_file_sections
        actual = get_file_sections(':MACRO: MAC(z, y) a~b~c| :END OF MACRO: d~e~f| g~h~i|')
        expected = [{'returns': [], 'params': ['z', 'y'], 'name': 'MAC', 'string': 'a~b~c|'},
                    {'returns': [], 'params': [], 'name': 'main', 'string': 'd~e~f| g~h~i|'}]
        self.assertEqual(actual, expected)

    def test_macro_with_returns(self):
        """ macro with return values """
        from pysd.vensim2py import get_file_sections
        actual = get_file_sections(':MACRO: MAC(z, y :x, w) a~b~c| :END OF MACRO: d~e~f| g~h~i|')
        expected = [{'returns': ['x', 'w'],
                     'params': ['z', 'y'],
                     'name': 'MAC',
                     'string': 'a~b~c|'},
                    {'returns': [],
                     'params': [],
                     'name': 'main',
                     'string': 'd~e~f| g~h~i|'}]
        self.assertEqual(actual, expected)

    def test_handle_encoding(self):
        """ Handle encoding """
        from pysd.vensim2py import get_file_sections
        actual = get_file_sections(r'{UTF-8} a~b~c| d~e~f| g~h~i|')
        expected = [{'returns': [], 'params': [], 'name': 'main', 'string': 'a~b~c| d~e~f| g~h~i|'}]
        self.assertEqual(actual, expected)

    def test_handle_encoding_like_strings(self):
        """ Handle encoding-like strings in other places in the file """
        from pysd.vensim2py import get_file_sections
        actual = get_file_sections(r'a~b~c| d~e~f{special}| g~h~i|')
        expected = [{'returns': [],
                     'params': [],
                     'name': 'main',
                     'string': 'a~b~c| d~e~f{special}| g~h~i|'}]
        self.assertEqual(actual, expected)


class TestEquationStringParsing(unittest.TestCase):
    """ Tests the 'get_equation_components function """
    def test_basics(self):
        from pysd.vensim2py import get_equation_components
        self.assertEqual(
            get_equation_components(r'constant = 25'),
            {'expr': '25', 'kind': 'component', 'subs': [], 'real_name': 'constant'}
        )

    def test_equals_handling(self):
        """ Parse cases with equal signs within the expression """
        from pysd.vensim2py import get_equation_components
        self.assertEqual(
            get_equation_components(r'Boolean = IF THEN ELSE(1 = 1, 1, 0)'),
            {'expr': 'IF THEN ELSE(1 = 1, 1, 0)', 'kind': 'component', 'subs': [],
             'real_name': 'Boolean'}
        )

    def test_whitespace_handling(self):
        """ Whitespaces should be shortened to a single space """
        from pysd.vensim2py import get_equation_components
        self.assertEqual(
            get_equation_components(r'''constant\t =
                                                        \t25\t '''),
            {'expr': '25', 'kind': 'component', 'subs': [], 'real_name': 'constant'}
        )

        # test eliminating vensim's line continuation character
        self.assertEqual(
            get_equation_components(r"""constant [Sub1, \\
                                     Sub2] = 10, 12; 14, 16;"""),
            {'expr': '10, 12; 14, 16;', 'kind': 'component', 'subs': ['Sub1', 'Sub2'],
             'real_name': 'constant'}
        )

    def test_subscript_definition_parsing(self):
        from pysd.vensim2py import get_equation_components
        self.assertEqual(
            get_equation_components(r'''Sub1: Entry 1, Entry 2, Entry 3 '''),
            {'expr': None, 'kind': 'subdef', 'subs': ['Entry 1', 'Entry 2', 'Entry 3'],
             'real_name': 'Sub1'}
        )

    def test_subscript_references(self):
        from pysd.vensim2py import get_equation_components
        self.assertEqual(
            get_equation_components(r'constant [Sub1, Sub2] = 10, 12; 14, 16;'),
            {'expr': '10, 12; 14, 16;', 'kind': 'component', 'subs': ['Sub1', 'Sub2'],
             'real_name': 'constant'}
        )

        self.assertEqual(
            get_equation_components(r'function [Sub1] = other function[Sub1]'),
            {'expr': 'other function[Sub1]', 'kind': 'component', 'subs': ['Sub1'],
             'real_name': 'function'}
        )

        self.assertEqual(
            get_equation_components(r'constant ["S1,b", "S1,c"] = 1, 2; 3, 4;'),
            {'expr': '1, 2; 3, 4;', 'kind': 'component', 'subs': ['"S1,b"', '"S1,c"'],
             'real_name': 'constant'}
        )

        self.assertEqual(
            get_equation_components(r'constant ["S1=b", "S1=c"] = 1, 2; 3, 4;'),
            {'expr': '1, 2; 3, 4;', 'kind': 'component', 'subs': ['"S1=b"', '"S1=c"'],
             'real_name': 'constant'}
        )

    def test_lookup_definitions(self):
        from pysd.vensim2py import get_equation_components
        self.assertEqual(
            get_equation_components(r'table([(0,-1)-(45,1)],(0,0),(5,0))'),
            {'expr': '([(0,-1)-(45,1)],(0,0),(5,0))', 'kind': 'lookup', 'subs': [],
             'real_name': 'table'}
        )

        self.assertEqual(
            get_equation_components(r'table2 ([(0,-1)-(45,1)],(0,0),(5,0))'),
            {'expr': '([(0,-1)-(45,1)],(0,0),(5,0))', 'kind': 'lookup', 'subs': [],
             'real_name': 'table2'}
        )

    def test_pathological_names(self):
        from pysd.vensim2py import get_equation_components
        self.assertEqual(
            get_equation_components(r'"silly-string" = 25'),
            {'expr': '25', 'kind': 'component', 'subs': [], 'real_name': '"silly-string"'}
        )

        self.assertEqual(
            get_equation_components(r'"pathological\\-string" = 25'),
            {'expr': '25', 'kind': 'component', 'subs': [],
             'real_name': r'"pathological\\-string"'}
        )


class TestParse_general_expression(unittest.TestCase):

    def test_arithmetic(self):
        from pysd.vensim2py import parse_general_expression
        res = parse_general_expression({'expr': '-10^3+4'})
        self.assertEqual(res[0]['py_expr'], '-10**3+4')

    def test_kind_assignment(self):
        from pysd.vensim2py import parse_general_expression
        res = parse_general_expression({'expr': '-10^3+4'})
        self.assertEqual(res[0]['kind'], 'constant')

        res = parse_general_expression({'expr': 'Abs(-3)'})
        self.assertEqual(res[0]['kind'], 'component')

        res = parse_general_expression({'expr': 'INTEG (FlowA, -10)',
                                        'py_name': 'test_stock',
                                        'subs': None},
                                       {'FlowA': 'flowa'})
        self.assertEqual(res[0]['kind'], 'component')
        self.assertEqual(res[1][0]['kind'], 'setup')
        self.assertEqual(res[1][1]['kind'], 'component')

    def test_builtins(self):
        from pysd.vensim2py import parse_general_expression
        res = parse_general_expression({'expr': 'Time'})
        self.assertEqual(res[0]['py_expr'], 'time()')
        self.assertDictContainsSubset({'kind': 'component', 'py_expr': '_t'},
                                      res[1][0])

    def test_caps_handling(self):
        from pysd.vensim2py import parse_general_expression
        res = parse_general_expression({'expr': 'Abs(-3)'})
        self.assertEqual(res[0]['py_expr'], 'abs(-3)')

        res = parse_general_expression({'expr': 'ABS(-3)'})
        self.assertEqual(res[0]['py_expr'], 'abs(-3)')

        res = parse_general_expression({'expr': 'aBS(-3)'})
        self.assertEqual(res[0]['py_expr'], 'abs(-3)')

    def test_delay_construction_function_no_subscripts(self):
        #todo: eventually make this case more rigorous, for now depending on integration test
        from pysd.vensim2py import parse_general_expression
        res = parse_general_expression({'expr': 'Const * DELAY1(Variable, DelayTime)',
                                        'subs': []},
                                       {'Const': 'const', 'Variable': 'variable',
                                        'DelayTime': 'delaytime'},
                                       )
        self.assertEqual(res[0]['py_expr'], 'const()*_variable_delay_1()')

    def test_function_calls(self):
        from pysd.vensim2py import parse_general_expression
        res = parse_general_expression({'expr': 'ABS(StockA)'}, {'StockA': 'stocka'})
        self.assertEqual(res[0]['py_expr'], 'abs(stocka())')

        res = parse_general_expression({'expr': 'If Then Else(A>B, 1, 0)'}, {'A': 'a', 'B':'b'})
        self.assertEqual(res[0]['py_expr'], 'functions.if_then_else(a()>b(),1,0)')

        # test that function calls are handled properly in arguments
        res = parse_general_expression({'expr': 'If Then Else(A>B,1,A)'}, {'A': 'a', 'B': 'b'})
        self.assertEqual(res[0]['py_expr'], 'functions.if_then_else(a()>b(),1,a())')

    def test_id_parsing(self):
        from pysd.vensim2py import parse_general_expression
        res = parse_general_expression({'expr': 'StockA'}, {'StockA': 'stocka'})
        self.assertEqual(res[0]['py_expr'], 'stocka()')

    def test_number_parsing(self):
        from pysd.vensim2py import parse_general_expression
        res = parse_general_expression({'expr': '20'})
        self.assertEqual(res[0]['py_expr'], '20')

        res = parse_general_expression({'expr': '3.14159'})
        self.assertEqual(res[0]['py_expr'], '3.14159')

        res = parse_general_expression({'expr': '+3.14159'})
        self.assertEqual(res[0]['py_expr'], '3.14159')

        res = parse_general_expression({'expr': '1.3e+10'})
        self.assertEqual(res[0]['py_expr'], '1.3e+10')

        res = parse_general_expression({'expr': '-1.3e-10'})
        self.assertEqual(res[0]['py_expr'], '-1.3e-10')

    def test_stock_construction_function_no_subscripts(self):
        from pysd.vensim2py import parse_general_expression
        actual = parse_general_expression({'expr': 'INTEG (FlowA, -10)',
                                      'py_name': 'test_stock',
                                      'subs': None},
                                     {'FlowA': 'flowa'})

        expected = (
            {'kind': 'component', 'py_expr': "_state['test_stock']", 'arguments': ''},
            [{'kind': 'setup',
              'subs': None,
              'doc': 'Provides initial conditions for test_stock function',
              'py_name': '_init_test_stock',
              'real_name': 'Implicit',
              'unit': 'See docs for test_stock',
              'py_expr': '-10',
              'arguments': ''},
             {'py_name': '_dtest_stock_dt',
              'subs': None,
              'doc': 'Provides derivative for test_stock function',
              'kind': 'component',
              'unit': 'See docs for test_stock',
              'py_expr': 'flowa()',
              'real_name': 'Implicit',
              'arguments': ''}])
        self.assertDictEqual(actual[0], expected[0])
        self.assertDictEqual(actual[1][0], expected[1][0])
        self.assertDictEqual(actual[1][1], expected[1][1])


    def test_smooth_construction_function_no_subscripts(self):
        # todo: improve this test
        from pysd.vensim2py import parse_general_expression
        res = parse_general_expression({'expr': 'Const * SMOOTH(Variable, DelayTime)',
                                        'subs': []},
                                       {'Const': 'const', 'Variable': 'variable',
                                        'DelayTime': 'delaytime'},
                                       )
        self.assertEqual(res[0]['py_expr'], 'const()*_variable_smooth_1()')

    def test_subscript_float_initialization(self):
        from pysd.vensim2py import parse_general_expression
        element = parse_general_expression({'expr': '3.32',
                                            'subs': ['Dim1']},
                                           {},
                                           {'Dim1': ['A', 'B', 'C'],
                                            'Dim2': ['D', 'E']})
        string = element[0]['py_expr']
        a = eval(string)
        self.assertDictEqual({key: list(val.values) for key, val in a.coords.iteritems()},
                             {'Dim1': ['A', 'B', 'C']})
        self.assertEqual(a.loc[{'Dim1': 'B'}], 3.32)

    def test_subscript_1d_constant(self):
        from pysd.vensim2py import parse_general_expression
        element = parse_general_expression({'expr': '1, 2, 3',
                                            'subs': ['Dim1']},
                                           {},
                                           {'Dim1': ['A', 'B', 'C'],
                                            'Dim2': ['D', 'E']})
        string = element[0]['py_expr']
        a = eval(string)
        self.assertDictEqual({key: list(val.values) for key, val in a.coords.iteritems()},
                             {'Dim1': ['A', 'B', 'C']})
        self.assertEqual(a.loc[{'Dim1': 'A'}], 1)

    def test_subscript_2d_constant(self):
        from pysd.vensim2py import parse_general_expression
        element = parse_general_expression({'expr': '1, 2; 3, 4; 5, 6;',
                                            'subs': ['Dim1', 'Dim2']},
                                           {},
                                           {'Dim1': ['A', 'B', 'C'],
                                            'Dim2': ['D', 'E']})
        string = element[0]['py_expr']
        a = eval(string)
        self.assertDictEqual({key: list(val.values) for key, val in a.coords.iteritems()},
                             {'Dim1': ['A', 'B', 'C'], 'Dim2': ['D', 'E']})
        self.assertEqual(a.loc[{'Dim1': 'A', 'Dim2': 'D'}], 1)
        self.assertEqual(a.loc[{'Dim1': 'B', 'Dim2': 'E'}], 4)

    def test_subscript_3d_depth(self):
        from pysd.vensim2py import parse_general_expression
        element = parse_general_expression({'expr': '1, 2; 3, 4; 5, 6;',
                                            'subs': ['Dim1', 'Dim2']},
                                           {},
                                           {'Dim1': ['A', 'B', 'C'],
                                            'Dim2': ['D', 'E']})
        string = element[0]['py_expr']
        a = eval(string)
        self.assertDictEqual({key: list(val.values) for key, val in a.coords.iteritems()},
                             {'Dim1': ['A', 'B', 'C'], 'Dim2': ['D', 'E']})
        self.assertEqual(a.loc[{'Dim1': 'A', 'Dim2': 'D'}], 1)
        self.assertEqual(a.loc[{'Dim1': 'B', 'Dim2': 'E'}], 4)

    def test_subscript_stock(self):
        from pysd.vensim2py import parse_general_expression
        res = parse_general_expression({'expr': 'INTEG (Flow[sub_D1,sub_D2], Init[sub_D1, sub_D2])',
                                      'py_name': 'stock_test', 'subs': ['sub_D1']},
                                     {'Init': 'init', 'Flow': 'flow'},
                                     {'sub_D1': ['Entry 1', 'Entry 2', 'Entry 3'],
                                      'sub_D2': ['Column 1', 'Column 2']})

        self.assertDictContainsSubset({'kind': 'component', 'py_expr': "_state['stock_test']"},
                                      res[0])

        self.assertDictContainsSubset({'kind': 'setup',
                                       'subs': ['sub_D1'],
                                       'py_name': '_init_stock_test',
                                       'py_expr': 'init()'},
                                      res[1][0])

        self.assertDictContainsSubset({'kind': 'component',
                                       'subs': ['sub_D1'],
                                       'py_name': '_dstock_test_dt',
                                       'py_expr': 'flow()'},
                                      res[1][1])

    def test_subscript_reference(self):
        from pysd.vensim2py import parse_general_expression
        res = parse_general_expression({'expr': 'Var A[Dim1, Dim2]'},
                                     {'Var A': 'var_a'},
                                     {'Dim1': ['A', 'B'],
                                      'Dim2': ['C', 'D', 'E']})

        self.assertEqual(res[0]['py_expr'], 'var_a()')

        res = parse_general_expression({'expr': 'Var B[Dim1, C]'},
                                     {'Var B': 'var_b'},
                                     {'Dim1': ['A', 'B'],
                                      'Dim2': ['C', 'D', 'E']})
        self.assertEqual(res[0]['py_expr'], "var_b().loc[{'Dim2': ['C']}]")

        res = parse_general_expression({'expr': 'Var C[Dim1, C, H]'},
                                     {'Var C': 'var_c'},
                                     {'Dim1': ['A', 'B'],
                                      'Dim2': ['C', 'D', 'E'],
                                      'Dim3': ['F', 'G', 'H', 'I']})
        self.assertEqual(res[0]['py_expr'], "var_c().loc[{'Dim2': ['C'], 'Dim3': ['H']}]")

    def test_subscript_ranges(self):
        from pysd.vensim2py import parse_general_expression
        res = parse_general_expression({'expr': 'Var D[Range1]'},
                                     {'Var D': 'var_c'},
                                     {'Dim1': ['A', 'B', 'C', 'D', 'E', 'F'],
                                      'Range1': ['C', 'D', 'E']})
        self.assertEqual(res[0]['py_expr'], "var_c().loc[{'Dim1': ['C', 'D', 'E']}]")


