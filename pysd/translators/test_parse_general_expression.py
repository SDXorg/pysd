import unittest


class TestParse_general_expression(unittest.TestCase):

    def test_id_parsing(self):
        from translators.vensim2py2 import parse_general_expression
        self.assertEqual(parse_general_expression({'expr': 'StockA'},
                                                  namespace={'StockA': 'stocka'}),
                         ({'kind': 'component', 'py_expr': 'stocka()'}, []))

    def test_number_parsing(self):
        from translators.vensim2py2 import parse_general_expression
        self.assertEqual(
            parse_general_expression({'expr': '20'}),
            ({'kind': 'constant', 'py_expr': '20'}, [])
        )

        self.assertEqual(
            parse_general_expression({'expr': '3.14159'}),
            ({'kind': 'constant', 'py_expr': '3.14159'}, [])
        )

        self.assertEqual(
            parse_general_expression({'expr': '+3.14159'}),
            ({'kind': 'constant', 'py_expr': '3.14159'}, [])
        )

        self.assertEqual(
            parse_general_expression({'expr': '1.3e-10'}),
            ({'kind': 'constant', 'py_expr': '1.3e-10'}, [])
        )

        self.assertEqual(
            parse_general_expression({'expr': '-1.3e-10'}),
            ({'kind': 'constant', 'py_expr': '-1.3e-10'}, [])
        )

    def test_arithmetic(self):
        from translators.vensim2py2 import parse_general_expression
        self.assertEqual(
            parse_general_expression({'expr': '-10^3+4'}),
            ({'kind': 'constant', 'py_expr': '-10**3+4'}, [])
        )

    def test_builtins(self):
        from translators.vensim2py2 import parse_general_expression
        self.assertEqual(
            parse_general_expression({'expr': 'Time'}),
            ({'kind': 'constant', 'py_expr': 'time()'}, [])
        )

    def test_function_calls(self):
        from translators.vensim2py2 import parse_general_expression
        self.assertEqual(
            parse_general_expression({'expr': 'ABS(StockA)'}, {'StockA': 'stocka'}),
            ({'kind': 'component', 'py_expr': 'abs(stocka())'}, [])
        )

        self.assertEqual(
            parse_general_expression({'expr': 'If Then Else(A>B, 1, 0)'}, {'A': 'a', 'B':'b'}),
            ({'kind': 'component', 'py_expr': 'functions.if_then_else(a()>b(), 1, 0)'}, [])
        )

        self.assertEqual(
            parse_general_expression({'expr': 'If Then Else(A>B, 1, A)'}, {'A': 'a', 'B':'b'}),
            ({'kind': 'component', 'py_expr': 'functions.if_then_else(a()>b(), 1, a())'}, [])
        )

    def test_stock_construction_function_no_subscripts(self):
        from translators.vensim2py2 import parse_general_expression
        self.assertEqual(
            parse_general_expression({'expr': 'INTEG (FlowA, -10)',
                                      'py_name':'test_stock',
                                      'subs':None},
                                     {'FlowA': 'flowa'}),
            ({'kind': 'component', 'py_expr': "_state['test_stock']"},
             [{'kind': 'implicit',
               'subs': None,
               'doc': 'Provides initial conditions for test_stock function',
               'py_name': '_init_test_stock',
               'real_name': None,
               'unit': 'See docs for test_stock',
               'py_expr': '-10'},
              {'py_name': '_dtest_stock_dt',
               'subs': None,
               'doc': 'Provides derivative for test_stock function',
               'kind': 'implicit',
               'unit': 'See docs for test_stock',
               'py_expr': 'flowa()',
               'real_name': None}])
        )

    @unittest.skip('not yet implemented')
    def test_delay_construction_function_no_subscripts(self):
        from translators.vensim2py2 import parse_general_expression
        self.assertEqual(
            parse_general_expression({'expr': 'Const * DELAY1(Variable, DelayTime)'},
                                     {'Const': 'const', 'Variable': 'variable', 'DelayTime':'delaytime'}),
            ({'kind': 'component', 'py_expr': 'abs(stocka)'}, [])
        )

        self.assertEqual(
            parse_general_expression({'expr': 'DELAY N(Inflow , delay , 0 , Order)'},
                                      {'Const': 'const', 'Variable': 'variable'}),
            ({'kind': 'component', 'py_expr': 'functions.if_then_else(a>b, 1, 0)'}, [])
        )

    @unittest.skip('not yet implemented')
    def test_smooth_construction_function_no_subscripts(self):
        from translators.vensim2py2 import parse_general_expression
        self.assertEqual(
            parse_general_expression({'expr': 'If Then Else(A>B, 1, 0)'}, {'A': 'a', 'B':'b'}),
            ({'kind': 'component', 'py_expr': 'functions.if_then_else(a>b, 1, 0)'}, [])
        )

    def test_subscript_constant(self):
        from translators.vensim2py2 import parse_general_expression
        self.assertEqual(
            parse_general_expression({'expr': '1, 2, 3, 4, 5, 6'}, {},
                                     {'One Dimensional Subscript': ['Entry 1', 'Entry 2', 'Entry 3'],
                                      'Second Dimension Subscript': ['Column 1', 'Column 2']}),
            ({'kind': 'constant', 'py_expr': '1,2,3,4,5,6'}, [])
        )

        self.assertEqual(
            parse_general_expression({'expr': '1, 2; 3, 4; 5, 6;'}, {},
                                     {'One Dimensional Subscript': ['Entry 1', 'Entry 2', 'Entry 3'],
                                      'Second Dimension Subscript': ['Column 1', 'Column 2']}),
            ({'kind': 'constant', 'py_expr': '[1,2],[3,4],[5,6]'}, [])
        )

    def test_subscript_stock(self):
        self.maxDiff = None
        from translators.vensim2py2 import parse_general_expression
        self.assertEqual(
            parse_general_expression({'expr': 'INTEG (Flow[sub_D1,sub_D2], Init[sub_D1, sub_D2])',
                                      'py_name': 'stock_test', 'subs': ['sub_D1']},
                                     {'Init': 'init', 'Flow': 'flow'},
                                     {'sub_D1': ['Entry 1', 'Entry 2', 'Entry 3'],
                                      'sub_D2': ['Column 1', 'Column 2']}),
            ({'kind': 'component', 'py_expr': "_state['stock_test']"},
             [{'kind': 'implicit',
               'subs': ['sub_D1'],
               'doc': 'Provides initial conditions for stock_test function',
               'py_name': '_init_stock_test',
               'real_name': None,
               'unit': 'See docs for stock_test',
               'py_expr': 'init()'},
              {'py_name': '_dstock_test_dt',
               'subs': ['sub_D1'],
               'kind': 'implicit',
               'py_expr': 'flow()',
               'unit': 'See docs for stock_test',
               'doc': 'Provides derivative for stock_test function',
               'real_name': None}])
        )

    #notes:
    # If the reference uses full subscript names in its reference, we don't have
    # to make any down-selection.
    # If the reference uses a subscript element, we MAY have to make a down-selection,
    # so do it by default?

    def test_subscript_reference(self):
        from translators.vensim2py2 import parse_general_expression
        self.assertEqual(
            parse_general_expression({'expr': 'Var A[Dim1, Dim2]'},
                                     {'Var A': 'var_a'},
                                     {'Dim1': ['A', 'B'],
                                      'Dim2': ['C', 'D', 'E']}),
            ({'kind': 'component', 'py_expr': 'var_a()'}, [])
        )

        self.assertEqual(
            parse_general_expression({'expr': 'Var B[Dim1, C]'},
                                     {'Var B': 'var_b'},
                                     {'Dim1': ['A', 'B'],
                                      'Dim2': ['C', 'D', 'E']}),
            ({'kind': 'component', 'py_expr': "var_b().loc[{'Dim2': ['C']}]"}, [])
        )

        self.assertEqual(
            parse_general_expression({'expr': 'Var C[Dim1, C, H]'},
                                     {'Var C': 'var_c'},
                                     {'Dim1': ['A', 'B'],
                                      'Dim2': ['C', 'D', 'E'],
                                      'Dim3': ['F', 'G', 'H', 'I']}),
            ({'kind': 'component', 'py_expr': "var_c().loc[{'Dim2': ['C'], 'Dim3': ['H']}]"}, [])
        )

    @unittest.skip('not yet implemented')
    def test_subscript_ranges(self):
        from translators.vensim2py2 import parse_general_expression
        self.assertEqual(
            parse_general_expression({'expr': 'Var D[Range1]'},
                                     {'Var D': 'var_c'},
                                     {'Dim1': ['A', 'B', 'C', 'D', 'E', 'F'],
                                      'Range1': ['C', 'D', 'E']}),
            ({'kind': 'constant', 'py_expr': "var_c().loc[{'Dim1': ['C', 'D', 'E']}]"}, [])
        )


