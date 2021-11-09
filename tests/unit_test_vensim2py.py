import unittest
import xarray as xr


class TestGetFileSections(unittest.TestCase):
    def test_normal_load(self):
        """normal model file with no macros"""
        from pysd.translation.vensim.vensim2py import get_file_sections

        actual = get_file_sections(r"a~b~c| d~e~f| g~h~i|")
        expected = [
            {
                "returns": [],
                "params": [],
                "name": "_main_",
                "string": "a~b~c| d~e~f| g~h~i|",
            }
        ]
        self.assertEqual(actual, expected)

    def test_macro_only(self):
        """ Macro Only """
        from pysd.translation.vensim.vensim2py import get_file_sections

        actual = get_file_sections(":MACRO: MAC(z) a~b~c| :END OF MACRO:")
        expected = [{"returns": [], "params": ["z"], "name": "MAC",
                     "string": "a~b~c|"}]
        self.assertEqual(actual, expected)

    def test_macro_and_model(self):
        """ basic macro and model """
        from pysd.translation.vensim.vensim2py import get_file_sections

        actual = get_file_sections(
            ":MACRO: MAC(z) a~b~c| :END OF MACRO: d~e~f| g~h~i|")
        expected = [
            {"returns": [], "params": ["z"], "name": "MAC",
             "string": "a~b~c|"},
            {"returns": [], "params": [], "name": "_main_",
             "string": "d~e~f| g~h~i|"},
        ]
        self.assertEqual(actual, expected)

    def test_macro_multiple_inputs(self):
        """ macro with multiple input parameters """
        from pysd.translation.vensim.vensim2py import get_file_sections

        actual = get_file_sections(
            ":MACRO: MAC(z, y) a~b~c| :END OF MACRO: d~e~f| g~h~i|"
        )
        expected = [
            {"returns": [], "params": ["z", "y"], "name": "MAC",
             "string": "a~b~c|"},
            {"returns": [], "params": [], "name": "_main_",
             "string": "d~e~f| g~h~i|"},
        ]
        self.assertEqual(actual, expected)

    def test_macro_with_returns(self):
        """ macro with return values """
        from pysd.translation.vensim.vensim2py import get_file_sections

        actual = get_file_sections(
            ":MACRO: MAC(z, y :x, w) a~b~c| :END OF MACRO: d~e~f| g~h~i|"
        )
        expected = [
            {
                "returns": ["x", "w"],
                "params": ["z", "y"],
                "name": "MAC",
                "string": "a~b~c|",
            },
            {"returns": [], "params": [], "name": "_main_",
             "string": "d~e~f| g~h~i|"},
        ]
        self.assertEqual(actual, expected)

    def test_handle_encoding(self):
        """ Handle encoding """
        from pysd.translation.vensim.vensim2py import get_file_sections

        actual = get_file_sections(r"{UTF-8} a~b~c| d~e~f| g~h~i|")
        expected = [
            {
                "returns": [],
                "params": [],
                "name": "_main_",
                "string": "a~b~c| d~e~f| g~h~i|",
            }
        ]
        self.assertEqual(actual, expected)

    def test_handle_encoding_like_strings(self):
        """ Handle encoding-like strings in other places in the file """
        from pysd.translation.vensim.vensim2py import get_file_sections

        actual = get_file_sections(r"a~b~c| d~e~f{special}| g~h~i|")
        expected = [
            {
                "returns": [],
                "params": [],
                "name": "_main_",
                "string": "a~b~c| d~e~f{special}| g~h~i|",
            }
        ]
        self.assertEqual(actual, expected)


class TestEquationStringParsing(unittest.TestCase):
    """ Tests the 'get_equation_components function """

    def test_basics(self):
        from pysd.translation.vensim.vensim2py import get_equation_components

        self.assertEqual(
            get_equation_components(r'constant = 25'),
            {
                'expr': '25',
                'kind': 'component',
                'subs': [],
                'subs_compatibility': {},
                'real_name': 'constant',
                'keyword': None
            }
        )

    def test_equals_handling(self):
        """ Parse cases with equal signs within the expression """
        from pysd.translation.vensim.vensim2py import get_equation_components

        self.assertEqual(
            get_equation_components(r"Boolean = IF THEN ELSE(1 = 1, 1, 0)"),
            {
                "expr": "IF THEN ELSE(1 = 1, 1, 0)",
                "kind": "component",
                "subs": [],
                "subs_compatibility": {},
                "real_name": "Boolean",
                "keyword": None,
            },
        )

    def test_whitespace_handling(self):
        """ Whitespaces should be shortened to a single space """
        from pysd.translation.vensim.vensim2py import get_equation_components

        self.assertEqual(
            get_equation_components(
                r"""constant\t =
                                                        \t25\t """
            ),
            {
                "expr": "25",
                "kind": "component",
                "subs": [],
                "subs_compatibility": {},
                "real_name": "constant",
                "keyword": None,
            },
        )

        # test eliminating vensim's line continuation character
        self.assertEqual(
            get_equation_components(
                r"""constant [Sub1, \\
                                     Sub2] = 10, 12; 14, 16;"""
            ),
            {
                "expr": "10, 12; 14, 16;",
                "kind": "component",
                "subs": ["Sub1", "Sub2"],
                "subs_compatibility": {},
                "real_name": "constant",
                "keyword": None,
            },
        )

    def test_subscript_definition_parsing(self):
        from pysd.translation.vensim.vensim2py import get_equation_components

        self.assertEqual(
            get_equation_components(r"""Sub1: Entry 1, Entry 2, Entry 3 """),
            {
                "expr": None,
                "kind": "subdef",
                "subs": ["Entry 1", "Entry 2", "Entry 3"],
                "subs_compatibility": {},
                "real_name": "Sub1",
                "keyword": None,
            },
        )

        with self.assertRaises(ValueError) as err:
            get_equation_components(r"""Sub2: (1-3) """)

        self.assertIn(
            "A numeric range must contain at least one letter.",
            str(err.exception))

        with self.assertRaises(ValueError) as err:
            get_equation_components(r"""Sub2: (a1-a1) """)

        self.assertIn(
            "The number of the first subscript value must be "
            "lower than the second subscript value in a "
            "subscript numeric range.",
            str(err.exception))

        with self.assertRaises(ValueError) as err:
            get_equation_components(r"""Sub2: (a1-b3) """)

        self.assertIn(
            "Only matching names ending in numbers are valid.",
            str(err.exception))

    def test_subscript_references(self):
        from pysd.translation.vensim.vensim2py import get_equation_components

        self.assertEqual(
            get_equation_components(
                r"constant [Sub1, Sub2] = 10, 12; 14, 16;"),
            {
                "expr": "10, 12; 14, 16;",
                "kind": "component",
                "subs": ["Sub1", "Sub2"],
                "subs_compatibility": {},
                "real_name": "constant",
                "keyword": None,
            },
        )

        self.assertEqual(
            get_equation_components(
                r"function [Sub1] = other function[Sub1]"),
            {
                "expr": "other function[Sub1]",
                "kind": "component",
                "subs": ["Sub1"],
                "subs_compatibility": {},
                "real_name": "function",
                "keyword": None,
            },
        )

        self.assertEqual(
            get_equation_components(
                r'constant ["S1,b", "S1,c"] = 1, 2; 3, 4;'),
            {
                "expr": "1, 2; 3, 4;",
                "kind": "component",
                "subs": ['"S1,b"', '"S1,c"'],
                "subs_compatibility": {},
                "real_name": "constant",
                "keyword": None,
            },
        )

        self.assertEqual(
            get_equation_components(
                r'constant ["S1=b", "S1=c"] = 1, 2; 3, 4;'),
            {
                "expr": "1, 2; 3, 4;",
                "kind": "component",
                "subs": ['"S1=b"', '"S1=c"'],
                "subs_compatibility": {},
                "real_name": "constant",
                "keyword": None,
            },
        )

    def test_lookup_definitions(self):
        from pysd.translation.vensim.vensim2py import get_equation_components

        self.assertEqual(
            get_equation_components(r"table([(0,-1)-(45,1)],(0,0),(5,0))"),
            {
                "expr": "([(0,-1)-(45,1)],(0,0),(5,0))",
                "kind": "lookup",
                "subs": [],
                "subs_compatibility": {},
                "real_name": "table",
                "keyword": None,
            },
        )

        self.assertEqual(
            get_equation_components(r"table2 ([(0,-1)-(45,1)],(0,0),(5,0))"),
            {
                "expr": "([(0,-1)-(45,1)],(0,0),(5,0))",
                "kind": "lookup",
                "subs": [],
                "subs_compatibility": {},
                "real_name": "table2",
                "keyword": None,
            },
        )

    def test_get_lookup(self):
        from pysd.translation.vensim.vensim2py import parse_lookup_expression

        res = parse_lookup_expression(
            {
                "expr": r"(GET DIRECT LOOKUPS('path2excel.xlsx', "
                + r"'SheetName', 'index'\ , 'values'))",
                "py_name": "get_lookup",
                "subs": [],
                "merge_subs": []
            },
            {}
        )[1][0]

        self.assertEqual(
            res["py_expr"],
            "ExtLookup('path2excel.xlsx', 'SheetName', 'index', 'values', "
            + "{},\n          _root, '_ext_lookup_get_lookup')",
        )

    def test_pathological_names(self):
        from pysd.translation.vensim.vensim2py import get_equation_components

        self.assertEqual(
            get_equation_components(r'"silly-string" = 25'),
            {
                "expr": "25",
                "kind": "component",
                "subs": [],
                "subs_compatibility": {},
                "real_name": '"silly-string"',
                "keyword": None,
            },
        )

        self.assertEqual(
            get_equation_components(r'"pathological\\-string" = 25'),
            {
                "expr": "25",
                "kind": "component",
                "subs": [],
                "subs_compatibility": {},
                "real_name": r'"pathological\\-string"',
                "keyword": None,
            },
        )

    def test_get_equation_components_error(self):
        from pysd.translation.vensim.vensim2py import get_equation_components

        defi = "NIF: NF<x-x>NF"
        try:
            get_equation_components(defi)
            self.assertFail()
        except ValueError as err:
            self.assertIn(
                "\nError when parsing definition:\n\t %s\n\n"
                "probably used definition is invalid or not integrated..."
                "\nSee parsimonious output above." % defi,
                err.args[0],
            )


class TestParse_general_expression(unittest.TestCase):
    def test_arithmetic(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        res = parse_general_expression({"expr": "-10^3+4"})
        self.assertEqual(res[0]["py_expr"], "-10**3+4")

    def test_arithmetic_scientific(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        res = parse_general_expression({"expr": "1e+4"})
        self.assertEqual(res[0]["py_expr"], "1e+4")

        res = parse_general_expression({"expr": "2e4"})
        self.assertEqual(res[0]["py_expr"], "2e4")

        res = parse_general_expression({"expr": "3.43e04"})
        self.assertEqual(res[0]["py_expr"], "3.43e04")

        res = parse_general_expression({"expr": "1.0E4"})
        self.assertEqual(res[0]["py_expr"], "1.0E4")

        res = parse_general_expression({"expr": "-2.0E43"})
        self.assertEqual(res[0]["py_expr"], "-2.0E43")

        res = parse_general_expression({"expr": "-2.0e-43"})
        self.assertEqual(res[0]["py_expr"], "-2.0e-43")

    def test_caps_handling(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        res = parse_general_expression({"expr": "Abs(-3)"})
        self.assertEqual(res[0]["py_expr"], "np.abs(-3)")

        res = parse_general_expression({"expr": "ABS(-3)"})
        self.assertEqual(res[0]["py_expr"], "np.abs(-3)")

        res = parse_general_expression({"expr": "aBS(-3)"})
        self.assertEqual(res[0]["py_expr"], "np.abs(-3)")

    def test_empty(self):
        from warnings import catch_warnings
        from pysd.translation.vensim.vensim2py import parse_general_expression

        with catch_warnings(record=True) as ws:
            res = parse_general_expression({"expr": "", "real_name": "Var"})
            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 1)
            self.assertIn("Empty expression for 'Var'", str(wu[0].message))

        self.assertEqual(res[0]["py_expr"], "None")

    def test_function_calls(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        res = parse_general_expression({"expr": "ABS(StockA)",
                                        "real_name": "AB",
                                        "eqn": "AB = ABS(StockA)"},
                                       {"StockA": "stocka"})
        self.assertEqual(res[0]["py_expr"], "np.abs(stocka())")

        res = parse_general_expression(
            {"expr": "If Then Else(A>B, 1, 0)",
             "real_name": "IFE",
             "eqn": "IFE = If Then Else(A>B, 1, 0)"},
            {"A": "a", "B": "b"}
        )
        self.assertEqual(
            res[0]["py_expr"], "if_then_else(a()>b(), lambda: 1, lambda: 0)"
        )

        # test that function calls are handled properly in arguments
        res = parse_general_expression(
            {"expr": "If Then Else(A>B,1,A)"}, {"A": "a", "B": "b"}
        )
        self.assertEqual(
            res[0]["py_expr"], "if_then_else(a()>b(), lambda: 1, lambda: a())"
        )

    def test_id_parsing(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        res = parse_general_expression({"expr": "StockA"},
                                       {"StockA": "stocka"})
        self.assertEqual(res[0]["py_expr"], "stocka()")

    def test_logicals(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        res = parse_general_expression(
            {'expr': 'IF THEN ELSE(1 :AND: 0,0,1)'})
        self.assertEqual(res[0]['py_expr'],
                         'if_then_else(logical_and(1,0), lambda: 0, lambda: 1)'
                         )

        res = parse_general_expression(
            {'expr': 'IF THEN ELSE(1 :OR: 0,0,1)'})
        self.assertEqual(
             res[0]['py_expr'],
             'if_then_else(logical_or(1,0), lambda: 0, lambda: 1)'
        )

        res = parse_general_expression(
            {'expr': 'IF THEN ELSE(1 :AND: 0 :and: 1,0,1)'})
        self.assertEqual(
             res[0]['py_expr'],
             'if_then_else(logical_and(1,0,1), lambda: 0, lambda: 1)'
        )

        res = parse_general_expression(
            {'expr': 'IF THEN ELSE(1 :or: 0 :OR: 1 :oR: 0,0,1)'})
        self.assertEqual(
             res[0]['py_expr'],
             'if_then_else(logical_or(1,0,1,0), lambda: 0, lambda: 1)'
        )

        res = parse_general_expression(
            {'expr': 'IF THEN ELSE(1 :AND: (0 :OR: 1),0,1)'})
        self.assertEqual(res[0]['py_expr'],
                         'if_then_else(logical_and(1,logical_or(0,1)),' +
                         ' lambda: 0, lambda: 1)')

        res = parse_general_expression(
            {'expr': 'IF THEN ELSE((1 :AND: 0) :OR: 1,0,1)'})
        self.assertEqual(res[0]['py_expr'],
                         'if_then_else(logical_or(logical_and(1,0),1),' +
                         ' lambda: 0, lambda: 1)')

        with self.assertRaises(ValueError):
            res = parse_general_expression(
                {'expr': 'IF THEN ELSE(1 :AND: 0 :OR: 1,0,1)',
                 'real_name': 'logical',
                 'eqn': 'logical = IF THEN ELSE(1 :AND: 0 :OR: 1,0,1)'})

    def test_number_parsing(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression
        res = parse_general_expression({'expr': '20'})
        self.assertEqual(res[0]['py_expr'], '20')

        res = parse_general_expression({"expr": "3.14159"})
        self.assertEqual(res[0]["py_expr"], "3.14159")

        res = parse_general_expression({"expr": "1.3e+10"})
        self.assertEqual(res[0]["py_expr"], "1.3e+10")

        res = parse_general_expression({"expr": "-1.3e-10"})
        self.assertEqual(res[0]["py_expr"], "-1.3e-10")

    def test_nan_parsing(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression
        from pysd.translation.builder import Imports

        Imports.reset()
        self.assertFalse(Imports._numpy)
        res = parse_general_expression({'expr': ':NA:'})
        self.assertEqual(res[0]['py_expr'], 'np.nan')
        self.assertTrue(Imports._numpy)

    def test_stock_construction_function_no_subscripts(self):
        """ stock construction should create a stateful variable and
        reference it """
        from pysd.translation.vensim.vensim2py import parse_general_expression
        from pysd.py_backend.statefuls import Integ

        res = parse_general_expression(
            {
                "expr": "INTEG (FlowA, -10)",
                "py_name": "test_stock",
                "subs": [],
                "merge_subs": []
            },
            {"FlowA": "flowa"}
        )

        self.assertEqual(res[1][0]["kind"], "stateful")
        a = eval(res[1][0]["py_expr"])
        self.assertIsInstance(a, Integ)

        # check the reference to that variable
        self.assertEqual(res[0]["py_expr"], res[1][0]["py_name"] + "()")

    def test_delay_construction_function_no_subscripts(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression
        from pysd.py_backend.statefuls import Delay

        res = parse_general_expression(
            {
                "expr": "DELAY1(Variable, DelayTime)",
                "py_name": "test_delay",
                "subs": [],
                "merge_subs": []
            },
            {
                "Variable": "variable",
                "DelayTime": "delaytime",
                "TIME STEP": "time_step",
            }
        )

        def time_step():
            return 0.5

        self.assertEqual(res[1][0]["kind"], "stateful")
        a = eval(res[1][0]["py_expr"])
        self.assertIsInstance(a, Delay)

        # check the reference to that variable
        self.assertEqual(res[0]["py_expr"], res[1][0]["py_name"] + "()")

    def test_forecast_construction_function_no_subscripts(self):
        """ Tests translation of 'forecast'

        This translation should create a new stateful object to hold the
        forecast elements, and then pass back a reference to that value
        """
        from pysd.translation.vensim.vensim2py import parse_general_expression
        from pysd.py_backend.statefuls import Forecast

        res = parse_general_expression(
            {
                "expr": "FORECAST(Variable, AverageTime, Horizon)",
                "py_name": "test_forecast",
                "subs": [],
                "merge_subs": []
            },
            {"Variable": "variable", "AverageTime": "averagetime",
             "Horizon": "horizon"},
            elements_subs_dict={"test_forecast": []},
        )

        # check stateful object creation
        self.assertEqual(res[1][0]["kind"], "stateful")
        a = eval(res[1][0]["py_expr"])
        self.assertIsInstance(a, Forecast)

        # check the reference to that variable
        self.assertEqual(res[0]["py_expr"], res[1][0]["py_name"] + "()")

    def test_smooth_construction_function_no_subscripts(self):
        """ Tests translation of 'smooth'

        This translation should create a new stateful object to hold the delay
        elements, and then pass back a reference to that value
        """
        from pysd.translation.vensim.vensim2py import parse_general_expression
        from pysd.py_backend.statefuls import Smooth

        res = parse_general_expression(
            {
                "expr": "SMOOTH(Variable, DelayTime)",
                "py_name": "test_smooth",
                "subs": [],
                "merge_subs": []
            },
            {"Variable": "variable", "DelayTime": "delaytime"},
        )

        # check stateful object creation
        self.assertEqual(res[1][0]["kind"], "stateful")
        a = eval(res[1][0]["py_expr"])
        self.assertIsInstance(a, Smooth)

        # check the reference to that variable
        self.assertEqual(res[0]["py_expr"], res[1][0]["py_name"] + "()")

    def test_subscript_float_initialization(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        _subscript_dict = {
            "Dim": ["A", "B", "C", "D", "E"],
            "Dim1": ["A", "B", "C"], "Dim2": ["D", "E"]
        }

        # case 1
        element = parse_general_expression(
            {"expr": "3.32", "subs": ["Dim1"], "py_name": "var",
             "merge_subs": ["Dim1"]}, {},
            _subscript_dict

        )
        string = element[0]["py_expr"]
        # TODO we should use a = eval(string)
        # hoewever eval is not detecting _subscript_dict variable
        self.assertEqual(
            string,
            "xr.DataArray(3.32,{'Dim1': _subscript_dict['Dim1']},['Dim1'])",
        )
        a = xr.DataArray(
            3.32, {dim: _subscript_dict[dim] for dim in ["Dim1"]}, ["Dim1"]
        )
        self.assertDictEqual(
            {key: list(val.values) for key, val in a.coords.items()},
            {"Dim1": ["A", "B", "C"]},
        )
        self.assertEqual(a.loc[{"Dim1": "B"}], 3.32)

        # case 2: xarray subscript is a subrange from the final subscript range
        element = parse_general_expression(
            {"expr": "3.32", "subs": ["Dim1"], "py_name": "var",
             "merge_subs": ["Dim"]}, {}, _subscript_dict
        )
        string = element[0]["py_expr"]
        # TODO we should use a = eval(string)
        # hoewever eval is not detecting _subscript_dict variable
        self.assertEqual(
            string,
            "xr.DataArray(3.32,{'Dim': _subscript_dict['Dim1']},['Dim'])",
        )
        a = xr.DataArray(
            3.32, {"Dim": _subscript_dict["Dim1"]}, ["Dim"]
        )
        self.assertDictEqual(
            {key: list(val.values) for key, val in a.coords.items()},
            {"Dim": ["A", "B", "C"]},
        )
        self.assertEqual(a.loc[{"Dim": "B"}], 3.32)

    def test_subscript_1d_constant(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        _subscript_dict = {"Dim1": ["A", "B", "C"], "Dim2": ["D", "E"]}
        element = parse_general_expression(
            {"expr": "1, 2, 3", "subs": ["Dim1"], "py_name": "var",
             "merge_subs": ["Dim1"]},
            {}, _subscript_dict
        )
        string = element[0]["py_expr"]
        # TODO we should use a = eval(string)
        # hoewever eval is not detecting _subscript_dict variable
        self.assertEqual(
            string,
            "xr.DataArray([1.,2.,3.],{'Dim1': _subscript_dict['Dim1']},"
            "['Dim1'])",
        )
        a = xr.DataArray([1.0, 2.0, 3.0],
                         {dim: _subscript_dict[dim] for dim in ["Dim1"]},
                         ["Dim1"])
        self.assertDictEqual(
            {key: list(val.values) for key, val in a.coords.items()},
            {"Dim1": ["A", "B", "C"]},
        )
        self.assertEqual(a.loc[{"Dim1": "A"}], 1)

    def test_subscript_2d_constant(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        _subscript_dict = {"Dim1": ["A", "B", "C"], "Dim2": ["D", "E"]}
        element = parse_general_expression(
            {"expr": "1, 2; 3, 4; 5, 6;", "subs": ["Dim1", "Dim2"],
             "merge_subs": ["Dim1", "Dim2"], "py_name": "var"},
            {}, _subscript_dict
        )
        string = element[0]["py_expr"]
        a = eval(string)
        self.assertDictEqual(
            {key: list(val.values) for key, val in a.coords.items()},
            {"Dim1": ["A", "B", "C"], "Dim2": ["D", "E"]},
        )
        self.assertEqual(a.loc[{"Dim1": "A", "Dim2": "D"}], 1)
        self.assertEqual(a.loc[{"Dim1": "B", "Dim2": "E"}], 4)

    def test_subscript_3d_depth(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        _subscript_dict = {"Dim1": ["A", "B", "C"], "Dim2": ["D", "E"]}
        element = parse_general_expression(
            {"expr": "1, 2; 3, 4; 5, 6;", "subs": ["Dim1", "Dim2"],
             "merge_subs": ["Dim1", "Dim2"], "py_name": "var"},
            {}, _subscript_dict,
        )
        string = element[0]["py_expr"]
        a = eval(string)
        self.assertDictEqual(
            {key: list(val.values) for key, val in a.coords.items()},
            {"Dim1": ["A", "B", "C"], "Dim2": ["D", "E"]},
        )
        self.assertEqual(a.loc[{"Dim1": "A", "Dim2": "D"}], 1)
        self.assertEqual(a.loc[{"Dim1": "B", "Dim2": "E"}], 4)

    def test_subscript_builder(self):
        """
        Testing how subscripts are translated when we have common subscript
        ranges.
        """
        from pysd.translation.vensim.vensim2py import\
            parse_general_expression, parse_lookup_expression

        _subscript_dict = {
            "Dim1": ["A", "B", "C"], "Dim2": ["B", "C"], "Dim3": ["B", "C"]
        }

        # case 1: subscript of the expr is in the final range, which is a
        # subrange of a greater range
        element = parse_general_expression(
            {"py_name": "var1", "subs": ["B"], "real_name": "var1", "eqn": "",
             "expr": "GET DIRECT CONSTANTS('input.xlsx', 'Sheet1', 'C20')",
             "merge_subs": ["Dim2"]},
            {},
            _subscript_dict
        )
        self.assertIn(
            "'Dim2': ['B']", element[1][0]['py_expr'])

        # case 1b: subscript of the expr is in the final range, which is a
        # subrange of a greater range
        element = parse_lookup_expression(
            {"py_name": "var1b", "subs": ["B"],
             "real_name": "var1b", "eqn": "",
             "expr": "(GET DIRECT LOOKUPS('input.xlsx', 'Sheet1',"
                     " '19', 'C20'))",
             "merge_subs": ["Dim2"]},
            _subscript_dict,
        )
        self.assertIn(
            "'Dim2': ['B']", element[1][0]['py_expr'])

        # case 2: subscript of the expr is a subscript subrange equal to the
        # final range, which is a subrange of a greater range
        element = parse_general_expression(
            {"py_name": "var2", "subs": ["Dim2"],
             "real_name": "var2", "eqn": "",
             "expr": "GET DIRECT CONSTANTS('input.xlsx', 'Sheet1', 'C20')",
             "merge_subs": ["Dim2"]},
            {},
            _subscript_dict
        )
        self.assertIn(
            "'Dim2': _subscript_dict['Dim2']", element[1][0]['py_expr'])

        # case 3: subscript of the expr is a subscript subrange equal to the
        # final range, which is a subrange of a greater range, but there is
        # a similar subrange before
        element = parse_general_expression(
            {"py_name": "var3", "subs": ["B"], "real_name": "var3", "eqn": "",
             "expr": "GET DIRECT CONSTANTS('input.xlsx', 'Sheet1', 'C20')",
             "merge_subs": ["Dim3"]},
            {},
            _subscript_dict
        )
        self.assertIn(
            "'Dim3': ['B']", element[1][0]['py_expr'])

        # case 4: subscript of the expr is a subscript subrange and the final
        # subscript is a greater range
        element = parse_general_expression(
            {"py_name": "var4", "subs": ["Dim2"],
             "real_name": "var4", "eqn": "",
             "expr": "GET DIRECT CONSTANTS('input.xlsx', 'Sheet1', 'C20')",
             "merge_subs": ["Dim1"]},
            {},
            _subscript_dict,
        )
        self.assertIn(
            "'Dim1': _subscript_dict['Dim2']", element[1][0]['py_expr'])

        # case 4b: subscript of the expr is a subscript subrange and the final
        # subscript is a greater range
        element = parse_general_expression(
            {"py_name": "var4b", "subs": ["Dim2"],
             "real_name": "var4b", "eqn": "",
             "expr": "GET DIRECT DATA('input.xlsx', 'Sheet1', '19', 'C20')",
             "keyword": None, "merge_subs": ["Dim1"]},
            {},
            _subscript_dict
        )
        self.assertIn(
            "'Dim1': _subscript_dict['Dim2']", element[1][0]['py_expr'])

        # case 4c: subscript of the expr is a subscript subrange and the final
        # subscript is a greater range
        element = parse_general_expression(
            {"py_name": "var4c", "subs": ["Dim2"],
             "real_name": "var4c", "eqn": "",
             "expr": "GET DIRECT LOOKUPS('input.xlsx', 'Sheet1',"
             " '19', 'C20')", "merge_subs": ["Dim1"]},
            {},
            _subscript_dict
        )
        self.assertIn(
            "'Dim1': _subscript_dict['Dim2']", element[1][0]['py_expr'])

        # case 4d: subscript of the expr is a subscript subrange and the final
        # subscript is a greater range
        element = parse_lookup_expression(
            {"py_name": "var4d", "subs": ["Dim2"],
             "real_name": "var4d", "eqn": "",
             "expr": "(GET DIRECT LOOKUPS('input.xlsx', 'Sheet1',"
                     " '19', 'C20'))", "merge_subs": ["Dim1"]},
            _subscript_dict
        )
        self.assertIn(
            "'Dim1': _subscript_dict['Dim2']", element[1][0]['py_expr'])

    def test_subscript_reference(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        res = parse_general_expression(
            {"expr": "Var A[Dim1, Dim2]", "real_name": "Var2", "eqn": ""},
            {"Var A": "var_a"},
            {"Dim1": ["A", "B"], "Dim2": ["C", "D", "E"]},
            None,
            {"var_a": ["Dim1", "Dim2"]}
        )

        self.assertEqual(res[0]["py_expr"], "var_a()")

        res = parse_general_expression(
            {"expr": "Var B[Dim1, C]"},
            {"Var B": "var_b"},
            {"Dim1": ["A", "B"], "Dim2": ["C", "D", "E"]},
            None,
            {"var_b": ["Dim1", "Dim2"]},
        )

        self.assertEqual(
            res[0]["py_expr"],
            "rearrange(var_b().loc[:, 'C'].reset_coords(drop=True),"
            "['Dim1'],_subscript_dict)",
        )

        res = parse_general_expression({'expr': 'Var B[A, C]'},
                                       {'Var B': 'var_b'},
                                       {'Dim1': ['A', 'B'],
                                        'Dim2': ['C', 'D', 'E']},
                                       None,
                                       {'var_b': ['Dim1', 'Dim2']})

        self.assertEqual(
            res[0]['py_expr'],
            "float(var_b().loc['A', 'C'])")

        res = parse_general_expression({'expr': 'Var C[Dim1, C, H]'},
                                       {'Var C': 'var_c'},
                                       {'Dim1': ['A', 'B'],
                                        'Dim2': ['C', 'D', 'E'],
                                        'Dim3': ['F', 'G', 'H', 'I']},
                                       None,
                                       {'var_c': ['Dim1', 'Dim2', 'Dim3']})
        self.assertEqual(
            res[0]["py_expr"],
            "rearrange(var_c().loc[:, 'C', 'H'].reset_coords(drop=True),"
            "['Dim1'],_subscript_dict)",
        )

        res = parse_general_expression({'expr': 'Var C[B, C, H]'},
                                       {'Var C': 'var_c'},
                                       {'Dim1': ['A', 'B'],
                                        'Dim2': ['C', 'D', 'E'],
                                        'Dim3': ['F', 'G', 'H', 'I']},
                                       None,
                                       {'var_c': ['Dim1', 'Dim2', 'Dim3']})

        self.assertEqual(
            res[0]['py_expr'],
            "float(var_c().loc['B', 'C', 'H'])")

    def test_subscript_ranges(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        res = parse_general_expression(
            {"expr": "Var D[Range1]"},
            {"Var D": "var_c"},
            {"Dim1": ["A", "B", "C", "D", "E", "F"],
             "Range1": ["C", "D", "E"]},
            None,
            {"var_c": ["Dim1"]},
        )

        self.assertEqual(
            res[0]["py_expr"], "rearrange(var_c(),['Range1'],_subscript_dict)"
        )

    def test_invert_matrix(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        res = parse_general_expression(
            {
                "expr": "INVERT MATRIX(A, 3)",
                "real_name": "A1",
                "py_name": "a1",
                "merge_subs": ["dim1", "dim2"]
            },
            {
                "A": "a",
                "A1": "a1",
            },
            subscript_dict={
                "dim1": ["a", "b", "c"], "dim2": ["a", "b", "c"]
            }
        )

        self.assertEqual(res[0]["py_expr"], "invert_matrix(a())")

    def test_subscript_elmcount(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        res = parse_general_expression(
            {
                "expr": "ELMCOUNT(dim1)",
                "real_name": "A",
                "py_name": "a",
                "merge_subs": []
            },
            {
                "A": "a",
            },
            subscript_dict={
                "dim1": ["a", "b", "c"], "dim2": ["a", "b", "c"]
            }
        )

        self.assertIn(
            "len(_subscript_dict['dim1'])",
            res[0]["py_expr"], )

    def test_subscript_logicals(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        res = parse_general_expression(
            {
                "expr": "IF THEN ELSE(dim1=dim2, 5, 0)",
                "real_name": "A",
                "py_name": "a",
                "merge_subs": ["dim1", "dim2"]
            },
            {
                "A": "a",
            },
            subscript_dict={
                "dim1": ["a", "b", "c"], "dim2": ["a", "b", "c"]
            }
        )

        self.assertIn(
            "xr.DataArray(_subscript_dict['dim1'],"
            "{'dim1': _subscript_dict['dim1']},'dim1')"
            "==xr.DataArray(_subscript_dict['dim2'],"
            "{'dim2': _subscript_dict['dim2']},'dim2')",
            res[0]["py_expr"], )

    def test_ref_with_subscript_prefix(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        # When parsing functions arguments first the subscript ranges are
        # parsed and later the general id is used, however, the if a reference
        # to a var starts with a subscript range name this could make the
        # parser crash
        res = parse_general_expression(
            {
                "expr": "ABS(Upper var)",
                "real_name": "A",
                "eqn": "A = ABS(Upper var)",
                "py_name": "a",
                "merge_subs": []
            },
            {
                "Upper var": "upper_var",
            },
            subscript_dict={
                "upper": ["a", "b", "c"]
            }
        )

        self.assertIn(
            "np.abs(upper_var())",
            res[0]["py_expr"], )

    def test_random_0_1(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        # When parsing functions arguments first the subscript ranges are
        # parsed and later the general id is used, however, the if a reference
        # to a var starts with a subscript range name this could make the
        # parser crash
        res = parse_general_expression(
            {
                "expr": "RANDOM 0 1()",
                "real_name": "A",
                "eqn": "A = RANDOM 0 1()",
                "py_name": "a",
                "merge_subs": [],
                "dependencies": set()
            },
            {
                "A": "a",
            }
        )

        self.assertIn(
            "np.random.uniform(0, 1)",
            res[0]["py_expr"], )

    def test_random_uniform(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        # When parsing functions arguments first the subscript ranges are
        # parsed and later the general id is used, however, the if a reference
        # to a var starts with a subscript range name this could make the
        # parser crash
        res = parse_general_expression(
            {
                "expr": "RANDOM UNIFORM(10, 15, 3)",
                "real_name": "A",
                "eqn": "A = RANDOM UNIFORM(10, 15, 3)",
                "py_name": "a",
                "merge_subs": [],
                "dependencies": set()
            },
            {
                "A": "a",
            }
        )

        self.assertIn(
            "np.random.uniform(10, 15)",
            res[0]["py_expr"], )

    def test_incomplete_expression(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression
        from warnings import catch_warnings

        with catch_warnings(record=True) as w:
            res = parse_general_expression(
                {
                    "expr": "A FUNCTION OF(Unspecified Eqn,Var A,Var B)",
                    "real_name": "Incomplete Func",
                    "py_name": "incomplete_func",
                    "eqn": "Incomplete Func = A FUNCTION OF(Unspecified "
                           + "Eqn,Var A,Var B)",
                    "subs": [],
                    "merge_subs": []
                },
                {
                    "Unspecified Eqn": "unspecified_eqn",
                    "Var A": "var_a",
                    "Var B": "var_b",
                }
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(
                "Incomplete Func has no equation specified" in
                str(w[-1].message)
            )

        self.assertEqual(res[0]["py_expr"],
                         "incomplete(unspecified_eqn(), var_a(), var_b())")

    def test_parse_general_expression_error(self):
        from pysd.translation.vensim.vensim2py import parse_general_expression

        element = {
            "expr": "NIF(1,3)",
            "real_name": "not implemented function",
            "eqn": "not implemented function=\tNIF(1,3)",
        }
        try:
            parse_general_expression(element)
            self.assertFail()
        except ValueError as err:
            self.assertIn(
                "\nError when parsing %s with equation\n\t %s\n\n"
                "probably a used function is not integrated..."
                "\nSee parsimonious output above."
                % (element["real_name"], element["eqn"]),
                err.args[0],
            )


class TestParse_sketch_line(unittest.TestCase):
    def test_parse_sketch_line(self):
        from pysd.translation.vensim.vensim2py import parse_sketch_line

        namespace = {'"var-n"': "varn", "Stock": "stock", '"rate-1"': "rate1"}
        lines = [
            '10,1,"var-n",332,344,21,12,0,3,0,32,1,0,0,0,-1--1--1,0-0-0' +
            ',@Malgun Gothic|12||0-0-0',  # normal variable with colors
            "10,2,Stock,497,237,40,20,3,3,0,0,0,0,0,0",  # stock
            '10,7,"rate-1",382,262,21,11,40,3,0,0,-1,0,0,0',  # normal variable
            '10,2,"var-n",235,332,27,11,8,2,0,3,-1,0,0,0,128-128-128,0-0-0,' +
            '|0||128-128-128',  # shadow variable
            "*Just another view",  # module definition
            "1,5,6,3,100,0,0,22,0,0,0,-1--1--1,,1|(341,243)|",  # arrow
            "This is a random comment."
        ]

        expected_var = [
            namespace['"var-n"'],
            namespace["Stock"],
            namespace['"rate-1"'],
            "",
            "",
            "",
            ""
        ]
        expected_mod = ["", "", "", "", "Just another view", "", ""]

        for num, line in enumerate(lines):
            res = parse_sketch_line(line.strip(), namespace)
            self.assertEqual(res["variable_name"], expected_var[num])
            self.assertEqual(res["view_name"], expected_mod[num])


class TestParse_private_functions(unittest.TestCase):
    def test__split_sketch_warning(self):
        import warnings
        from pysd.translation.vensim.vensim2py import _split_sketch

        model_str = "this is my model"

        with warnings.catch_warnings(record=True) as ws:
            text, sketch = _split_sketch(model_str)

            # use only user warnings
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 1)
            self.assertTrue(
                "Your model does not have a sketch." in str(wu[0].message))

            self.assertEqual(text, model_str)
            self.assertEqual(sketch, "")
