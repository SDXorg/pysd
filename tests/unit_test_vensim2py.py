import unittest
import xarray as xr


class TestGetFileSections(unittest.TestCase):
    def test_normal_load(self):
        """normal model file with no macros"""
        from pysd.py_backend.vensim.vensim2py import get_file_sections

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
        from pysd.py_backend.vensim.vensim2py import get_file_sections

        actual = get_file_sections(":MACRO: MAC(z) a~b~c| :END OF MACRO:")
        expected = [{"returns": [], "params": ["z"], "name": "MAC", "string": "a~b~c|"}]
        self.assertEqual(actual, expected)

    def test_macro_and_model(self):
        """ basic macro and model """
        from pysd.py_backend.vensim.vensim2py import get_file_sections

        actual = get_file_sections(":MACRO: MAC(z) a~b~c| :END OF MACRO: d~e~f| g~h~i|")
        expected = [
            {"returns": [], "params": ["z"], "name": "MAC", "string": "a~b~c|"},
            {"returns": [], "params": [], "name": "_main_", "string": "d~e~f| g~h~i|"},
        ]
        self.assertEqual(actual, expected)

    def test_macro_multiple_inputs(self):
        """ macro with multiple input parameters """
        from pysd.py_backend.vensim.vensim2py import get_file_sections

        actual = get_file_sections(
            ":MACRO: MAC(z, y) a~b~c| :END OF MACRO: d~e~f| g~h~i|"
        )
        expected = [
            {"returns": [], "params": ["z", "y"], "name": "MAC", "string": "a~b~c|"},
            {"returns": [], "params": [], "name": "_main_", "string": "d~e~f| g~h~i|"},
        ]
        self.assertEqual(actual, expected)

    def test_macro_with_returns(self):
        """ macro with return values """
        from pysd.py_backend.vensim.vensim2py import get_file_sections

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
            {"returns": [], "params": [], "name": "_main_", "string": "d~e~f| g~h~i|"},
        ]
        self.assertEqual(actual, expected)

    def test_handle_encoding(self):
        """ Handle encoding """
        from pysd.py_backend.vensim.vensim2py import get_file_sections

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
        from pysd.py_backend.vensim.vensim2py import get_file_sections

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
        from pysd.py_backend.vensim.vensim2py import get_equation_components

        self.assertEqual(
            get_equation_components(r"constant = 25"),
            {
                "expr": "25",
                "kind": "component",
                "subs": [],
                "subs_compatibility": {},
                "real_name": "constant",
                "keyword": None,
            },
        )

    def test_equals_handling(self):
        """ Parse cases with equal signs within the expression """
        from pysd.py_backend.vensim.vensim2py import get_equation_components

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
        from pysd.py_backend.vensim.vensim2py import get_equation_components

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
        from pysd.py_backend.vensim.vensim2py import get_equation_components

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

    def test_subscript_references(self):
        from pysd.py_backend.vensim.vensim2py import get_equation_components

        self.assertEqual(
            get_equation_components(r"constant [Sub1, Sub2] = 10, 12; 14, 16;"),
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
            get_equation_components(r"function [Sub1] = other function[Sub1]"),
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
            get_equation_components(r'constant ["S1,b", "S1,c"] = 1, 2; 3, 4;'),
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
            get_equation_components(r'constant ["S1=b", "S1=c"] = 1, 2; 3, 4;'),
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
        from pysd.py_backend.vensim.vensim2py import get_equation_components

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
        from pysd.py_backend.vensim.vensim2py import parse_lookup_expression

        res = parse_lookup_expression(
            {
                "expr": r"(GET DIRECT LOOKUPS('path2excel.xlsx', "
                + r"'SheetName', 'index'\ , 'values'))",
                "py_name": "get_lookup",
                "subs": [],
            },
            {},
        )[1][0]

        self.assertEqual(
            res["py_expr"],
            "ExtLookup('path2excel.xlsx', 'SheetName', 'index', 'values', "
            + "{},\n          _root, '_ext_lookup_get_lookup')",
        )

    def test_pathological_names(self):
        from pysd.py_backend.vensim.vensim2py import get_equation_components

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
        from pysd.py_backend.vensim.vensim2py import get_equation_components

        defi = "NIF: NF<x-x>NF"
        try:
            get_equation_components(defi)
            self.assertFail()
        except ValueError as err:
            self.assertIn(
                "\nError when parsing definition:\n\t %s\n\n"
                "probably used definition is not integrated..."
                "\nSee parsimonious output above." % defi,
                err.args[0],
            )


class TestParse_general_expression(unittest.TestCase):
    def test_arithmetic(self):
        from pysd.py_backend.vensim.vensim2py import parse_general_expression

        res = parse_general_expression({"expr": "-10^3+4"})
        self.assertEqual(res[0]["py_expr"], "-10**3+4")

    def test_caps_handling(self):
        from pysd.py_backend.vensim.vensim2py import parse_general_expression

        res = parse_general_expression({"expr": "Abs(-3)"})
        self.assertEqual(res[0]["py_expr"], "abs(-3)")

        res = parse_general_expression({"expr": "ABS(-3)"})
        self.assertEqual(res[0]["py_expr"], "abs(-3)")

        res = parse_general_expression({"expr": "aBS(-3)"})
        self.assertEqual(res[0]["py_expr"], "abs(-3)")

    def test_function_calls(self):
        from pysd.py_backend.vensim.vensim2py import parse_general_expression

        res = parse_general_expression({"expr": "ABS(StockA)"}, {"StockA": "stocka"})
        self.assertEqual(res[0]["py_expr"], "abs(stocka())")

        res = parse_general_expression(
            {"expr": "If Then Else(A>B, 1, 0)"}, {"A": "a", "B": "b"}
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
        from pysd.py_backend.vensim.vensim2py import parse_general_expression

        res = parse_general_expression({"expr": "StockA"}, {"StockA": "stocka"})
        self.assertEqual(res[0]["py_expr"], "stocka()")

    def test_number_parsing(self):
        from pysd.py_backend.vensim.vensim2py import parse_general_expression

        res = parse_general_expression({"expr": "20"})
        self.assertEqual(res[0]["py_expr"], "20")

        res = parse_general_expression({"expr": "3.14159"})
        self.assertEqual(res[0]["py_expr"], "3.14159")

        res = parse_general_expression({"expr": "1.3e+10"})
        self.assertEqual(res[0]["py_expr"], "1.3e+10")

        res = parse_general_expression({"expr": "-1.3e-10"})
        self.assertEqual(res[0]["py_expr"], "-1.3e-10")

    def test_stock_construction_function_no_subscripts(self):
        """ stock construction should create a stateful variable and reference it """
        from pysd.py_backend.vensim.vensim2py import parse_general_expression
        from pysd.py_backend.functions import Integ

        res = parse_general_expression(
            {"expr": "INTEG (FlowA, -10)", "py_name": "test_stock", "subs": []},
            {"FlowA": "flowa"},
            elements_subs_dict={"test_stock": []},
        )

        self.assertEqual(res[1][0]["kind"], "stateful")
        a = eval(res[1][0]["py_expr"])
        self.assertIsInstance(a, Integ)

        # check the reference to that variable
        self.assertEqual(res[0]["py_expr"], res[1][0]["py_name"] + "()")

    def test_delay_construction_function_no_subscripts(self):
        from pysd.py_backend.vensim.vensim2py import parse_general_expression
        from pysd.py_backend.functions import Delay

        res = parse_general_expression(
            {
                "expr": "DELAY1(Variable, DelayTime)",
                "py_name": "test_delay",
                "subs": [],
            },
            {
                "Variable": "variable",
                "DelayTime": "delaytime",
                "TIME STEP": "time_step",
            },
            elements_subs_dict={"test_delay": {}},
        )

        def time_step():
            return 0.5

        self.assertEqual(res[1][0]["kind"], "stateful")
        a = eval(res[1][0]["py_expr"])
        self.assertIsInstance(a, Delay)

        # check the reference to that variable
        self.assertEqual(res[0]["py_expr"], res[1][0]["py_name"] + "()")

    def test_smooth_construction_function_no_subscripts(self):
        """ Tests translation of 'smooth'

        This translation should create a new stateful object to hold the delay
        elements, and then pass back a reference to that value
        """
        from pysd.py_backend.vensim.vensim2py import parse_general_expression
        from pysd.py_backend.functions import Smooth

        res = parse_general_expression(
            {
                "expr": "SMOOTH(Variable, DelayTime)",
                "py_name": "test_smooth",
                "subs": [],
            },
            {"Variable": "variable", "DelayTime": "delaytime"},
            elements_subs_dict={"test_smooth": []},
        )

        # check stateful object creation
        self.assertEqual(res[1][0]["kind"], "stateful")
        a = eval(res[1][0]["py_expr"])
        self.assertIsInstance(a, Smooth)

        # check the reference to that variable
        self.assertEqual(res[0]["py_expr"], res[1][0]["py_name"] + "()")

    def test_subscript_float_initialization(self):
        from pysd.py_backend.vensim.vensim2py import parse_general_expression

        _subscript_dict = {"Dim1": ["A", "B", "C"], "Dim2": ["D", "E"]}
        element = parse_general_expression(
            {"expr": "3.32", "subs": ["Dim1"]}, {}, _subscript_dict
        )
        string = element[0]["py_expr"]
        # TODO we should use a = eval(string)
        # hoewever eval is not detecting _subscript_dict variable
        self.assertEqual(
            string,
            "xr.DataArray(3.32,{dim: "
            + "_subscript_dict[dim] for dim in "
            + "['Dim1']},['Dim1'])",
        )
        a = xr.DataArray(
            3.32, {dim: _subscript_dict[dim] for dim in ["Dim1"]}, ["Dim1"]
        )
        self.assertDictEqual(
            {key: list(val.values) for key, val in a.coords.items()},
            {"Dim1": ["A", "B", "C"]},
        )
        self.assertEqual(a.loc[{"Dim1": "B"}], 3.32)

    def test_subscript_1d_constant(self):
        from pysd.py_backend.vensim.vensim2py import parse_general_expression

        _subscript_dict = {"Dim1": ["A", "B", "C"], "Dim2": ["D", "E"]}
        element = parse_general_expression(
            {"expr": "1, 2, 3", "subs": ["Dim1"]}, {}, _subscript_dict
        )
        string = element[0]["py_expr"]
        # TODO we should use a = eval(string)
        # hoewever eval is not detecting _subscript_dict variable
        self.assertTrue(
            string,
            "xr.DataArray([1.,2.,3.],"
            + "{dim: _subscript_dict[dim]"
            + " for dim in ['Dim1']}, ['Dim1'])",
        )
        a = xr.DataArray(
            [1.0, 2.0, 3.0], {dim: _subscript_dict[dim] for dim in ["Dim1"]}, ["Dim1"]
        )
        self.assertDictEqual(
            {key: list(val.values) for key, val in a.coords.items()},
            {"Dim1": ["A", "B", "C"]},
        )
        self.assertEqual(a.loc[{"Dim1": "A"}], 1)

    def test_subscript_2d_constant(self):
        from pysd.py_backend.vensim.vensim2py import parse_general_expression

        _subscript_dict = {"Dim1": ["A", "B", "C"], "Dim2": ["D", "E"]}
        element = parse_general_expression(
            {"expr": "1, 2; 3, 4; 5, 6;", "subs": ["Dim1", "Dim2"]}, {}, _subscript_dict
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
        from pysd.py_backend.vensim.vensim2py import parse_general_expression

        _subscript_dict = {"Dim1": ["A", "B", "C"], "Dim2": ["D", "E"]}
        element = parse_general_expression(
            {"expr": "1, 2; 3, 4; 5, 6;", "subs": ["Dim1", "Dim2"]}, {}, _subscript_dict
        )
        string = element[0]["py_expr"]
        a = eval(string)
        self.assertDictEqual(
            {key: list(val.values) for key, val in a.coords.items()},
            {"Dim1": ["A", "B", "C"], "Dim2": ["D", "E"]},
        )
        self.assertEqual(a.loc[{"Dim1": "A", "Dim2": "D"}], 1)
        self.assertEqual(a.loc[{"Dim1": "B", "Dim2": "E"}], 4)

    def test_subscript_reference(self):
        from pysd.py_backend.vensim.vensim2py import parse_general_expression

        res = parse_general_expression(
            {"expr": "Var A[Dim1, Dim2]"},
            {"Var A": "var_a"},
            {"Dim1": ["A", "B"], "Dim2": ["C", "D", "E"]},
            None,
            {"var_a": ["Dim1", "Dim2"]},
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

        res = parse_general_expression(
            {"expr": "Var C[Dim1, C, H]"},
            {"Var C": "var_c"},
            {"Dim1": ["A", "B"], "Dim2": ["C", "D", "E"], "Dim3": ["F", "G", "H", "I"]},
            None,
            {"var_c": ["Dim1", "Dim2", "Dim3"]},
        )

        self.assertEqual(
            res[0]["py_expr"],
            "rearrange(var_c().loc[:, 'C', 'H'].reset_coords(drop=True),"
            "['Dim1'],_subscript_dict)",
        )

    def test_subscript_ranges(self):
        from pysd.py_backend.vensim.vensim2py import parse_general_expression

        res = parse_general_expression(
            {"expr": "Var D[Range1]"},
            {"Var D": "var_c"},
            {"Dim1": ["A", "B", "C", "D", "E", "F"], "Range1": ["C", "D", "E"]},
            None,
            {"var_c": ["Dim1"]},
        )

        self.assertEqual(
            res[0]["py_expr"], "rearrange(var_c(),['Range1'],_subscript_dict)"
        )

    def test_incomplete_expression(self):
        from pysd.py_backend.vensim.vensim2py import parse_general_expression
        from warnings import catch_warnings

        with catch_warnings(record=True) as w:
            res = parse_general_expression(
                {
                    "expr": "A FUNCTION OF(Unspecified Eqn,Var A,Var B)",
                    "real_name": "Incomplete Func",
                    "py_name": "incomplete_func",
                },
                {
                    "Unspecified Eqn": "unspecified_eqn",
                    "Var A": "var_a",
                    "Var B": "var_b",
                },
                elements_subs_dict={"incomplete_func": []},
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(
                "Incomplete Func has no equation specified" in str(w[-1].message)
            )

        self.assertEqual(
            res[0]["py_expr"], "incomplete(unspecified_eqn(), var_a(), var_b())"
        )

    def test_parse_general_expression_error(self):
        from pysd.py_backend.vensim.vensim2py import parse_general_expression

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
    def test_parse_variable_name(self):
        from pysd.py_backend.vensim.vensim2py import parse_sketch_line

        namespace = {'"var-n"': "varn", "Stock": "stock", '"rate-1"': "rate1"}
        lines = [
            '10,1,"var-n",332,344,21,12,0,3,0,32,1,0,0,0,-1--1--1,0-0-0,@Malgun Gothic|12||0-0-0',  # normal variable with colors
            "10,2,Stock,497,237,40,20,3,3,0,0,0,0,0,0",  # stock
            '10,7,"rate-1",382,262,21,11,40,3,0,0,-1,0,0,0',  # normal variable
            '10,2,"var-n",235,332,27,11,8,2,0,3,-1,0,0,0,128-128-128,0-0-0,|0||128-128-128',  # shadow variable
            "*Just another view",  # module definition
            "1,5,6,3,100,0,0,22,0,0,0,-1--1--1,,1|(341,243)|",  # arrow
        ]

        expected_var = [
            namespace['"var-n"'],
            namespace["Stock"],
            namespace['"rate-1"'],
            "",
            "",
            "",
        ]
        expected_mod = ["", "", "", "", "Just another view", ""]

        for num, line in enumerate(lines):
            res = parse_sketch_line(line.strip(), namespace)
            self.assertEqual(res["variable_name"], expected_var[num])
            self.assertEqual(res["module_name"], expected_mod[num])

