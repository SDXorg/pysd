import warnings
import shutil
import pytest

from pysd.tools.benchmarking import runner, assert_frames_close

# TODO add warnings catcher per test

vensim_test = {
    "abs": {
        "folder": "abs",
        "file": "test_abs.mdl"
    },
    "active_initial": {
        "folder": "active_initial",
        "file": "test_active_initial.mdl"
    },
    "active_initial_circular": {
        "folder": "active_initial_circular",
        "file": "test_active_initial_circular.mdl"
    },
    "allocate_available": {
        "folder": "allocate_available",
        "file": "test_allocate_available.mdl",
        "rtol": 2e-2
    },
    "allocate_by_priority": {
        "folder": "allocate_by_priority",
        "file": "test_allocate_by_priority.mdl"
    },
    "arithmetics": {
        "folder": "arithmetics",
        "file": "test_arithmetics.mdl"
    },
    "arithmetics_exp": {
        "folder": "arithmetics_exp",
        "file": "test_arithmetics_exp.mdl"
    },
    "arguments": {
        "folder": "arguments",
        "file": "test_arguments.mdl",
        "rtol": 1e-2  # TODO test why it is failing with smaller tolerance
    },
    "array_with_line_break": {
        "folder": "array_with_line_break",
        "file": "test_array_with_line_break.mdl"
    },
    "builtin_max": {
        "folder": "builtin_max",
        "file": "builtin_max.mdl"
    },
    "builtin_min": {
        "folder": "builtin_min",
        "file": "builtin_min.mdl"
    },
    "chained_initialization": {
        "folder": "chained_initialization",
        "file": "test_chained_initialization.mdl"
    },
    "conditional_subscripts": {
        "folder": "conditional_subscripts",
        "file": "test_conditional_subscripts.mdl"
    },
    "constant_expressions": {
        "folder": "constant_expressions",
        "file": "test_constant_expressions.mdl"
    },
    "control_vars": {
        "folder": "control_vars",
        "file": "test_control_vars.mdl"
    },
    "data_from_other_model": {
        "folder": "data_from_other_model",
        "file": "test_data_from_other_model.mdl",
        "data_files": "data.tab"
    },
    "delay_fixed": {
        "folder": "delay_fixed",
        "file": "test_delay_fixed.mdl"
    },
    "delay_numeric_error": {
        "folder": "delay_numeric_error",
        "file": "test_delay_numeric_error.mdl"
    },
    "delay_parentheses": {
        "folder": "delay_parentheses",
        "file": "test_delay_parentheses.mdl"
    },
    "delay_pipeline": {
        "folder": "delay_pipeline",
        "file": "test_pipeline_delays.mdl"
    },
    "delays": {
        "folder": "delays",
        "file": "test_delays.mdl"
    },
    "dynamic_final_time": {
        "folder": "dynamic_final_time",
        "file": "test_dynamic_final_time.mdl"
    },
    "elm_count": {
        "folder": "elm_count",
        "file": "test_elm_count.mdl"
    },
    "euler_step_vs_saveper": {
        "folder": "euler_step_vs_saveper",
        "file": "test_euler_step_vs_saveper.mdl"
    },
    "except": {
        "folder": "except",
        "file": "test_except.mdl"
    },
    "except_multiple": {
        "folder": "except_multiple",
        "file": "test_except_multiple.mdl"
    },
    "except_subranges": {
        "folder": "except_subranges",
        "file": "test_except_subranges.mdl"
    },
    "exp": {
        "folder": "exp",
        "file": "test_exp.mdl"
    },
    "exponentiation": {
        "folder": "exponentiation",
        "file": "exponentiation.mdl"
    },
    "forecast": {
        "folder": "forecast",
        "file": "test_forecast.mdl"
    },
    "function_capitalization": {
        "folder": "function_capitalization",
        "file": "test_function_capitalization.mdl"
    },
    "fully_invalid_names": {
        "folder": "fully_invalid_names",
        "file": "test_fully_invalid_names.mdl"
    },
    "game": {
        "folder": "game",
        "file": "test_game.mdl"
    },
    "get_constants": pytest.param({
        "folder": "get_constants",
        "file": "test_get_constants.mdl"
    },  marks=pytest.mark.xfail(reason="csv files not implemented")),
    "get_constants_subranges": {
        "folder": "get_constants_subranges",
        "file": "test_get_constants_subranges.mdl"
    },
    "get_data": pytest.param({
        "folder": "get_data",
        "file": "test_get_data.mdl"
    },  marks=pytest.mark.xfail(reason="csv files not implemented")),
    "get_data_args_3d_xls": {
        "folder": "get_data_args_3d_xls",
        "file": "test_get_data_args_3d_xls.mdl"
    },
    "get_lookups_data_3d_xls": {
        "folder": "get_lookups_data_3d_xls",
        "file": "test_get_lookups_data_3d_xls.mdl"
    },
    "get_lookups_subscripted_args": {
        "folder": "get_lookups_subscripted_args",
        "file": "test_get_lookups_subscripted_args.mdl"
    },
    "get_lookups_subset": {
        "folder": "get_lookups_subset",
        "file": "test_get_lookups_subset.mdl"
    },
    "get_mixed_definitions": {
        "folder": "get_mixed_definitions",
        "file": "test_get_mixed_definitions.mdl"
    },
    "get_subscript_3d_arrays_xls": {
        "folder": "get_subscript_3d_arrays_xls",
        "file": "test_get_subscript_3d_arrays_xls.mdl"
    },
    "get_time_value": {
        "folder": "get_time_value",
        "file": "test_get_time_value_simple.mdl"
    },
    "get_values_order": {
        "folder": "get_values_order",
        "file": "test_get_values_order.mdl"
    },
    "get_with_missing_values_xlsx": {
        "folder": "get_with_missing_values_xlsx",
        "file": "test_get_with_missing_values_xlsx.mdl"
    },
    "get_xls_cellrange": {
        "folder": "get_xls_cellrange",
        "file": "test_get_xls_cellrange.mdl"
    },
    "if_stmt": {
        "folder": "if_stmt",
        "file": "if_stmt.mdl"
    },
    "initial_function": {
        "folder": "initial_function",
        "file": "test_initial.mdl"
    },
    "input_functions": {
        "folder": "input_functions",
        "file": "test_inputs.mdl"
    },
    "invert_matrix": {
        "folder": "invert_matrix",
        "file": "test_invert_matrix.mdl"
    },
    "limits": {
        "folder": "limits",
        "file": "test_limits.mdl"
    },
    "line_breaks": {
        "folder": "line_breaks",
        "file": "test_line_breaks.mdl"
    },
    "line_continuation": {
        "folder": "line_continuation",
        "file": "test_line_continuation.mdl"
    },
    "ln": {
        "folder": "ln",
        "file": "test_ln.mdl"
    },
    "log": {
        "folder": "log",
        "file": "test_log.mdl"
    },
    "logicals": {
        "folder": "logicals",
        "file": "test_logicals.mdl"
    },
    "lookups": {
        "folder": "lookups",
        "file": "test_lookups.mdl"
    },
    "lookups_funcnames": {
        "folder": "lookups_funcnames",
        "file": "test_lookups_funcnames.mdl"
    },
    "lookups_inline": {
        "folder": "lookups_inline",
        "file": "test_lookups_inline.mdl"
    },
    "lookups_inline_bounded": {
        "folder": "lookups_inline_bounded",
        "file": "test_lookups_inline_bounded.mdl"
    },
    "lookups_inline_spaces": {
        "folder": "lookups_inline_spaces",
        "file": "test_lookups_inline_spaces.mdl"
    },
    "lookups_with_expr": {
        "folder": "lookups_with_expr",
        "file": "test_lookups_with_expr.mdl"
    },
    "lookups_without_range": {
        "folder": "lookups_without_range",
        "file": "test_lookups_without_range.mdl"
    },
    "macro_cross_reference": {
        "folder": "macro_cross_reference",
        "file": "test_macro_cross_reference.mdl"
    },
    "macro_expression": {
        "folder": "macro_expression",
        "file": "test_macro_expression.mdl"
    },
    "macro_multi_expression": {
        "folder": "macro_multi_expression",
        "file": "test_macro_multi_expression.mdl"
    },
    "macro_multi_macros": {
        "folder": "macro_multi_macros",
        "file": "test_macro_multi_macros.mdl"
    },
    "macro_stock": {
        "folder": "macro_stock",
        "file": "test_macro_stock.mdl"
    },
    "macro_trailing_definition": {
        "folder": "macro_trailing_definition",
        "file": "test_macro_trailing_definition.mdl"
    },
    "model_doc": {
        "folder": "model_doc",
        "file": "model_doc.mdl"
    },
    "multiple_lines_def": {
        "folder": "multiple_lines_def",
        "file": "test_multiple_lines_def.mdl"
    },
    "na": {
        "folder": "na",
        "file": "test_na.mdl"
    },
    "nested_functions": {
        "folder": "nested_functions",
        "file": "test_nested_functions.mdl"
    },
    "number_handling": {
        "folder": "number_handling",
        "file": "test_number_handling.mdl"
    },
    "odd_number_quotes": {
        "folder": "odd_number_quotes",
        "file": "teacup_3quotes.mdl"
    },
    "parentheses": {
        "folder": "parentheses",
        "file": "test_parens.mdl"
    },
    "partial_range_definitions": {
        "folder": "partial_range_definitions",
        "file": "test_partial_range_definitions.mdl"
    },
    "reality_checks": {
        "folder": "reality_checks",
        "file": "test_reality_checks.mdl"
    },
    "reference_capitalization": {
        "folder": "reference_capitalization",
        "file": "test_reference_capitalization.mdl"
    },
    "repeated_subscript": {
        "folder": "repeated_subscript",
        "file": "test_repeated_subscript.mdl"
    },
    "rounding": {
        "folder": "rounding",
        "file": "test_rounding.mdl"
    },
    "sample_if_true": {
        "folder": "sample_if_true",
        "file": "test_sample_if_true.mdl"
    },
    "smaller_range": {
        "folder": "smaller_range",
        "file": "test_smaller_range.mdl"
    },
    "smooth": {
        "folder": "smooth",
        "file": "test_smooth.mdl"
    },
    "smooth_and_stock": {
        "folder": "smooth_and_stock",
        "file": "test_smooth_and_stock.mdl"
    },
    "special_characters": {
        "folder": "special_characters",
        "file": "test_special_variable_names.mdl"
    },
    "sqrt": {
        "folder": "sqrt",
        "file": "test_sqrt.mdl"
    },
    "subrange_merge": {
        "folder": "subrange_merge",
        "file": "test_subrange_merge.mdl"
    },
    "subscript_1d_arrays": {
        "folder": "subscript_1d_arrays",
        "file": "test_subscript_1d_arrays.mdl"
    },
    "subscript_2d_arrays": {
        "folder": "subscript_2d_arrays",
        "file": "test_subscript_2d_arrays.mdl"
    },
    "subscript_3d_arrays": {
        "folder": "subscript_3d_arrays",
        "file": "test_subscript_3d_arrays.mdl"
    },
    "subscript_3d_arrays_lengthwise": {
        "folder": "subscript_3d_arrays_lengthwise",
        "file": "test_subscript_3d_arrays_lengthwise.mdl"
    },
    "subscript_3d_arrays_widthwise": {
        "folder": "subscript_3d_arrays_widthwise",
        "file": "test_subscript_3d_arrays_widthwise.mdl"
    },
    "subscript_aggregation": {
        "folder": "subscript_aggregation",
        "file": "test_subscript_aggregation.mdl"
    },
    "subscript_constant_call": {
        "folder": "subscript_constant_call",
        "file": "test_subscript_constant_call.mdl"
    },
    "subscript_copy": {
        "folder": "subscript_copy",
        "file": "test_subscript_copy.mdl"
    },
    "subscript_copy2": {
        "folder": "subscript_copy",
        "file": "test_subscript_copy2.mdl"
    },
    "subscript_definition": {
        "folder": "subscript_definition",
        "file": "test_subscript_definition.mdl"
    },
    "subscript_docs": {
        "folder": "subscript_docs",
        "file": "subscript_docs.mdl"
    },
    "subscript_element_name": {
        "folder": "subscript_element_name",
        "file": "test_subscript_element_name.mdl"
    },
    "subscript_individually_defined_1_of_2d_arrays": {
        "folder": "subscript_individually_defined_1_of_2d_arrays",
        "file": "subscript_individually_defined_1_of_2d_arrays.mdl"
    },
    "subscript_individually_defined_1_of_2d_arrays_from_floats": {
        "folder": "subscript_individually_defined_1_of_2d_arrays_from_floats",
        "file": "subscript_individually_defined_1_of_2d_arrays_from_floats.mdl"
    },
    "subscript_individually_defined_1d_arrays": {
        "folder": "subscript_individually_defined_1d_arrays",
        "file": "subscript_individually_defined_1d_arrays.mdl"
    },
    "subscript_individually_defined_stocks": {
        "folder": "subscript_individually_defined_stocks",
        "file": "test_subscript_individually_defined_stocks.mdl"
    },
    "subscript_logicals": {
        "folder": "subscript_logicals",
        "file": "test_subscript_logicals.mdl"
    },
    "subscript_mapping_simple": {
        "folder": "subscript_mapping_simple",
        "file": "test_subscript_mapping_simple.mdl"
    },
    "subscript_mapping_vensim": {
        "folder": "subscript_mapping_vensim",
        "file": "test_subscript_mapping_vensim.mdl"
    },
    "subscript_mixed_assembly": {
        "folder": "subscript_mixed_assembly",
        "file": "test_subscript_mixed_assembly.mdl"
    },
    "subscript_multiples": {
        "folder": "subscript_multiples",
        "file": "test_multiple_subscripts.mdl"
    },
    "subscript_numeric_range": {
        "folder": "subscript_numeric_range",
        "file": "test_subscript_numeric_range.mdl"
    },
    "subscript_selection": {
        "folder": "subscript_selection",
        "file": "subscript_selection.mdl"
    },
    "subscript_subranges": {
        "folder": "subscript_subranges",
        "file": "test_subscript_subrange.mdl"
    },
    "subscript_subranges_equal": {
        "folder": "subscript_subranges_equal",
        "file": "test_subscript_subrange_equal.mdl"
    },
    "subscript_switching": {
        "folder": "subscript_switching",
        "file": "subscript_switching.mdl"
    },
    "subscript_transposition": {
        "folder": "subscript_transposition",
        "file": "test_subscript_transposition.mdl"
    },
    "subscript_updimensioning": {
        "folder": "subscript_updimensioning",
        "file": "test_subscript_updimensioning.mdl"
    },
    "subscripted_delays": {
        "folder": "subscripted_delays",
        "file": "test_subscripted_delays.mdl"
    },
    "subscripted_flows": {
        "folder": "subscripted_flows",
        "file": "test_subscripted_flows.mdl"
    },
    "subscripted_if_then_else": {
        "folder": "subscripted_if_then_else",
        "file": "test_subscripted_if_then_else.mdl"
    },
    "subscripted_logicals": {
        "folder": "subscripted_logicals",
        "file": "test_subscripted_logicals.mdl"
    },
    "subscripted_lookups": {
        "folder": "subscripted_lookups",
        "file": "test_subscripted_lookups.mdl"
    },
    "subscripted_ramp_step": {
        "folder": "subscripted_ramp_step",
        "file": "test_subscripted_ramp_step.mdl"
    },
    "subscripted_round": {
        "folder": "subscripted_round",
        "file": "test_subscripted_round.mdl"
    },
    "subscripted_smooth": {
        "folder": "subscripted_smooth",
        "file": "test_subscripted_smooth.mdl"
    },
    "subscripted_trend": {
        "folder": "subscripted_trend",
        "file": "test_subscripted_trend.mdl"
    },
    "subscripted_trig": {
        "folder": "subscripted_trig",
        "file": "test_subscripted_trig.mdl"
    },
    "subscripted_xidz": {
        "folder": "subscripted_xidz",
        "file": "test_subscripted_xidz.mdl"
    },
    "subset_duplicated_coord": {
        "folder": "subset_duplicated_coord",
        "file": "test_subset_duplicated_coord.mdl"
    },
    "tabbed_arrays": {
        "folder": "tabbed_arrays",
        "file": "tabbed_arrays.mdl"
    },
    "time": {
        "folder": "time",
        "file": "test_time.mdl"
    },
    "trend": {
        "folder": "trend",
        "file": "test_trend.mdl"
    },
    "trig": {
        "folder": "trig",
        "file": "test_trig.mdl"
    },
    "unchangeable_constant": {
        "folder": "unchangeable_constant",
        "file": "test_unchangeable_constant.mdl"
    },
    "unicode_characters": {
        "folder": "unicode_characters",
        "file": "unicode_test_model.mdl"
    },
    "variable_ranges": {
        "folder": "variable_ranges",
        "file": "test_variable_ranges.mdl"
    },
    "vector_order": {
        "folder": "vector_order",
        "file": "test_vector_order.mdl"
    },
    "vector_select": {
        "folder": "vector_select",
        "file": "test_vector_select.mdl"
    },
    "with_lookup": {
        "folder": "with_lookup",
        "file": "test_with_lookup.mdl"
    },
    "xidz_zidz": {
        "folder": "xidz_zidz",
        "file": "xidz_zidz.mdl"
    },
    "zeroled_decimals": {
        "folder": "zeroled_decimals",
        "file": "test_zeroled_decimals.mdl"
    }
}


@pytest.mark.parametrize(
    "test_data",
    [item for item in vensim_test.values()],
    ids=list(vensim_test)
)
class TestIntegrateVensim:
    """
    Test for splitting Vensim views in modules and submodules
    """
    @pytest.fixture
    def test_folder(self, tmp_path, _test_models, test_data):
        """
        Copy test folder to a temporary folder therefore we avoid creating
        PySD model files in the original folder
        """
        test_folder = tmp_path.joinpath(test_data["folder"])
        shutil.copytree(
            _test_models.joinpath(test_data["folder"]),
            test_folder
        )
        return test_folder

    @pytest.fixture
    def model_path(self, test_folder, test_data):
        """Return model path"""
        return test_folder.joinpath(test_data["file"])

    @pytest.fixture
    def data_path(self, test_folder, test_data):
        """Fixture for models with data_path"""
        if "data_files" in test_data:
            if isinstance(test_data["data_files"], str):
                return test_folder.joinpath(test_data["data_files"])
            elif isinstance(test_data["data_files"], list):
                return [
                    test_folder.joinpath(file)
                    for file in test_data["data_files"]
                ]
            else:
                return {
                    test_folder.joinpath(file): values
                    for file, values in test_data["data_files"].items()
                }
        else:
            return None

    @pytest.fixture
    def kwargs(self, test_data):
        """Fixture for atol and rtol"""
        kwargs = {}
        if "atol" in test_data:
            kwargs["atol"] = test_data["atol"]
        if "rtol" in test_data:
            kwargs["rtol"] = test_data["rtol"]
        return kwargs

    def test_read_vensim_file(self, model_path, data_path, kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output, canon = runner(model_path, data_files=data_path)
        assert_frames_close(output, canon, verbose=True, **kwargs)
