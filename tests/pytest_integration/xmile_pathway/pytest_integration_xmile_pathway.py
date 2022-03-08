import pytest
import shutil
from pysd.tools.benchmarking import runner, assert_frames_close

# TODO add warnings catcher per test


xmile_test = {
    "abs": {
        "folder": "abs",
        "file": "test_abs.xmile"
    },
    "active_initial": pytest.param({
        "folder": "active_initial",
        "file": "test_active_initial.xmile"
    },  marks=pytest.mark.xfail(reason="failing originally")),
    "builtin_max": {
        "folder": "builtin_max",
        "file": "builtin_max.xmile"
    },
    "builtin_min": {
        "folder": "builtin_min",
        "file": "builtin_min.xmile"
    },
    "chained_initialization": {
        "folder": "chained_initialization",
        "file": "test_chained_initialization.xmile"
    },
    "comparisons": {
        "folder": "comparisons",
        "file": "comparisons.xmile"
    },
    "constant_expressions": {
        "folder": "constant_expressions",
        "file": "test_constant_expressions.xmile"
    },
    "euler_step_vs_saveper": pytest.param({
        "folder": "euler_step_vs_saveper",
        "file": "test_euler_step_vs_saveper.xmile"
    },  marks=pytest.mark.xfail(reason="failing originally")),
    "eval_order": {
        "folder": "eval_order",
        "file": "eval_order.xmile"
    },
    "exp": {
        "folder": "exp",
        "file": "test_exp.xmile"
    },
    "exponentiation": {
        "folder": "exponentiation",
        "file": "exponentiation.xmile"
    },
    "function_capitalization": {
        "folder": "function_capitalization",
        "file": "test_function_capitalization.xmile"
    },
    "game": {
        "folder": "game",
        "file": "test_game.xmile"
    },
    "if_stmt": {
        "folder": "if_stmt",
        "file": "if_stmt.xmile"
    },
    "initial_function": {
        "folder": "initial_function",
        "file": "test_initial.xmile"
    },
    "limits": {
        "folder": "limits",
        "file": "test_limits.xmile"
    },
    "line_breaks": {
        "folder": "line_breaks",
        "file": "test_line_breaks.xmile"
    },
    "line_continuation": {
        "folder": "line_continuation",
        "file": "test_line_continuation.xmile"
    },
    "ln": {
        "folder": "ln",
        "file": "test_ln.xmile"
    },
    "log": {
        "folder": "log",
        "file": "test_log.xmile"
    },
    "logicals": {
        "folder": "logicals",
        "file": "test_logicals.xmile"
    },
    "lookups": {
        "folder": "lookups",
        "file": "test_lookups.xmile"
    },
    "lookups_no-indirect": pytest.param({
        "folder": "lookups",
        "file": "test_lookups_no-indirect.xmile"
    },  marks=pytest.mark.xfail(reason="failing originally")),
    "lookups_xpts_sep": {
        "folder": "lookups",
        "file": "test_lookups_xpts_sep.xmile"
    },
    "lookups_xscale": {
        "folder": "lookups",
        "file": "test_lookups_xscale.xmile"
    },
    "lookups_ypts_sep": {
        "folder": "lookups",
        "file": "test_lookups_ypts_sep.xmile"
    },
    "lookups_inline": pytest.param({
        "folder": "lookups_inline",
        "file": "test_lookups_inline.xmile"
    },  marks=pytest.mark.xfail(reason="failing originally")),
    "macro_expression": pytest.param({
        "folder": "macro_expression",
        "file": "test_macro_expression.xmile"
    },  marks=pytest.mark.xfail(reason="failing originally")),
    "macro_multi_expression": pytest.param({
        "folder": "macro_multi_expression",
        "file": "test_macro_multi_expression.xmile"
    },  marks=pytest.mark.xfail(reason="failing originally")),
    "macro_multi_macros":  pytest.param({
        "folder": "macro_multi_macros",
        "file": "test_macro_multi_macros.xmile"
    },  marks=pytest.mark.xfail(reason="failing originally")),
    "macro_stock":  pytest.param({
        "folder": "macro_stock",
        "file": "test_macro_stock.xmile"
    },  marks=pytest.mark.xfail(reason="failing originally")),
    "model_doc": {
        "folder": "model_doc",
        "file": "model_doc.xmile"
    },
    "number_handling": {
        "folder": "number_handling",
        "file": "test_number_handling.xmile"
    },
    "parentheses": {
        "folder": "parentheses",
        "file": "test_parens.xmile"
    },
    "reference_capitalization": {
        "folder": "reference_capitalization",
        "file": "test_reference_capitalization.xmile"
    },
    "smooth_and_stock": pytest.param({
        "folder": "smooth_and_stock",
        "file": "test_smooth_and_stock.xmile"
    },  marks=pytest.mark.xfail(reason="failing originally")),
    "special_characters":  pytest.param({
        "folder": "special_characters",
        "file": "test_special_variable_names.xmile"
    },  marks=pytest.mark.xfail(reason="failing originally")),
    "sqrt": {
        "folder": "sqrt",
        "file": "test_sqrt.xmile"
    },
    "subscript_1d_arrays": {
        "folder": "subscript_1d_arrays",
        "file": "test_subscript_1d_arrays.xmile"
    },
    "subscript_constant_call": {
        "folder": "subscript_constant_call",
        "file": "test_subscript_constant_call.xmile"
    },
    "subscript_individually_defined_1d_arrays": {
        "folder": "subscript_individually_defined_1d_arrays",
        "file": "subscript_individually_defined_1d_arrays.xmile"
    },
    "subscript_mixed_assembly": {
        "folder": "subscript_mixed_assembly",
        "file": "test_subscript_mixed_assembly.xmile"
    },
    "subscript_multiples": {
        "folder": "subscript_multiples",
        "file": "test_multiple_subscripts.xmile"
    },
    "subscript_subranges": {
        "folder": "subscript_subranges",
        "file": "test_subscript_subrange.xmile"
    },
    "subscript_subranges_equal": {
        "folder": "subscript_subranges_equal",
        "file": "test_subscript_subrange_equal.xmile"
    },
    "subscript_updimensioning": {
        "folder": "subscript_updimensioning",
        "file": "test_subscript_updimensioning.xmile"
    },
    "subscripted_flows": {
        "folder": "subscripted_flows",
        "file": "test_subscripted_flows.xmile"
    },
    "trig": {
        "folder": "trig",
        "file": "test_trig.xmile"
    },
    "xidz_zidz": {
        "folder": "xidz_zidz",
        "file": "xidz_zidz.xmile"
    },
}


@pytest.mark.parametrize(
    "test_data",
    [item for item in xmile_test.values()],
    ids=list(xmile_test)
)
class TestIntegrateXmile:
    """
    Test for full translation and integration of models
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
    def kwargs(self, test_data):
        """Fixture for atol and rtol"""
        kwargs = {}
        if "atol" in test_data:
            kwargs["atol"] = test_data["atol"]
        if "rtol" in test_data:
            kwargs["rtol"] = test_data["rtol"]
        return kwargs

    def test_read_vensim_file(self, model_path, kwargs):
        output, canon = runner(model_path)
        assert_frames_close(output, canon, **kwargs)
