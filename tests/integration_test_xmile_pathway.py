
import os
import unittest
from pysd.tools.benchmarking import runner, assert_frames_close

rtol = .05

_root = os.path.dirname(__file__)
test_models = os.path.join(_root, "test-models/tests")


class TestIntegrationExamples(unittest.TestCase):

    def test_abs(self):
        output, canon = runner(test_models + '/abs/test_abs.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('error in model file')
    def test_active_initial(self):
        output, canon = runner(test_models + '/active_initial/test_active_initial.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing model file')
    def test_arguments(self):
        output, canon = runner(test_models + '/arguments/test_arguments.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    def test_builtin_max(self):
        output, canon = runner(test_models + '/builtin_max/builtin_max.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_builtin_min(self):
        output, canon = runner(test_models + '/builtin_min/builtin_min.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_chained_initialization(self):
        output, canon = runner(
            test_models + '/chained_initialization/test_chained_initialization.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_comparisons(self):
        output, canon = runner(
            test_models + '/comparisons/comparisons.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_constant_expressions(self):
        output, canon = runner(
            test_models + '/constant_expressions/test_constant_expressions.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_delay_parentheses(self):
        output, canon = runner(
            test_models + '/delay_parentheses/test_delay_parentheses.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_delays(self):
        output, canon = runner(test_models + '/delays/test_delays.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_euler_step_vs_saveper(self):
        output, canon = runner(
            test_models + '/euler_step_vs_saveper/test_euler_step_vs_saveper.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_eval_order(self):
        output, canon = runner(
            test_models + '/eval_order/eval_order.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_exp(self):
        output, canon = runner(test_models + '/exp/test_exp.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_exponentiation(self):
        output, canon = runner(test_models + '/exponentiation/exponentiation.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_function_capitalization(self):
        output, canon = runner(
            test_models + '/function_capitalization/test_function_capitalization.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('not sure if this is implemented in xmile?')
    def test_game(self):
        output, canon = runner(test_models + '/game/test_game.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_if_stmt(self):
        output, canon = runner(test_models + '/if_stmt/if_stmt.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_initial_function(self):
        output, canon = runner(test_models + '/initial_function/test_initial.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile model')
    def test_input_functions(self):
        output, canon = runner(test_models + '/input_functions/test_inputs.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    def test_limits(self):
        output, canon = runner(test_models + '/limits/test_limits.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_line_breaks(self):
        output, canon = runner(test_models + '/line_breaks/test_line_breaks.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_line_continuation(self):
        output, canon = runner(test_models + '/line_continuation/test_line_continuation.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_ln(self):
        output, canon = runner(test_models + '/ln/test_ln.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_log(self):
        output, canon = runner(test_models + '/log/test_log.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_logicals(self):
        output, canon = runner(test_models + '/logicals/test_logicals.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_lookups(self):
        output, canon = runner(test_models + '/lookups/test_lookups.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_lookups_xscale(self):
        output, canon = runner(test_models + '/lookups/test_lookups_xscale.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_lookups_xpts_sep(self):
        output, canon = runner(test_models + '/lookups/test_lookups_xpts_sep.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_lookups_ypts_sep(self):
        output, canon = runner(test_models + '/lookups/test_lookups_ypts_sep.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_lookups_funcnames(self):
        output, canon = runner(test_models + '/lookups_funcnames/test_lookups_funcnames.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    def test_lookups_inline(self):
        output, canon = runner(test_models + '/lookups_inline/test_lookups_inline.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_lookups_inline_bounded(self):
        output, canon = runner(
            test_models + '/lookups_inline_bounded/test_lookups_inline_bounded.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_macro_cross_reference(self):
        output, canon = runner(test_models + '/macro_cross_reference/test_macro_cross_reference.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_macro_expression(self):
        output, canon = runner(test_models + '/macro_expression/test_macro_expression.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_macro_multi_expression(self):
        output, canon = runner(
            test_models + '/macro_multi_expression/test_macro_multi_expression.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_macro_multi_macros(self):
        output, canon = runner(
            test_models + '/macro_multi_macros/test_macro_multi_macros.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_macro_output(self):
        output, canon = runner(test_models + '/macro_output/test_macro_output.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_macro_stock(self):
        output, canon = runner(test_models + '/macro_stock/test_macro_stock.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('do we need this?')
    def test_macro_trailing_definition(self):
        output, canon = runner(test_models + '/macro_trailing_definition/test_macro_trailing_definition.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    def test_model_doc(self):
        output, canon = runner(test_models + '/model_doc/model_doc.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_number_handling(self):
        output, canon = runner(test_models + '/number_handling/test_number_handling.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_parentheses(self):
        output, canon = runner(test_models + '/parentheses/test_parens.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_reference_capitalization(self):
        """A properly formatted Vensim model should never create this failure"""
        output, canon = runner(
            test_models + '/reference_capitalization/test_reference_capitalization.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('in branch')
    def test_rounding(self):
        output, canon = runner(test_models + '/rounding/test_rounding.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_smooth(self):
        output, canon = runner(test_models + '/smooth/test_smooth.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_smooth_and_stock(self):
        output, canon = runner(test_models + '/smooth_and_stock/test_smooth_and_stock.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_special_characters(self):
        output, canon = runner(
            test_models + '/special_characters/test_special_variable_names.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_sqrt(self):
        output, canon = runner(test_models + '/sqrt/test_sqrt.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_subscript_multiples(self):
        output, canon = runner(
            test_models + '/subscript multiples/test_multiple_subscripts.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_subscript_1d_arrays(self):
        output, canon = runner(
            test_models + '/subscript_1d_arrays/test_subscript_1d_arrays.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_2d_arrays(self):
        output, canon = runner(
            test_models + '/subscript_2d_arrays/test_subscript_2d_arrays.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_3d_arrays(self):
        output, canon = runner(test_models + '/subscript_3d_arrays/test_subscript_3d_arrays.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_3d_arrays_lengthwise(self):
        output, canon = runner(test_models + '/subscript_3d_arrays_lengthwise/test_subscript_3d_arrays_lengthwise.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_3d_arrays_widthwise(self):
        output, canon = runner(test_models + '/subscript_3d_arrays_widthwise/test_subscript_3d_arrays_widthwise.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('in branch')
    def test_subscript_aggregation(self):
        output, canon = runner(test_models + '/subscript_aggregation/test_subscript_aggregation.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_subscript_constant_call(self):
        output, canon = runner(
            test_models + '/subscript_constant_call/test_subscript_constant_call.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_docs(self):
        output, canon = runner(test_models + '/subscript_docs/subscript_docs.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_individually_defined_1_of_2d_arrays(self):
        output, canon = runner(test_models + '/subscript_individually_defined_1_of_2d_arrays/subscript_individually_defined_1_of_2d_arrays.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_individually_defined_1_of_2d_arrays_from_floats(self):
        output, canon = runner(test_models + '/subscript_individually_defined_1_of_2d_arrays_from_floats/subscript_individually_defined_1_of_2d_arrays_from_floats.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_individually_defined_1d_arrays(self):
        output, canon = runner(test_models + '/subscript_individually_defined_1d_arrays/subscript_individually_defined_1d_arrays.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_individually_defined_stocks(self):
        output, canon = runner(test_models + '/subscript_individually_defined_stocks/test_subscript_individually_defined_stocks.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_mixed_assembly(self):
        output, canon = runner(test_models + '/subscript_mixed_assembly/test_subscript_mixed_assembly.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_selection(self):
        output, canon = runner(test_models + '/subscript_selection/subscript_selection.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_subscript_subranges(self):
        output, canon = runner(
            test_models + '/subscript_subranges/test_subscript_subrange.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_subscript_subranges_equal(self):
        output, canon = runner(
            test_models + '/subscript_subranges_equal/test_subscript_subrange_equal.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_switching(self):
        output, canon = runner(test_models + '/subscript_switching/subscript_switching.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_subscript_updimensioning(self):
        output, canon = runner(
            test_models + '/subscript_updimensioning/test_subscript_updimensioning.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscripted_delays(self):
        output, canon = runner(test_models + '/subscripted_delays/test_subscripted_delays.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscripted_flows(self):
        output, canon = runner(test_models + '/subscripted_flows/test_subscripted_flows.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_time(self):
        output, canon = runner(test_models + '/time/test_time.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    def test_trig(self):
        output, canon = runner(test_models + '/trig/test_trig.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_trend(self):
        output, canon = runner(test_models + '/trend/test_trend.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_xidz_zidz(self):
        output, canon = runner(test_models + '/xidz_zidz/xidz_zidz.xmile')
        assert_frames_close(output, canon, rtol=rtol)


