

import unittest

rtol = .05


class TestIntegrationExamples(unittest.TestCase):

    def test_abs(self):
        from .test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/abs/test_abs.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('error in model file')
    def test_active_initial(self):
        from .test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/active_initial/test_active_initial.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing model file')
    def test_arguments(self):
        from .test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/arguments/test_arguments.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    def test_builtin_max(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/builtin_max/builtin_max.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_builtin_min(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/builtin_min/builtin_min.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_chained_initialization(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/chained_initialization/test_chained_initialization.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_comparisons(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/comparisons/comparisons.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_constant_expressions(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/constant_expressions/test_constant_expressions.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_delay_parentheses(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/delay_parentheses/test_delay_parentheses.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_delays(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/delays/test_delays.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_euler_step_vs_saveper(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/euler_step_vs_saveper/test_euler_step_vs_saveper.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_eval_order(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/eval_order/eval_order.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_exp(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/exp/test_exp.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_exponentiation(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/exponentiation/exponentiation.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_function_capitalization(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/function_capitalization/test_function_capitalization.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('not sure if this is implemented in xmile?')
    def test_game(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/game/test_game.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_if_stmt(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/if_stmt/if_stmt.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_initial_function(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/initial_function/test_initial.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile model')
    def test_input_functions(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/input_functions/test_inputs.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    def test_limits(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/limits/test_limits.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_line_breaks(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/line_breaks/test_line_breaks.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_line_continuation(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/line_continuation/test_line_continuation.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_ln(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/ln/test_ln.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_log(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/log/test_log.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_logicals(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/logicals/test_logicals.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_lookups(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/lookups/test_lookups.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_lookups_xscale(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/lookups/test_lookups_xscale.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_lookups_xpts_sep(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/lookups/test_lookups_xpts_sep.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_lookups_ypts_sep(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/lookups/test_lookups_ypts_sep.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_lookups_funcnames(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/lookups_funcnames/test_lookups_funcnames.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    def test_lookups_inline(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/lookups_inline/test_lookups_inline.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_lookups_inline_bounded(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/lookups_inline_bounded/test_lookups_inline_bounded.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_macro_cross_reference(self):
        from .test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/macro_cross_reference/test_macro_cross_reference.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_macro_expression(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/macro_expression/test_macro_expression.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_macro_multi_expression(self):
        from .test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/macro_multi_expression/test_macro_multi_expression.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_macro_multi_macros(self):
        from .test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/macro_multi_macros/test_macro_multi_macros.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_macro_output(self):
        from .test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/macro_output/test_macro_output.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_macro_stock(self):
        from .test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/macro_stock/test_macro_stock.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('do we need this?')
    def test_macro_trailing_definition(self):
        from .test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/macro_trailing_definition/test_macro_trailing_definition.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    def test_model_doc(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/model_doc/model_doc.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_number_handling(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/number_handling/test_number_handling.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_parentheses(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/parentheses/test_parens.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_reference_capitalization(self):
        """A properly formatted Vensim model should never create this failure"""
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/reference_capitalization/test_reference_capitalization.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('in branch')
    def test_rounding(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/rounding/test_rounding.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_smooth(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/smooth/test_smooth.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_smooth_and_stock(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/smooth_and_stock/test_smooth_and_stock.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_special_characters(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/special_characters/test_special_variable_names.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    def test_sqrt(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/sqrt/test_sqrt.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_subscript_multiples(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/subscript multiples/test_multiple_subscripts.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_subscript_1d_arrays(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/subscript_1d_arrays/test_subscript_1d_arrays.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_2d_arrays(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/subscript_2d_arrays/test_subscript_2d_arrays.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_3d_arrays(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_3d_arrays/test_subscript_3d_arrays.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_3d_arrays_lengthwise(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_3d_arrays_lengthwise/test_subscript_3d_arrays_lengthwise.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_3d_arrays_widthwise(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_3d_arrays_widthwise/test_subscript_3d_arrays_widthwise.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('in branch')
    def test_subscript_aggregation(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_aggregation/test_subscript_aggregation.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_subscript_constant_call(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/subscript_constant_call/test_subscript_constant_call.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_docs(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_docs/subscript_docs.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_individually_defined_1_of_2d_arrays(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_individually_defined_1_of_2d_arrays/subscript_individually_defined_1_of_2d_arrays.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_individually_defined_1_of_2d_arrays_from_floats(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_individually_defined_1_of_2d_arrays_from_floats/subscript_individually_defined_1_of_2d_arrays_from_floats.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_individually_defined_1d_arrays(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_individually_defined_1d_arrays/subscript_individually_defined_1d_arrays.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_individually_defined_stocks(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_individually_defined_stocks/test_subscript_individually_defined_stocks.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_mixed_assembly(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_mixed_assembly/test_subscript_mixed_assembly.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_selection(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_selection/subscript_selection.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_subscript_subranges(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/subscript_subranges/test_subscript_subrange.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_subscript_subranges_equal(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/subscript_subranges_equal/test_subscript_subrange_equal.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscript_switching(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_switching/subscript_switching.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('missing test model')
    def test_subscript_updimensioning(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner(
            'test-models/tests/subscript_updimensioning/test_subscript_updimensioning.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscripted_delays(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscripted_delays/test_subscripted_delays.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_subscripted_flows(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscripted_flows/test_subscripted_flows.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_time(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/time/test_time.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    def test_trig(self):
        from.test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/trig/test_trig.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_trend(self):
        from .test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/trend/test_trend.xmile')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('no xmile')
    def test_xidz_zidz(self):
        from .test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/xidz_zidz/xidz_zidz.xmile')
        assert_frames_close(output, canon, rtol=rtol)


