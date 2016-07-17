import unittest


rtol = .05


class TestIntegrationExamples(unittest.TestCase):

    def test_abs(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/abs/test_abs.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('James Working')
    def test_active_initial(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/active_initial/test_active_initial.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_builtin_max(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/builtin_max/builtin_max.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_builtin_min(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/builtin_min/builtin_min.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_chained_initialization(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/chained_initialization/test_chained_initialization.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('Not Yet Implemented')
    def test_delays(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/delays/test_delays.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_euler_step_vs_saveper(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/euler_step_vs_saveper/test_euler_step_vs_saveper.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_exp(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/exp/test_exp.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('James Working')
    def test_exponentiation(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/exponentiation/exponentiation.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_function_capitalization(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/function_capitalization/test_function_capitalization.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_if_stmt(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/if_stmt/if_stmt.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('Not Yet Implemented')
    def test_initial_function(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/initial_function/test_initial.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('Not Yet Implemented')
    def test_input_functions(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/input_functions/test_inputs.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('James Working')
    def test_line_continuation(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/line_continuation/test_line_continuation.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('James Working')
    def test_ln(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/ln/test_ln.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_logicals(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/logicals/test_logicals.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('Not Yet Implemented')
    def test_lookups(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/lookups/test_lookups.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_number_handling(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/number_handling/test_number_handling.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('James Working')
    def test_rounding(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/rounding/test_rounding.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('Not Yet Implemented')
    def test_smooth(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/smooth/test_smooth.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('James Working')
    def test_special_characters(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/special_characters/test_special_variable_names.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_sqrt(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/sqrt/test_sqrt.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_subscript_multiples(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript multiples/test_multiple_subscripts.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_subscript_1d_arrays(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_1d_arrays/test_subscript_1d_arrays.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_subscript_2d_arrays(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_2d_arrays/test_subscript_2d_arrays.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('James Working')
    def test_subscript_3d_arrays(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_3d_arrays/test_subscript_3d_arrays.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('Not Yet Implemented')
    def test_subscript_aggregation(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_aggregation/test_subscript_aggregation.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('James Working')
    def test_subscript_docs(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_docs/subscript_docs.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_subscript_individually_defined_1_of_2d_arrays(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_individually_defined_1_of_2d_arrays/subscript_individually_defined_1_of_2d_arrays.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_subscript_individually_defined_1d_arrays(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_individually_defined_1d_arrays/subscript_individually_defined_1d_arrays.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_subscript_subranges(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_subranges/test_subscript_subrange.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('James Working')
    def test_subscript_subranges_equal(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscript_subranges_equal/test_subscript_subrange_equal.mdl')
        assert_frames_close(output, canon, rtol=rtol)

    @unittest.skip('Not Yet Implemented')
    def test_subscripted_delays(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscripted_delays/test_subscripted_delays.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_subscripted_flows(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/subscripted_flows/test_subscripted_flows.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    
    def test_trig(self):
        from test_utils import runner, assert_frames_close
        output, canon = runner('test-models/tests/trig/test_trig.mdl')
        assert_frames_close(output, canon, rtol=rtol)
    

