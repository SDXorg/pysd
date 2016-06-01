
from unittest import TestCase

rtol = .05


class TestIntegrationExamples(TestCase):

    def test_abs(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/abs/test_abs.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_active_initial(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/active_initial/test_active_initial.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_builtin_max(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/builtin_max/builtin_max.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_builtin_min(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/builtin_min/builtin_min.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_chained_initialization(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/chained_initialization/test_chained_initialization.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_delays(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/delays/test_delays.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_euler_step_vs_saveper(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/euler_step_vs_saveper/test_euler_step_vs_saveper.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_exp(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/exp/test_exp.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_exponentiation(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/exponentiation/exponentiation.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_function_capitalization(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/function_capitalization/test_function_capitalization.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_if_stmt(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/if_stmt/if_stmt.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_initial_function(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/initial_function/test_initial.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_input_functions(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/input_functions/test_inputs.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_line_continuation(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/line_continuation/test_line_continuation.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_ln(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/ln/test_ln.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_logicals(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/logicals/test_logicals.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_lookups(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/lookups/test_lookups.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_number_handling(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/number_handling/test_number_handling.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_rounding(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/rounding/test_rounding.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_smooth(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/smooth/test_smooth.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_special_characters(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/special_characters/test_special_variable_names.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_sqrt(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/sqrt/test_sqrt.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_subscript_multiples(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/subscript multiples/test_multiple_subscripts.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_subscript_1d_arrays(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/subscript_1d_arrays/test_subscript_1d_arrays.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_subscript_2d_arrays(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/subscript_2d_arrays/test_subscript_2d_arrays.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_subscript_3d_arrays(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/subscript_3d_arrays/test_subscript_3d_arrays.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_subscript_aggregation(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/subscript_aggregation/test_subscript_aggregation.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_subscript_docs(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/subscript_docs/subscript_docs.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_subscript_individually_defined_1_of_2d_arrays(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/subscript_individually_defined_1_of_2d_arrays/subscript_individually_defined_1_of_2d_arrays.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_subscript_individually_defined_1d_arrays(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/subscript_individually_defined_1d_arrays/subscript_individually_defined_1d_arrays.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_subscript_subranges(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/subscript_subranges/test_subscript_subrange.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_subscript_subranges_equal(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/subscript_subranges_equal/test_subscript_subrange_equal.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_subscripted_delays(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/subscripted_delays/test_subscripted_delays.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_subscripted_flows(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/subscripted_flows/test_subscripted_flows.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    
    def test_trig(self):
        from test_utils import runner, assertFramesClose
        output, canon = runner('test-models/tests/trig/test_trig.mdl')
        assertFramesClose(output, canon, rtol=rtol)
    

