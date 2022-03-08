
import os
import unittest
from pysd.tools.benchmarking import runner, assert_frames_close

rtol = .05

_root = os.path.dirname(__file__)
test_models = os.path.join(_root, "test-models/tests")


class TestIntegrationExamples(unittest.TestCase):

    def test_abs(self):
        output, canon = runner(test_models + '/abs/test_abs.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_builtin_max(self):
        output, canon = runner(test_models + '/builtin_max/builtin_max.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_builtin_min(self):
        output, canon = runner(test_models + '/builtin_min/builtin_min.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_chained_initialization(self):
        output, canon = runner(
            test_models + '/chained_initialization/test_chained_initialization.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_comparisons(self):
        output, canon = runner(
            test_models + '/comparisons/comparisons.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_constant_expressions(self):
        output, canon = runner(
            test_models + '/constant_expressions/test_constant_expressions.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_eval_order(self):
        output, canon = runner(
            test_models + '/eval_order/eval_order.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_exp(self):
        output, canon = runner(test_models + '/exp/test_exp.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_exponentiation(self):
        output, canon = runner(test_models + '/exponentiation/exponentiation.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_function_capitalization(self):
        output, canon = runner(
            test_models + '/function_capitalization/test_function_capitalization.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_if_stmt(self):
        output, canon = runner(test_models + '/if_stmt/if_stmt.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_initial_function(self):
        output, canon = runner(test_models + '/initial_function/test_initial.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_limits(self):
        output, canon = runner(test_models + '/limits/test_limits.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_line_breaks(self):
        output, canon = runner(test_models + '/line_breaks/test_line_breaks.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_line_continuation(self):
        output, canon = runner(test_models + '/line_continuation/test_line_continuation.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_ln(self):
        output, canon = runner(test_models + '/ln/test_ln.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_log(self):
        output, canon = runner(test_models + '/log/test_log.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_logicals(self):
        output, canon = runner(test_models + '/logicals/test_logicals.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_lookups(self):
        output, canon = runner(test_models + '/lookups/test_lookups.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_lookups_xscale(self):
        output, canon = runner(test_models + '/lookups/test_lookups_xscale.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_lookups_xpts_sep(self):
        output, canon = runner(test_models + '/lookups/test_lookups_xpts_sep.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_lookups_ypts_sep(self):
        output, canon = runner(test_models + '/lookups/test_lookups_ypts_sep.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_lookups_inline(self):
        output, canon = runner(test_models + '/lookups_inline/test_lookups_inline.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_model_doc(self):
        output, canon = runner(test_models + '/model_doc/model_doc.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_number_handling(self):
        output, canon = runner(test_models + '/number_handling/test_number_handling.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_parentheses(self):
        output, canon = runner(test_models + '/parentheses/test_parens.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_reference_capitalization(self):
        """A properly formatted Vensim model should never create this failure"""
        output, canon = runner(
            test_models + '/reference_capitalization/test_reference_capitalization.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_sqrt(self):
        output, canon = runner(test_models + '/sqrt/test_sqrt.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)

    def test_trig(self):
        output, canon = runner(test_models + '/trig/test_trig.xmile', old=True)
        assert_frames_close(output, canon, rtol=rtol)
