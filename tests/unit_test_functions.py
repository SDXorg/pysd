import unittest
import numpy as np

class TestInputFunctions(unittest.TestCase):
    def test_ramp(self):
        """Test functions.ramp"""
        from pysd import functions

        functions.time = lambda: 14
        self.assertEqual(functions.ramp(.5, 10, 18), 2)

        functions.time = lambda: 4
        self.assertEqual(functions.ramp(.5, 10, 18), 0)

        functions.time = lambda: 24
        self.assertEqual(functions.ramp(.5, 10, 18), 4)

    def test_step(self):
        from pysd import functions

        functions.time = lambda: 5
        self.assertEqual(functions.step(1, 10), 0)

        functions.time = lambda: 15
        self.assertEqual(functions.step(1, 10), 1)

        functions.time = lambda: 10
        self.assertEqual(functions.step(1, 10), 1)

    def test_pulse(self):
        from pysd import functions

        functions.time = lambda: 0
        self.assertEqual(functions.pulse(1, 3), 0)

        functions.time = lambda: 1
        self.assertEqual(functions.pulse(1, 3), 1)

        functions.time = lambda: 2
        self.assertEqual(functions.pulse(1, 3), 1)

        functions.time = lambda: 4
        self.assertEqual(functions.pulse(1, 3), 0)

        functions.time = lambda: 5
        self.assertEqual(functions.pulse(1, 3), 0)

    def test_pulse_chain(self):
        from pysd import functions
        # before train starts
        functions.time = lambda: 0
        self.assertEqual(functions.pulse_train(1, 3, 5, 12), 0)
        # on train start
        functions.time = lambda: 1
        self.assertEqual(functions.pulse_train(1, 3, 5, 12), 1)
        # within first pulse
        functions.time = lambda: 2
        self.assertEqual(functions.pulse_train(1, 3, 5, 12), 1)
        # end of first pulse
        functions.time = lambda: 4
        self.assertEqual(functions.pulse_train(1, 3, 5, 12), 0)
        # after first pulse before second
        functions.time = lambda: 5
        self.assertEqual(functions.pulse_train(1, 3, 5, 12), 0)
        # on start of second pulse
        functions.time = lambda: 6
        self.assertEqual(functions.pulse_train(1, 3, 5, 12), 1)
        # within second pulse
        functions.time = lambda: 7
        self.assertEqual(functions.pulse_train(1, 3, 5, 12), 1)
        # after second pulse
        functions.time = lambda: 10
        self.assertEqual(functions.pulse_train(1, 3, 5, 12), 0)
        # on third pulse
        functions.time = lambda: 11
        self.assertEqual(functions.pulse_train(1, 3, 5, 12), 1)
        # on train end
        functions.time = lambda: 12
        self.assertEqual(functions.pulse_train(1, 3, 5, 12), 0)
        # after train
        functions.time = lambda: 15
        self.assertEqual(functions.pulse_train(1, 3, 5, 13), 0)


class TestStatsFunctions(unittest.TestCase):
    def test_bounded_normal(self):
        from pysd.functions import bounded_normal
        min = -4
        max = .2
        mean = -1
        std = .05
        seed = 1
        results = np.array(
            [bounded_normal(min, max, mean, std, seed)
             for _ in range(1000)])

        self.assertGreaterEqual(results.min(), min)
        self.assertLessEqual(results.max(), max)
        self.assertAlmostEqual(results.mean(), mean, delta=std)
        self.assertAlmostEqual(results.std(), std, delta=std)  # this is a pretty loose test
        self.assertGreater(len(np.unique(results)), 100)


class TestLogicFunctions(unittest.TestCase):
    def test_if_then_else_basic(self):
        """ If Then Else function"""
        from pysd.functions import if_then_else
        self.assertEqual(if_then_else(True, 1, 0), 1)
        self.assertEqual(if_then_else(False, 1, 0), 0)

    @unittest.skip('Not Yet Implemented')
    def test_if_then_else_with_subscripted_output(self):
        """ We may have a subscripted value passed as an output """
        self.fail()

    @unittest.skip('Not Yet Implemented')
    def test_if_then_else_with_subscripted_input_basic_output(self):
        """ What do we do if the expression yields a subscripted array of true and false values,
         and the output options are singular?"""
        self.fail()

    @unittest.skip('Not Yet Implemented')
    def test_if_then_else_with_subscripted_input_output(self):
        """ What do we do if the expression yields a subscripted array of true and false values,
        and the output values are subscripted? """
        self.fail()