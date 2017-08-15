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

    def test_xidz(self):
        from pysd import functions
        # functions.time = lambda: 5 ## any time will and should do
        self.assertEqual(functions.xidz(1, -0.00000001, 5), 5)
        self.assertEqual(functions.xidz(1, 0, 5), 5)
        self.assertEqual(functions.xidz(1, 8, 5), 0.125)

    def test_zidz(self):
        from pysd import functions
        # functions.time = lambda: 5 ## any time will and should do
        self.assertEqual(functions.zidz(1, -0.00000001), 0)
        self.assertEqual(functions.zidz(1, 0), 0)
        self.assertEqual(functions.zidz(1, 8), 0.125)


class TestStatsFunctions(unittest.TestCase):
    def test_bounded_normal(self):
        from pysd.functions import bounded_normal
        min_val = -4
        max_val = .2
        mean = -1
        std = .05
        seed = 1
        results = np.array(
            [bounded_normal(min_val, max_val, mean, std, seed)
             for _ in range(1000)])

        self.assertGreaterEqual(results.min(), min_val)
        self.assertLessEqual(results.max(), max_val)
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


class TestStateful(unittest.TestCase):

    def test_integ(self):
        import pysd.functions

        ddt_val = 5
        init_val = 10

        def ddt_func():
            return ddt_val

        def init_func():
            return init_val

        stock = pysd.functions.Integ(lambda: ddt_func(),
                                     lambda: init_func())

        stock.initialize()

        self.assertEqual(stock(), 10)
        self.assertEqual(stock.ddt(), 5)

        dt = .1
        stock.update(stock() + dt * stock.ddt())

        self.assertEqual(stock(), 10.5)

        ddt_val = 43
        self.assertEqual(stock.ddt(), 43)

        init_val = 11
        self.assertEqual(stock(), 10.5)

        stock.initialize()
        self.assertEqual(stock(), 11)

    def test_stateful_identification(self):
        import pysd.functions

        stock = pysd.functions.Integ(lambda: 5,
                                     lambda: 7)

        self.assertIsInstance(stock,
                              pysd.functions.Stateful)

    def test_delay(self):
        import pysd.functions

        delay_a = pysd.functions.Delay(delay_input=lambda: 5,
                                       delay_time=lambda: 3,
                                       initial_value=lambda: 4.234,
                                       order=lambda: 3)

        delay_a.initialize()

        self.assertEqual(delay_a(), 4.234)

        self.assertEqual(delay_a.ddt()[0], 5-(4.234*3/3))

    def test_initial(self):
        from pysd import functions
        a = 1
        b = 2

        def func1():
            return a

        def func2():
            return b

        # test that the function returns the first value
        # after it is called once
        f1_0 = func1()
        initial_f10 = functions.Initial(lambda: func1())
        initial_f10.initialize()
        f1_i0 = initial_f10()
        self.assertEqual(f1_0, f1_i0)

        f2_0 = func2()
        initial_f20 = functions.Initial(func2)
        initial_f20.initialize()
        f2_i0 = initial_f20()
        self.assertEqual(f2_0, f2_i0)

        a = 5
        b = 6

        # test that the function returns the first value
        # after it is called a second time
        f1_1 = func1()
        f1_i1 = initial_f10()
        self.assertNotEqual(f1_1, f1_i1)
        self.assertEqual(f1_i1, f1_0)

        f2_1 = func2()
        f2_i1 = initial_f20()
        self.assertNotEqual(f2_1, f2_i1)
        self.assertEqual(f2_i1, f2_0)


