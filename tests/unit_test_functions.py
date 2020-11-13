import unittest

import numpy as np


class TestInputFunctions(unittest.TestCase):
    def test_ramp(self):
        """Test functions.ramp"""
        from pysd import functions

        self.assertEqual(functions.ramp(lambda: 14, .5, 10, 18), 2)

        self.assertEqual(functions.ramp(lambda: 4, .5, 10, 18), 0)

        self.assertEqual(functions.ramp(lambda: 24, .5, 10, 18), 4)

    def test_step(self):
        from pysd import functions

        self.assertEqual(functions.step(lambda: 5, 1, 10), 0)

        self.assertEqual(functions.step(lambda: 15, 1, 10), 1)

        self.assertEqual(functions.step(lambda: 10, 1, 10), 1)

    def test_pulse(self):
        from pysd import functions

        self.assertEqual(functions.pulse(lambda: 0, 1, 3), 0)

        self.assertEqual(functions.pulse(lambda: 1, 1, 3), 1)

        self.assertEqual(functions.pulse(lambda: 2, 1, 3), 1)

        self.assertEqual(functions.pulse(lambda: 4, 1, 3), 0)

        self.assertEqual(functions.pulse(lambda: 5, 1, 3), 0)

    def test_pulse_chain(self):
        from pysd import functions
        # before train starts
        self.assertEqual(functions.pulse_train(lambda: 0, 1, 3, 5, 12), 0)
        # on train start
        self.assertEqual(functions.pulse_train(lambda: 1, 1, 3, 5, 12), 1)
        # within first pulse
        self.assertEqual(functions.pulse_train(lambda: 2, 1, 3, 5, 12), 1)
        # end of first pulse
        self.assertEqual(functions.pulse_train(lambda: 4, 1, 3, 5, 12), 0)
        # after first pulse before second
        self.assertEqual(functions.pulse_train(lambda: 5, 1, 3, 5, 12), 0)
        # on start of second pulse
        self.assertEqual(functions.pulse_train(lambda: 6, 1, 3, 5, 12), 1)
        # within second pulse
        self.assertEqual(functions.pulse_train(lambda: 7, 1, 3, 5, 12), 1)
        # after second pulse
        self.assertEqual(functions.pulse_train(lambda: 10, 1, 3, 5, 12), 0)
        # on third pulse
        self.assertEqual(functions.pulse_train(lambda: 11, 1, 3, 5, 12), 1)
        # on train end
        self.assertEqual(functions.pulse_train(lambda: 12, 1, 3, 5, 12), 0)
        # after train
        self.assertEqual(functions.pulse_train(lambda: 15, 1, 3, 5, 13), 0)

    def test_pulse_magnitude(self):
        from pysd import functions

        # Pulse function with repeat time
        # before first impulse
        self.assertEqual(functions.pulse_magnitude(functions.Time(0, 1), 10, 2, 5), 0)
        # first impulse
        self.assertEqual(functions.pulse_magnitude(functions.Time(2, 1), 10, 2, 5), 10)
        # after first impulse and before second
        self.assertEqual(functions.pulse_magnitude(functions.Time(4, 1), 10, 2, 5), 0)
        # second impulse
        self.assertEqual(functions.pulse_magnitude(functions.Time(7, 1), 10, 2, 5), 10)
        # after second and before third impulse
        self.assertEqual(functions.pulse_magnitude(functions.Time(9, 1), 10, 2, 5), 0)
        # third impulse
        self.assertEqual(functions.pulse_magnitude(functions.Time(12, 1), 10, 2, 5), 10)
        # after third impulse
        self.assertEqual(functions.pulse_magnitude(functions.Time(14, 1), 10, 2, 5), 0)

        # Pulse function without repeat time
        # before first impulse
        self.assertEqual(functions.pulse_magnitude(functions.Time(0, 1), 10, 2), 0)
        # first impulse
        self.assertEqual(functions.pulse_magnitude(functions.Time(2, 1), 10, 2), 10)
        # after first impulse and before second
        self.assertEqual(functions.pulse_magnitude(functions.Time(4, 1), 10, 2), 0)
        # second impulse
        self.assertEqual(functions.pulse_magnitude(functions.Time(7, 1), 10, 2), 0)
        # after second and before third impulse
        self.assertEqual(functions.pulse_magnitude(functions.Time(9, 1), 10, 2), 0)

    def test_xidz(self):
        from pysd import functions
        self.assertEqual(functions.xidz(1, -0.00000001, 5), 5)
        self.assertEqual(functions.xidz(1, 0, 5), 5)
        self.assertEqual(functions.xidz(1, 8, 5), 0.125)

    def test_zidz(self):
        from pysd import functions
        self.assertEqual(functions.zidz(1, -0.00000001), 0)
        self.assertEqual(functions.zidz(1, 0), 0)
        self.assertEqual(functions.zidz(1, 8), 0.125)


class TestStatsFunctions(unittest.TestCase):
    def test_bounded_normal(self):
        import pysd
        min_val = -4
        max_val = .2
        mean = -1
        std = .05
        seed = 1
        results = np.array(
            [pysd.functions.bounded_normal(min_val, max_val, mean, std, seed)
             for _ in range(1000)])

        self.assertGreaterEqual(results.min(), min_val)
        self.assertLessEqual(results.max(), max_val)
        self.assertAlmostEqual(results.mean(), mean, delta=std)
        self.assertAlmostEqual(results.std(), std, delta=std)  # this is a pretty loose test
        self.assertGreater(len(np.unique(results)), 100)


class TestLogicFunctions(unittest.TestCase):
    def test_if_then_else_basic(self):
        """ If Then Else function"""
        import pysd
        self.assertEqual(pysd.functions.if_then_else(True, 1, 0), 1)
        self.assertEqual(pysd.functions.if_then_else(False, 1, 0), 0)

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


class TestLookup(unittest.TestCase):
    def test_lookup(self):
        import pysd

        xpts = [0, 1, 2, 3,  5,  6, 7, 8]
        ypts = [0, 0, 1, 1, -1, -1, 0, 0]

        expected_xpts = np.arange(-0.5, 8.6, 0.5)
        expected_ypts = [
            0,
            0, 0,
            0, 0.5,
            1, 1,
            1, 0.5, 0, -0.5, -1,
            -1, -1, -0.5,
            0, 0, 0, 0
        ]

        for index in range(0, len(xpts)):
            x = xpts[index]
            y = ypts[index]

            result = pysd.functions.lookup(x, xpts, ypts)

            self.assertEqual(y, result, "Wrong result at X=" + str(x))

    def test_lookup_extrapolation_inbounds(self):
        import pysd

        xpts = [0, 1, 2, 3,  5,  6, 7, 8]
        ypts = [0, 0, 1, 1, -1, -1, 0, 0]

        expected_xpts = np.arange(-0.5, 8.6, 0.5)
        expected_ypts = [
            0,
            0, 0,
            0, 0.5,
            1, 1,
            1, 0.5, 0, -0.5, -1,
            -1, -1, -0.5,
            0, 0, 0, 0
        ]

        for index in range(0, len(expected_xpts)):
            x = expected_xpts[index]
            y = expected_ypts[index]

            result = pysd.functions.lookup_extrapolation(x, xpts, ypts)

            self.assertEqual(y, result, "Wrong result at X=" + str(x))

    def test_lookup_extrapolation_two_points(self):
        import pysd

        xpts = [0, 1]
        ypts = [0, 1]

        expected_xpts = np.arange(-0.5, 1.6, 0.5)
        expected_ypts = [-0.5, 0.0, 0.5, 1.0, 1.5]

        for index in range(0, len(expected_xpts)):
            x = expected_xpts[index]
            y = expected_ypts[index]

            result = pysd.functions.lookup_extrapolation(x, xpts, ypts)

            self.assertEqual(y, result, "Wrong result at X=" + str(x))

    def test_lookup_extrapolation_outbounds(self):
        import pysd

        xpts = [0, 1, 2, 3]
        ypts = [0, 1, 1, 0]

        expected_xpts = np.arange(-0.5, 3.6, 0.5)
        expected_ypts = [
            -0.5,
            0.0, 0.5, 1.0,
            1.0, 1.0,
            0.5, 0,
            -0.5
        ]

        for index in range(0, len(expected_xpts)):
            x = expected_xpts[index]
            y = expected_ypts[index]

            result = pysd.functions.lookup_extrapolation(x, xpts, ypts)

            self.assertEqual(y, result, "Wrong result at X=" + str(x))

    def test_lookup_discrete(self):
        import pysd

        xpts = [0, 1, 2, 3,  5,  6, 7, 8]
        ypts = [0, 0, 1, 1, -1, -1, 0, 0]

        expected_xpts = np.arange(-0.5, 8.6, 0.5)
        expected_ypts = [
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1,
            -1, -1, -1, -1,
            0, 0, 0, 0
        ]

        for index in range(0, len(expected_xpts)):
            x = expected_xpts[index]
            y = expected_ypts[index]

            result = pysd.functions.lookup_discrete(x, xpts, ypts)

            self.assertEqual(y, result, "Wrong result at X=" + str(x))


class TestStateful(unittest.TestCase):

    def test_integ(self):
        import pysd

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
        import pysd

        stock = pysd.functions.Integ(lambda: 5,
                                     lambda: 7)

        self.assertIsInstance(stock,
                              pysd.functions.Stateful)

    def test_delay(self):
        import pysd

        delay_a = pysd.functions.Delay(delay_input=lambda: 5,
                                       delay_time=lambda: 3,
                                       initial_value=lambda: 4.234,
                                       order=lambda: 3)

        delay_a.initialize()

        self.assertEqual(delay_a(), 4.234)

        self.assertEqual(delay_a.ddt()[0], 5-4.234)

    def test_delay_subscript(self):
        """
        Test for subscripted delay
        """
        import pysd
        import xarray as xr

        coords = {'d1':[9,1],'d2':[2,4]}
        dims = ['d1', 'd2']
        xr_input = xr.DataArray([[1,2],[3,4]], coords, dims)
        xr_initial = xr.DataArray([10, 0.5], {'d1':[9,1]}, ['d1'])
        _, xr_initial_e = xr.broadcast(xr_input, xr_initial)
        xr_delay_time = xr.DataArray([3,2], {'d2':[2,4]}, ['d2'])
        _, xr_delay_time_e = xr.broadcast(xr_input, xr_delay_time)
        
        # if only the delay_input is xarray
        delay_a = pysd.functions.Delay(delay_input=lambda: xr_input,
                                       delay_time=lambda: 3,
                                       initial_value=lambda: 4.234,
                                       order=lambda: 3,
                                       coords=coords,
                                       dims=dims)

        delay_a.initialize()

        self.assertTrue(delay_a().equals(xr.DataArray(4.234, coords,dims)))
        delay_ddt = delay_a.ddt()[0].reset_coords('delay', drop=True)
        self.assertTrue(delay_ddt.equals(xr_input-4.234))
        
        # if delay input and initial_value are xarray (differents shapes checked)
        delay_b = pysd.functions.Delay(delay_input=lambda: xr_input,
                                       delay_time=lambda: 3,
                                       initial_value=lambda: xr_initial,
                                       order=lambda: 3,
                                       coords=coords,
                                       dims=dims)

        delay_b.initialize()
        
        self.assertTrue(delay_b().equals(xr_initial_e))
        delay_ddt = delay_b.ddt()[0].reset_coords('delay', drop=True)

        self.assertTrue(delay_ddt.equals(xr_input-xr_initial_e))
        
        # if delay input and delay_time are xarray (differents shapes checked)        
        delay_c = pysd.functions.Delay(delay_input=lambda: xr_input,
                                       delay_time=lambda: xr_delay_time,
                                       initial_value=lambda: 4.234,
                                       order=lambda: 3,
                                       coords=coords,
                                       dims=dims)

        delay_c.initialize()   
              
    def test_initial(self):
        import pysd
        a = 1
        b = 2

        def func1():
            return a

        def func2():
            return b

        # test that the function returns the first value
        # after it is called once
        f1_0 = func1()
        initial_f10 = pysd.functions.Initial(lambda: func1())
        initial_f10.initialize()
        f1_i0 = initial_f10()
        self.assertEqual(f1_0, f1_i0)

        f2_0 = func2()
        initial_f20 = pysd.functions.Initial(func2)
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


