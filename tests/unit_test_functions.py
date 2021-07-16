import unittest
import warnings

import numpy as np
import xarray as xr


class TestInputFunctions(unittest.TestCase):
    def test_ramp(self):
        """Test functions.ramp"""
        from pysd import functions

        self.assertEqual(functions.ramp(lambda: 14, .5, 10, 18), 2)

        self.assertEqual(functions.ramp(lambda: 4, .5, 10, 18), 0)

        self.assertEqual(functions.ramp(lambda: 24, .5, 10, 18), 4)

        self.assertEqual(functions.ramp(lambda: 24, .5, 10, -1), 7)

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
        self.assertEqual(pysd.functions.if_then_else(True,
            lambda: 1, lambda: 0), 1)
        self.assertEqual(pysd.functions.if_then_else(False,
            lambda: 1, lambda: 0), 0)

        # Ensure lazzy evaluation
        self.assertEqual(pysd.functions.if_then_else(True,
            lambda: 1, lambda: 1/0), 1)
        self.assertEqual(pysd.functions.if_then_else(False,
            lambda: 1/0, lambda: 0), 0)

        with self.assertRaises(ZeroDivisionError):
            pysd.functions.if_then_else(True, lambda: 1/0, lambda: 0)
        with self.assertRaises(ZeroDivisionError):
            pysd.functions.if_then_else(False, lambda: 1, lambda: 1/0)

    def test_if_then_else_with_subscripted(self):
        # this test only test the lazzy evaluation and basics
        # subscripted_if_then_else test all the possibilities

        import pysd

        coords = {'dim1': [0, 1], 'dim2': [0, 1]}
        dims = list(coords)

        xr_true = xr.DataArray([[True, True], [True, True]], coords, dims)
        xr_false = xr.DataArray([[False, False], [False, False]], coords, dims)
        xr_mixed = xr.DataArray([[True, False], [False, True]], coords, dims)

        out_true = xr.DataArray([[1, 1], [1, 1]], coords, dims)
        out_false = xr.DataArray([[0, 0], [0, 0]], coords, dims)
        out_mixed = xr.DataArray([[1, 0], [0, 1]], coords, dims)

        self.assertEqual(pysd.functions.if_then_else(xr_true,
            lambda: 1, lambda: 0), 1)
        self.assertEqual(pysd.functions.if_then_else(xr_false,
            lambda: 1, lambda: 0), 0)
        self.assertTrue(pysd.functions.if_then_else(xr_mixed,
            lambda: 1, lambda: 0).equals(out_mixed))

        # Ensure lazzy evaluation
        self.assertEqual(pysd.functions.if_then_else(xr_true,
            lambda: 1, lambda: 1/0), 1)
        self.assertEqual(pysd.functions.if_then_else(xr_false,
            lambda: 1/0, lambda: 0), 0)

        with self.assertRaises(ZeroDivisionError):
            pysd.functions.if_then_else(xr_true, lambda: 1/0, lambda: 0)
        with self.assertRaises(ZeroDivisionError):
            pysd.functions.if_then_else(xr_false,  lambda: 1, lambda: 1/0)
        with self.assertRaises(ZeroDivisionError):
            pysd.functions.if_then_else(xr_mixed,  lambda: 1/0, lambda: 0)
        with self.assertRaises(ZeroDivisionError):
            pysd.functions.if_then_else(xr_mixed,  lambda: 1, lambda: 1/0)

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

        stock.initialize(30)
        self.assertEqual(stock(), 30)

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
                                       order=lambda: 3,
                                       tstep=lambda: 0.5,
                                       py_name='delay_a')

        delay_a.initialize()

        self.assertEqual(delay_a(), 4.234)

        self.assertEqual(float(delay_a.ddt()[0]), (5-4.234)*3)

        delay_a.initialize(6)
        self.assertEqual(delay_a(), 6)
        self.assertEqual(float(delay_a.ddt()[0]), (5-6)*3)

        delay_b = pysd.functions.DelayN(delay_input=lambda: 5,
                                        delay_time=lambda: 3,
                                        initial_value=lambda: 4.234,
                                        order=lambda: 3,
                                        tstep=lambda: 0.5,
                                        py_name='delay_b')

        delay_b.initialize()

        self.assertEqual(delay_b(), 4.234)

        self.assertEqual(float(delay_b.ddt()[0]), (5-4.234)*3)

        delay_b.initialize(6)
        self.assertEqual(delay_b(), 6)
        self.assertEqual(float(delay_b.ddt()[0]), (5-6)*3)

        delay_c = pysd.functions.DelayFixed(delay_input=lambda: 5,
                                            delay_time=lambda: 3,
                                            initial_value=lambda: 4.234,
                                            tstep=lambda: 0.5,
                                            py_name='delay_c')

        delay_c.initialize()

        self.assertEqual(delay_c(), 4.234)

        delay_c.initialize(6)
        self.assertEqual(delay_c(), 6)

    def test_delay_subscript(self):
        """
        Test for subscripted delay
        """
        import pysd

        coords = {'d1': [9, 1], 'd2': [2, 4]}
        dims = ['d1', 'd2']
        xr_input = xr.DataArray([[1, 2], [3, 4]], coords, dims)
        xr_initial = xr.DataArray([[10, 10], [0.5, 0.5]], coords, dims)
        xr_delay_time = xr.DataArray([[3, 2], [3, 2]], coords, dims)

        delay = pysd.functions.Delay(delay_input=lambda: xr_input,
                                     delay_time=lambda: xr_delay_time,
                                     initial_value=lambda: xr_initial,
                                     order=lambda: 2,
                                     tstep=lambda: 0.5,
                                     py_name='delay')

        delay.initialize()

        self.assertTrue(delay().equals(xr_initial))
        delay_ddt = delay.ddt()[0].reset_coords('_delay', drop=True)
        print(delay_ddt)

        self.assertTrue(delay_ddt.equals((xr_input-xr_initial)*2))

    def test_delay_order(self):
        import pysd

        # order 3 to 2
        delay1 = pysd.functions.Delay(delay_input=lambda: 10,
                                      delay_time=lambda: 1,
                                      initial_value=lambda: 0,
                                      order=lambda: 3,
                                      tstep=lambda: 0.4,
                                      py_name='delay1')

        # order 3 to 2
        delay2 = pysd.functions.DelayN(delay_input=lambda: 10,
                                       delay_time=lambda: 1,
                                       initial_value=lambda: 0,
                                       order=lambda: 3,
                                       tstep=lambda: 0.4,
                                       py_name='delay2')

        # 1.5 to 1
        delay3 = pysd.functions.Delay(delay_input=lambda: 10,
                                      delay_time=lambda: 1,
                                      initial_value=lambda: 0,
                                      order=lambda: 1.5,
                                      tstep=lambda: 0.4,
                                      py_name='delay3')

        # 1.5  to 1
        delay4 = pysd.functions.DelayN(delay_input=lambda: 10,
                                       delay_time=lambda: 1,
                                       initial_value=lambda: 0,
                                       order=lambda: 1.5,
                                       tstep=lambda: 0.4,
                                       py_name='delay4')
        # 1.5 rounded to 2
        delay5 = pysd.functions.DelayFixed(delay_input=lambda: 10,
                                           delay_time=lambda: 0.75,
                                           initial_value=lambda: 0,
                                           tstep=lambda: 0.5,
                                           py_name='delay5')

        with warnings.catch_warnings(record=True) as ws:
            delay1.initialize()
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 1)
            self.assertIn('Delay time very small, casting delay order '
                          + 'from 3 to 2',
                          str(wu[0].message))

        with warnings.catch_warnings(record=True) as ws:
            delay2.initialize()
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 1)
            self.assertIn('Delay time very small, casting delay order '
                          + 'from 3 to 2',
                          str(wu[0].message))

        with warnings.catch_warnings(record=True) as ws:
            delay3.initialize()
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 1)
            self.assertIn("Casting delay order from 1.5 to 1",
                          str(wu[0].message))

        with warnings.catch_warnings(record=True) as ws:
            delay4.initialize()
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 1)
            self.assertIn("Casting delay order from 1.5 to 1",
                          str(wu[0].message))

        with warnings.catch_warnings(record=True) as ws:
            delay5.initialize()
            wu = [w for w in ws if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wu), 1)
            self.assertIn("Casting delay order from 1.500000 to 2",
                          str(wu[0].message))

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

        # test change initial condition
        initial_f20.initialize(123)
        self.assertEqual(initial_f20(), 123)

    def test_smooth(self):
        import pysd
        smooth = pysd.functions.Smooth(smooth_input=lambda: 5,
                                       smooth_time=lambda: 3,
                                       initial_value=lambda: 4.234,
                                       order=lambda: 3,
                                       py_name='smooth')

        smooth.initialize()
        self.assertEqual(smooth(), 4.234)

        smooth.initialize(6)
        self.assertEqual(smooth(), 6)

    def test_trend(self):
        import pysd
        trend = pysd.functions.Trend(trend_input=lambda: 5,
                                     average_time=lambda: 3,
                                     initial_trend=lambda: 4.234,
                                     py_name='trend')

        trend.initialize()
        self.assertEqual(trend(), 4.234)

        trend.initialize(6)
        self.assertEqual(round(trend(), 8), 6)

    def test_sampleiftrue(self):
        import pysd

        bool_v = False

        def condition():
            return bool_v

        sif = pysd.functions.SampleIfTrue(condition=condition,
                                          actual_value=lambda: 3,
                                          initial_value=lambda: 4.234,
                                          py_name='sampleiftrue')

        sif.initialize()
        self.assertEqual(sif(), 4.234)

        sif.initialize(6)
        self.assertEqual(sif(), 6)

        bool_v = True
        self.assertEqual(sif(), 3)


class TestFunctions(unittest.TestCase):

    def test_sum(self):
        """
        Test for sum function
        """
        import pysd

        coords = {'d1': [9, 1], 'd2': [2, 4]}
        coords_d1, coords_d2 = {'d1': [9, 1]}, {'d2': [2, 4]}
        dims = ['d1', 'd2']

        data = xr.DataArray([[1, 2], [3, 4]], coords, dims)

        self.assertTrue(pysd.functions.sum(data,
            dim=['d1']).equals(xr.DataArray([4, 6], coords_d2, ['d2'])))
        self.assertTrue(pysd.functions.sum(data,
            dim=['d2']).equals(xr.DataArray([3, 7], coords_d1, ['d1'])))
        self.assertEqual(pysd.functions.sum(data, dim=['d1', 'd2']), 10)
        self.assertEqual(pysd.functions.sum(data), 10)

    def test_prod(self):
        """
        Test for sum function
        """
        import pysd

        coords = {'d1': [9, 1], 'd2': [2, 4]}
        coords_d1, coords_d2 = {'d1': [9, 1]}, {'d2': [2, 4]}
        dims = ['d1', 'd2']

        data = xr.DataArray([[1, 2], [3, 4]], coords, dims)

        self.assertTrue(pysd.functions.prod(data,
            dim=['d1']).equals(xr.DataArray([3, 8], coords_d2, ['d2'])))
        self.assertTrue(pysd.functions.prod(data,
            dim=['d2']).equals(xr.DataArray([2, 12], coords_d1, ['d1'])))
        self.assertEqual(pysd.functions.prod(data, dim=['d1', 'd2']), 24)
        self.assertEqual(pysd.functions.prod(data), 24)

    def test_vmin(self):
        """
        Test for vmin function
        """
        import pysd

        coords = {'d1': [9, 1], 'd2': [2, 4]}
        coords_d1, coords_d2 = {'d1': [9, 1]}, {'d2': [2, 4]}
        dims = ['d1', 'd2']

        data = xr.DataArray([[1, 2], [3, 4]], coords, dims)

        self.assertTrue(pysd.functions.vmin(data,
            dim=['d1']).equals(xr.DataArray([1, 2], coords_d2, ['d2'])))
        self.assertTrue(pysd.functions.vmin(data,
            dim=['d2']).equals(xr.DataArray([1, 3], coords_d1, ['d1'])))
        self.assertEqual(pysd.functions.vmin(data, dim=['d1', 'd2']), 1)
        self.assertEqual(pysd.functions.vmin(data), 1)

    def test_vmax(self):
        """
        Test for vmax function
        """
        import pysd

        coords = {'d1': [9, 1], 'd2': [2, 4]}
        coords_d1, coords_d2 = {'d1': [9, 1]}, {'d2': [2, 4]}
        dims = ['d1', 'd2']

        data = xr.DataArray([[1, 2], [3, 4]], coords, dims)

        self.assertTrue(pysd.functions.vmax(data,
            dim=['d1']).equals(xr.DataArray([3, 4], coords_d2, ['d2'])))
        self.assertTrue(pysd.functions.vmax(data,
            dim=['d2']).equals(xr.DataArray([2, 4], coords_d1, ['d1'])))
        self.assertEqual(pysd.functions.vmax(data, dim=['d1', 'd2']), 4)
        self.assertEqual(pysd.functions.vmax(data), 4)

    def test_invert_matrix(self):
        """
        Test for invert_matrix function
        """
        import pysd

        coords1 = {'d1': ['a', 'b'], 'd2': ['a', 'b']}
        coords2 = {'d0': ['1', '2'], 'd1': ['a', 'b'], 'd2': ['a', 'b']}
        coords3 = {'d0': ['1', '2'],
                   'd1': ['a', 'b', 'c'],
                   'd2': ['a', 'b', 'c']}

        data1 = xr.DataArray([[1, 2], [3, 4]], coords1, ['d1', 'd2'])
        data2 = xr.DataArray([[[1, 2], [3, 4]], [[-1, 2], [5, 4]]],
                             coords2,
                             ['d0', 'd1', 'd2'])
        data3 = xr.DataArray([[[1, 2, 3], [3, 7, 2], [3, 4, 6]],
                              [[-1, 2, 3], [4, 7, 3], [5, 4, 6]]],
                             coords3,
                             ['d0', 'd1', 'd2'])

        for data in [data1, data2, data3]:
            datai = pysd.functions.invert_matrix(data)
            self.assertEqual(data.dims, datai.dims)

            if len(data.shape) == 2:
                # two dimensions xarrays
                self.assertTrue((
                    abs(np.dot(data, datai) - np.dot(datai, data))
                    < 1e-14
                    ).all())
                self.assertTrue((
                    abs(np.dot(data, datai) - np.identity(data.shape[-1]))
                    < 1e-14
                    ).all())
            else:
                # three dimensions xarrays
                for i in range(data.shape[0]):
                    self.assertTrue((
                        abs(np.dot(data[i], datai[i])
                            - np.dot(datai[i], data[i]))
                        < 1e-14
                        ).all())
                    self.assertTrue((
                        abs(np.dot(data[i], datai[i])
                            - np.identity(data.shape[-1]))
                        < 1e-14
                        ).all())

    def test_incomplete(self):
        import pysd
        from warnings import catch_warnings

        with catch_warnings(record=True) as w:
            pysd.functions.incomplete()
            self.assertEqual(len(w), 1)
            self.assertTrue('Call to undefined function' in str(w[-1].message))

    def test_not_implemented_function(self):
        import pysd

        with self.assertRaises(NotImplementedError):
            pysd.functions.not_implemented_function("NIF")
