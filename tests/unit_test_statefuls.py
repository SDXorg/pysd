import unittest
import warnings

import xarray as xr


class TestStateful(unittest.TestCase):

    def test_integ(self):
        from pysd.py_backend.statefuls import Integ

        ddt_val = 5
        init_val = 10

        def ddt_func():
            return ddt_val

        def init_func():
            return init_val

        stock = Integ(lambda: ddt_func(), lambda: init_func(), "stock")

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
        from pysd.py_backend.statefuls import Integ, Stateful

        stock = Integ(lambda: 5, lambda: 7, "stock")

        self.assertIsInstance(stock, Stateful)

    def test_delay(self):
        from pysd.py_backend.statefuls import Delay, DelayN, DelayFixed

        delay_a = Delay(delay_input=lambda: 5,
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

        delay_b = DelayN(delay_input=lambda: 5,
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

        delay_c = DelayFixed(delay_input=lambda: 5,
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
        from pysd.py_backend.statefuls import Delay

        coords = {'d1': [9, 1], 'd2': [2, 4]}
        dims = ['d1', 'd2']
        xr_input = xr.DataArray([[1, 2], [3, 4]], coords, dims)
        xr_initial = xr.DataArray([[10, 10], [0.5, 0.5]], coords, dims)
        xr_delay_time = xr.DataArray([[3, 2], [3, 2]], coords, dims)

        delay = Delay(delay_input=lambda: xr_input,
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
        from pysd.py_backend.statefuls import Delay, DelayN, DelayFixed

        # order 3 to 2
        delay1 = Delay(delay_input=lambda: 10,
                       delay_time=lambda: 1,
                       initial_value=lambda: 0,
                       order=lambda: 3,
                       tstep=lambda: 0.4,
                       py_name='delay1')

        # order 3 to 2
        delay2 = DelayN(delay_input=lambda: 10,
                        delay_time=lambda: 1,
                        initial_value=lambda: 0,
                        order=lambda: 3,
                        tstep=lambda: 0.4,
                        py_name='delay2')

        # 1.5 to 1
        delay3 = Delay(delay_input=lambda: 10,
                       delay_time=lambda: 1,
                       initial_value=lambda: 0,
                       order=lambda: 1.5,
                       tstep=lambda: 0.4,
                       py_name='delay3')

        # 1.5  to 1
        delay4 = DelayN(delay_input=lambda: 10,
                        delay_time=lambda: 1,
                        initial_value=lambda: 0,
                        order=lambda: 1.5,
                        tstep=lambda: 0.4,
                        py_name='delay4')

        # 1.5 rounded to 2
        delay5 = DelayFixed(delay_input=lambda: 10,
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

    def test_forecast(self):
        from pysd.py_backend.statefuls import Forecast

        input_val = 5

        def input():
            return input_val

        frcst = Forecast(forecast_input=input,
                         average_time=lambda: 3,
                         horizon=lambda: 10,
                         py_name='forecast')

        frcst.initialize()
        self.assertEqual(frcst(), input_val)

        frcst.state = frcst.state + 0.1*frcst.ddt()
        input_val = 20
        self.assertEqual(frcst(), 220)

        frcst.state = frcst.state + 0.1*frcst.ddt()
        input_val = 35.5
        self.assertEqual(
            frcst(),
            input_val*(1+(input_val-frcst.state)/(3*frcst.state)*10))

        input_val = 7
        init_val = 6
        frcst.initialize(init_val)
        self.assertEqual(
            frcst(),
            input_val*(1+(input_val-init_val)/(3*init_val)*10))

    def test_initial(self):
        from pysd.py_backend.statefuls import Initial
        a = 1
        b = 2

        def func1():
            return a

        def func2():
            return b

        # test that the function returns the first value
        # after it is called once
        f1_0 = func1()
        initial_f10 = Initial(lambda: func1(), "initial_f10")
        initial_f10.initialize()
        f1_i0 = initial_f10()
        self.assertEqual(f1_0, f1_i0)

        f2_0 = func2()
        initial_f20 = Initial(func2, "initial_f20")
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

    def test_sampleiftrue(self):
        from pysd.py_backend.statefuls import SampleIfTrue

        bool_v = False

        def condition():
            return bool_v

        sif = SampleIfTrue(condition=condition,
                           actual_value=lambda: 3,
                           initial_value=lambda: 4.234,
                           py_name='sampleiftrue')

        sif.initialize()
        self.assertEqual(sif(), 4.234)

        sif.initialize(6)
        self.assertEqual(sif(), 6)

        bool_v = True
        self.assertEqual(sif(), 3)

    def test_smooth(self):
        from pysd.py_backend.statefuls import Smooth
        smooth = Smooth(smooth_input=lambda: 5,
                        smooth_time=lambda: 3,
                        initial_value=lambda: 4.234,
                        order=lambda: 3,
                        py_name='smooth')

        smooth.initialize()
        self.assertEqual(smooth(), 4.234)

        smooth.initialize(6)
        self.assertEqual(smooth(), 6)

    def test_trend(self):
        from pysd.py_backend.statefuls import Trend
        trend = Trend(trend_input=lambda: 5,
                      average_time=lambda: 3,
                      initial_trend=lambda: 4.234,
                      py_name='trend')

        trend.initialize()
        self.assertEqual(trend(), 4.234)

        trend.initialize(6)
        self.assertEqual(round(trend(), 8), 6)


class TestStatefulErrors(unittest.TestCase):
    def test_not_initialized_object(self):
        from pysd.py_backend.statefuls import Stateful

        obj = Stateful()
        obj.py_name = "my_object"
        with self.assertRaises(AttributeError) as err:
            obj.state
            self.assertIn(
                "my_object\nAttempt to call stateful element"
                + " before it is initialized.",
                err.args[0])
