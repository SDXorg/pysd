import pytest

import numpy as np
import xarray as xr

from pysd.py_backend.statefuls import Stateful, Integ, Delay, DelayN,\
    DelayFixed, Forecast, Initial, SampleIfTrue, Smooth, Trend


class TestStateful():

    def test_integ(self):
        ddt_val = 5
        init_val = 10

        def ddt_func():
            return ddt_val

        def init_func():
            return init_val

        stock = Integ(lambda: ddt_func(), lambda: init_func(), "stock")

        stock.initialize()

        assert stock() == 10
        assert stock.ddt() == 5

        dt = .1
        stock.update(stock() + dt * stock.ddt())

        assert stock() == 10.5

        ddt_val = 43
        assert stock.ddt() == 43

        init_val = 11
        assert stock() == 10.5

        stock.initialize()
        assert stock() == 11

        stock.initialize(30)
        assert stock() == 30

    def test_stateful_identification(self):
        stock = Integ(lambda: 5, lambda: 7, "stock")

        assert isinstance(stock, Stateful)

    def test_delay(self):

        delay_a = Delay(delay_input=lambda: 5,
                        delay_time=lambda: 3,
                        initial_value=lambda: 4.234,
                        order=lambda: 3,
                        tstep=lambda: 0.5,
                        py_name='delay_a')

        delay_a.initialize()

        assert delay_a() == 4.234

        assert float(delay_a.ddt()[0]) == (5-4.234)*3

        delay_a.initialize(6)
        assert delay_a() == 6
        assert float(delay_a.ddt()[0]) == (5-6)*3

        delay_b = DelayN(delay_input=lambda: 5,
                         delay_time=lambda: 3,
                         initial_value=lambda: 4.234,
                         order=lambda: 3,
                         tstep=lambda: 0.5,
                         py_name='delay_b')

        delay_b.initialize()

        assert delay_b() == 4.234

        assert float(delay_b.ddt()[0]) == (5-4.234)*3

        delay_b.initialize(6)
        assert delay_b() == 6
        assert float(delay_b.ddt()[0]) == (5-6)*3

        delay_c = DelayFixed(delay_input=lambda: 5,
                             delay_time=lambda: 3,
                             initial_value=lambda: 4.234,
                             tstep=lambda: 0.5,
                             py_name='delay_c')

        delay_c.initialize()

        assert delay_c() == 4.234

        delay_c.initialize(6)
        assert delay_c() == 6

        for p, i in enumerate(np.arange(0, 2.5, 0.5)):
            delay_c.update(None)
            assert delay_c() == 6
            assert delay_c.pointer == p+1

        delay_c.update(None)
        assert delay_c() == 5
        assert delay_c.pointer == 0

        # check that the pointer is set to 0 after initialization
        delay_c.update(None)
        assert delay_c.pointer != 0
        delay_c.initialize()
        assert delay_c.pointer == 0


    def test_delay_subscript(self):
        """
        Test for subscripted delay
        """

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

        assert delay().equals(xr_initial)
        delay_ddt = delay.ddt()[0].reset_coords('_delay', drop=True)

        assert delay_ddt.equals((xr_input-xr_initial)*2)

    def test_delay_order(self):
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

        warning_message = "Delay time very small, casting delay order "\
            "from 3 to 2"
        with pytest.warns(UserWarning, match=warning_message):
            delay1.initialize()

        warning_message = "Delay time very small, casting delay order "\
            "from 3 to 2"
        with pytest.warns(UserWarning, match=warning_message):
            delay2.initialize()

        warning_message = r"Casting delay order from 1\.5 to 1"
        with pytest.warns(UserWarning, match=warning_message):
            delay3.initialize()

        warning_message = r"Casting delay order from 1\.5 to 1"
        with pytest.warns(UserWarning, match=warning_message):
            delay4.initialize()

        warning_message = r"Casting delay order from 1\.500000 to 2"
        with pytest.warns(UserWarning, match=warning_message):
            delay5.initialize()

    def test_forecast(self):
        input_val = 5

        def input():
            return input_val

        frcst = Forecast(forecast_input=input,
                         average_time=lambda: 3,
                         horizon=lambda: 10,
                         initial_trend=lambda: 0,
                         py_name='forecast')

        frcst.initialize()
        assert frcst() == input_val

        frcst.state = frcst.state + 0.1*frcst.ddt()
        input_val = 20
        assert frcst() == 220

        frcst.state = frcst.state + 0.1*frcst.ddt()
        input_val = 35.5
        assert frcst()\
            == input_val*(1+(input_val-frcst.state)/(3*frcst.state)*10)

        input_val = 7
        init_trend = 6

        frcst.initialize(init_trend)
        assert frcst()\
            == input_val * (1+(input_val-input_val/(1+init_trend))
                            / (3*input_val/(1+init_trend))*10)

    def test_initial(self):
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
        assert f1_0 == f1_i0

        f2_0 = func2()
        initial_f20 = Initial(func2, "initial_f20")
        initial_f20.initialize()
        f2_i0 = initial_f20()
        assert f2_0 == f2_i0

        a = 5
        b = 6

        # test that the function returns the first value
        # after it is called a second time
        f1_1 = func1()
        f1_i1 = initial_f10()
        assert f1_1 != f1_i1
        assert f1_i1 == f1_0

        f2_1 = func2()
        f2_i1 = initial_f20()
        assert f2_1 != f2_i1
        assert f2_i1 == f2_0

        # test change initial condition
        initial_f20.initialize(123)
        assert initial_f20() == 123

    def test_sampleiftrue(self):
        bool_v = False

        def condition():
            return bool_v

        sif = SampleIfTrue(condition=condition,
                           actual_value=lambda: 3,
                           initial_value=lambda: 4.234,
                           py_name='sampleiftrue')

        sif.initialize()
        assert sif() == 4.234

        sif.initialize(6)
        assert sif() == 6

        bool_v = True
        assert sif() == 3

    def test_smooth(self):
        smooth = Smooth(smooth_input=lambda: 5,
                        smooth_time=lambda: 3,
                        initial_value=lambda: 4.234,
                        order=lambda: 3,
                        py_name='smooth')

        smooth.initialize()
        assert smooth() == 4.234

        smooth.initialize(6)
        assert smooth() == 6

    def test_trend(self):
        trend = Trend(trend_input=lambda: 5,
                      average_time=lambda: 3,
                      initial_trend=lambda: 4.234,
                      py_name='trend')

        trend.initialize()
        assert trend() == 4.234

        trend.initialize(6)
        assert round(trend(), 8) == 6


class TestStatefulErrors():
    def test_not_initialized_object(self):
        obj = Stateful()
        obj.py_name = "my_object"
        error_message = r"my_object\nAttempt to call stateful element"\
            r" before it is initialized\."
        with pytest.raises(AttributeError, match=error_message):
            obj.state


class TestMacroMethods():
    def test_get_elements_to_initialize(self, _root):
        from pysd import read_vensim
        from pysd.py_backend.model import Macro

        test_model = _root.joinpath("test-models/samples/teacup/teacup.mdl")
        read_vensim(test_model)
        macro = Macro(test_model.with_suffix(".py"))

        macro.stateful_initial_dependencies = {
            "A": {"B", "C"},
            "B": {"C"},
            "C": {},
            "D": {"E"},
            "E": {}
        }

        assert macro._get_elements_to_initialize(["A"]) == set()
        assert macro._get_elements_to_initialize(["B"]) == {"A"}
        assert macro._get_elements_to_initialize(["B", "A"]) == set()
        assert macro._get_elements_to_initialize(["C"]) == {"A", "B"}
        assert macro._get_elements_to_initialize(["A", "C"]) == {"B"}
        assert macro._get_elements_to_initialize(["C", "E"]) == {"A", "B", "D"}
