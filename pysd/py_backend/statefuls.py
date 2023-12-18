"""
The Stateful objects are used and updated each time step with an update
method. This include Integs, Delays, Forecasts, Smooths, and Trends,
between others. The Macro class and Model class are also Stateful type.
However, they are defined appart as they are more complex.
"""
import warnings

import numpy as np
import xarray as xr

from .functions import zidz, if_then_else


SMALL_VENSIM = 1e-6  # What is considered zero according to Vensim Help


class Stateful(object):
    # the integrator needs to be able to 'get' the current state of the object,
    # and get the derivative. It calculates the new state, and updates it.
    # The state can be any object which is subject to basic (element-wise)
    # algebraic operations
    def __init__(self):
        self._state = None
        self.shape_info = None
        self.py_name = ""

    def __call__(self, *args, **kwargs):
        return self.state

    @property
    def state(self):
        if self._state is None:
            raise AttributeError(
                self.py_name
                + "\nAttempt to call stateful element"
                + " before it is initialized.")
        return self._state

    @state.setter
    def state(self, new_value):
        if self.shape_info:
            self._state = xr.DataArray(data=new_value, **self.shape_info)
        else:
            self._state = new_value


class DynamicStateful(Stateful):

    def __init__(self):
        super().__init__()

    def update(self, state):
        try:
            self.state = state
        except Exception as err:
            raise ValueError(err.args[0] + "\n\n"
                             + "Could not update the value of "
                             + self.py_name)


class Integ(DynamicStateful):
    """
    Implements INTEG function.

    Parameters
    ----------
    ddt: callable
        Derivate to integrate.
    initial_value: callable
        Initial value.
    py_name: str
        Python name to identify the object.

    Attributes
    ----------
    state: float or xarray.DataArray
        Current state of the object. Value of the stock.

    """
    def __init__(self, ddt, initial_value, py_name):
        super().__init__()
        self.init_func = initial_value
        self.ddt = ddt
        self.shape_info = None
        self.py_name = py_name

    def initialize(self, init_val=None):
        if init_val is None:
            self.state = self.init_func()
        else:
            self.state = init_val
        if isinstance(self.state, xr.DataArray):
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}

    def export(self):
        return {'state': self.state, 'shape_info': self.shape_info}


class NonNegativeInteg(Integ):
    """
    Implements non negative INTEG function.

    Parameters
    ----------
    ddt: callable
        Derivate to integrate.
    initial_value: callable
        Initial value.
    py_name: str
        Python name to identify the object.

    Attributes
    ----------
    state: float or xarray.DataArray
        Current state of the object. Value of the stock.

    """
    def __init__(self, ddt, initial_value, py_name):
        super().__init__(ddt, initial_value, py_name)

    def update(self, state):
        self.state = np.maximum(state, 0)


class Delay(DynamicStateful):
    """
    Implements DELAY function.

    Parameters
    ----------
    delay_input: callable
        Input of the delay.
    delay_time: callable
        Delay time.
    initial_value: callable
        Initial value.
    order: callable
        Delay order.
    tsetp: callable
        The time step of the model.
    py_name: str
        Python name to identify the object.

    Attributes
    ----------
    state: numpy.array or xarray.DataArray
        Current state of the object. Array of the delays values multiplied
        by their corresponding average time.

    """
    # note that we could have put the `delay_input` argument as a parameter to
    # the `__call__` function, and more closely mirrored the vensim syntax.
    # However, people may get confused this way in thinking that they need
    # only one delay object and can call it with various arguments to delay
    # whatever is convenient. This method forces them to acknowledge that
    # additional structure is being created in the delay object.

    def __init__(self, delay_input, delay_time, initial_value, order, tstep,
                 py_name):
        super().__init__()
        self.init_func = initial_value
        self.delay_time_func = delay_time
        self.input_func = delay_input
        self.order_func = order
        self.order = None
        self.tstep = tstep
        self.shape_info = None
        self.py_name = py_name

    def initialize(self, init_val=None):
        order = self.order_func()

        if order != int(order):
            warnings.warn(self.py_name + '\n' +
                          'Casting delay order '
                          + f'from {order} to {int(order)}')

        self.order = int(order)  # The order can only be set once
        if self.order*self.tstep() > np.min(self.delay_time_func()):
            while self.order*self.tstep() > np.min(self.delay_time_func()):
                self.order -= 1
            warnings.warn(self.py_name + '\n' +
                          'Delay time very small, casting delay order '
                          + f'from {int(order)} to {self.order}')

        if init_val is None:
            init_state_value = self.init_func() * self.delay_time_func()
        else:
            init_state_value = init_val * self.delay_time_func()

        if isinstance(init_state_value, xr.DataArray):
            # broadcast self.state
            self.state = init_state_value.expand_dims({
                '_delay': np.arange(self.order)}, axis=0)
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}
        else:
            self.state = np.array([init_state_value] * self.order)

    def __call__(self):
        if self.shape_info:
            return self.state[-1].reset_coords('_delay', drop=True)\
                   / self.delay_time_func()
        else:
            return self.state[-1] / self.delay_time_func()

    def ddt(self):
        outflows = self.state / self.delay_time_func()
        inflows = np.roll(outflows, 1, axis=0)
        if self.shape_info:
            inflows[0] = self.input_func().values
        else:
            inflows[0] = self.input_func()
        return (inflows - outflows) * self.order

    def export(self):
        return {'state': self.state, 'shape_info': self.shape_info}


class DelayN(DynamicStateful):
    """
    Implements DELAY N function.

    Parameters
    ----------
    delay_input: callable
        Input of the delay.
    delay_time: callable
        Delay time.
    initial_value: callable
        Initial value.
    order: callable
        Delay order.
    tsetp: callable
        The time step of the model.
    py_name: str
        Python name to identify the object.

    Attributes
    ----------
    state: numpy.array or xarray.DataArray
        Current state of the object. Array of the delays values multiplied
        by their corresponding average time.

    times: numpy.array or xarray.DataArray
        Array of delay times used for computing the delay output.
        If delay_time is constant, this array will be constant and
        DelayN will behave ad Delay.

    """
    # note that we could have put the `delay_input` argument as a parameter to
    # the `__call__` function, and more closely mirrored the vensim syntax.
    # However, people may get confused this way in thinking that they need
    # only one delay object and can call it with various arguments to delay
    # whatever is convenient. This method forces them to acknowledge that
    # additional structure is being created in the delay object.

    def __init__(self, delay_input, delay_time, initial_value, order, tstep,
                 py_name):
        super().__init__()
        self.init_func = initial_value
        self.delay_time_func = delay_time
        self.input_func = delay_input
        self.order_func = order
        self.order = None
        self.times = None
        self.tstep = tstep
        self.shape_info = None
        self.py_name = py_name

    def initialize(self, init_val=None):
        order = self.order_func()

        if order != int(order):
            warnings.warn(self.py_name + '\n' +
                          'Casting delay order '
                          + f'from {order} to {int(order)}')

        self.order = int(order)  # The order can only be set once
        if self.order*self.tstep() > np.min(self.delay_time_func()):
            while self.order*self.tstep() > np.min(self.delay_time_func()):
                self.order -= 1
            warnings.warn(self.py_name + '\n' +
                          'Delay time very small, casting delay order '
                          + f'from {int(order)} to {self.order}')

        if init_val is None:
            init_state_value = self.init_func() * self.delay_time_func()
        else:
            init_state_value = init_val * self.delay_time_func()

        if isinstance(init_state_value, xr.DataArray):
            # broadcast self.state
            self.state = init_state_value.expand_dims({
                '_delay': np.arange(self.order)}, axis=0)
            self.times = self.delay_time_func().expand_dims({
                '_delay': np.arange(self.order)}, axis=0)
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}
        else:
            self.state = np.array([init_state_value] * self.order)
            self.times = np.array([self.delay_time_func()] * self.order)

    def __call__(self):
        if self.shape_info:
            return self.state[-1].reset_coords('_delay', drop=True)\
                   / self.times[0].reset_coords('_delay', drop=True)
        else:
            return self.state[-1] / self.times[0]

    def ddt(self):
        if self.shape_info:
            # if is xarray need to preserve coords
            self.times = self.times.roll({'_delay': 1}, False)
            self.times[0] = self.delay_time_func()
            outflows = self.state / self.times
            inflows = outflows.roll({'_delay': 1}, False)
        else:
            # if is float use numpy.roll
            self.times = np.roll(self.times, 1, axis=0)
            self.times[0] = self.delay_time_func()
            outflows = self.state / self.times
            inflows = np.roll(outflows, 1, axis=0)

        inflows[0] = self.input_func()
        return (inflows - outflows)*self.order

    def export(self):
        return {'state': self.state, 'times': self.times,
                'shape_info': self.shape_info}


class DelayFixed(DynamicStateful):
    """
    Implements DELAY FIXED function.

    Parameters
    ----------
    delay_input: callable
        Input of the delay.
    delay_time: callable
        Delay time.
    initial_value: callable
        Initial value.
    tsetp: callable
        The time step of the model.
    py_name: str
        Python name to identify the object.

    Attributes
    ----------
    state: float or xarray.DataArray
        Current state of the object, equal to pipe[pointer].
    pipe: list
        List of the delays values.
    pointer: int
        Pointer to the last value in the pipe

    """

    def __init__(self, delay_input, delay_time, initial_value, tstep,
                 py_name):
        super().__init__()
        self.init_func = initial_value
        self.delay_time_func = delay_time
        self.input_func = delay_input
        self.tstep = tstep
        self.order = None
        self.pointer = 0
        self.py_name = py_name

    def initialize(self, init_val=None):
        order = max(self.delay_time_func()/self.tstep(), 1)

        if order != int(order):
            warnings.warn(
                self.py_name + '\n'
                + 'Casting delay order from %f to %i' % (
                    order, round(order + SMALL_VENSIM)))

        # need to add a small decimal to ensure that 0.5 is rounded to 1
        # The order can only be set once
        self.order = round(order + SMALL_VENSIM)

        # set the pointer to 0
        self.pointer = 0

        if init_val is None:
            init_state_value = self.init_func()
        else:
            init_state_value = init_val

        self.state = init_state_value
        self.pipe = [init_state_value] * self.order

    def __call__(self):
        return self.state

    def ddt(self):
        return np.nan

    def update(self, state):
        self.pipe[self.pointer] = self.input_func()
        self.pointer = (self.pointer + 1) % self.order
        self.state = self.pipe[self.pointer]

    def export(self):
        return {'state': self.state, 'pointer': self.pointer,
                'pipe': self.pipe}


class Forecast(DynamicStateful):
    """
    Implements FORECAST function.

    Parameters
    ----------
    forecast_input: callable
        Input of the forecast.
    average_time: callable
        Average time.
    horizon: callable
        Forecast horizon.
    initial_trend: callable
        Initial trend of the forecast.
    py_name: str
        Python name to identify the object.

    Attributes
    ----------
    state: float or xarray.DataArray
        Current state of the object. AV value by Vensim docs.

    """
    def __init__(self, forecast_input, average_time, horizon, initial_trend,
                 py_name):
        super().__init__()
        self.horizon = horizon
        self.average_time = average_time
        self.input = forecast_input
        self.initial_trend = initial_trend
        self.py_name = py_name

    def initialize(self, init_trend=None):

        # self.state = AV in the vensim docs
        if init_trend is None:
            self.state = self.input() / (1 + self.initial_trend())
        else:
            self.state = self.input() / (1 + init_trend)

        if isinstance(self.state, xr.DataArray):
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}

    def __call__(self):
        return self.input() * (
            1 + zidz(self.input() - self.state,
                     self.average_time() * self.state
                     )*self.horizon()
        )

    def ddt(self):
        return (self.input() - self.state) / self.average_time()

    def export(self):
        return {'state': self.state, 'shape_info': self.shape_info}


class Smooth(DynamicStateful):
    """
    Implements SMOOTH function.

    Parameters
    ----------
    smooth_input: callable
        Input of the smooth.
    smooth_time: callable
        Smooth time.
    initial_value: callable
        Initial value.
    order: callable
        Delay order.
    py_name: str
        Python name to identify the object.

    Attributes
    ----------
    state: numpy.array or xarray.DataArray
        Current state of the object. Array of the inputs having the
        value to return in the last position.

    """
    def __init__(self, smooth_input, smooth_time, initial_value, order,
                 py_name):
        super().__init__()
        self.init_func = initial_value
        self.smooth_time_func = smooth_time
        self.input_func = smooth_input
        self.order_func = order
        self.order = None
        self.shape_info = None
        self.py_name = py_name

    def initialize(self, init_val=None):
        self.order = self.order_func()  # The order can only be set once

        if init_val is None:
            init_state_value = self.init_func()
        else:
            init_state_value = init_val

        if isinstance(init_state_value, xr.DataArray):
            # broadcast self.state
            self.state = init_state_value.expand_dims({
                '_smooth': np.arange(self.order)}, axis=0)
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}
        else:
            self.state = np.array([init_state_value] * self.order)

    def __call__(self):
        if self.shape_info:
            return self.state[-1].reset_coords('_smooth', drop=True)
        else:
            return self.state[-1]

    def ddt(self):
        targets = np.roll(self.state, 1, axis=0)
        if self.shape_info:
            targets[0] = self.input_func().values
        else:
            targets[0] = self.input_func()
        return (targets - self.state) * self.order / self.smooth_time_func()

    def export(self):
        return {'state': self.state, 'shape_info': self.shape_info}


class Trend(DynamicStateful):
    """
    Implements TREND function.

    Parameters
    ----------
    trend_input: callable
        Input of the trend.
    average_time: callable
        Average time.
    initial_trend: callable
        Initial trend.
    py_name: str
        Python name to identify the object.

    Attributes
    ----------
    state: float or xarray.DataArray
        Current state of the object. AV value by Vensim docs.

    """
    def __init__(self, trend_input, average_time, initial_trend, py_name):
        super().__init__()
        self.init_func = initial_trend
        self.average_time_function = average_time
        self.input_func = trend_input
        self.py_name = py_name

    def initialize(self, init_trend=None):
        if init_trend is None:
            self.state = self.input_func()\
                / (1 + self.init_func()*self.average_time_function())
        else:
            self.state = self.input_func()\
                / (1 + init_trend*self.average_time_function())

        if isinstance(self.state, xr.DataArray):
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}

    def __call__(self):
        return zidz(self.input_func() - self.state,
                    self.average_time_function() * np.abs(self.state))

    def ddt(self):
        return (self.input_func() - self.state) / self.average_time_function()

    def export(self):
        return {'state': self.state, 'shape_info': self.shape_info}


class SampleIfTrue(DynamicStateful):
    """
    Implements SAMPLE IF TRUE function.

    Parameters
    ----------
    condition: callable
        Condition for sample.
    actual_value: callable
        Value to update if condition is true.
    initial_value: callable
        Initial value.
    py_name: str
        Python name to identify the object.

    Attributes
    ----------
    state: float or xarray.DataArray
        Current state of the object. Last actual_value when condition
        was true or the initial_value if condition has never been true.

    """
    def __init__(self, condition, actual_value, initial_value, py_name):
        super().__init__()
        self.condition = condition
        self.actual_value = actual_value
        self.init_func = initial_value
        self.py_name = py_name

    def initialize(self, init_val=None):
        if init_val is None:
            self.state = self.init_func()
        else:
            self.state = init_val
        if isinstance(self.state, xr.DataArray):
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}

    def __call__(self):
        return if_then_else(self.condition(),
                            self.actual_value,
                            lambda: self.state)

    def ddt(self):
        return np.nan

    def update(self, state):
        self.state = self.state*0 + if_then_else(self.condition(),
                                                 self.actual_value,
                                                 lambda: self.state)

    def export(self):
        return {'state': self.state, 'shape_info': self.shape_info}


class Initial(Stateful):
    """
    Implements INITIAL function.

    Parameters
    ----------
    initial_value: callable
        Initial value.
    py_name: str
        Python name to identify the object.

    Attributes
    ----------
    state: float or xarray.DataArray
        Current state of the object, which will always be the initial_value.

    """
    def __init__(self, initial_value, py_name):
        super().__init__()
        self.init_func = initial_value
        self.py_name = py_name

    def initialize(self, init_val=None):
        if init_val is None:
            self.state = self.init_func()
        else:
            self.state = init_val

    def export(self):
        return {'state': self.state}
