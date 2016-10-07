"""
functions.py

These are supports for functions that are included in modeling software but have no
straightforward equivalent in python.

"""

from __future__ import division
import numpy as np
from functools import wraps


try:
    import scipy.stats as stats

    def bounded_normal(minimum, maximum, mean, std, seed):
        """ Implements vensim's BOUNDED NORMAL function """
        # np.random.seed(seed)  # we could bring this back later, but for now, ignore
        return stats.truncnorm.rvs(minimum, maximum, loc=mean, scale=std)

except ImportError:
    import warnings
    warnings.warn("Scipy required for functions:"
                  "- Bounded Normal (falling back to unbounded normal)")

    def bounded_normal(minimum, maximum, mean, std, seed):
        """ Warning: using unbounded normal due to no scipy """
        return np.random.normal(mean, std)


def cache(horizon):
    """
    Put a wrapper around a model function

    Decorators with parameters are tricky, you have to
    essentially create a decorator that returns a decorator,
    which itself then returns the function wrapper.

    Parameters
    ----------
    horizon: string
        - 'step' means cache just until the next timestep
        - 'run' means cache until the next initialization of the model

    Returns
    -------
    new_func: decorated function
        function wrapping the original function, handling caching

    """
    def cache_step(func):
        """ Decorator for caching at a step level"""
        @wraps(func)
        def cached(*args):
            """Step wise cache function"""
            try:  # fails if cache is out of date or not instantiated
                assert cached.t == func.__globals__['_t']
                assert hasattr(cached, 'cache_val')
            except (AssertionError, AttributeError):
                cached.cache_val = func(*args)
                cached.cache_t = func.__globals__['_t']
            return cached.cache_val

        return cached

    def cache_run(func):
        """ Decorator for caching at  the run level"""
        @wraps(func)
        def cached(*args):
            """Run wise cache function"""
            try:  # fails if cache is not instantiated
                return cached.cache_val
            except AttributeError:
                cached.cache_val = func(*args)
                return cached.cache_val
        return cached

    if horizon == 'step':
        return cache_step

    elif horizon == 'run':
        return cache_run

    else:
        raise(AttributeError('Bad horizon for cache decorator'))


class Stateful(object):
    # the integrator needs to be able to 'get' the current state of the object,
    # and get the derivative. It calculates the new state, and updates it. The state
    # can be any object which is subject to basic (element-wise) algebraic operations
    def __init__(self):
        self._state = None

    def __call__(self, *args, **kwargs):
        return self.state

    def initialize(self):
        raise NotImplementedError

    @property
    def state(self):
        if self._state is None:
            raise AttributeError('Attempt to call stateful element before it is initialized.')
        return self._state

    @state.setter
    def state(self, new_value):
        self._state = new_value

    def update(self, state):
        self.state = state


class Integ(Stateful):
    def __init__(self, ddt, initial_value):
        """

        Parameters
        ----------
        ddt: function
            This will become an attribute of the object
        initial_value
        """
        super(Integ, self).__init__()
        self.init_func = initial_value
        self.ddt = ddt

    def initialize(self):
        self.state = self.init_func()


class Delay(Stateful):
    # note that we could have put the `delay_input` argument as a parameter to
    # the `__call__` function, and more closely mirrored the vensim syntax.
    # However, people may get confused this way in thinking that they need only one
    # delay object and can call it with various arguments to delay whatever is convenient.
    # This method forces them to acknowledge that additional structure is being created
    # in the delay object.

    def __init__(self, delay_input, delay_time, initial_value, order):
        """

        Parameters
        ----------
        delay_input: function
        delay_time: function
        initial_value: function
        order: function
        """
        super(Delay, self).__init__()
        self.init_func = initial_value
        self.delay_time_func = delay_time
        self.input_func = delay_input
        self.order_func = order
        self.order = None

    def initialize(self):
        self.order = self.order_func()  # The order can only be set once
        init_state_value = self.init_func() * self.delay_time_func() / self.order
        self.state = np.array([init_state_value] * self.order)

    def __call__(self):
        return self.state[-1] / (self.delay_time_func() / self.order)

    def ddt(self):
        outflows = self.state / (self.delay_time_func() / self.order)
        inflows = np.roll(outflows, 1)
        inflows[0] = self.input_func()
        return inflows - outflows


class Smooth(Stateful):
    def __init__(self, smooth_input, smooth_time, initial_value, order):
        super(Smooth, self).__init__()
        self.init_func = initial_value
        self.smooth_time_func = smooth_time
        self.input_func = smooth_input
        self.order_func = order
        self.order = None

    def initialize(self):
        self.order = self.order_func()  # The order can only be set once
        self.state = np.array([self.init_func()] * self.order)

    def __call__(self):
        return self.state[-1]

    def ddt(self):
        targets = np.roll(self.state, 1)
        targets[0] = self.input_func()
        return (targets - self.state) * self.order / self.smooth_time_func()


class Initial(Stateful):
    def __init__(self, func):
        super(Initial, self).__init__()
        self.func = func

    def initialize(self):
        self.state = self.func()

    def ddt(self):
        return 0

    def update(self, state):
        # this doesn't change once it's set up.
        pass


def ramp(slope, start, finish):
    """
    Implements vensim's RAMP function

    Parameters
    ----------
    slope: float
        The slope of the ramp starting at zero at time start
    start: float
        Time at which the ramp begins
    finish: float
        Time at which the ramo ends

    Returns
    -------
    response: float
        If prior to ramp start, returns zero
        If after ramp ends, returns top of ramp
    Examples
    --------

    """

    t = time()
    if t < start:
        return 0
    elif t > finish:
        return slope * (finish-start)
    else:
        return slope * (t-start)


def step(value, tstep):
    """"
    Implements vensim's STEP function

    Parameters
    ----------
    value: float
        The height of the step
    tstep: float
        The time at and after which `result` equals `value`

    Returns
    -------
    - In range [-inf, tstep) returns 0
    - In range [tstep, +inf] returns `value`
    """
    return value if time() >= tstep else 0


def pulse(start, duration):
    """ Implements vensim's PULSE function

    In range [-inf, start) returns 0
    In range [start, start + duration) returns 1
    In range [start + duration, +inf] returns 0
    """
    t = time()
    return 1 if start <= t < start + duration else 0


def pulse_train(start, duration, repeat_time, end):
    """ Implements vensim's PULSE TRAIN function

    In range [-inf, start) returns 0
    In range [start + n * repeat_time, start + n * repeat_time + duration) return 1
    In range [start + n * repeat_time + duration, start + (n+1) * repeat_time) return 0
    """
    t = time()
    if start <= t < end:
        return 1 if (t-start) % repeat_time < duration else 0
    else:
        return 0


def lookup(x, xs, ys):
    """ Provides the working mechanism for lookup functions the builder builds """
    return np.interp(x, xs, ys)


def if_then_else(condition, val_if_true, val_if_false):
    return np.where(condition, val_if_true, val_if_false)


def xidz(numerator, denominator, value_if_denom_is_zero):
    """
    Implements Vensim's XIDZ function.
    This function executes a division, robust to denominator being zero.
    In the case of zero denominator, the final argument is returned.

    Parameters
    ----------
    numerator: float
    denominator: float
        Components of the division operation
    value_if_denom_is_zero: float
        The value to return if the denominator is zero

    Returns
    -------
    numerator / denominator if denominator > 1e-6
    otherwise, returns value_if_denom_is_zero
    """
    small = 1e-6  # What is considered zero according to Vensim Help
    if abs(denominator) < small:
        return value_if_denom_is_zero
    else:
        return numerator*1.0/denominator


def zidz(numerator, denominator):
    """
    This function bypasses divide-by-zero errors,
    implementing Vensim's ZIDZ function

    Parameters
    ----------
    numerator: float
        value to be divided
    denominator: float
        value to devide by

    Returns
    -------
    result of division numerator/denominator if denominator is not zero,
    otherwise zero.
    """
    # Todo: make this work for arrays
    small = 1e-6  # What is considered zero according to Vensim Help
    if abs(denominator) < small:
        return 0
    else:
        return numerator*1.0/denominator


def active_initial(expr, init_val):
    """
    Implements vensim's ACTIVE INITIAL function
    Parameters
    ----------
    expr
    init_val

    Returns
    -------

    """
    if _stage() == 'Initialization':
        return init_val
    else:
        return expr

def random_uniform(m, x, s):
    return np.random.uniform(m, x)
