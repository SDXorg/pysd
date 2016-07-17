"""
functions.py

These are supports for functions that are included in modeling software but have no
straightforward equivalent in python.

"""

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
                assert cached.t == func.func_globals['_t']
            except (AssertionError, AttributeError):
                cached.cache = func(*args)
                cached.t = func.func_globals['_t']
            return cached.cache
        return cached

    def cache_run(func):
        """ Decorator for caching at  the run level"""
        @wraps(func)
        def cached(*args):
            """Run wise cache function"""
            try:  # fails if cache is not instantiated
                return cached.cache
            except AttributeError:
                cached.cache = func(*args)
                return cached.cache
        return cached

    if horizon == 'step':
        return cache_step

    elif horizon == 'run':
        return cache_run

    else:
        raise(AttributeError('Bad horizon for cache decorator'))

@cache('run')
def initial(value):
    """
    This function returns the first value passed in,
    regardless of how many times it is called

    Uses the @cache('run') functionality.

    Parameters
    ----------
    value

    Returns
    -------
    The first value of `value` after the caches are reset
    """
    return value

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

def xidz(numerator,denominator,value_if_denom_is_zero):
    """ Implements Vensim's XIDZ function, which takes as arguments numerator, denominator of a fraction and a value to return if the denominator is close to zero, returning the fraction otherwise. This function bypasses divide-by-zero errors
    """
    small = 1e-6 ## What is considered zero according to Vensim Help
    if abs(denominator) < small:
        return value_if_denom_is_zero
    else:
        return numerator*1.0/denominator

def zidz(numerator,denominator):
    """ Implements Vensim's ZIDZ function, which takes as arguments numerator and denominator of a fraction to be returned if the denominator is not close to zero, and zero otherwise. This function bypasses divide-by-zero errors
    """
    small = 1e-6 ## What is considered zero according to Vensim Help
    if abs(denominator) < small:
        return 0
    else:
        return numerator*1.0/denominator

def active_initial(expr, initval):
    """
    Implements vensim's ACTIVE INITIAL function
    Parameters
    ----------
    expr
    initval

    Returns
    -------

    """
    if time() == initial_time():
        return initval
    else:
        return expr

