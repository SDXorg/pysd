"""
These functions have no direct analog in the standard python data analytics
stack, or require information about the internal state of the system beyond
what is present in the function call. We provide them in a structure that
makes it easy for the model elements to call.
"""

import warnings

import numpy as np
import xarray as xr
import scipy.stats as stats

small_vensim = 1e-6  # What is considered zero according to Vensim Help


def ramp(time, slope, start, finish=0):
    """
    Implements vensim's and xmile's RAMP function.

    Parameters
    ----------
    time: function
        The current time of modelling.
    slope: float
        The slope of the ramp starting at zero at time start.
    start: float
        Time at which the ramp begins.
    finish: float
        Optional. Time at which the ramp ends.

    Returns
    -------
    response: float
        If prior to ramp start, returns zero.
        If after ramp ends, returns top of ramp.

    """
    t = time()
    if t < start:
        return 0
    else:
        if finish <= 0:
            return slope * (t - start)
        elif t > finish:
            return slope * (finish - start)
        else:
            return slope * (t - start)


def step(time, value, tstep):
    """"
    Implements vensim's STEP function.

    Parameters
    ----------
    value: float
        The height of the step.
    tstep: float
        The time at and after which `result` equals `value`.

    Returns
    -------
    float:
        - In range [-inf, tstep):
            returns 0
        - In range [tstep, +inf]:
            returns `value`
    """
    return value if time() >= tstep else 0


def pulse(time, start, duration):
    """
    Implements vensim's PULSE function.

    Parameters
    ----------
    time: function
        Function that returns the current time.

    start: float
        Starting time of the pulse.

    duration: float
        Duration of the pulse.

    Returns
    -------
    float:
        - In range [-inf, start):
            returns 0
        - In range [start, start + duration):
            returns 1
        - In range [start + duration, +inf]:
            returns 0

    """
    t = time()
    return 1 if start <= t < start + duration else 0


def pulse_train(time, start, duration, repeat_time, end):
    """
    Implements vensim's PULSE TRAIN function.

    Parameters
    ----------
    time: function
        Function that returns the current time.

    start: float
        Starting time of the pulse.

    duration: float
        Duration of the pulse.

    repeat_time: float
        Time interval of the pulse repetition.

    end: float
        Final time of the pulse.

    Returns
    -------
    float:
        - In range [-inf, start):
            returns 0
        - In range [start + n*repeat_time, start + n*repeat_time + duration):
            returns 1
        - In range [start + n*repeat_time + duration,
                    start + (n+1)*repeat_time):
            returns 0

    """
    t = time()
    if start <= t < end:
        return 1 if (t - start) % repeat_time < duration else 0
    else:
        return 0


def pulse_magnitude(time, magnitude, start, repeat_time=0):
    """
    Implements xmile's PULSE function. Generate a one-DT wide pulse
    at the given time.

    Parameters
    ----------
    time: function
        Function that returns the current time.

    magnitude:
        Magnitude of the pulse.

    start: float
        Starting time of the pulse.

    repeat_time: float (optional)
        Time interval of the pulse repetition.  Default is 0, only one
        pulse will be generated.

    Notes
    -----
    PULSE(time(), 20, 12, 5) generates a pulse value of 20/DT at
    time 12, 17, 22, etc.

    Returns
    -------
    float:
        - In rage [-inf, start):
            returns 0
        - In range [start + n*repeat_time, start + n*repeat_time + dt):
            returns magnitude/dt
        - In rage [start + n*repeat_time + dt, start + (n+1)*repeat_time):
            returns 0

    """
    t = time()
    if repeat_time <= small_vensim:
        if abs(t - start) < time.step():
            return magnitude * time.step()
        else:
            return 0
    else:
        if abs((t - start) % repeat_time) < time.step():
            return magnitude * time.step()
        else:
            return 0


def lookup(x, xs, ys):
    """
    Intermediate values are calculated with linear interpolation between
    the intermediate points. Out-of-range values are the same as the
    closest endpoint (i.e, no extrapolation is performed).
    """
    return np.interp(x, xs, ys)


def lookup_extrapolation(x, xs, ys):
    """
    Intermediate values are calculated with linear interpolation between
    the intermediate points. Out-of-range values are calculated with linear
    extrapolation from the last two values at either end.
    """
    if x < xs[0]:
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        k = dy / dx
        return ys[0] + (x - xs[0]) * k
    if x > xs[-1]:
        dx = xs[-1] - xs[-2]
        dy = ys[-1] - ys[-2]
        k = dy / dx
        return ys[-1] + (x - xs[-1]) * k
    return np.interp(x, xs, ys)


def lookup_discrete(x, xs, ys):
    """
    Intermediate values take on the value associated with the next lower
    x-coordinate (also called a step-wise function). The last two points
    of a discrete graphical function must have the same y value.
    Out-of-range values are the same as the closest endpoint
    (i.e, no extrapolation is performed).
    """
    for index in range(0, len(xs)):
        if x < xs[index]:
            return ys[index - 1] if index > 0 else ys[index]
    return ys[-1]


def if_then_else(condition, val_if_true, val_if_false):
    """
    Implements Vensim's IF THEN ELSE function.
    https://www.vensim.com/documentation/20475.htm

    Parameters
    ----------
    condition: bool or xarray.DataArray of bools
    val_if_true: function
        Value to evaluate and return when condition is true.
    val_if_false: function
        Value to evaluate and return when condition is false.

    Returns
    -------
    The value depending on the condition.

    """
    if isinstance(condition, xr.DataArray):
        if condition.all():
            return val_if_true()
        elif not condition.any():
            return val_if_false()

        return xr.where(condition, val_if_true(), val_if_false())

    return val_if_true() if condition else val_if_false()


def logical_and(*args):
    """
    Implements Vensim's :AND: method for two or several arguments.

    Parameters
    ----------
    *args: arguments
        The values to compare with and operator

    Returns
    -------
    result: bool or xarray.DataArray
        The result of the comparison.

    """
    current = args[0]
    for arg in args[1:]:
        current = np.logical_and(arg, current)
    return current


def logical_or(*args):
    """
    Implements Vensim's :OR: method for two or several arguments.

    Parameters
    ----------
    *args: arguments
        The values to compare with and operator

    Returns
    -------
    result: bool or xarray.DataArray
        The result of the comparison.

    """
    current = args[0]
    for arg in args[1:]:
        current = np.logical_or(arg, current)
    return current


def xidz(numerator, denominator, value_if_denom_is_zero):
    """
    Implements Vensim's XIDZ function.
    https://www.vensim.com/documentation/fn_xidz.htm

    This function executes a division, robust to denominator being zero.
    In the case of zero denominator, the final argument is returned.

    Parameters
    ----------
    numerator: float or xarray.DataArray
    denominator: float or xarray.DataArray
        Components of the division operation
    value_if_denom_is_zero: float or xarray.DataArray
        The value to return if the denominator is zero

    Returns
    -------
    numerator / denominator if denominator > 1e-6
    otherwise, returns value_if_denom_is_zero

    """
    if isinstance(denominator, xr.DataArray):
        return xr.where(np.abs(denominator) < small_vensim,
                        value_if_denom_is_zero,
                        numerator * 1.0 / denominator)

    if abs(denominator) < small_vensim:
        return value_if_denom_is_zero
    else:
        return numerator * 1.0 / denominator


def zidz(numerator, denominator):
    """
    This function bypasses divide-by-zero errors,
    implementing Vensim's ZIDZ function
    https://www.vensim.com/documentation/fn_zidz.htm

    Parameters
    ----------
    numerator: float or xarray.DataArray
        value to be divided
    denominator: float or xarray.DataArray
        value to devide by

    Returns
    -------
    result of division numerator/denominator if denominator is not zero,
    otherwise zero.

    """
    if isinstance(denominator, xr.DataArray):
        return xr.where(np.abs(denominator) < small_vensim,
                        0,
                        numerator * 1.0 / denominator)

    if abs(denominator) < small_vensim:
        return 0
    else:
        return numerator * 1.0 / denominator


def active_initial(time, expr, init_val):
    """
    Implements vensim's ACTIVE INITIAL function
    Parameters
    ----------
    time: function
        The current time function
    expr
    init_val

    Returns
    -------

    """
    if time.stage == 'Initialization':
        return init_val
    else:
        return expr()


def bounded_normal(minimum, maximum, mean, std, seed):
    """
    Implements vensim's BOUNDED NORMAL function
    """
    # np.random.seed(seed)
    # we could bring this back later, but for now, ignore
    return stats.truncnorm.rvs(minimum, maximum, loc=mean, scale=std)


def incomplete(*args):
    warnings.warn(
        'Call to undefined function, calling dependencies and returning NaN',
        RuntimeWarning, stacklevel=2)

    return np.nan


def not_implemented_function(*args):
    raise NotImplementedError(
        'Not implemented function {}'.format(args[0]))


def log(x, base):
    """
    Implements Vensim's LOG function with change of base.

    Parameters
    ----------
    x: input value
    base: base of the logarithm

    Returns
    -------
    float
      the log of 'x' in base 'base'
    """
    return np.log(x) / np.log(base)


def integer(x):
    """
    Implements Vensim's INTEGER function.

    Parameters
    ----------
    x: float or xarray.DataArray
        Input value.

    Returns
    -------
    Returns integer part of x.

    """
    if isinstance(x, xr.DataArray):
        return x.astype(int)
    else:
        return int(x)


def quantum(a, b):
    """
    Implements Vensim's QUANTUM function.

    Parameters
    ----------
    a: float or xarray.DataArray
        Input value.
    b: float or xarray.DataArray
        Input value.

    Returns
    -------
    quantum: float or xarray.DataArray
        If b > 0 returns b * integer(a/b). Otherwise, returns a.

    """
    if isinstance(b, xr.DataArray):
        return xr.where(b < small_vensim, a, b*integer(a/b))
    if b < small_vensim:
        return a
    else:
        return b*integer(a/b)


def modulo(x, m):
    """
    Implements Vensim's MODULO function.

    Parameters
    ----------
    x: float or xarray.DataArray
        Input value.

    m: float or xarray.DataArray
        Modulo to compute.

    Returns
    -------
    Returns x modulo m, if x is smaller than 0 the result is given in
    the range (-m, 0] as Vensim does. x - quantum(x, m)

    """
    return x - quantum(x, m)


def sum(x, dim=None):
    """
    Implements Vensim's SUM function.

    Parameters
    ----------
    x: xarray.DataArray
      Input value
    dim: list of strs (optional)
      Dimensions to apply the function over.
      If not given the function will be applied over all dimensions

    Returns
    -------
    xarray.DataArray or float
      The result of the sum operation in the given dimensions

    """
    # float returned if the function is applied over all the dimensions
    if dim is None or set(x.dims) == set(dim):
        return float(x.sum())

    return x.sum(dim=dim)


def prod(x, dim=None):
    """
    Implements Vensim's PROD function.

    Parameters
    ----------
    x: xarray.DataArray
      Input value
    dim: list of strs (optional)
      Dimensions to apply the function over.
      If not given the function will be applied over all dimensions

    Returns
    -------
    xarray.DataArray or float
      The result of the product operation in the given dimensions

    """
    # float returned if the function is applied over all the dimensions
    if dim is None or set(x.dims) == set(dim):
        return float(x.prod())

    return x.prod(dim=dim)


def vmin(x, dim=None):
    """
    Implements Vensim's Vmin function.

    Parameters
    ----------
    x: xarray.DataArray
      Input value
    dim: list of strs (optional)
      Dimensions to apply the function over.
      If not given the function will be applied over all dimensions

    Returns
    -------
    xarray.DataArray or float
      The result of the minimum value over the given dimensions

    """
    # float returned if the function is applied over all the dimensions
    if dim is None or set(x.dims) == set(dim):
        return float(x.min())

    return x.min(dim=dim)


def vmax(x, dim=None):
    """
    Implements Vensim's VMAX function.

    Parameters
    ----------
    x: xarray.DataArray
      Input value
    dim: list of strs (optional)
      Dimensions to apply the function over.
      If not given the function will be applied over all dimensions

    Returns
    -------
    xarray.DataArray or float
      The result of the maximum value over the dimensions

    """
    # float returned if the function is applied over all the dimensions
    if dim is None or set(x.dims) == set(dim):
        return float(x.max())

    return x.max(dim=dim)


def invert_matrix(mat):
    """
    Implements Vensim's INVERT MATRIX function.

    Invert the matrix defined by the last two dimensions of xarray.DataArray.

    Paramteters
    -----------
    mat: xarray.DataArray
        The matrix to invert.

    Returns
    -------
    mat1: xarray.DataArray
        Inverted matrix.

    """
    return xr.DataArray(np.linalg.inv(mat.values), mat.coords, mat.dims)
