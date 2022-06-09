"""
The provided functions have no direct analog in the standard Python data
analytics stack, or require information about the internal state of the
system beyond what is present in the function call. They are provided
in a structure that makes it easy for the model elements to call. The
functions may be similar to the original functions given by Vensim or
Stella, but sometimes the number or order of arguments may change.
"""
import warnings

import numpy as np
import xarray as xr

from . import utils

small_vensim = 1e-6  # What is considered zero according to Vensim Help


def ramp(time, slope, start, finish=None):
    """
    Implements vensim's and xmile's RAMP function.

    Parameters
    ----------
    time: callable
        Function that returns the current time.
    slope: float
        The slope of the ramp starting at zero at time start.
    start: float
        Time at which the ramp begins.
    finish: float or None (oprional)
        Time at which the ramp ends. If None the ramp will never end.
        Default is None.

    Returns
    -------
    float or xarray.DataArray:
        If prior to ramp start, returns zero.
        If after ramp ends, returns top of ramp.

    """
    t = time()
    if t < start:
        return 0
    else:
        if finish is None:
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
    time: callable
        Function that returns the current time.
    value: float
        The height of the step.
    tstep: float
        The time at and after which `result` equals `value`.

    Returns
    -------
    float or xarray.DataArray:
        - In range [-inf, tstep):
            returns 0
        - In range [tstep, +inf]:
            returns `value`
    """
    return value if time() >= tstep else 0


def pulse(time, start, repeat_time=0, width=None, magnitude=None, end=None):
    """
    Implements Vensim's PULSE and PULSE TRAIN functions and Xmile's PULSE
    function.

    Parameters
    ----------
    time: callable
        Function that returns the current time.
    start: float
        Starting time of the pulse.
    repeat_time: float (optional)
        Time interval of the pulse repetition. If 0 it will return a
        single pulse. Default is 0.
    width: float or None (optional)
        Duration of the pulse. If None only one-time_step pulse will be
        generated. Default is None.
    magnitude: float or None (optional)
        The magnitude of the pulse. If None it will return 1 when the
        pulse happens, similar to magnitude=time_step(). Default is None.
    end: float or None (optional)
        Final time of the pulse. If None there is no final time.
        Default is None.

    Returns
    -------
    float or xarray.DataArray:
        - In range [-inf, start):
            returns 0
        - In range [start + n*repeat_time, start + n*repeat_time + width):
            returns magnitude/time_step or 1
        - In range [start + n*repeat_time + width, start + (n+1)*repeat_time):
            returns 0

    """
    t = time()
    width = .5*time.time_step() if width is None else width
    out = magnitude/time.time_step() if magnitude is not None else 1
    if repeat_time == 0:
        return out if start - small_vensim <= t < start + width else 0
    elif start <= t and (end is None or t < end):
        return out if (t - start + small_vensim) % repeat_time < width else 0
    else:
        return 0


def if_then_else(condition, val_if_true, val_if_false):
    """
    Implements Vensim's IF THEN ELSE function.
    https://www.vensim.com/documentation/20475.htm

    Parameters
    ----------
    condition: bool or xarray.DataArray of bools
    val_if_true: callable
        Value to evaluate and return when condition is true.
    val_if_false: callable
        Value to evaluate and return when condition is false.

    Returns
    -------
    float or xarray.DataArray:
        The value depending on the condition.

    """
    # NUMPY: replace xr by np
    if isinstance(condition, xr.DataArray):
        # NUMPY: neccessarry for keep the same shape always
        # if condition.all():
        #    value = val_if_true()
        # elif not condition.any():
        #    value = val_if_false()
        # else:
        #    return np.where(condition, val_if_true(), val_if_false())
        #
        # if isinstance(value, np.ndarray):
        #    return value
        # return np.full_like(condition, value)
        if condition.all():
            return val_if_true()
        elif not condition.any():
            return val_if_false()

        return xr.where(condition, val_if_true(), val_if_false())

    return val_if_true() if condition else val_if_false()


def xidz(numerator, denominator, x):
    """
    Implements Vensim's XIDZ function.
    https://www.vensim.com/documentation/fn_xidz.htm

    This function executes a division, robust to denominator being zero.
    In the case of zero denominator, the final argument is returned.

    Parameters
    ----------
    numerator: float or xarray.DataArray
        Numerator of the operation.
    denominator: float or xarray.DataArray
        Denominator of the operation.
    x: float or xarray.DataArray
        The value to return if the denominator is zero.

    Returns
    -------
    float or xarray.DataArray:
        - numerator/denominator if denominator > small_vensim
        - value_if_denom_is_zero otherwise

    """
    # NUMPY: replace DataArray by np.ndarray, xr.where -> np.where
    if isinstance(denominator, xr.DataArray):
        return xr.where(np.abs(denominator) < small_vensim,
                        x,
                        numerator/denominator)

    if abs(denominator) < small_vensim:
        # NUMPY: neccessarry for keep the same shape always
        # if isinstance(numerator, np.ndarray):
        #    return np.full_like(numerator, x)
        return x
    else:
        # NUMPY: neccessarry for keep the same shape always
        # if isinstance(x, np.ndarray):
        #    return np.full_like(x, numerator/denominator)
        return numerator/denominator


def zidz(numerator, denominator):
    """
    This function bypasses divide-by-zero errors,
    implementing Vensim's ZIDZ function.
    https://www.vensim.com/documentation/fn_zidz.htm

    Parameters
    ----------
    numerator: float or xarray.DataArray
        value to be divided
    denominator: float or xarray.DataArray
        value to devide by

    Returns
    -------
    float or xarray.DataArray:
        - numerator/denominator if denominator > small_vensim
        - 0 or 0s array otherwise

    """
    # NUMPY: replace DataArray by np.ndarray, xr.where -> np.where
    if isinstance(denominator, xr.DataArray):
        return xr.where(np.abs(denominator) < small_vensim,
                        0,
                        numerator/denominator)

    if abs(denominator) < small_vensim:
        # NUMPY: neccessarry for keep the same shape always
        # if isinstance(denominator, np.ndarray):
        #    return np.zeros_like(denominator)
        if isinstance(numerator, xr.DataArray):
            return xr.DataArray(0, numerator.coords, numerator.dims)
        return 0
    else:
        return numerator/denominator


def active_initial(stage, expr, init_val):
    """
    Implements vensim's ACTIVE INITIAL function

    Parameters
    ----------
    stage: str
        The stage of the model.
    expr: callable
        Running stage value
    init_val: float or xarray.DataArray
        Initialization stage value.

    Returns
    -------
    float or xarray.DataArray:
        - inti_val if stage='Initialization'
        - expr() otherwise
    """
    # NUMPY: both must have same dimensions in inputs, remove time.stage
    if stage == 'Initialization':
        return init_val
    else:
        return expr()


def incomplete(*args):
    """
    Implements an incomplete functions.
    Prompts a RuntimeWarning.

    Parameters
    ----------
    *args: arguments

    Returns
    -------
    numpy.nan

    """
    warnings.warn(
        'Call to undefined function, calling dependencies and returning NaN',
        RuntimeWarning, stacklevel=2)

    return np.nan


def not_implemented_function(*args):
    """
    Implements a not implemented functions.
    Raises a NotImplementedError if it is called.

    Parameters
    ----------
    *args: arguments
        The first argument must be the name of the function as str to
        properly print the error message.

    """
    raise NotImplementedError(f"Not implemented function '{args[0]}'")


def integer(x):
    """
    Implements Vensim's INTEGER function.

    Parameters
    ----------
    x: float or xarray.DataArray
        Input value.

    Returns
    -------
    integer: float or xarray.DataArray
        Returns integer part of x.

    """
    # NUMPY: replace xr by np
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
    # NUMPY: replace xr by np
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
    modulo: float or xarray.DataArray
        Returns x modulo m, if x is smaller than 0 the result is given
        in the range (-m, 0] as Vensim does. x - quantum(x, m)

    """
    return x - quantum(x, m)


def sum(x, dim=None):
    """
    Implements Vensim's SUM function.

    Parameters
    ----------
    x: xarray.DataArray
        Input value.
    dim: list of strs (optional)
        Dimensions to apply the function over.
        If not given the function will be applied over all dimensions.

    Returns
    -------
    sum: xarray.DataArray or float
        The result of the sum operation in the given dimensions.

    """
    # NUMPY: replace by np.sum(x, axis=axis) put directly in the file
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
        Input value.
    dim: list of strs (optional)
        Dimensions to apply the function over.
        If not given the function will be applied over all dimensions.

    Returns
    -------
    prod: xarray.DataArray or float
        The result of the product operation in the given dimensions.

    """
    # NUMPY: replace by np.prod(x, axis=axis) put directly in the file
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
        Input value.
    dim: list of strs (optional)
        Dimensions to apply the function over.
        If not given the function will be applied over all dimensions.

    Returns
    -------
    vmin: xarray.DataArray or float
        The result of the minimum value over the given dimensions.

    """
    # NUMPY: replace by np.min(x, axis=axis) put directly in the file
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
        Input value.
    dim: list of strs (optional)
        Dimensions to apply the function over.
        If not given the function will be applied over all dimensions.

    Returns
    -------
    vmax: xarray.DataArray or float
        The result of the maximum value over the dimensions.

    """
    # NUMPY: replace by np.max(x, axis=axis) put directly in the file
    # float returned if the function is applied over all the dimensions
    if dim is None or set(x.dims) == set(dim):
        return float(x.max())

    return x.max(dim=dim)


def invert_matrix(mat):
    """
    Implements Vensim's INVERT MATRIX function.

    Invert the matrix defined by the last two dimensions of xarray.DataArray.

    Parameters
    -----------
    mat: xarray.DataArray
        The matrix to invert.

    Returns
    -------
    mat1: xarray.DataArray
        Inverted matrix.

    """
    # NUMPY: avoid converting to xarray, put directly the expression
    # in the model
    return xr.DataArray(np.linalg.inv(mat.values), mat.coords, mat.dims)


def vector_sort_order(vector, direction):
    """
    Implements Vensim's VECTOR SORT ORDER function. Sorting is done on
    the complete vector relative to the last subscript.
    https://www.vensim.com/documentation/fn_vector_sort_order.html

    Parameters
    -----------
    vector: xarray.DataArray
        The vector to sort.
    direction: float
        The direction to sort the vector. If direction > 1 it will sort
        the vector entries from smallest to biggest, otherwise from
        biggest to smallest.

    Returns
    -------
    vector_sorted: xarray.DataArray
        The sorted vector.

    """
    # TODO: can direction be an array? In this case this will fail
    if direction <= 0:
        # NUMPY: return flip directly
        flip = np.flip(vector.argsort(), axis=-1)
        return xr.DataArray(flip.values, vector.coords, vector.dims)
    return vector.argsort()


def vector_reorder(vector, svector):
    """
    Implements Vensim's VECTOR REORDER function. Reordering is done on
    the complete vector relative to the last subscript.
    https://www.vensim.com/documentation/fn_vector_reorder.html

    Parameters
    -----------
    vector: xarray.DataArray
        The vector to sort.
    svector: xarray.DataArray
        The vector to specify the order.

    Returns
    -------
    vector_sorted: xarray.DataArray
        The sorted vector.

    """
    # NUMPY: Use directly numpy sort functions, no need to assign coords later
    if len(svector.dims) > 1:
        # TODO this may be simplified
        new_vector = vector.copy()
        dims = svector.dims
        # create an empty array to hold the orderings (only last dim)
        arr = xr.DataArray(
            np.nan,
            {dims[-1]: vector.coords[dims[-1]].values},
            dims[-1:]
        )
        # split the ordering array in 0-dim arrays
        svectors = utils.xrsplit(svector)
        orders = {}
        for sv in svectors:
            # regrup the ordering arrays using last dimensions
            pos = {dim: str(sv.coords[dim].values) for dim in dims[:-1]}
            key = ";".join(pos.values())
            if key not in orders.keys():
                orders[key] = (pos, arr.copy())
            orders[key][1].loc[sv.coords[dims[-1]]] = sv.values

        for pos, array in orders.values():
            # get the reordered array
            values = [vector.loc[pos].values[int(i)] for i in array.values]
            new_vector.loc[pos] = values

        return new_vector

    return vector[svector.values].assign_coords(vector.coords)


def vector_rank(vector, direction):
    """
    Implements Vensim's VECTOR RANK function. Ranking is done on the
    complete vector relative to the last subscript.
    https://www.vensim.com/documentation/fn_vector_rank.html

    Parameters
    -----------
    vector: xarray.DataArray
        The vector to sort.
    direction: float
        The direction to sort the vector. If direction > 1 it will rank
        the vector entries from smallest to biggest, otherwise from
        biggest to smallest.

    Returns
    -------
    vector_rank: xarray.DataArray
        The rank of the vector.

    """
    return vector_sort_order(vector, direction).argsort() + 1
