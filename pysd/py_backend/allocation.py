"""
The provided allocation functions have no direct analog in the standard
Python data analytics stack. They are provided in a structure that makes
it easy for the model elements to call. The functions may be similar to
the original functions given by Vensim or Stella, but sometimes the
number or order of arguments may change. The allocation functions may
call a protected function or class method thatintegrates the algorithm
to compute the allocation. The algorithms are briefly explained in these
functions docstring.
"""
import itertools
from math import erfc

import numpy as np
import xarray as xr
from scipy.optimize import fsolve


class Priorities:
    @classmethod
    def get_functions(cls, array):
        return [cls.get_function(array[i]) for i in range(array.shape[0])]

    @classmethod
    def get_function(cls, array):
        if array[0] == 0:
            return cls.fixed_quantity(*array[1:])
        elif array[0] == 1:
            return cls.rectangular(*array[1:])
        elif array[0] == 2:
            return cls.triangular(*array[1:])
        elif array[0] == 3:
            return cls.normal(*array[1:])
        elif array[0] == 4:
            return cls.exponential(*array[1:])
        elif array[0] == 5:
            return cls.constant_elasticity(*array[1:])
        else:
            raise ValueError("")

    @staticmethod
    def fixed_quantity(ppriority, pwidth, pextra):
        raise NotImplementedError

    @staticmethod
    def rectangular(ppriority, pwidth, pextra):
        """
        The curve will be shaped as the integral of a rectangle.

        Parameters
        ----------
        ppriority: float
            Specifies the midpoint of the curve.
        pwidth: float
            Determines the speed with which the curve goes from 0 to the
            specified quantity.

        Returns
        -------
        priority_func: function
            The priority function.

        """
        def priority_func(x):
            if x < ppriority - pwidth*.5:
                return 0
            elif x < ppriority + pwidth*.5:
                return (x-ppriority+pwidth*.5)/pwidth
            else:
                return 1

        return priority_func

    @staticmethod
    def triangular(ppriority, pwidth, pextra):
        """
        The curve will be shaped as the integral of a triangle.

        Parameters
        ----------
        ppriority: float
            Specifies the midpoint of the curve.
        pwidth: float
            Determines the speed with which the curve goes from 0 to the
            specified quantity.

        Returns
        -------
        priority_func: function
            The priority function.

        """
        def priority_func(x):
            if x < ppriority - pwidth*.5:
                return 0
            elif x < ppriority:
                return 2*(x - ppriority + pwidth*.5)**2 / pwidth**2
            elif x < ppriority + pwidth*.5:
                return 1 - 2*(ppriority + pwidth*.5 - x)**2 / pwidth**2
            else:
                return 1

        return priority_func

    @staticmethod
    def normal(ppriority, pwidth, pextra):
        """
        The curve will be shaped as the integral of a normal distribution.

        Parameters
        ----------
        ppriority: float
            Specifies the midpoint of the curve (the mean of the
            underlying distribution).
        pwidth: float
            Standard deviation of the underlying distribution.

        Returns
        -------
        priority_func: function
            The priority function.

        """
        def priority_func(x):
            return .5*(2-erfc((x-ppriority)/(np.sqrt(2)*pwidth)))

        return priority_func

    @staticmethod
    def exponential(ppriority, pwidth, pextra):
        """
        The curve will be shaped as the integral of an exponential
        distribution that is symmetric around its mean
        (0.5*exp(-ABS(x-ppriority)/pwidth) on -∞ to ∞).

        Parameters
        ----------
        ppriority: float
            Specifies the midpoint of the curve (the mean of the
            underlying distribution).
        pwidth: float
            Multiplier on x in the underlying distribution.

        Returns
        -------
        priority_func: function
            The priority function.

        """
        def priority_func(x):
            if x < ppriority:
                return .5*np.exp((x-ppriority)/pwidth)
            else:
                return 1-.5*np.exp((ppriority-x)/pwidth)

        return priority_func

    @staticmethod
    def constant_elasticity(ppriority, pwidth, pextra):
        """
        The curve will be a constant elasticity curve.

        Parameters
        ----------
        ppriority: float
            Specifies the midpoint of the curve (the mean of the
            underlying distribution).
        pwidth: float
            Standard deviation of the underlying distribution.
        pextra: positive float

        Returns
        -------
        priority_func: function
            The priority function.

        """
        raise NotImplementedError


def _allocate_available_1d(request, pp, aval):
    if aval >= np.sum(request):
        return request

    n_obj = len(request)

    priorities = Priorities.get_functions(pp)
    x0 = np.mean(pp[:, 1])

    priority = fsolve(
        lambda x: np.sum(
            [(1-priorities[i](x))*request[i] for i in range(n_obj)]
            ) - aval,
        x0)[0]

    return [(1-priorities[i](priority))*request[i] for i in range(n_obj)]


def allocate_available(request, pp, aval):
    if len(request.shape) == 1:
        # NUMPY: avoid '.values' and return directly the result of the
        # function call
        return xr.DataArray(
            _allocate_available_1d(
                request.values, pp.values, aval),
            request.coords
        )

    # NUMPY: use np.empty_like and remove '.values'
    out = xr.zeros_like(request, dtype=float)
    for comb in itertools.product(*[range(i) for i in aval.shape]):
        out.values[comb] = _allocate_by_priority_1d(
            request.values[comb], pp.values[comb], aval.values[comb])

    return out


def _allocate_by_priority_1d(request, priority, width, supply):
    """
    This function implements the algorithm for allocate_by_priority
    to be passed for 1d numpy.arrays. The algorithm works as follows:

    0. If supply > sum(request): return request.
    1. Order the request and priorities from bigger to lower priorities.
    2. Compute the 'distances' between the target, the 'distance' is
       defined as the difference between priorities divided by width
       multiplied by the target request. If the difference in priorities
       over width is bigger than 1, set it to 1, having the distance
       equal to the request. For example, priorities = [10, 9, 7, 2],
       request = [3, 6, 2.5, 2] and width = 3 will have the following
       'distances' vector, distance = [(10-9)/3*3, (9-7)/3*6, 1*2.5]
       = [1, 4, 2.5]
    3. The supply is assigned with linear functions. The fraction
       (or slope) of supply that a target receives is its total request
       divided by the request of all the targets that are receiving
       supply at that point.
    4. The supply is assigned from bigger to lower priority. Starts
       assigning the supply to the first target, when it reaches the
       quantity of the 'distance,' to the second target, the second
       target will start receiving its supply. When the second target
       receives its 'distance' to the third target, the third target
       will start receiving supply and so on.
    5. Each time a target reaches its request or a new target starts
       receiving supply the slope of each target is computed again.
    6. It finishes when all the supply is distributed between targets

    Parameters
    ----------
    request: numpy.ndarray (1D)
        The request by target. Values must be non-negative.
    priority: numpy.ndarray (1D)
        The priority of each target.
    width: float
        The width between priorities. Must be positive.
    supply: float
        The available supply. Must be non-negative.

    Returns
    -------
    out: numpy.ndarray (1D)
        The distribution of the supply.

    """
    if supply >= np.sum(request):
        # All targets receive their request
        return request
    # Remove request 0 targets and order by priority
    is_0 = request == 0
    sort = (-priority[~is_0]).argsort()
    request = request[~is_0].astype(float)[sort]
    priority = priority[~is_0][sort]
    # Create the outputs array
    out = np.zeros_like(request, dtype=float)
    # Compute the distances between target supply and next target start
    distances = np.full_like(request, np.nan, dtype=float)
    # last target will have an numpy.nan as distances as there are no
    # more targets after
    distances[:-1] = np.minimum(-np.diff(priority)/width, 1)*request[:-1]
    # Create a vector of the current active targets
    active = np.zeros_like(request, dtype=bool)
    active[0] = True
    # Create a vector of the last activated target
    c_i = 0
    while supply > 0:
        # Compute the slopes of the active targets of supply
        slopes = request*active
        slopes /= np.sum(slopes)
        # Compute how much supply much be given to any target reach its request
        dx_next_top = np.nanmin((request-out)[active]/slopes[active])
        # Compute how much supply is needed to start next target
        # (last target will return a numpy.nan)
        dx_next_start = (distances[c_i]-out[c_i])/slopes[c_i]
        # Compute where the next change in allocation function will change
        # this will happen when a target reaches is request, or when
        # the next target starts or when the supply is totally distributed
        dx = np.nanmin((dx_next_top, dx_next_start, supply))
        # Assing the supply to the targets
        out += slopes*dx
        if dx == dx_next_start:
            # A new target will start in the next loop
            c_i += 1
            # Active the next targetif its request is different than 0
            active[c_i] = request[c_i] != 0
        if dx == dx_next_top:
            # One or more target have reached their request
            active[out == request] = False
        supply -= dx
    # Return the distributed supply in the original order
    # adding to it again the rquest 0 if the where removed
    return np.insert(out[sort.argsort()], np.where(is_0)[0], 0)


def allocate_by_priority(request, priority, width, supply):
    """
    Implements Vensim's ALLOCATE BY PRIORITY function.
    https://www.vensim.com/documentation/fn_allocate_by_priority.html

    Parameters
    -----------
    request: xarray.DataArray
        Request of each target. Its shape should be the same as
        priority. width and supply must have the same shape except the
        last dimension.
    priority: xarray.DataArray
        Priority of each target. Its shape should be the same as
        request. width and supply must have the same shape except the
        last dimension.
    width: float or xarray.DataArray
        Specifies how big a gap in priority is required to have the
        allocation go first to higher priority with only leftovers going
        to lower priority. When the distance between any two priorities
        exceeds width and the higher priority does not receive its full
        request the lower priority will receive nothing. Its shape
        should be the same as supply.
    supply: float or xarray.DataArray
        The total supply available to fulfill all requests. If the
        supply exceeds total requests, all requests are filled, but
        none are overfilled.  If you wish to conserve material you must
        compute supply minus total allocations explicitly. Its shape
        should be the same as width.

    Returns
    -------
    out: xarray.DataArray
        The distribution of the supply.

    """
    if np.any(request < 0):
        raise ValueError(
            "There are some negative request values. Ensure that "
            "your request is always non-negative. Allocation requires "
            f"all quantities to be positive or 0.\n{request}")

    if np.any(width <= 0):
        raise ValueError(
            f"width={width} is not allowed. width should be greater than 0."
        )

    if np.any(supply < 0):
        raise ValueError(
            f"supply={supply} is not allowed. supply should be non-negative."
        )

    if len(request.shape) == 1:
        # NUMPY: avoid '.values' and return directly the result of the
        # function call
        return xr.DataArray(
            _allocate_by_priority_1d(
                request.values, priority.values, width, supply),
            request.coords
        )

    # NUMPY: use np.empty_like and remove '.values'
    out = xr.zeros_like(request, dtype=float)
    for comb in itertools.product(*[range(i) for i in supply.shape]):
        out.values[comb] = _allocate_by_priority_1d(
            request.values[comb], priority.values[comb],
            width.values[comb], supply.values[comb])

    return out
