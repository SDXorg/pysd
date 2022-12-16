"""
The provided allocation functions have no direct analog in the standard
Python data analytics stack. They are provided in a structure that makes
it easy for the model elements to call. The functions may be similar to
the original functions given by Vensim, but sometimes the number or
order of arguments may change. The allocation functions may call a
protected function or class method thatintegrates the algorithm to
compute the allocation. The algorithms are briefly explained in these
functions docstring.

Note
----
The Allocation functions basis is explained in the Vensim documentation.
https://www.vensim.com/documentation/allocation_overview.html

Warning
-------
Some allocation function's results may differ from the result given by
Vensim as optimization functions are used to solve the allocation
problems. Those algorithms may not work in the same way or may
have differences in the numerical error propagation.

"""
import itertools
from math import erfc

import numpy as np
import xarray as xr
from scipy.optimize import least_squares
import portion as p


class Priorities:
    @classmethod
    def get_functions(cls, q0, pp, kind):
        """
        Get priority functions based on the demand/supply and priority profile.

        Parameters
        ----------
        q0: numpy.array
            values of maximum demand or supply of each component.
            Its shape should be (n,)
        pp: numpy.array
            pp values array. Its shape should be (n, m).
        kind: str ("demand" or "supply")
            The kind of priority "demand" or "supply".

        Returns
        -------
        functions: list of functions
            List of allocation supply or demand function for each element.

        full_allocation: function
            Full allocation function. It is the result function of
            addying all the functions.

        def_intervals: list of tuples
            List of (supply interval, priority interval, mean priority)
            where the full_allocation function is extrictly monotonous
            (injective). Givin a supply value, this is used to compute
            the limits and starting point of the optimization problem.

        """
        if np.any(pp[:, 2] <= 0):
            # pwidth values smaller than 0
            raise ValueError("pwidth values must be positive.")

        if kind == "demand":
            # Get the list of priority functions and the intervals where
            # they are strictly monotonous (injective function)
            func_int = [
                cls.get_function_demand(q0[i], pp[i])
                for i in range(pp.shape[0])
            ]
            # In order to get the range of the full_allocation function,
            # we need to flip the lower and the upper value, as it is a
            # decreasing function for demand
            int_attr = {"lower": "upper", "upper": "lower"}
        elif kind == "supply":  # pragma: no cover
            # Get the list of priority functions and the intervals where
            # they are strictly monotonous (injective function)
            func_int = [
                cls.get_function_supply(q0[i], pp[i])
                for i in range(pp.shape[0])
            ]
            # In order to get the range of the full_allocation function,
            # we need to keep the lower and the upper value, as it is a
            # increasing function for supply
            int_attr = {"lower": "lower", "upper": "upper"}
        else:
            raise ValueError(
                f"kind='{kind}' is not allowed. kind should be "
                "'demand' or 'supply'.")

        functions = [fi[0] for fi in func_int]
        intervals = [fi[1] for fi in func_int]

        # Join the intervals of all functions to get the intervals where
        # the sum of the functions is strictly monotonous (injective
        # function), therefore we can solve the minimization problem in
        # strictly monotonous areas of the function, avoiding the crash
        # of the algorithm
        interval = intervals[0]
        for i in intervals[1:]:
            interval = interval.union(i)

        # Full allocation function -> function to solve
        def full_allocation(x): return np.sum([func(x) for func in functions])

        def_intervals = []
        for subinterval in interval:
            # Iterate over disjoint interval sections
            # Each interval section will be converted in supply interval
            # and compute the starting point for the supply interval
            # as the midpoint in the priority interval
            def_intervals.append((
                p.closed(
                    full_allocation(getattr(subinterval, int_attr["lower"])),
                    full_allocation(getattr(subinterval, int_attr["upper"]))
                ),
                subinterval,
                .5*(subinterval.upper+subinterval.lower)
            ))

        return functions, full_allocation, def_intervals

    @classmethod
    def get_function_demand(cls, q0, pp):
        """
        Get priority functions for demand based on the priority profile.

        Parameters
        ----------
        q0: float [0, +np.inf)
            The demand of the target.
        pp: numpy.array
            pp values array.

        Returns
        -------
        priority_func: function
            Priority function.
        interval: portion.interval
            The interval where the priority function is strictly monotonous.

        """
        if q0 == 0:
            # No demand is requested return a 0 function with an empty interval
            return lambda x: 0, p.empty()
        if pp[0] == 0:
            # Fixed quantity demand
            return cls.fixed_quantity(q0, *pp[1:])
        elif pp[0] == 1:
            # Rectangular demand
            return cls.rectangular(q0, *pp[1:])
        elif pp[0] == 2:
            # Triangular demand
            return cls.triangular(q0, *pp[1:])
        elif pp[0] == 3:
            # Normal distribution demand
            return cls.normal(q0, *pp[1:])
        elif pp[0] == 4:
            # Exponential distribution demand
            return cls.exponential(q0, *pp[1:])
        elif pp[0] == 5:
            # Constant elasticity demand
            return cls.constant_elasticity_demand(q0, *pp[1:])
        else:
            raise ValueError(
                f"The priority function for pprofile={pp[0]} is not valid.")

    @classmethod
    def get_function_supply(cls, q0, pp):
        """
        Get priority functions for supply based on the priority profile.

        Parameters
        ----------
        q0: float [0, +np.inf)
            The supply of the producer.
        pp: numpy.array
            pp values array.

        Returns
        -------
        priority_func: function
            Priority function.
        interval: portion.interval
            The interval where the priority function is strictly monotonous.

        """
        # TODO: This function should be similar to the demand function
        # it is neccessary for the many-to-many allocation given by
        # the set FIND MARKET PLACE, DEMAND AT PRICE, SUPPLY AT PRICE
        raise NotImplementedError("get_function_supply is not implemented.")

    @staticmethod
    def fixed_quantity(q0, ppriority, pwidth, pextra):
        raise NotImplementedError(
            "fixed_quantity priority profile is not implemented.")

    @staticmethod
    def rectangular(q0, ppriority, pwidth, pextra):
        """
        Demand curve for rectangular shape.
        The supply curve will be shaped as the integral of a rectangle.

        Parameters
        ----------
        q0: float
            The total demand/supply of the element.
        ppriority: float
            Specifies the midpoint of the curve.
        pwidth: float
            Determines the speed with which the curve goes from 0 to
            the specified quantity.
        pextra: float
            Ignore.

        Returns
        -------
        priority_func: function
            The priority function.

        """
        def priority_func(x):
            if x <= ppriority - pwidth*.5:
                return q0
            elif x < ppriority + pwidth*.5:
                return q0*(1-(x-ppriority+pwidth*.5)/pwidth)
            else:
                return 0

        return (
            priority_func,
            p.open(ppriority - pwidth*.5, ppriority + pwidth*.5)
        )

    @staticmethod
    def triangular(q0, ppriority, pwidth, pextra):
        """
        Demand curve for triangular shape.
        The supply curve will be shaped as the integral of a triangle.

        Parameters
        ----------
        q0: float
            The total demand/supply of the element.
        ppriority: float
            Specifies the midpoint of the curve.
        pwidth: float
            Determines the speed with which the curve goes from 0 to the
            specified quantity.
        pextra: float
            Ignore.

        Returns
        -------
        priority_func: function
            The priority function.

        """
        def priority_func(x):
            if x <= ppriority - pwidth*.5:
                return q0
            elif x < ppriority:
                return q0*(1-2*(x-ppriority+pwidth*.5)**2/pwidth**2)
            elif x < ppriority + pwidth*.5:
                return 2*q0*(ppriority+pwidth*.5-x)**2/pwidth**2
            else:
                return 0

        return (
            priority_func,
            p.open(ppriority - pwidth*.5, ppriority + pwidth*.5)
        )

    @staticmethod
    def normal(q0, ppriority, pwidth, pextra):
        """
        Demand curve for normal shape.
        The supply curve will be shaped as the integral of a normal
        distribution.

        Parameters
        ----------
        q0: float
            The total demand/supply of the element.
        ppriority: float
            Specifies the midpoint of the curve (the mean of the
            underlying distribution).
        pwidth: float
            Standard deviation of the underlying distribution.
        pextra: float
            Ignore.

        Returns
        -------
        priority_func: function
            The priority function.

        """
        def priority_func(x):
            return q0*.5*(2-erfc((ppriority-x)/(np.sqrt(2)*pwidth)))

        # Normal distribution CDF is stricty monotonous in (-inf, inf).
        # However, numerically it is only in a the range ~ (-8.29*sd, 8.29*sd)
        return (
            priority_func,
            p.open(
                ppriority-8.2923611*pwidth,
                ppriority+8.2923611*pwidth
            )
        )

    @staticmethod
    def exponential(q0, ppriority, pwidth, pextra):
        """
        Demand curve for exponential shape
        The supply curve will be shaped as the integral of an
        exponential distribution that is symmetric around its mean
        (0.5*exp(-ABS(x-ppriority)/pwidth) on -∞ to ∞).

        Parameters
        ----------
        q0: float
            The total demand/supply of the element.
        ppriority: float
            Specifies the midpoint of the curve (the mean of the
            underlying distribution).
        pwidth: float
            Multiplier on x in the underlying distribution.
        pextra: float
            Ignore.

        Returns
        -------
        priority_func: function
            The priority function.

        """
        def priority_func(x):
            if x < ppriority:
                return q0*(1-.5*np.exp((x-ppriority)/pwidth))
            else:
                return q0*.5*np.exp((ppriority-x)/pwidth)

        # Exponential distribution CDF is stricty monotonous in (-inf, inf).
        # However, numerically it is only in a the range ~ (-36.7*sd, 36.7*sd)
        return (
            priority_func,
            p.open(
                ppriority-36.7368005696*pwidth,
                ppriority+36.7368005696*pwidth
            )
        )

    @staticmethod
    def constant_elasticity_demand(q0, ppriority, pwidth, pextra):
        """
        Demand constant elasticity curve.
        The curve will be a constant elasticity curve.

        Parameters
        ----------
        q0: float
            The total demand/supply of the element.
        ppriority: float
            Specifies the midpoint of the curve (the mean of the
            underlying distribution).
        pwidth: float
            Standard deviation of the underlying distribution.
        pextra: positive float
            Elasticity exponent.

        Returns
        -------
        priority_func: function
            The priority function.

        """
        raise NotImplementedError(
            "Some results for Vensim showed some bugs when using this "
            "priority curve. Therefore, the curve is not implemented in "
            "PySD as it cannot be properly tested."
        )

    @staticmethod
    def constant_elasticity_supply(ppriority, pwidth,
                                   pextra):   # pragma: no cover
        """
        Supply constant elasticity curve.
        The curve will be a constant elasticity curve.

        Parameters
        ----------
        q0: float
            The total demand/supply of the element.
        ppriority: float
            Specifies the midpoint of the curve (the mean of the
            underlying distribution).
        pwidth: float
            Standard deviation of the underlying distribution.
        pextra: positive float
            Elasticity exponent.

        Returns
        -------
        priority_func: function
            The priority function.

        """
        raise NotImplementedError(
            "Some results for Vensim showed some bugs when using this "
            "priority curve. Therefore, the curve is not implemented in "
            "PySD as it cannot be properly tested."
        )


def _allocate_available_1d(request, pp, avail):
    """
    This function implements the algorithm for allocate_available
    to be passed for 1d numpy.arrays. The algorithm works as follows:

    0. If supply > sum(request): return request. In the same way,
       if supply = 0: return request*0
    1. Based on the priority profiles and demands, the priority profiles
       are computed. This profiles are returned with the interval where
       each of them is strictly monotonous (or injective).
    2. Using the intervals of injectivity the initial guess is
       selected depending on the available supply.
    3. The initial guess and injectivity interval are used to compute
       the value where the sum of all priority functions is equal to
       the avilable supply. This porcess is done using a least_squares
       optimization function.
    4. The output from the previous step is used to compute the supply
       to each target.

    Parameters
    ----------
    request: numpy.ndarray (1D)
        The request by target. Values must be non-negative.
    pp: numpy.ndarray (2D)
        The priority profiles of each target.
    avail: float
        The available supply. Must be non-negative.

    Returns
    -------
    out: numpy.ndarray (1D)
        The distribution of the supply.

    """
    if avail >= np.sum(request):
        return request
    if avail == 0:
        return np.zeros_like(request)

    priorities, full_allocation, intervals =\
        Priorities.get_functions(request, pp, "demand")

    for interval, x_interval, x0 in intervals:
        if avail in interval:
            break
    priority = least_squares(
        lambda x: full_allocation(x) - avail,
        x0,
        bounds=(x_interval.lower, x_interval.upper),
        method='dogbox',
        tr_solver='exact',
        ).x[0]

    return [allocate(priority) for allocate in priorities]


def allocate_available(request, pp, avail):
    """
    Implements Vensim's ALLOCATE AVAILABLE function.
    https://www.vensim.com/documentation/fn_allocate_available.html

    Parameters
    -----------
    request: xarray.DataArray
        Request of each target. Its shape should be the one of the
        expected output of the function, having the allocation dimension
        in the last position.
    pp: xarray.DataArray
        Priority of each target. Its shape should be the same as
        request with an extra dimension for the priority profiles
        in the last position. See Vensim's documentation for more
        information https://www.vensim.com/documentation/24335.html
    avail: float or xarray.DataArray
        The total supply available to fulfill all requests. If the
        supply exceeds total requests, all requests are filled, but
        none are overfilled. If you wish to conserve material you must
        compute supply minus total allocations explicitly. Its shape,
        should be the same of request without the last dimension.

    Returns
    -------
    out: xarray.DataArray
        The distribution of the supply.

    Warning
    -------
    This function uses an optimization method for resolution and the
    given solution could differ from the one from Vensim. Particularly,
    when close to the boundaries of the defined priority profiles.

    """
    if np.any(request < 0):
        raise ValueError(
            "There are some negative request values. Ensure that "
            "your request is always non-negative. Allocation requires "
            f"all quantities to be positive or 0.\n{request}")

    if np.any(avail < 0):
        raise ValueError(
            f"avail={avail} is not allowed. avail should be non-negative."
        )

    if len(request.shape) == 1:
        # NUMPY: avoid '.values' and return directly the result of the
        # function call
        return xr.DataArray(
            _allocate_available_1d(
                request.values, pp.values, avail),
            request.coords
        )

    # NUMPY: use np.empty_like and remove '.values'
    out = xr.zeros_like(request, dtype=float)
    for comb in itertools.product(*[range(i) for i in avail.shape]):
        out.values[comb] = _allocate_available_1d(
            request.values[comb], pp.values[comb], avail.values[comb])

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
    elif supply == 0:
        # No supply, all targets receive 0
        return np.zeros_like(request)

    # Remove request 0 targets and order by priority
    is_0 = request == 0
    sort = (-priority[~is_0]).argsort()
    request = request[~is_0].astype(float)[sort]
    priority = priority[~is_0][sort]
    # Create the outputs array
    out_return = np.zeros_like(is_0, dtype=float)
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
        if np.isclose(dx, dx_next_start, rtol=1e-10, atol=1e-16):
            # A new target will start in the next loop
            c_i += 1
            # Active the next targetif its request is different than 0
            active[c_i] = True
        if dx == dx_next_top:
            # One or more target have reached their request
            active[out == request] = False
        supply -= dx
    # Return the distributed supply in the original order
    # adding to it again the request 0 if the where removed
    out_return[~is_0] = out[sort.argsort()]
    return out_return


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
            f"width={width} \n is not allowed. width must be greater than 0.")

    if np.any(supply < 0):
        raise ValueError(
            f"supply={supply} \n is not allowed. supply must not be negative.")

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
