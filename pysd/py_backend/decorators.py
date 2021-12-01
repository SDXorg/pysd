"""
These are the decorators used by the functions in the model file.
functions.py
"""
from functools import wraps
import inspect
import xarray as xr


def subs(dims, subcoords):
    """
    This decorators returns the python object with the correct dimensions
    xarray.DataArray. The algorithm is a simple version of utils.rearrange
    """
    def decorator(function):
        function.dims = dims

        @wraps(function)
        def wrapper(*args):
            data = function(*args)
            coords = {dim: subcoords[dim] for dim in dims}

            if isinstance(data, xr.DataArray):
                dacoords = {coord: list(data.coords[coord].values)
                            for coord in data.coords}
                if data.dims == tuple(dims) and dacoords == coords:
                    # If the input data already has the output format
                    # return it.
                    return data

                # The coordinates are expanded or transposed
                return xr.DataArray(0, coords, dims) + data

            return xr.DataArray(data, coords, dims)

        return wrapper
    return decorator


class Cache(object):
    """
    This is the class for the chache. Several cache types can be saved
    in dictionaries and acces using cache.data[cache_type].
    """
    def __init__(self):
        self.cached_funcs = set()
        self.data = {}

    def __call__(self, func, *args):
        """ Decorator for caching """
        func.args = inspect.getfullargspec(func)[0]

        @wraps(func)
        def cached_func(*args):
            """ Cache function """
            try:
                return self.data[func.__name__]
            except KeyError:
                value = func(*args)
                self.data[func.__name__] = value
                return value
        return cached_func

    def clean(self):
        """ Cleans the cache """
        self.data = {}


def constant_cache(function, *args):
    """
    Constant cache decorator for all the run
    The original function is saved in 'function' attribuite so we can
    recover it later.
    """
    function.function = function
    function.value = function(*args)

    @wraps(function)
    def wrapper(*args):
        return function.value

    return wrapper
