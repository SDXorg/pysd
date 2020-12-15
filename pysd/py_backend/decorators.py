from functools import wraps
from .utils import compute_shape
import xarray as xr
import numpy as np


def subs(dims, subcoords):
    """
    This decorators returns the python object with the correct dimensions
    xarray.DataArray. The algorithm is a simple version of utils.rearrange
    """

    def decorator(function):
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
                # TODO replace cleaner version for Python 3 (when deprecate Py2)
                # return xr.DataArray(0, coords, dims)
                return xr.DataArray(np.zeros(compute_shape(coords, dims)),
                                    coords, dims) + data

            else:
                # TODO replace cleaner version for Python 3 (when deprecate Py2)
                # return xr.DataArray(float(data), coords, dims)
                return xr.DataArray(np.full(compute_shape(coords, dims),
                                            float(data)), coords, dims)

        return wrapper
    return decorator


class Cache(object):
    """
    This is the class for the chache. Several cache types can be saved
    in dictionaries and acces using cache.data[cache_type].
    """
    def __init__(self):
        self.types = ['run', 'step']
        self.data = {t:{} for t in self.types}
        self.time = None

    def __call__(self, horizon):
        """
        THIS CALL HAS BEEN ADDED TO MAKE PYSD BACKWARDS COMPATIBLE:
        IN A FUTURE UPDATE WHEN PYSD IS NO MORE BACKWARDS COMPATIBLE
        IT SHOULD BE ERASED
        THE NENW FUNCTIONS ARE ADDED BELLOW WHICH SHOULD BE CALLED AS:
            @cache.run, @cache.steep
        INSTEAF OF:
            @cache("run"), @cache("step")
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
        cached_func: decorated function
            function wrapping the original function, handling caching
        """

        def _run(func):
            """ Decorator for caching at a run level"""
            @wraps(func)
            def cached_func(*args):
                """Run wise cache function"""
                try: # fails if cache is not instantiated
                    return self.data['run'][func.__name__]
                except KeyError:
                    value = func(*args)
                    self.data['run'][func.__name__] = value
                    return value
            return cached_func

        def _step(func):
            """ Decorator for caching at a step level"""
            @wraps(func)
            def cached_func(*args):
                """Step wise cache function"""
                try: # fails if cache is not instantiated or if it is None
                    value = self.data['step'][func.__name__]
                    assert value is not None
                except (KeyError, AssertionError):
                    value = func(*args)
                    self.data['step'][func.__name__] = value
                return value
            return cached_func

        if horizon == 'step':
            return _step
        elif horizon == 'run':
            return _run
        else:
            raise (AttributeError('Bad horizon for cache decorator'))

    def run(self, func, *args):
        """ Decorator for caching at a run level"""
        @wraps(func)
        def cached_func(*args):
            """Run wise cache function"""
            try: # fails if cache is not instantiated
                return self.data['run'][func.__name__]
            except KeyError:
                value = func(*args)
                self.data['run'][func.__name__] = value
                return value
        return cached_func

    def step(self, func, *args):
        """ Decorator for caching at a step level"""
        @wraps(func)
        def cached_func(*args):
            """Step wise cache function"""
            try: # fails if cache is not instantiated or if it is None
                value = self.data['step'][func.__name__]
                assert value is not None
            except (KeyError, AssertionError):
                value = func(*args)
                self.data['step'][func.__name__] = value
            return value
        return cached_func

    def reset(self, time):
        """
        Resets the time to the given one and cleans the step cache.

        Parameters
        ----------
        time: float
          The time to be set.

        """
        for key in self.data['step']:
            self.data['step'][key] = None

        self.time = time

    def clean(self, horizon=None):
        """
        Cleans the cache.

        Parameters
        ----------
        horizon: str or list of str (optional)
          Name(s) of the cache(s) to clean.

        """
        if horizon is None:
            horizon = self.types
        elif isinstance(horizon, str):
            horizon = [horizon]

        for k in horizon:
            self.data[k] = {}


cache = Cache()
