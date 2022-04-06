"""
These are the decorators used by the functions in the model file.
functions.py
"""
from functools import wraps
import inspect


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
