from pysd import cache, external
from pysd.py_backend.functions import Integ, Delay

_subscript_dict = {}
_namespace = {'integ': 'integ', 'delay': 'delay'}
__pysd_version__ = "1.1.1"

__data = {'scope': None, 'time': lambda: 0}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


def time():
    return __data["time"]()


def time_step():
    return 0.5


def initial_time():
    return 0


def integ():
    return _integ_integ()


def delay():
    return _delay_delay()


_integ_integ = Integ(lambda: 2, lambda: delay(), '_integ_integ')

_delay_delay = Delay(lambda: 2, lambda: 1,
                     lambda: integ(), 1, time_step, '_delay_delay')