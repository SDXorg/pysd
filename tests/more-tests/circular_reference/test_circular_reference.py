from pysd import cache
from pysd.py_backend.statefuls import Integ, Delay

_subscript_dict = {}
_namespace = {'integ': 'integ', 'delay': 'delay'}
_dependencies = {
    'integ': {'_integ_integ'},
    'delay': {'_delay_delay'},
    '_integ_integ': {'initial': {'delay'}, 'step': None},
    '_delay_delay': {'initial': {'integ'}, 'step': None}
}
__pysd_version__ = "2.0.0"

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