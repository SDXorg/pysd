from pysd.py_backend.statefuls import Integ, Delay

_subscript_dict = {}
_namespace = {'integ': 'integ', 'delay': 'delay'}
_dependencies = {
    'integ': {'_integ_integ': 1},
    'delay': {'_delay_delay': 1},
    '_integ_integ': {'initial': {'delay': 1}, 'step': {}},
    '_delay_delay': {'initial': {'integ': 1}, 'step': {}}
}
__pysd_version__ = "2.0.0"

__data = {'scope': None, 'time': lambda: 0}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


def time():
    return __data["time"]()


def _time_step():
    return 0.5


def _initial_time():
    return 0


def _final_time():
    return 0.5


def _saveper():
    return 0.5


def time_step():
    return __data["time"].step()


def initial_time():
    return __data["time"].initial()


def final_time():
    return __data["time"].final()


def saveper():
    return __data["time"].save()


def integ():
    return _integ_integ()


def delay():
    return _delay_delay()


_integ_integ = Integ(lambda: 2, lambda: delay(), '_integ_integ')

_delay_delay = Delay(lambda: 2, lambda: 1,
                     lambda: integ(), 1, time_step, '_delay_delay')