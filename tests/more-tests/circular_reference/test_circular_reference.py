from pysd.py_backend.statefuls import Integ, Delay
from pysd import Component

_subscript_dict = {}
_dependencies = {
    'integ': {'_integ_integ': 1},
    'delay': {'_delay_delay': 1},
    '_integ_integ': {'initial': {'delay': 1}, 'step': {}},
    '_delay_delay': {'initial': {'integ': 1}, 'step': {}}
}
__pysd_version__ = "3.0.0"

__data = {'scope': None, 'time': lambda: 0}

component = Component()

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 0.5,
    "time_step": lambda: 0.5,
    "saveper": lambda: time_step(),
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


@component.add(name="Time")
def time():
    return __data["time"]()


@component.add(name="Time step")
def time_step():
    return __data["time"].step()


@component.add(name="Initial time")
def initial_time():
    return __data["time"].initial()


@component.add(name="Final time")
def final_time():
    return __data["time"].final()


@component.add(name="Saveper")
def saveper():
    return __data["time"].save()


@component.add(name="Integ")
def integ():
    return _integ_integ()


@component.add(name="Delay")
def delay():
    return _delay_delay()


_integ_integ = Integ(lambda: 2, lambda: delay(), '_integ_integ')

_delay_delay = Delay(lambda: 2, lambda: 1,
                     lambda: integ(), 1, time_step, '_delay_delay')
