"""
Python model 'trying_to_reproduce_bug.py'
Translated using PySD
"""


from pysd.py_backend.statefuls import Integ
from pysd import Component

__pysd_version__ = "3.0.0"

_subscript_dict = {}

_dependencies = {
    'initial_time': {},
    'final_time': {},
    'time_step': {},
    'saveper': {'time_step': 1},
    'initial_par': {},
    'stock_a': {'_integ_stock_a': 1},
    'stock_b': {'_integ_stock_b': 1},
    '_integ_stock_a': {'initial': {'initial_par': 1}, 'step': {}},
    '_integ_stock_b': {'initial': {'stock_a': 1}, 'step': {}}
}

__data = {"scope": None, "time": lambda: 0}

component = Component()

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 20,
    "time_step": lambda: 1,
    "saveper": lambda: time_step()
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


@component.add(name="Time")
def time():
    return __data["time"]()


@component.add(name="Initial time")
def initial_time():
    return __data["time"].initial_time()


@component.add(name="Final time")
def final_time():
    return __data["time"].final_time()


@component.add(name="Time step")
def time_step():
    return __data["time"].time_step()


@component.add(name="Saveper")
def saveper():
    return __data["time"].saveper()


@component.add(name="Stock B")
def stock_b():
    return _integ_stock_b()


@component.add(name="Stock A")
def stock_a():
    return _integ_stock_a()


@component.add(name="Initial par")
def initial_par():
    return 42


_integ_stock_b = Integ(lambda: 1, lambda: stock_a(), "_integ_stock_b")


_integ_stock_a = Integ(lambda: 1, lambda: initial_par(), "_integ_stock_a")
