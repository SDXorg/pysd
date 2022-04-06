"""
Python model 'trying_to_reproduce_bug.py'
Translated using PySD
"""


from pysd.py_backend.statefuls import Integ
from pysd import component

__pysd_version__ = "3.0.0"

_subscript_dict = {}

_namespace = {
    "TIME": "time",
    "Time": "time",
    "Stock B": "stock_b",
    "Stock A": "stock_a",
    "Initial Parameter": "initial_parameter",
    "FINAL TIME": "final_time",
    "INITIAL TIME": "initial_time",
    "SAVEPER": "saveper",
    "TIME STEP": "time_step",
}

_dependencies = {
    'initial_time': {},
    'final_time': {},
    'time_step': {},
    'saveper': {'time_step': 1},
    'initial_parameter': {},
    'stock_a': {'_integ_stock_a': 1},
    'stock_b': {'_integ_stock_b': 1},
    '_integ_stock_a': {'initial': {'initial_parameter': 1}, 'step': {}},
    '_integ_stock_b': {'initial': {'stock_a': 1}, 'step': {}}
}

__data = {"scope": None, "time": lambda: 0}

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 20,
    "time_step": lambda: 1,
    "saveper": lambda: time_step()
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


@component(name="Time")
def time():
    return __data["time"]()


@component(name="Initial time")
def initial_time():
    return __data["time"].initial_time()


@component(name="Final time")
def final_time():
    return __data["time"].final_time()


@component(name="Time step")
def time_step():
    return __data["time"].time_step()


@component(name="Saveper")
def saveper():
    return __data["time"].saveper()


@component(name="Stock B")
def stock_b():
    return _integ_stock_b()


@component(name="Stock A")
def stock_a():
    return _integ_stock_a()


@component(name="Initial parameter")
def initial_parameter():
    return 42


_integ_stock_b = Integ(lambda: 1, lambda: stock_a(), "_integ_stock_b")


_integ_stock_a = Integ(lambda: 1, lambda: initial_parameter(), "_integ_stock_a")
