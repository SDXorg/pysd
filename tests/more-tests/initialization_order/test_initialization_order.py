"""
Python model 'trying_to_reproduce_bug.py'
Translated using PySD
"""


from pysd.py_backend.statefuls import Integ

__pysd_version__ = "2.0.0"

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


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


def time():
    return __data["time"]()


def stock_b():
    """
    Real Name: Stock B
    Original Eqn: INTEG(1, Stock A)
    Units:
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return _integ_stock_b()


def stock_a():
    """
    Real Name: Stock A
    Original Eqn: INTEG (1, Initial Parameter)
    Units:
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return _integ_stock_a()


def initial_parameter():
    """
    Real Name: Initial Parameter
    Original Eqn: 42
    Units:
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 42


def _time_step():
    return 0.5


def _initial_time():
    return 0


def _final_time():
    return 0.5


def _saveper():
    return 0.5


_integ_stock_b = Integ(lambda: 1, lambda: stock_a(), "_integ_stock_b")


_integ_stock_a = Integ(lambda: 1, lambda: initial_parameter(), "_integ_stock_a")
