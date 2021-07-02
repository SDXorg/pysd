"""
Python model 'trying_to_reproduce_bug.py'
Translated using PySD
"""


from pysd.py_backend.functions import Integ
from pysd import cache

__pysd_version__ = "1.6.0"

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

__data = {"scope": None, "time": lambda: 0}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


def time():
    return __data["time"]()


@cache.step
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


@cache.step
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


@cache.run
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


@cache.run
def final_time():
    """
    Real Name: FINAL TIME
    Original Eqn: 1
    Units: Month
    Limits: (None, None)
    Type: constant
    Subs: None

    The final time for the simulation.
    """
    return 1


@cache.run
def initial_time():
    """
    Real Name: INITIAL TIME
    Original Eqn: 0
    Units: Month
    Limits: (None, None)
    Type: constant
    Subs: None

    The initial time for the simulation.
    """
    return 0


@cache.step
def saveper():
    """
    Real Name: SAVEPER
    Original Eqn: TIME STEP
    Units: Month
    Limits: (0.0, None)
    Type: component
    Subs: None

    The frequency with which output is stored.
    """
    return time_step()


@cache.run
def time_step():
    """
    Real Name: TIME STEP
    Original Eqn: 1
    Units: Month
    Limits: (0.0, None)
    Type: constant
    Subs: None

    The time step for the simulation.
    """
    return 1


_integ_stock_b = Integ(lambda: 1, lambda: stock_a(), "_integ_stock_b")


_integ_stock_a = Integ(lambda: 1, lambda: initial_parameter(), "_integ_stock_a")
