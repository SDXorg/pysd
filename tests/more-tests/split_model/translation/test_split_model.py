"""
Python model "test_split_model.py"
Translated using PySD
"""

__pysd_version__ = "1.4.1"


from os import path
from pysd.py_backend.utils import load_model_data, open_module

from pysd.py_backend.functions import Integ
from pysd import cache


_root = path.dirname(__file__)

__data = {"scope": None, "time": lambda: 0}

_namespace, _subscript_dict, _modules = load_model_data(_root, "test_split_model")

# loading modules from the modules_test_split_model directory
for module in _modules:
    exec(open_module(_root, "test_split_model", module))


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


def time():
    return __data["time"]()


@cache.run
def final_time():
    """
    Real Name: FINAL TIME
    Original Eqn: 100
    Units: Month
    Limits: (None, None)
    Type: constant
    Subs: None

    The final time for the simulation.
    """
    return 100


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
