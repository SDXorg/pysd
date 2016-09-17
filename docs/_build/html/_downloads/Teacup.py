
"""
Python model tests/test-models/samples/teacup/teacup.py
Translated using PySD version 0.7.1
"""
from __future__ import division
import numpy as np
from pysd import utils
import xarray as xr

from pysd.functions import cache
from pysd import functions

_subscript_dict = {}

_namespace = {
    'INITIAL TIME': 'initial_time',
    'Characteristic Time': 'characteristic_time',
    'Heat Loss to Room': 'heat_loss_to_room',
    'TIME': 'time',
    'SAVEPER': 'saveper',
    'FINAL TIME': 'final_time',
    'Room Temperature': 'room_temperature',
    'Teacup Temperature': 'teacup_temperature',
    'Time': 'time',
    'TIME STEP': 'time_step'}


@cache('step')
def teacup_temperature():
    """
    Teacup Temperature
    ------------------
    (teacup_temperature)
    Degrees

    """
    return integ_teacup_temperature()


@cache('run')
def final_time():
    """
    FINAL TIME
    ----------
    (final_time)
    Minute
    The final time for the simulation.
    """
    return 30


integ_teacup_temperature = functions.Integ(lambda: -heat_loss_to_room(), lambda: 180)


@cache('run')
def room_temperature():
    """
    Room Temperature
    ----------------
    (room_temperature)


    """
    return 70


@cache('step')
def heat_loss_to_room():
    """
    Heat Loss to Room
    -----------------
    (heat_loss_to_room)
    Degrees/Minute
    This is the rate at which heat flows from the cup into the room. We can
                ignore it at this point.
    """
    return (teacup_temperature() - room_temperature()) / characteristic_time()


@cache('run')
def characteristic_time():
    """
    Characteristic Time
    -------------------
    (characteristic_time)
    Minutes

    """
    return 10


@cache('step')
def saveper():
    """
    SAVEPER
    -------
    (saveper)
    Minute [0,?]
    The frequency with which output is stored.
    """
    return time_step()


@cache('run')
def initial_time():
    """
    INITIAL TIME
    ------------
    (initial_time)
    Minute
    The initial time for the simulation.
    """
    return 0


@cache('step')
def time():
    """
    TIME
    ----
    (time)
    None
    The time of the model
    """
    return _t


@cache('run')
def time_step():
    """
    TIME STEP
    ---------
    (time_step)
    Minute [0,?]
    The time step for the simulation.
    """
    return 0.125


def time():
    return _t
functions.time = time
functions._stage = lambda: _stage
