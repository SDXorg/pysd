import numpy as np
import xarray as xr

from pysd import Component

__pysd_version__ = "3.14.3"

__data = {"scope": None, "time": lambda: 0}


_subscript_dict = {"sex": ["Female", "Male"], "education": ["Low", "High"]}

component = Component()

#######################################################################
#                          CONTROL VARIABLES                          #
#######################################################################

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 0,
    "time_step": lambda: 1,
    "saveper": lambda: 1,
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]

#######################################################################
#                           MODEL VARIABLES                           #
#######################################################################


@component.add(
    name="Stock",
    units="Dimensionless",
    subscripts=["sex", "education"],
    comp_type="Constant",
    comp_subtype="Normal",
)
def stock():
    value = xr.DataArray(
        np.nan,
        {"sex": _subscript_dict["sex"], "education": _subscript_dict["education"]},
        ["sex", "education"],
    )
    return value
