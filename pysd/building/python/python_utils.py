import re
import warnings
import numpy as np

# used to create python safe names with the variable reserved_words
from keyword import kwlist
from builtins import __dir__ as bidir
from pysd.py_backend.components import __dir__ as cdir
from pysd.py_backend.data import __dir__ as ddir
from pysd.py_backend.decorators import __dir__ as dedir
from pysd.py_backend.external import __dir__ as edir
from pysd.py_backend.functions import __dir__ as fdir
from pysd.py_backend.statefuls import __dir__ as sdir
from pysd.py_backend.utils import __dir__ as udir


reserved_words = set(
    dir() + bidir() + cdir() + ddir() + dedir() + edir() + fdir()
    + sdir() + udir()).union(kwlist)


def simplify_subscript_input(coords, subscript_dict, return_full, merge_subs):
    """
    Parameters
    ----------
    coords: dict
        Coordinates to write in the model file.

    subscript_dict: dict
        The subscript dictionary of the model file.

    return_full: bool
        If True the when coords == subscript_dict, '_subscript_dict'
        will be returned

    merge_subs: list of strings
        List of the final subscript range of the python array after
        merging with other objects

    Returns
    -------
    coords: str
        The equations to generate the coord dicttionary in the model file.

    """

    if coords == subscript_dict and return_full:
        # variable defined with all the subscripts
        return "_subscript_dict"

    coordsp = []
    for ndim, (dim, coord) in zip(merge_subs, coords.items()):
        # find dimensions can be retrieved from _subscript_dict
        if coord == subscript_dict[dim]:
            # use _subscript_dict
            coordsp.append(f"'{ndim}': _subscript_dict['{dim}']")
        else:
            # write whole dict
            coordsp.append(f"'{ndim}': {coord}")

    return "{" + ", ".join(coordsp) + "}"
