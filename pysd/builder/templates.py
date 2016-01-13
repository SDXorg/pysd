"""
templates.py
james.p.houghton@gmail.com
Jan 12, 2016

Specifies the structure of python blocks corresponding with various model elements.
"""

from string import Template

templates = dict()

templates['new file'] = Template(
"""
from __future__ import division
import numpy as np
from pysd import functions

def time():
    return _t
""")
# todo: figure out a better place for the 'time' function to live


# Stock initial conditions and derivative functions are given leading underscores not because
# we specifically want them to be private to PySD, but because they have to exist in the same
# namespace as other model components, which for all we know, might have similar names. We know
# that no element from Vensim will have a leading underscore, however, as Vensim won't allow it,
#  so this is a reasonable way to make them 'safe'.

# The 'init' needs to be a function in its own right (as opposed to just an attribute of the stock
# accessor) because its expression may reference other parts of the model. We want the initialization
# to happen at run-time, not import-time, so we need to defer execution.

templates['stock'] = Template(
"""
def ${identifier}():
    return _state['${identifier}']

def _${identifier}_init():
    return ${initial_condition}

def _d${identifier}_dt():
    return ${expression}
""")


templates['flaux'] = Template(
"""
def ${identifier}():
    \"""
    ${docstring}
    \"""

    ${expression}
    return output
""")


templates['lookup'] = Template(
"""
def ${identifier}(x):
    return functions.lookup(x,
                            ${identifier}.xs,
                            ${identifier}.ys)

${identifier}.xs = ${xs_str}
${identifier}.ys = ${ys_str}
""")
