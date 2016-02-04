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
# Todo: np, functions may want to be private, to make sure the namespace is safe...
# Todo: figure out a better place for the 'time' function to live
# Todo: rework this with the builder, things are getting out of hand...


# Track subscripts names in the model file
# What we do is at the top of the model file, create a dictionary with all of
# the subscript families and their associated elements, also any subranges. i.e.
#
# subscript_list = {'one_dimensional_subscript': {'entry_1': 0, 'entry_2': 1, 'entry_3': 2},
#                   'second_dimension_subscript': {'column_1': 0, 'column_2': 1},
#                   'third_dimension_subscript': {'depth_1': 0, 'depth_2': 1}}
#
# Then, as an attribute to a subscripted variable, we include another
# dictionary which says which families are used that variable, and what axis each
# corresponds to:
#
# function.dimension_dir = {'one_dimensional_subscript':0, 'second_dimension_subscript':1}



templates['subscript_dict'] = Template(
"""
_subscript_dict = ${dictofsubs}
"""
)

# Stock initial conditions and derivative functions are given leading underscores not because
# we specifically want them to be private to PySD, but because they have to exist in the same
# namespace as other model components, which for all we know, might have similar names. We know
# that no element from Vensim will have a leading underscore, however, as Vensim won't allow it,
#  so this is a reasonable way to make them 'safe'.
# Todo: safe the init and dt in another way
# we use the leading underscore to safe things like _funcs, etc. Either those need to be
# safed in a different way, or these do.

# The 'init' needs to be a function in its own right (as opposed to just an attribute of the stock
# accessor) because its expression may reference other parts of the model. We want the
# initialization to happen at run-time, not import-time, so we need to defer execution.




templates['stock'] = Template(
"""
def ${identifier}():
    return _state['${identifier}']

def _${identifier}_init():
    try:
        loc_dimension_dir = ${identifier}.dimension_dir
    except:
        loc_dimension_dir = 0
    return ${initial_condition}

def _d${identifier}_dt():
    try:
        loc_dimension_dir = ${identifier}.dimension_dir
    except:
        loc_dimension_dir = 0
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
