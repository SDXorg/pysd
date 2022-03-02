"""
These are general utilities used by the builder.py, functions.py or the
model file. Vensim's function equivalents should not go here but in
functions.py
"""

import warnings
from collections.abc import Mapping

import regex as re
import numpy as np

# used to create python safe names with the variable reserved_words
from keyword import kwlist
from builtins import __dir__ as bidir
from ..py_backend.components import __dir__ as cdir
from ..py_backend.data import __dir__ as ddir
from ..py_backend.decorators import __dir__ as dedir
from ..py_backend.external import __dir__ as edir
from ..py_backend.functions import __dir__ as fdir
from ..py_backend.statefuls import __dir__ as sdir
from ..py_backend.utils import __dir__ as udir


reserved_words = set(
    dir() + bidir() + cdir() + ddir() + dedir() + edir() + fdir()
    + sdir() + udir())
reserved_words = reserved_words.union(kwlist)


def find_subscript_name(subscript_dict, element, avoid=[]):
    """
    Given a subscript dictionary, and a member of a subscript family,
    return the first key of which the member is within the value list.
    If element is already a subscript name, return that.

    Parameters
    ----------
    subscript_dict: dict
        Follows the {'subscript name':['list','of','subscript','elements']}
        format.

    element: str

    avoid: list (optional)
        List of subscripts to avoid. Default is an empty list.

    Returns
    -------

    Examples
    --------
    >>> find_subscript_name({'Dim1': ['A', 'B'],
    ...                      'Dim2': ['C', 'D', 'E'],
    ...                      'Dim3': ['F', 'G', 'H', 'I']},
    ...                      'D')
    'Dim2'
    >>> find_subscript_name({'Dim1': ['A', 'B'],
    ...                      'Dim2': ['A', 'B'],
    ...                      'Dim3': ['A', 'B']},
    ...                      'B')
    'Dim1'
    >>> find_subscript_name({'Dim1': ['A', 'B'],
    ...                      'Dim2': ['A', 'B'],
    ...                      'Dim3': ['A', 'B']},
    ...                      'B',
    ...                      avoid=['Dim1'])
    'Dim2'
    """
    if element in subscript_dict.keys():
        return element

    for name, elements in subscript_dict.items():
        if element in elements and name not in avoid:
            return name


def make_coord_dict(subs, subscript_dict, terse=True):
    """
    This is for assisting with the lookup of a particular element, such that
    the output of this function would take the place of %s in this expression.

    `variable.loc[%s]`

    Parameters
    ----------
    subs: list of strings
        coordinates, either as names of dimensions, or positions within
        a dimension.

    subscript_dict: dict
        the full dictionary of subscript names and values.

    terse: bool (optional)
        If True, includes only elements that do not cover the full range of
        values in their respective dimension.If False, returns all dimensions.
        Default is True.

    Returns
    -------
    coordinates: dict
        Coordinates needed to access the xarray quantities we're interested in.

    Examples
    --------
    >>> make_coord_dict(['Dim1', 'D'], {'Dim1': ['A', 'B', 'C'],
    ...                                 'Dim2': ['D', 'E', 'F']})
    {'Dim2': ['D']}
    >>> make_coord_dict(['Dim1', 'D'], {'Dim1': ['A', 'B', 'C'],
    ...                                 'Dim2':['D', 'E', 'F']}, terse=False)
    {'Dim2': ['D'], 'Dim1': ['A', 'B', 'C']}

    """
    sub_elems_list = [y for x in subscript_dict.values() for y in x]
    coordinates = {}
    for sub in subs:
        if sub in sub_elems_list:
            name = find_subscript_name(subscript_dict, sub, avoid=subs)
            coordinates[name] = [sub]
        elif not terse:
            coordinates[sub] = subscript_dict[sub]
    return coordinates


def make_merge_list(subs_list, subscript_dict, element=""):
    """
    This is for assisting when building xrmerge. From a list of subscript
    lists returns the final subscript list after mergin. Necessary when
    merging variables with subscripts comming from different definitions.

    Parameters
    ----------
    subs_list: list of lists of strings
        Coordinates, either as names of dimensions, or positions within
        a dimension.

    subscript_dict: dict
        The full dictionary of subscript names and values.

    element: str (optional)
        Element name, if given it will be printed with any error or
        warning message. Default is "".

    Returns
    -------
    dims: list
        Final subscripts after merging.

    Examples
    --------
    >>> make_merge_list([['upper'], ['C']], {'all': ['A', 'B', 'C'],
    ...                                      'upper': ['A', 'B']})
    ['all']

    """
    coords_set = [set() for i in range(len(subs_list[0]))]
    coords_list = [
        make_coord_dict(subs, subscript_dict, terse=False)
        for subs in subs_list
    ]

    # update coords set
    [[coords_set[i].update(coords[dim]) for i, dim in enumerate(coords)]
     for coords in coords_list]

    dims = [None] * len(coords_set)
    # create an array with the name of the subranges for all merging elements
    dims_list = np.array([list(coords) for coords in coords_list]).transpose()
    indexes = np.arange(len(dims))

    for i, coord2 in enumerate(coords_set):
        dims1 = [
            dim for dim in dims_list[i]
            if dim is not None and set(subscript_dict[dim]) == coord2
        ]
        if dims1:
            # if the given coordinate already matches return it
            dims[i] = dims1[0]
        else:
            # find a suitable coordinate
            other_dims = dims_list[indexes != i]
            for name, elements in subscript_dict.items():
                if coord2 == set(elements) and name not in other_dims:
                    dims[i] = name
                    break

            if not dims[i]:
                # the dimension is incomplete use the smaller
                # dimension that completes it
                for name, elements in subscript_dict.items():
                    if coord2.issubset(set(elements))\
                      and name not in other_dims:
                        dims[i] = name
                        warnings.warn(
                            element
                            + "\nDimension given by subscripts:"
                            + "\n\t{}\nis incomplete ".format(coord2)
                            + "using {} instead.".format(name)
                            + "\nSubscript_dict:"
                            + "\n\t{}".format(subscript_dict)
                        )
                        break

            if not dims[i]:
                for name, elements in subscript_dict.items():
                    if coord2 == set(elements):
                        j = 1
                        while name + str(j) in subscript_dict.keys():
                            j += 1
                        subscript_dict[name + str(j)] = elements
                        dims[i] = name + str(j)
                        warnings.warn(
                            element
                            + "\nAdding new subscript range to"
                            + " subscript_dict:\n"
                            + name + str(j) + ": " + ', '.join(elements))
                        break

            if not dims[i]:
                # not able to find the correct dimension
                raise ValueError(
                    element
                    + "\nImpossible to find the dimension that contains:"
                    + "\n\t{}\nFor subscript_dict:".format(coord2)
                    + "\n\t{}".format(subscript_dict)
                )

    return dims


def make_python_identifier(string, namespace=None):
    """
    Takes an arbitrary string and creates a valid Python identifier.

    If the input string is in the namespace, return its value.

    If the python identifier created is already in the namespace,
    but the input string is not (ie, two similar strings resolve to
    the same python identifier)

    or if the identifier is a reserved word in the reserved_words
    list, or is a python default reserved word,
    adds _1, or if _1 is in the namespace, _2, etc.

    Parameters
    ----------
    string: str
        The text to be converted into a valid python identifier.

    namespace: dict
        Map of existing translations into python safe identifiers.
        This is to ensure that two strings are not translated into
        the same python identifier. If string is already in the namespace
        its value will be returned. Otherwise, namespace will be mutated
        adding string as a new key and its value.

    Returns
    -------
    identifier: str
        A vaild python identifier based on the input string.

    Examples
    --------
    >>> make_python_identifier('Capital')
    'capital'

    >>> make_python_identifier('multiple words')
    'multiple_words'

    >>> make_python_identifier('multiple     spaces')
    'multiple_spaces'

    When the name is a python keyword, add '_1' to differentiate it
    >>> make_python_identifier('for')
    'for_1'

    Remove leading and trailing whitespace
    >>> make_python_identifier('  whitespace  ')
    'whitespace'

    Remove most special characters outright:
    >>> make_python_identifier('H@t tr!ck')
    'ht_trck'

    remove leading digits
    >>> make_python_identifier('123abc')
    'nvs_123abc'

    already in namespace
    >>> make_python_identifier('Var$', namespace={'Var$': 'var'})
    ''var'

    namespace conflicts
    >>> make_python_identifier('Var@', namespace={'Var$': 'var'})
    'var_1'

    >>> make_python_identifier('Var$', namespace={'Var@': 'var',
    ...                                           'Var%':'var_1'})
    'var_2'

    References
    ----------
    Identifiers must follow the convention outlined here:
        https://docs.python.org/2/reference/lexical_analysis.html#identifiers

    """
    if namespace is None:
        namespace = dict()

    if string in namespace:
        return namespace[string]

    # create a working copy (and make it lowercase, while we're at it)
    s = string.lower()

    # remove leading and trailing whitespace
    s = s.strip()

    # Make spaces into underscores
    s = re.sub(r"[\s\t\n]+", "_", s)

    # Remove invalid characters
    s = re.sub(r"[^\p{l}\p{m}\p{n}_]", "", s)

    # If leading character is not a letter add nvs_.
    # Only letters can be leading characters.
    if re.findall(r"^[^\p{l}_]", s):
        s = "nvs_" + s
    elif re.findall(r"^_", s):
        s = "nvs" + s

    # reserved the names of PySD functions and methods and other vars
    # in the namespace
    used_words = reserved_words.union(namespace.values())

    # Check that the string is not a python identifier
    identifier = s
    i = 1
    while identifier in used_words:
        identifier = s + '_' + str(i)
        i += 1

    namespace[string] = identifier

    return identifier


def make_add_identifier(identifier, build_names):
    """
    Takes an existing used Python identifier and attatch a unique
    identifier with ADD_# ending.

    Used for add new information to an existing external object.
    build_names will be updated inside this functions as a set
    is mutable.

    Parameters
    ----------
    identifier: str
      Existing python identifier.

    build_names: set
      Set of the already used identifiers for external objects.

    Returns
    -------
    identifier: str
      A vaild python identifier based on the input indentifier
      and the existing ones.

    """
    identifier += "ADD_"
    number = 1
    # iterate until finding a non-used identifier
    while identifier + str(number) in build_names:
        number += 1

    # update identifier
    identifier += str(number)

    # update the build names
    build_names.add(identifier)

    return identifier


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


def add_entries_underscore(*dictionaries):
    """
    Expands dictionaries adding new keys underscoring the white spaces
    in the old ones. As the dictionaries are mutable objects this functions
    will add the new entries to the already existing dictionaries with
    no need to return a new one.

    Parameters
    ----------
    *dictionaries: dict(s)
        The dictionary or dictionaries to add the entries with underscore.

    Return
    ------
    None

    """
    for dictionary in dictionaries:
        keys = list(dictionary)
        for name in keys:
            dictionary[re.sub(" ", "_", name)] = dictionary[name]
    return


def clean_file_names(*args):
    """
    Removes special characters and makes clean file names

    Parameters
    ----------
    *args: tuple
        Any number of strings to to clean

    Returns
    -------
    clean: list
        List containing the clean strings
    """
    clean = []
    for name in args:
        clean.append(re.sub(
                            r"[\W]+", "", name.replace(" ", "_")
                            ).lstrip("0123456789")
                     )
    return clean


def merge_nested_dicts(original_dict, dict_to_merge):
    """
    Merge dictionaries recursively, preserving common keys.

    Parameters
    ----------
    original_dict: dict
        Dictionary onto which the merge is executed.

    dict_to_merge: dict
        Dictionary to be merged to the original_dict.

    Returns
    -------
    None

    """
    for k, v in dict_to_merge.items():
        if (k in original_dict and isinstance(original_dict[k], dict)
                and isinstance(dict_to_merge[k], Mapping)):
            merge_nested_dicts(original_dict[k], dict_to_merge[k])
        else:
            original_dict[k] = dict_to_merge[k]


def update_dependency(dependency, deps_dict):
    """
    Update dependency in dependencies dict.

    Parameters
    ----------
    dependency: str
        The dependency to add to the dependency dict.

    deps_dict: dict
        The dictionary of dependencies. If dependency is in deps_dict add 1
        to its value. Otherwise, add dependency to deps_dict with value 1.

    Returns
    -------
    None

    """
    if dependency in deps_dict:
        deps_dict[dependency] += 1
    else:
        deps_dict[dependency] = 1
