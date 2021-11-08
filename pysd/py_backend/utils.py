"""
These are general utilities used by the builder.py, functions.py or the
model file. Vensim's function equivalents should not go here but in
functions.py
"""

import os
import warnings
import json
from collections.abc import Mapping

import regex as re
import progressbar
import numpy as np
import xarray as xr
import pandas as pd

# used to create python safe names with the variable reserved_words
from keyword import kwlist
from builtins import __dir__ as bidir
from .decorators import __dir__ as ddir
from .external import __dir__ as edir
from .functions import __dir__ as fdir
from .statefuls import __dir__ as sdir


reserved_words = set(dir() + fdir() + edir() + ddir() + sdir() + bidir())
reserved_words = reserved_words.union(kwlist)


def xrmerge(*das):
    """
    Merges xarrays with different dimension sets.

    Parameters
    ----------
    *das: xarray.DataArrays
        The data arrays to merge.


    Returns
    -------
    da: xarray.DataArray
        Merged data array.

    References
    ----------
    Thanks to @jcmgray
    https://github.com/pydata/xarray/issues/742#issue-130753818

    In the future, we may not need this as xarray may provide the merge for us.
    """
    da = das[0]
    for new_da in das[1:]:
        da = da.combine_first(new_da)

    return da


def xrsplit(array):
    """
    Split an array to a list of all the components.

    Parameters
    ----------
    array: xarray.DataArray
        Array to split.

    Returns
    -------
    sp_list: list of xarray.DataArrays
        List of shape 0 xarray.DataArrays with coordinates.

    """
    sp_list = [sa for sa in array]
    if sp_list[0].shape:
        sp_list = [ssa for sa in sp_list for ssa in xrsplit(sa)]
    return sp_list


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


def get_return_elements(return_columns, namespace):
    """
    Takes a list of return elements formatted in vensim's format
    Varname[Sub1, SUb2]
    and returns first the model elements (in python safe language)
    that need to be computed and collected, and secondly the addresses
    that each element in the return columns list translates to

    Parameters
    ----------
    return_columns: list of strings

    namespace: dict

    Returns
    -------
    capture_elements
    return_addresses

    """
    capture_elements = list()
    return_addresses = dict()
    for col in return_columns:
        if col[0] == col[-1] and col[0] == '"':
            name = col
            address = None
        elif "[" in col:
            name, location = col.strip("]").split("[")
            address = tuple([loc.strip() for loc in location.split(",")])
        else:
            name = col
            address = None

        if name in namespace:
            py_name = namespace[name]
        else:
            if name in namespace.values():
                py_name = name
            else:
                raise KeyError(name + " not found as model element")

        if py_name not in capture_elements:
            capture_elements += [py_name]

        return_addresses[col] = (py_name, address)

    return list(capture_elements), return_addresses


def make_flat_df(df, return_addresses, flatten=False):
    """
    Takes a dataframe from the outputs of the integration processes,
    renames the columns as the given return_adresses and splits xarrays
    if needed.

    Parameters
    ----------
    df: Pandas.DataFrame
        Output from the integration.

    return_addresses: dict
        Keys will be column names of the resulting dataframe, and are what the
        user passed in as 'return_columns'. Values are a tuple:
        (py_name, {coords dictionary}) which tells us where to look for the
        value to put in that specific column.

    flatten: bool (optional)
            If True, once the output dataframe has been formatted will
            split the xarrays in new columns following vensim's naming
            to make a totally flat output. Default is False.

    Returns
    -------
    new_df: pandas.DataFrame
        Formatted dataframe.

    """
    new_df = {}
    for real_name, (pyname, address) in return_addresses.items():
        if address:
            # subset the specific address
            values = [x.loc[address] for x in df[pyname].values]
        else:
            # get the full column
            values = df[pyname].to_list()

        is_dataarray = len(values) != 0 and isinstance(values[0], xr.DataArray)

        if is_dataarray and values[0].size == 1:
            # some elements are returned as 0-d arrays, convert
            # them to float
            values = [float(x) for x in values]

        if flatten and is_dataarray:
            _add_flat(new_df, real_name, values)
        else:
            new_df[real_name] = values

    return pd.DataFrame(index=df.index, data=new_df)


def _add_flat(savedict, name, values):
    """
    Add float lists from a list of xarrays to a provided dictionary.

    Parameters
    ----------
    savedict: dict
        Dictionary to save the data on.

    name: str
        The base name of the variable to save the data.

    values: list
        List of xarrays to convert to split in floats.

    Returns
    -------
    None

    """
    # remove subscripts from name if given
    name = re.sub(r'\[.*\]', '', name)
    # split values in xarray.DataArray
    lval = [xrsplit(val) for val in values]
    for i, ar in enumerate(lval[0]):
        vals = [float(v[i]) for v in lval]
        subs = '[' + ','.join([str(ar.coords[dim].values)
                               for dim in list(ar.coords)]) + ']'
        savedict[name+subs] = vals


def compute_shape(coords, reshape_len=None, py_name=""):
    """
    Computes the 'shape' of a coords dictionary.
    Function used to rearange data in xarrays and
    to compute the number of rows/columns to be read in a file.

    Parameters
    ----------
    coords: dict
      Ordered dictionary of the dimension names as a keys with their values.

    reshape_len: int (optional)
      Number of dimensions of the output shape.
      The shape will ony compute the corresponent table
      dimensions to read from Excel, then, the dimensions
      with length one will be ignored at first.
      Lately, it will complete with 1 on the left of the shape
      if the reshape_len value is bigger than the length of shape.
      Will raise a ValueError if we try to reshape to a reshape_len
      smaller than the initial shape.

    py_name: str
      Name to print if an error is raised.

    Returns
    -------
    shape: list
      Shape of the ordered dictionary or of the desired table or vector.

    Note
    ----
    Dictionaries in Python >= 3.7 are ordered, which means that
    we could remove dims if there is a not backward compatible
    version of the library which only works in Python 3.7+. For now,
    the dimensions list is passed to make it work properly for all the users.

    """
    if not reshape_len:
        return [len(coord) for coord in coords.values()]

    # get the shape of the coordinates bigger than 1
    shape = [len(coord) for coord in coords.values() if len(coord) > 1]

    shape_len = len(shape)

    # return an error when the current shape is bigger than the requested one
    if shape_len > reshape_len:
        raise ValueError(
            py_name
            + "\n"
            + "The shape of the coords to read in a "
            + " external file must be at most "
            + "{} dimensional".format(reshape_len)
        )

    # complete with 1s on the left
    return [1] * (reshape_len - shape_len) + shape


def get_value_by_insensitive_key_or_value(key, dict):
    lower_key = key.lower()
    for real_key, real_value in dict.items():
        if real_key.lower() == lower_key:
            return dict[real_key]
        if real_value.lower() == lower_key:
            return real_value

    return None


def rearrange(data, dims, coords):
    """
    Returns a xarray.DataArray object with the given coords and dims

    Parameters
    ---------
    data: float or xarray.DataArray
        The input data to rearrange.

    dims: list
        Ordered list of the dimensions.

    coords: dict
        Dictionary of the dimension names as a keys with their values.

    Returns
    -------
    xarray.DataArray

    """
    # subset used coords in general coords will be the subscript_dict
    coords = {dim: coords[dim] for dim in dims}
    if isinstance(data, xr.DataArray):
        shape = tuple(compute_shape(coords))
        if data.shape == shape:
            # Allows switching dimensions names and transpositions
            return xr.DataArray(data=data.values, coords=coords, dims=dims)
        elif np.prod(shape) < np.prod(data.shape):
            # Allows subscripting a subrange
            return data.rename(
                {dim: new_dim for dim, new_dim in zip(data.dims, dims)}
            ).loc[coords]

        # The coordinates are expanded or transposed
        return xr.DataArray(0, coords, dims) + data

    elif data is not None:
        return xr.DataArray(data, coords, dims)

    return None


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


def load_model_data(root_dir, model_name):

    """
    Used for models split in several files.
    Loads subscripts_dic, namespace and modules dictionaries

    Parameters
    ----------
    root_dir: str
        Path to the model file.

    model_name: str
        Name of the model without file type extension (e.g. "my_model").

    Returns
    -------
    namespace: dict
        Translation from original model element names (keys) to python safe
        function identifiers (values).

    subscripts: dict
        Dictionary describing the possible dimensions of the stock's
        subscripts.

    modules: dict
        Dictionary containing view (module) names as keys and a list of the
        corresponding variables as values.

    """

    with open(os.path.join(root_dir, "_subscripts_" + model_name + ".json")
              ) as subs:
        subscripts = json.load(subs)

    with open(os.path.join(root_dir, "_namespace_" + model_name + ".json")
              ) as names:
        namespace = json.load(names)
    with open(os.path.join(root_dir, "_dependencies_" + model_name + ".json")
              ) as deps:
        dependencies = json.load(deps)

    # the _modules.json in the sketch_var folder shows to which module each
    # variable belongs
    with open(os.path.join(root_dir, "modules_" + model_name, "_modules.json")
              ) as mods:
        modules = json.load(mods)

    return namespace, subscripts, dependencies, modules


def load_modules(module_name, module_content, work_dir, submodules):
    """
    Used to load model modules from the main model file, when
    split_views=True in the read_vensim function. This function is used
    to iterate over the different layers of the nested dictionary that
    describes which model variables belong to each module/submodule.

    Parameters
    ----------
    module_name: str
        Name of the module to load.

    module_content: dict or list
        Content of the module. If it's a dictionary, it means that the
        module has submodules, whereas if it is a list it means that that
        particular module/submodule is a final one.

    work_dir: str
        Path to the module file.

    submodules: list
        This list gets updated at every recursive iteration, and each element
        corresponds to the string representation of each module/submodule that
        is read.

    Returns
    -------
    str:
        String representations of the modules/submodules to execute in the main
        model file.

    """
    if isinstance(module_content, list):
        with open(os.path.join(work_dir, module_name + ".py"), "r") as mod:
            submodules.append(mod.read())
    else:
        for submod_name, submod_content in module_content.items():
            load_modules(
                submod_name, submod_content,
                os.path.join(work_dir, module_name),
                submodules)

    return "\n\n".join(submodules)


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


class ProgressBar:
    """
    Progress bar for integration
    """

    def __init__(self, maxval=None):

        self.maxval = maxval
        if self.maxval is None:
            return

        self.counter = 0

        self.bar = progressbar.ProgressBar(
            maxval=self.maxval,
            widgets=[
                progressbar.ETA(),
                " ",
                progressbar.Bar("#", "[", "]", "-"),
                progressbar.Percentage(),
            ],
        )

        self.bar.start()

    def update(self):
        """Update progress bar"""
        try:
            self.counter += 1
            self.bar.update(self.counter)
        except AttributeError:
            # Error if bar is not imported
            pass

    def finish(self):
        """Finish progress bar"""
        try:
            self.bar.finish()
        except AttributeError:
            # Error if bar is not imported
            pass
