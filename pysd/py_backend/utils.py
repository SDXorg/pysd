"""
These are general utilities used by the builder.py, functions.py or the
model file. Vensim's function equivalents should not go here but in
functions.py
"""

import warnings
import keyword
import regex as re
import os
import json

import xarray as xr

# used to create python safe names
from .decorators import __dir__ as ddir
from .external import __dir__ as edir
from .functions import __dir__ as fdir


def xrmerge(das, accept_new=True):
    """
    Merges xarrays with different dimension sets
    Parameters
    ----------
    das : list of data_arrays

    accept_new

    Returns
    -------
    da : an xarray that is the merge of das

    References
    ----------
    Thanks to @jcmgray https://github.com/pydata/xarray/issues/742#issue-130753818

    In the future, we may not need this as xarray may provide the merge for us.
    """
    da = das[0]
    for new_da in das[1:]:
        # Expand both to have same dimensions, padding with NaN
        da, new_da = xr.align(da, new_da, join="outer")
        # Fill NaNs one way or the other re. accept_new
        da = new_da.fillna(da) if accept_new else da.fillna(new_da)
    return da


def xrsplit(array):
    """
        Split an array to a list of all the components

        Parameters
        ----------
        array: xarray.DataArray
            Array to split.

        Returns
        -------
        sp_list: list
            List of shape 0 xarray.DataArrays with coordinates
        """
    sp_list = [sa for sa in array]
    if sp_list[0].shape:
        sp_list = [ssa for sa in sp_list for ssa in xrsplit(sa)]
    return sp_list


def find_subscript_name(subscript_dict, element, avoid=[]):
    """
    Given a subscript dictionary, and a member of a subscript family,
    return the first key of which the member is within the value list.
    If element is already a subscript name, return that

    Parameters
    ----------
    subscript_dict: dictionary
        Follows the {'subscript name':['list','of','subscript','elements']} format

    element: string

    avoid: list (optional)
        List of subscripts to avoid. Default is an empty list.

    Returns
    -------

    Examples:
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
    the output of this function would take the place of %s in this expression

    `variable.loc[%s]`

    Parameters
    ----------
    subs: list of strings
        coordinates, either as names of dimensions, or positions within
        a dimension
    subscript_dict: dict
        the full dictionary of subscript names and values
    terse: Binary Flag
        - If true, includes only elements that do not cover the full range of
          values in their respective dimension
        - If false, returns all dimensions

    Returns
    -------
    coordinates: dictionary
        Coordinates needed to access the xarray quantities we're interested in.

    Examples
    --------
    >>> make_coord_dict(['Dim1', 'D'], {'Dim1': ['A', 'B', 'C'], 'Dim2': ['D', 'E', 'F']})
    {'Dim2': ['D']}
    >>> make_coord_dict(['Dim1', 'D'], {'Dim1': ['A', 'B', 'C'], 'Dim2':['D', 'E', 'F']},
    >>>                 terse=False)
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


def make_merge_list(subs_list, subscript_dict):
    """
    This is for assisting when building xrmerge. From a list of subscript
    lists returns the final subscript list after mergin. Necessary when
    merging variables with subscripts comming from different definitions.

    Parameters
    ----------
    subs_list: list of lists of strings
        coordinates, either as names of dimensions, or positions within
        a dimension
    subscript_dict: dict
        the full dictionary of subscript names and values

    Returns
    -------
    dims: list
        Final subscripts after merging.

    Examples
    --------
    >>> make_merge_list([['upper'], ['C']], {'all': ['A', 'B', 'C'], 'upper': ['A', 'B']})
    ['all']

    """
    coords_set = [set() for i in range(len(subs_list[0]))]
    for subs in subs_list:
        coords = make_coord_dict(subs, subscript_dict, terse=False)
        [coords_set[i].update(coords[dim]) for i, dim in enumerate(coords)]

    dims = [None] * len(coords_set)
    for i, (coord1, coord2) in enumerate(zip(coords, coords_set)):
        if set(coords[coord1]) == coord2:
            # if the given coordinate already matches return it
            dims[i] = coord1
        else:
            # find a suitable coordinate
            for name, elements in subscript_dict.items():
                if coord2 == set(elements) and name not in subs_list[0]:
                    dims[i] = name
                    break

            if not dims[i]:
                # the dimension is incomplete use the smaller
                # dimension that completes it
                for name, elements in subscript_dict.items():
                    if coord2.issubset(set(elements))\
                      and name not in subs_list[0]:
                        dims[i] = name
                        warnings.warn(
                            "\nDimension given by subscripts:"
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
                            "\nAdding new subscript range to subscript_dict:\n"
                            + name + str(j) + ": " + ', '.join(elements))
                        break

            if not dims[i]:
                # not able to find the correct dimension
                raise ValueError(
                    "\nImpossible to find the dimension that contains:"
                    + "\n\t{}\nFor subscript_dict:".format(coord2)
                    + "\n\t{}".format(subscript_dict)
                )

    return dims


def make_python_identifier(
    string, namespace=None, reserved_words=None, convert="drop", handle="force"
):
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
    string : <basestring>
        The text to be converted into a valid python identifier
    namespace : <dictionary>
        Map of existing translations into python safe identifiers.
        This is to ensure that two strings are not translated into
        the same python identifier
    reserved_words : <list of strings>
        List of words that are reserved (because they have other meanings
        in this particular program, such as also being the names of
        libraries, etc.
    convert : <string>
        Tells the function what to do with characters that are not
        valid in python identifiers
        - 'hex' implies that they will be converted to their hexidecimal
                representation. This is handy if you have variables that
                have a lot of reserved characters, or you don't want the
                name to be dependent on when things were added to the
                namespace
        - 'drop' implies that they will just be dropped altogether
    handle : <string>
        Tells the function how to deal with namespace conflicts
        - 'force' will create a representation which is not in conflict
                  by appending _n to the resulting variable where n is
                  the lowest number necessary to avoid a conflict
        - 'throw' will raise an exception

    Returns
    -------
    identifier : <string>
        A vaild python identifier based on the input string
    namespace : <dictionary>
        An updated map of the translations of words to python identifiers,
        including the passed in 'string'.

    Examples
    --------
    >>> make_python_identifier('Capital')
    ('capital', {'Capital': 'capital'})

    >>> make_python_identifier('multiple words')
    ('multiple_words', {'multiple words': 'multiple_words'})

    >>> make_python_identifier('multiple     spaces')
    ('multiple_spaces', {'multiple     spaces': 'multiple_spaces'})

    When the name is a python keyword, add '_1' to differentiate it
    >>> make_python_identifier('for')
    ('for_1', {'for': 'for_1'})

    Remove leading and trailing whitespace
    >>> make_python_identifier('  whitespace  ')
    ('whitespace', {'  whitespace  ': 'whitespace'})

    Remove most special characters outright:
    >>> make_python_identifier('H@t tr!ck')
    ('ht_trck', {'H@t tr!ck': 'ht_trck'})

    Replace special characters with their hex representations
    >>> make_python_identifier('H@t tr!ck', convert='hex')
    ('h40t_tr21ck', {'H@t tr!ck': 'h40t_tr21ck'})

    remove leading digits
    >>> make_python_identifier('123abc')
    ('abc', {'123abc': 'abc'})

    already in namespace
    >>> make_python_identifier('Variable$', namespace={'Variable$': 'variable'})
    ('variable', {'Variable$': 'variable'})

    namespace conflicts
    >>> make_python_identifier('Variable$', namespace={'Variable@': 'variable'})
    ('variable_1', {'Variable@': 'variable', 'Variable$': 'variable_1'})

    >>> make_python_identifier('Variable$', namespace={'Variable@': 'variable',
    >>>                                                'Variable%': 'variable_1'})
    ('variable_2', {'Variable@': 'variable', 'Variable%': 'variable_1', 'Variable$': 'variable_2'})

    throw exception instead
    >>> make_python_identifier('Variable$', namespace={'Variable@': 'variable'}, handle='throw')
    Traceback (most recent call last):
     ...
    NameError: variable already exists in namespace or is a reserved word


    References
    ----------
    Identifiers must follow the convention outlined here:
        https://docs.python.org/2/reference/lexical_analysis.html#identifiers
    """

    if namespace is None:
        namespace = dict()

    if reserved_words is None:
        reserved_words = list()

    # reserved the names of PySD functions and methods
    reserved_words += dir() + fdir() + edir() + ddir()

    if string in namespace:
        return namespace[string], namespace

    # create a working copy (and make it lowercase, while we're at it)
    s = string.lower()

    # remove leading and trailing whitespace
    s = s.strip()

    # Make spaces into underscores
    s = re.sub(r"[\s\t\n]+", "_", s)

    if convert == "hex":
        # Convert invalid characters to hex. Note: \p{l} designates all
        # Unicode letter characters (any language), \p{m} designates all
        # mark symbols (e.g., vowel marks in Indian scrips, such as the final)
        # and \p{n} designates all numbers. We allow any of these to be
        # present in the regex.
        s = "".join(
            [c.encode("hex") if re.findall(r"[^\p{l}\p{m}\p{n}_]", c)
             else c for c in s]
        )

    elif convert == "drop":
        # Remove invalid characters
        s = re.sub(r"[^\p{l}\p{m}\p{n}_]", "", s)

    # If leading characters are not a letter or underscore add nvs_.
    # Only letters can be leading characters.
    if re.findall(r"^[^\p{l}_]+", s):
        s = "nvs_" + s

    # Check that the string is not a python identifier
    while (s in keyword.kwlist or
           s in namespace.values() or
           s in reserved_words):
        if handle == "throw":
            raise NameError(
                s + " already exists in namespace or is a reserved word")
        if handle == "force":
            if re.match(r".*?_\d+$", s):
                i = re.match(r".*?_(\d+)$", s).groups()[0]
                s = s.strip("_" + i) + "_" + str(int(i) + 1)
            else:
                s += "_1"

    namespace[string] = s

    return s, namespace


def make_add_identifier(identifier, build_names):
    """
    Takes an existing used Python identifier and attatch a unique
    identifier with ADD_# ending.

    Used for add new information to an existing external object.
    build_names will be updated inside this functions as a set
    is mutable.

    Parameters
    ----------
    string : <basestring>
      Existing python identifier
    build_names : <set>
      Set of the already used identifiers for external objects.

    Returns
    -------
    identifier : <string>
      A vaild python identifier based on the input indentifier
      and the existing ones
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
    namespace

    Returns
    -------
    capture_elements
    return_addresses

    Examples
    --------

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
    df: pandas.DataFrame
        Formatted dataframe.

    """
    cols_to_remove = set()
    rename_cols = {}
    for real_name, (pyname, address) in return_addresses.items():
        if address:
            cols_to_remove.add(pyname)
            # subset the value and add it to a new column
            xrval = [x.loc[address] for x in df[pyname].values]
            if xrval[0].size > 1:
                df[real_name] = xrval
            else:
                df[real_name] = [float(x) for x in xrval]
        else:
            # save the name to change it in the dataframe
            try:
                # some elements are returned as 0-d arrays, convert
                # them to float
                df[pyname] = [float(x) for x in df[pyname].values]
            except TypeError:
                pass
            rename_cols[pyname] = real_name

    df.rename(columns=rename_cols, inplace=True)

    if cols_to_remove:
        # remove the columns of the subset values
        df.drop(cols_to_remove, axis='columns', inplace=True)

    if flatten:
        # create a totally flat df (no xarray.DataArray)
        cols_to_remove.clear()
        for col in df.columns:
            if isinstance(df[col].values[0], xr.DataArray):
                # remove subscripts from name if given
                name = re.sub(r'\[.*\]', '', col)
                # split values in xarray.DataArray
                lval = [xrsplit(val) for val in df[col].values]
                for i, ar in enumerate(lval[0]):
                    vals = [float(v[i]) for v in lval]
                    subs = '[' + ','.join([str(ar.coords[dim].values)
                                           for dim in list(ar.coords)]) + ']'
                    df[name+subs] = vals
                    cols_to_remove.add(col)

        if cols_to_remove:
            # remove the columns of the subset values
            df.drop(cols_to_remove, axis='columns', inplace=True)

    return df


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
      Shape of the ordered dictionary or of the desired table or vector

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
      switch: bool
        Flag to denote if the dimensions can be switched. Default True,
        The False is used to rearrange general expressions

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
        elif len(shape) == len(data.shape) and all(
            [shape[i] < data.shape[i] for i in range(len(shape))]
        ):
            # Allows subscripting a subrange
            return data.rename(
                {dim: new_dim for dim, new_dim in zip(data.dims, dims)}
            ).loc[coords]

        # The coordinates are expanded or transposed
        return xr.DataArray(0, coords, dims) + data

    elif data is not None:
        return xr.DataArray(data, coords, dims)

    return None


def round_(x):
    """
    Redefinition of round function to make it work with floats and xarrays
    """
    if isinstance(x, xr.DataArray):
        return x.round()

    return round(x)


def add_entries_underscore(*dictionaries):
    """
    Expands dictionaries adding new keys underscoring the white spaces
    in the old ones. As the dictionaries are mutable objects this functions
    will add the new entries to the already existing dictionaries with
    no need to return a new one.

    Parameters
    ----------
    *dictionaries: Dictionary or dictionaries
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
        Path to the model file
    model_name: str
        Name of the model without file type extension (e.g. "my_model")

    Returns
    -------
    namespace: dict
        Translation from original model element names (keys) to python safe
        function identifiers (values).

    subscripts: dict
        Dictionary describing the possible dimensions of the stock's subscripts

    modules: dict
        Dictionary containing view (module) names as keys and a list of the
        corresponding variables as values.

    """

    with open(os.path.join(root_dir, "_subscripts_" + model_name + ".json")) as subs:
        subscripts = json.load(subs)

    with open(os.path.join(root_dir, "_namespace_" + model_name + ".json")) as names:
        namespace = json.load(names)

    # the _modules.json in the sketch_var folder shows to which module each variable belongs
    with open(os.path.join(root_dir, "modules_" + model_name, "_modules.json")) as mods:
        modules = json.load(mods)

    return namespace, subscripts, modules


def open_module(root_dir, model_name, module):
    """
    Used to load model modules from the main model file, when
    split_modules=True in the read_vensim function.

    Parameters
    ----------
    root_dir: str
        Path to the model file
    model_name: str
        Name of the model without file type extension (e.g. "my_model")
    module: str
        Name of the module to open

    Returns
    -------
    str:
        Model file content.

    """
    return open(
        os.path.join(root_dir, "modules_" + model_name, module + ".py")
        ).read()


class ProgressBar:
    """
    Progress bar for integration
    """

    def __init__(self, maxval=None):

        self.maxval = maxval
        if self.maxval is None:
            return

        self.counter = 0

        # this way we made the package optional
        import progressbar

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
