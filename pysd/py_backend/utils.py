import keyword
import re

import numpy as np
import pandas as pd
import xarray as xr


def dict_find(in_dict, value):
    """ Helper function for looking up directory keys by their values.
     This isn't robust to repeated values

    Parameters
    ----------
    in_dict : dictionary
        A dictionary containing `value`

    value : any type
        What we wish to find in the dictionary

    Returns
    -------
    key: basestring
        The key at which the value can be found

    Examples
    --------
    >>> dict_find({'Key1': 'A', 'Key2': 'B'}, 'B')
    'Key2'

    """
    # Todo: make this robust to repeated values
    # Todo: make this robust to missing values
    return list(in_dict.keys())[list(in_dict.values()).index(value)]


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
        da, new_da = xr.align(da, new_da, join='outer')
        # Fill NaNs one way or the other re. accept_new
        da = new_da.fillna(da) if accept_new else da.fillna(new_da)
    return da


def find_subscript_name(subscript_dict, element):
    """
    Given a subscript dictionary, and a member of a subscript family,
    return the first key of which the member is within the value list.
    If element is already a subscript name, return that

    Parameters
    ----------
    subscript_dict: dictionary
        Follows the {'subscript name':['list','of','subscript','elements']} format

    element: string

    Returns
    -------

    Examples:
    >>> find_subscript_name({'Dim1': ['A', 'B'],
    ...                      'Dim2': ['C', 'D', 'E'],
    ...                      'Dim3': ['F', 'G', 'H', 'I']},
    ...                      'D')
    'Dim2'
    """
    if element in subscript_dict.keys():
        return element

    for name, elements in subscript_dict.items():
        if element in elements:
            return name


def make_coord_dict(subs, subscript_dict, terse=True):
    """
    This is for assisting with the lookup of a particular element, such that the output
    of this function would take the place of %s in this expression

    `variable.loc[%s]`

    Parameters
    ----------
    subs: list of strings
        coordinates, either as names of dimensions, or positions within a dimension
    subscript_dict: dict
        the full dictionary of subscript names and values
    terse: Binary Flag
        - If true, includes only elements that do not cover the full range of values in their
          respective dimension
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
            name = find_subscript_name(subscript_dict, sub)
            coordinates[name] = [sub]
        elif not terse:
            coordinates[sub] = subscript_dict[sub]
    return coordinates


def make_python_identifier(string, namespace=None, reserved_words=None,
                           convert='drop', handle='force'):
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

    if string in namespace:
        return namespace[string], namespace

    # create a working copy (and make it lowercase, while we're at it)
    s = string.lower()

    # remove leading and trailing whitespace
    s = s.strip()

    # Make spaces into underscores
    s = re.sub('[\\s\\t\\n]+', '_', s)

    if convert == 'hex':
        # Convert invalid characters to hex
        s = ''.join([c.encode("hex") if re.findall('[^0-9a-zA-Z_]', c) else c for c in s])

    elif convert == 'drop':
        # Remove invalid characters
        s = re.sub('[^0-9a-zA-Z_]', '', s)

    # Remove leading characters until we find a letter or underscore
    s = re.sub('^[^a-zA-Z_]+', '', s)

    # Check that the string is not a python identifier
    while (s in keyword.kwlist or
                   s in namespace.values() or
                   s in reserved_words):
        if handle == 'throw':
            raise NameError(s + ' already exists in namespace or is a reserved word')
        if handle == 'force':
            if re.match(".*?_\d+$", s):
                i = re.match(".*?_(\d+)$", s).groups()[0]
                s = s.strip('_' + i) + '_' + str(int(i) + 1)
            else:
                s += '_1'

    namespace[string] = s

    return s, namespace


def get_return_elements(return_columns, namespace, subscript_dict):
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
    subscript_dict

    Returns
    -------
    capture_elements
    return_addresses

    Examples
    --------

    """
    # Todo: clean up duplicated code
    capture_elements = list()
    return_addresses = dict()
    for col in return_columns:
        if '[' in col:
            name, location = col.strip(']').split('[')
            subs = [l.strip() for l in location.split(',')]
            try:
                py_name = namespace[name]
            except KeyError:
                if name in namespace.values():
                    py_name = name
                else:
                    raise KeyError(name + " not found as model element")

            if py_name not in capture_elements:
                capture_elements += [py_name]
            address = make_coord_dict(subs, subscript_dict)
            return_addresses[col] = (py_name, address)
        else:
            try:
                py_name = namespace[col]
            except KeyError:
                if col in namespace.values():
                    py_name = col
                else:
                    raise KeyError(col + " not found as model element")

            if py_name not in capture_elements:
                capture_elements += [py_name]
            return_addresses[col] = (py_name, {})

    return list(capture_elements), return_addresses


def make_flat_df(frames, return_addresses):
    """
    Takes a list of dictionaries, each representing what is returned from the
    model at a particular time, and creates a dataframe whose columns correspond
    to the keys of `return addresses`

    Parameters
    ----------
    frames: list of dictionaries
        each dictionary represents the result of a prticular time in the model
    return_addresses: a dictionary,
        keys will be column names of the resulting dataframe, and are what the
        user passed in as 'return_columns'. Values are a tuple:
        (py_name, {coords dictionary}) which tells us where to look for the value
        to put in that specific column.

    Returns
    -------

    """

    # Todo: could also try a list comprehension here, or parallel apply
    visited = list(map(lambda x: visit_addresses(x, return_addresses), frames))
    return pd.DataFrame(visited)


def visit_addresses(frame, return_addresses):
    """
    Visits all of the addresses, returns a new dict
    which contains just the addressed elements


    Parameters
    ----------
    frame
    return_addresses: a dictionary,
        keys will be column names of the resulting dataframe, and are what the
        user passed in as 'return_columns'. Values are a tuple:
        (py_name, {coords dictionary}) which tells us where to look for the value
        to put in that specific column.

    Returns
    -------
    outdict: dictionary

    """
    outdict = dict()
    for real_name, (pyname, address) in return_addresses.items():
        if address:
            xrval = frame[pyname].loc[address]
            if xrval.size > 1:
                outdict[real_name] = xrval
            else:
                outdict[real_name] = float(np.squeeze(xrval.values))
        else:
            outdict[real_name] = frame[pyname]

    return outdict
