"""
These are general utilities used by the builder.py, functions.py or the
model file. Vensim's function equivalents should not go here but in
functions.py
"""

import json
from datetime import datetime
from pathlib import Path
from chardet.universaldetector import UniversalDetector

import progressbar
import numpy as np
import xarray as xr
import pandas as pd


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


def get_current_computer_time():
    """
    Returns the current machine time. Needed to mock the machine time in
    the tests.

    Parameters
    ---------
    None

    Returns
    -------
    datetime.now(): datetime.datetime
        Current machine time.

    """
    return datetime.now()


def get_return_elements(return_columns, namespace):
    """
    Takes a list of return elements formatted in vensim's format
    Varname[Sub1, SUb2]
    and returns first the model elements (in Python safe language)
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


def get_key_and_value_by_insensitive_key_or_value(key, dict):
    """
    Providing a key or value in a dictionary search for the real key and value
    in the dictionary ignoring case sensitivity.

    Parameters
    ----------
    key: str
        Key or value to look for in the dictionary.
    dict: dict
        Dictionary to search in.

    Returns
    -------
    real key, real value: (str, str) or (None, None)
        The real key and value that appear in the dictionary or a tuple
        of Nones if the input key is not in the dictionary.

    """
    lower_key = key.lower()
    for real_key, real_value in dict.items():
        if real_key.lower() == lower_key or real_value.lower() == lower_key:
            return real_key, real_value

    return None, None


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
            return data.rename({
                dim: new_dim for dim, new_dim in zip(data.dims, dims)
                if dim != new_dim
            }).loc[coords]

        # The coordinates are expanded or transposed
        return xr.DataArray(0, coords, dims) + data

    elif data is not None:
        return xr.DataArray(data, coords, dims)

    return None


def load_model_data(root, model_name):

    """
    Used for models split in several files.
    Loads subscripts and modules dictionaries

    Parameters
    ----------
    root: pathlib.Path
        Path to the model file.

    model_name: str
        Name of the model without file type extension (e.g. "my_model").

    Returns
    -------
    subscripts: dict
        Dictionary describing the possible dimensions of the stock's
        subscripts.

    modules: dict
        Dictionary containing view (module) names as keys and a list of the
        corresponding variables as values.

    """
    with open(root.joinpath("_subscripts_" + model_name + ".json")) as subs:
        subscripts = json.load(subs)

    # the _modules.json in the sketch_var folder shows to which module each
    # variable belongs
    with open(root.joinpath("modules_" + model_name, "_modules.json")) as mods:
        modules = json.load(mods)

    return subscripts, modules


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

    work_dir: pathlib.Path
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
        with open(work_dir.joinpath(module_name + ".py"), "r",
                  encoding="UTF-8") as mod:
            submodules.append(mod.read())
    else:
        for submod_name, submod_content in module_content.items():
            load_modules(
                submod_name, submod_content,
                work_dir.joinpath(module_name),
                submodules)

    return "\n\n".join(submodules)


def load_outputs(file_name, transpose=False, columns=None, encoding=None):
    """
    Load outputs file

    Parameters
    ----------
    file_name: str
        Output file to read. Must be csv or tab.

    transpose: bool (optional)
        If True reads transposed outputs file, i.e. one variable per row.
        Default is False.

    columns: list or None (optional)
        List of the column names to load. If None loads all the columns.
        Default is None.
        NOTE: if transpose=False, the loading will be faster as only
        selected columns will be loaded. If transpose=True the whole
        file must be read and it will be subselected later.

    encoding: str or None (optional)
        Encoding type to read output file. Needed if the file has special
        characters. Default is None.

    Returns
    -------
    pandas.DataFrame
        A pandas.DataFrame with the outputs values.

    """
    read_func = {'.csv': pd.read_csv, '.tab': pd.read_table}

    if isinstance(file_name, str):
        file_name = Path(file_name)

    if columns:
        columns = set(columns)
        if not transpose:
            columns.add("Time")

    for end, func in read_func.items():
        if file_name.suffix.lower() == end:
            if transpose:
                out = func(file_name,
                           encoding=encoding,
                           index_col=0).T
                if columns:
                    out = out[list(columns)]
            else:
                out = func(file_name,
                           encoding=encoding,
                           usecols=columns,
                           index_col="Time")

            out.index = out.index.astype(float)
            # return the dataframe removing nan index values
            return out[~np.isnan(out.index)]

    raise ValueError(
        f"\nNot able to read '{file_name}'. "
        + f"Only {', '.join(list(read_func))} files are accepted.")


def detect_encoding(filename):
    """
    Detects the encoding of a file.

    Parameters
    ----------
    filename: str
        Name of the file to detect the encoding.

    Returns
    -------
    encoding: str
        The encoding of the file.

    """
    detector = UniversalDetector()
    with open(filename, 'rb') as file:
        for line in file.readlines():
            detector.feed(line)
    detector.close()
    return detector.result['encoding']


def print_objects_format(object_set, text):
    """
    Return a printable version of the variables in object_sect with the
    header given with text.
    """
    text += " (total %(n_obj)s):\n\t%(objs)s\n" % {
        "n_obj": len(object_set),
        "objs": ", ".join(object_set)
    }
    return text


class ProgressBar:
    """
    Progress bar for integration
    """

    def __init__(self, max_value=None):

        self.max_value = max_value
        if self.max_value is None:
            return

        self.counter = 0

        self.bar = progressbar.ProgressBar(
            max_value=self.max_value,
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
