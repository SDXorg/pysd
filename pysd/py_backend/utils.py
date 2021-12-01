"""
These are general utilities used by the builder.py, functions.py or the
model file. Vensim's function equivalents should not go here but in
functions.py
"""

import os
import json
from chardet.universaldetector import UniversalDetector

import regex as re
import progressbar
import numpy as np
import xarray as xr
import pandas as pd


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
            return data.rename(
                {dim: new_dim for dim, new_dim in zip(data.dims, dims)}
            ).loc[coords]

        # The coordinates are expanded or transposed
        return xr.DataArray(0, coords, dims) + data

    elif data is not None:
        return xr.DataArray(data, coords, dims)

    return None


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

    if columns:
        columns = set(columns)
        if not transpose:
            columns.add("Time")

    for end, func in read_func.items():
        if file_name.lower().endswith(end):
            if transpose:
                out = func(file_name,
                           encoding=encoding,
                           index_col=0).T
                if columns:
                    out = out[columns]
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
    for line in open(filename, 'rb').readlines():
        detector.feed(line)
    detector.close()
    return detector.result['encoding']


def print_objects_format(object_set, text):
    text += " (total %(n_obj)s):\n\t%(objs)s\n" % {
        "n_obj": len(object_set),
        "objs": ", ".join(object_set)
    }
    return text

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
