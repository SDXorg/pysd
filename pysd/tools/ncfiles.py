"""
Tools for importing and converting netCDF files generated from simulations run
using PySD.
"""

from csv import QUOTE_NONE
from lib2to3.pgen2.token import OP
from tkinter.ttk import Progressbar
import xarray as xr
import pandas as pd
from pathlib import Path
import warnings
from typing import Union, Optional
import itertools


def open_nc_file(ncfile: Union[str, Path],
                 parallel: Optional[bool]=False) -> xr.Dataset:
    """
    Loads netCDF file into xarray Dataset. It's basically a wrapper to
    xr.open_dataset to simplify the interface for pysd use case (loading
    simulation results).

    Parameters
    ----------
    ncfile: str or pathlib.Path
        Path to the netCDF file to process.
    parallel: bool
        When True, the Dataset is opened using chunks=-1 (see xarray
        documentation for details), and when it's False, chunks=None.

    Returns
    -------
    xarray.Dataset

    """

    # we run all these checks because xarrays Exceptions are not very clear.
    if not isinstance(ncfile, (str, Path)):
        raise TypeError(f"Invalid file path type: {type(ncfile)}.\n"
                        "Please provide string or pathlib Path")

    if isinstance(ncfile, str):
        ncfile = Path(ncfile)

    if not ncfile.is_file():
        raise FileNotFoundError(f"{ncfile} could not be found.")

    if not ncfile.suffix == ".nc":
        raise TypeError("Input file must have nc extension.")

    if parallel:
        return xr.open_dataset(ncfile, engine="netcdf4", chunks=-1)

    return xr.open_dataset(ncfile, engine="netcdf4")

def _validate_ds_subset(ds: xr.Dataset, subset: list) -> list:
    """
    If subset=None, it returns a list with all variable names in the ds.
    If var names in subset are present in ds, it returns them, else it warns
    the user.

    Parameters
    ----------
    subset: list
        Subset of variable names in the xarray Dataset.

    """
    # process subset list
    if not subset: # use all names
        new_subset = [name for name in ds.data_vars.keys()]
    else:
        if not isinstance(subset, list) or \
             not all(map(lambda x: isinstance(x, str), subset)):
            raise TypeError("Subset argument must be a list of strings.")

        new_subset =[]
        for name in subset:
            if name in ds.data_vars.keys():
                new_subset.append(name)
            else:
                warnings.warn(f"{name} not in Dataset.")

        if not new_subset:
            raise ValueError("None of the elements of the subset are present"
                             "in the Dataset.")
    return new_subset

def _index_da_by_coord_labels(da: xr.DataArray,
                              dims: list,
                              coords: tuple
                              ) -> tuple:
    """
    Function to generate variable names, combining the actual name
    with the coordinate names between brackets and separated by commas,
    and to index the DataArray by the coordinate names specified in
    the coords argument.

    Parameters
    ----------
    da: xr.Dataset
        Dataset to be indexed.
    dims: list
        Dimensions along which the DataArray will be indexed.
    coords: tuple
        Coordinate names for each of the dimensons in the dims list.

    Returns
    -------
    A tuple consisting of the string
    var_name[dim_1_coord_j, ..., dim_n_coord_k] in the first index, and the
    indexed data as the second index.

    """
    name = da.attrs["Py Name"]
    idx = dict(zip(dims, coords))
    subs = "[" + ",".join(coords) + "]"
    return name + subs, da.loc[idx].values

def _get_da_dims_coords(da: xr.DataArray, exclude_dims: list=[]) -> tuple:
    """
    Returns the dimension names and coordinate labels in two separate lists.
    If a dimension name is in the exclude_dims lists, the returned dims and
    coords will not include it.

    Parameters
    ----------
    exclude_dims: list
        List with the names of dimensions to exclude.

    Returns
    -------
    dims: list
        List containing the names of the DataArray dimensions.
    coords: list
        List of lists of coordinates for each dimension.

    """
    dims, coords = [], []

    for dim in da.dims:
        if dim not in exclude_dims:
            dims.append(dim)
            coords.append(da.coords[dim].values)

    return dims, coords

def split_da(da: xr.DataArray) -> dict:
    """
    Splits a DataArray into a dictionary, with keys equal to the name of
    the variable plus all combinations of the cartesian product of
    coordinates within brackets, and values equal to the data corresponding
    to those coordinates.
    """
    dims, coords = _get_da_dims_coords(da, exclude_dims=["time"])

    l = []
    for coords_prod in itertools.product(*coords):
        l.append(_index_da_by_coord_labels(da, dims, coords_prod))

    return dict(l)

def split_da_delayed(da: xr.DataArray) -> dict:

    """
    Same as split_da, but using dask delayed and compute.
    This function runs much faster when da is a dask array (chunked).
    """

    dims, coords = _get_da_dims_coords(da, exclude_dims=["time"])

    l = []
    for coords_prod in itertools.product(*coords):
        x = delayed(_index_da_by_coord_labels)(da, dims, coords_prod)
        l.append(x)

    with ProgressBar():
        res = compute(*l)

    return dict(res)

def _dict_to_df(savedict: dict, time_in_row: bool) -> pd.DataFrame:
    """
    Convert a dict to a pandas Dataframe.

    Parameters
    ----------
    savedict: dict
        Dictionary to convert to pandas DataFrame.
    time_in_row: bool
        Whether time increases along a column or a row.

    """
    # process time_in_row argument
    if not isinstance(time_in_row, bool):
        raise ValueError("time_in_row argument takes boolen values.")

    df = pd.DataFrame(savedict)

    if time_in_row:
        df = df.transpose()

    return df

def df_to_csv(df: pd.DataFrame, outfile: Union[str, Path]) -> None:
    """
    Store pandas DataFrame into csv or tab file.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame to save as csv or tab file.
    outfile: pathlib.Path
        Path of the output file.

    Returns
    -------
    None
    """
        # process output file path
    if isinstance(outfile, str):
        outfile = Path(outfile)

    out_fmt = outfile.suffix
    if out_fmt not in [".csv", ".tab"]:
        raise TypeError("Invalid output file format {out_fmt}\n"
                        "Supported formats are csv and tab.")

    # TODO fix the creation of the parent folders if they don't exist
    outfile.parent. mkdir(parents=True, exist_ok=True)

    if outfile.suffix == ".csv":
        sep = ","
    elif outfile.suffix == ".tab":
        sep = "\t"
    else:
        raise ValueError("Wrong export file type.")

    df.to_csv(outfile, sep=sep, index_label="Time",
              quoting=QUOTE_NONE,
              escapechar="\\")

def ds_to_df(ds: xr.Dataset,
             subset: Optional[list]=None,
             time_in_row: Optional[bool]=False,
             parallel: Optional[bool]=False
             ) -> pd.DataFrame:
    """
    Convert xarray.Dataset into a pandas DataFrame.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset object.
    subset: list
        List of variables to export from the Dataset.
    time_in_row: bool
        Whether time increases along row. Default is False.
    parallel: bool
        When True, DataArrays are processed in parallel using dask delayed.
        Dask is not included as a requirement for pysd, hence it must be
        installed separately. Setting parallel=True is highly recommended
        when DataArrays are large and multidimensional.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with all colums specified in subset.

    """
    subset = _validate_ds_subset(ds, subset)

    savedict = {}

    if parallel:
        global delayed, compute, ProgressBar
        from dask import delayed, compute
        from dask.diagnostics import ProgressBar

        for name in subset:
            print(f"\nProcessing variable {name}")
            da = ds[name]

            if da.dims:
                savedict.update(split_da_delayed(da))
            else:
                savedict.update({name: da.values.tolist()})
    else:
        for name in subset:
            print(f"\nProcessing variable {name}")
            da = ds[name]

            if da.dims:
                savedict.update(split_da(da))
            else:
                savedict.update({name: da.values.tolist()})

    return _dict_to_df(savedict, time_in_row)

def nc_to_df(ncfile: Union[str, Path],
             subset: Optional[list]=None,
             time_in_row: Optional[bool]=False,
             parallel: Optional[bool]=False
             ) -> pd.DataFrame:
    """
    Convert netCDF file contents into a pandas DataFrame.

    Parameters
    ----------
    ncfile: str
        Path to the netCDF file to process.
    subset: list
        List of variables to export from the netCDF.
    time_in_row: bool
        Whether time increases along row.
        Default is False.
    parallel: bool
        When True, the Dataset is opened using chunks=-1 (see xarray
        documentation for details) and DataArrays are processed in parallel
        using dask delayed. Dask is not included as a requirement for pysd,
        hence it must be installed separately. Setting parallel=True is
        highly recommended when the Dataset contains large multidimensional
        DataArrays.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with all colums specified in subset.

    """

    ds = open_nc_file(ncfile, parallel)

    return ds_to_df(ds, subset, time_in_row, parallel)

def nc_to_csv(ncfile: Union[str, Path],
              outfile: Optional[Union[str, Path]]="result.tab",
              subset: Optional[list]=None,
              time_in_row: Optional[bool]=False,
              parallel: Optional[bool]=False
              ) -> pd.DataFrame:
    """
    Convert netCDF file contents into comma or tab separated file.

    Parameters
    ----------
    ncfile: str
        Path to the netCDF file to process.
    outfile: str
        Path to the output file.
    subset: list
        List of variables to export from the netCDF.
    time_in_row: bool
        Whether time increases along row.
        Default is False.
    parallel: bool
        When True, the Dataset is opened using chunks=-1 (see xarray
        documentation for details) and DataArrays are processed in parallel
        using dask delayed. Dask is not included as a requirement for pysd,
        hence it must be installed separately. Setting parallel=True is
        highly recommended when the Dataset contains large multidimensional
        DataArrays.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with all colums specified in subset.

    """
    df = nc_to_df(ncfile, subset=subset, time_in_row=time_in_row,
                  parallel=parallel
                  )

    df_to_csv(df, outfile)

    return df


