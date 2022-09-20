"""
Tools for importing and converting netCDF files generated from simulations run
using PySD.
"""

from csv import QUOTE_NONE
from lib2to3.pgen2.token import OP
import xarray as xr
import pandas as pd
from pathlib import Path
import warnings
from typing import Union, Optional
import itertools

TypeChunks = Union[int, dict, str]

def load_nc_file(ncfile: Union[str, Path],
                 chunks: Optional[TypeChunks]=None) -> xr.Dataset:
    """
    Loads netCDF file into xarray Dataset.

    Parameters
    ----------
    ncfile: str or pathlib.Path
        Path to the netCDF file to process.
    chunks: int, dict, str, None (optional)
        Check xarray.open_dataset for potential values for this argument.
        Frequent values are -1, to load each DataArray in a single chunk, or
        None, to not use dask array. Using chunks requires installing dask.

    Returns
    -------
    xarray.Dataset

    """
    if not isinstance(ncfile, (str, Path)):
        raise TypeError(f"Invalid file path type: {type(ncfile)}.\n"
                        "Please provide string or pathlib Path")

    if isinstance(ncfile, str):
        ncfile = Path(ncfile)

    if not ncfile.is_file():
        raise FileNotFoundError(f"{ncfile} could not be found.")

    if not ncfile.suffix == ".nc":
        raise TypeError("Input file must have nc extension.")

    if chunks:
        return xr.open_dataset(ncfile, engine="netcdf4", chunks=chunks)

    return xr.open_dataset(ncfile, engine="netcdf4")

def _validate_ds_subset(ds: xr.Dataset, subset: list) -> list:
    """
    If subset = None, it returns a list with all variables in the ds Dataset.
    Else, if vars in subset are present in ds, it returns them, else it warns
    the user.

    Parameters
    ----------
    subset: list
        Subset of variable names in the xarray Dataset.

    """
    # process subset list
    if not subset: # use all names
        new_subset = [name for name  in ds.data_vars.keys()]
    else:
        if not isinstance(subset, list) or \
             not all(map(lambda x: isinstance(x, str), subset)):
            raise TypeError("Subset argument must be a list of strings.")

        new_subset = [name if name in ds.data_vars.keys()
                      else warnings.warn(f"{name} not in Dataset.")
                      for name in subset]
    return new_subset

def _time_dim_from_da(da: xr.DataArray,
                      dims: list,
                      coords: tuple
                      ) -> tuple:
    """
    Function to extract all values along the time dimension with fixed
    coordinates for the other dimensions as defined in coords.


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
    var_name[dim1_coord, dim2_coord, ....] in the first index, and the
    indexed data as the second index.

    """
    name = da.attrs["Py Name"]
    idx = dict(zip(dims, coords))
    subs = "[" + ",".join(coords) + "]"
    return name + subs, da.loc[idx].values

def split_data_array(da):

    if not da.dims:
        savedict = {}
        name = da.attrs["Py Name"]
        savedict[name] = da.values.tolist()
    else:
        dims, coords = [], []
        [(dims.append(dim), coords.append(da.coords[dim].values))
         for dim in da.dims if dim != "time"]

        l = []
        for coords_prod in itertools.product(*coords):
            l.append(_time_dim_from_da(da, dims, coords_prod))
        savedict = dict(l)

    return savedict

def split_data_array_parallel(da):

    if not da.dims:
        savedict = {}
        name = da.attrs["Py Name"]
        savedict[name] = da.values.tolist()
    else:
        dims, coords = [], []
        [(dims.append(dim), coords.append(da.coords[dim].values))
         for dim in da.dims if dim != "time"]

        l = []
        for coords_prod in itertools.product(*coords):
            x = delayed(_time_dim_from_da)(da, dims, coords_prod)
            l.append(x)

        res = compute(*l)
        savedict = dict(res)

    return savedict

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
        List of variables to export from the netCDF.
    time_in_row: bool
        Whether time increases along row.
        Default is False.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with all colums specified in subset.

    """
    subset = _validate_ds_subset(ds, subset)

    savedict = {}

    if parallel:
        global delayed, compute
        from dask import delayed, compute

        for name in subset:
            savedict.update(split_data_array_parallel(ds[name]))
    else:
        for name in subset:
            savedict.update(split_data_array(ds[name]))

    return _dict_to_df(savedict, time_in_row)

def nc_to_df(ncfile: Union[str, Path],
             subset: Optional[list]=None,
             time_in_row: Optional[bool]=False,
             chunks: Optional[TypeChunks]=None,
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
    chunks: int, dict, str, None (optional)
        Check xarray.open_dataset for potential values for this argument.
        Frequent values are -1, to load each DataArray in a single chunk, or
        None, to not use dask array. Using chunks requires installing dask.
    parallel: bool
        When True, DataArrays are processed in parallel using dask. Dask is
        not included as a requiremetn for pysd, hence must be installed
        separately. Setting parallel=True is highly recommended for large
        multidimensional DataArrays.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with all colums specified in subset.

    """

    ds = load_nc_file(ncfile, chunks)

    return ds_to_df(ds, subset, time_in_row, parallel)

def nc_to_csv(ncfile: Union[str, Path],
              outfile: Optional[Union[str, Path]]="result.tab",
              subset: Optional[list]=None,
              time_in_row: Optional[bool]=False,
              chunks: Optional[TypeChunks]=None,
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
    chunks: int, dict, str, None (optional)
        Check xarray.open_dataset for potential values for this argument.
        Frequent values are -1, to load each DataArray in a single chunk, or
        None, to not use dask array. Using chunks requires installing dask.
    parallel: bool
        When True, DataArrays are processed in parallel using dask. Dask is
        not included as a requiremetn for pysd, hence must be installed
        separately. Setting parallel=True is highly recommended for large
        multidimensional DataArrays.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with all colums specified in subset.

    """
    df = nc_to_df(ncfile, subset=subset, time_in_row=time_in_row,
                  chunks=chunks, parallel=parallel
                  )

    df_to_csv(df, outfile)

    return df


