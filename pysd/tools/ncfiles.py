"""
Tools for importing and converting netCDF files generated from simulations run
using PySD.
"""

from csv import QUOTE_NONE
import xarray as xr
import pandas as pd
from pysd.py_backend.utils import xrsplit
from pathlib import Path
import warnings
from typing import Union, Optional

ChunkTypes = Union[str, int, dict]

def load_nc_file(ncfile: Union[str, Path],
                 chunks: Optional[ChunkTypes]={}) -> xr.Dataset:
    """
    Loads netCDF file into xarray Dataset.

    Parameters
    ----------
    ncfile: str or pathlib.Path
        Path to the netCDF file to process.
    chunks: int, dict, auto or None (optional)
        chunks argument from the xarray.load_dataset method.
        Used to load the dataset into dask arrays. The default
        value of this argument will load the dataset with dask
        using engine preferred chunks if exposed by the backend,
        otherwise with a single chunk for all arrays. See dask
        chunking for more details.

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

    # open netCDF file
    return xr.open_dataset(ncfile, engine="netcdf4", chunks=chunks)

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

def _ds_var_to_dict_item(savedict: dict, ds: xr.Dataset, name: str) -> None:
    """
    Convert xarray Dataset variable to dict items for each combination of
    dimensions.

    Parameters
    ----------
    savedict: dict
        Dictionary to which keys and values are added.
    ds: xr.Dataset
        Dataset containing the variable to convert into a dictionary.
    name: str
        Name of the variable to extract from the Dataset.

    Returns
    -------
    None

    """
    da = ds[name]
    dims = da.dims

    if not dims:
        savedict[name] = da.values.tolist()
    else:
        for elem in xrsplit(da):
            coords = {dim: str(elem.coords[dim].values)
                      for dim in dims if dim != "time"
                      }
            subs = "[" + ",".join(coords.values()) + "]"
            savedict[name + subs] = da.loc[coords].values

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

    df.to_csv(outfile, sep, index_label="Time", quoting=QUOTE_NONE,
              escapechar="\\")

def ds_to_df(ds: xr.Dataset,
             subset: Optional[list]=None,
             time_in_row: Optional[bool]=False
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
    for name in subset:
        _ds_var_to_dict_item(savedict, ds, name)

    return _dict_to_df(savedict, time_in_row)

def nc_to_df(ncfile: Union[str, Path],
             subset: Optional[list]=None,
             time_in_row: Optional[bool]=False,
             chunks: Optional[ChunkTypes]={}
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
    chunks: int, dict, auto or None (optional)
        chunks argument from the xarray.load_dataset method.
        Used to load the dataset into dask arrays. The default
        value of this argument will load the dataset with dask
        using engine preferred chunks if exposed by the backend,
        otherwise with a single chunk for all arrays. See dask
        chunking for more details.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with all colums specified in subset.

    """
    ds = load_nc_file(ncfile, chunks=chunks)

    return ds_to_df(ds, subset, time_in_row)

def nc_to_csv(ncfile: Union[str, Path],
              outfile: Optional[Union[str, Path]]="result.tab",
              subset: Optional[list]=None,
              time_in_row: Optional[bool]=False,
              chunks: Optional[ChunkTypes]={}
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
    chunks: int, dict, auto or None (optional)
        chunks argument from the xarray.load_dataset method.
        Used to load the dataset into dask arrays. The default
        value of this argument will load the dataset with dask
        using engine preferred chunks if exposed by the backend,
        otherwise with a single chunk for all arrays. See dask
        chunking for more details.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with all colums specified in subset.

    """
    df = nc_to_df(ncfile, subset, time_in_row, chunks)

    df_to_csv(df, outfile)

    return df


