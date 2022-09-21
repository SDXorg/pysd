"""
Tools for importing and converting netCDF files generated from simulations run
using PySD.
"""

from csv import QUOTE_NONE
import xarray as xr
import pandas as pd
from pathlib import Path
import warnings
from typing import Union, Optional
import itertools

class NCFile():
    """
    Helper class to extract data from netCDF files.

    Parameters
    ----------
    ncfile: str or pathlib.Path
        Path to the netCDF file to process.
    parallel: bool
        When True, the Dataset is opened using chunks=-1 (see xarray
        documentation for details) and DataArrays are processed in parallel
        using dask delayed. Dask is not included as a requirement for pysd,
        hence it must be installed separately. Setting parallel=True is
        highly recommended when the Dataset contains large multidimensional
        DataArrays.

    """

    valid_export_file_types = [".csv", ".tab"]

    def __init__(self, filename: Union[str, Path],
                 parallel: bool=False) -> None:

        self.ncfile = NCFile._validate_nc_path(filename)
        self.parallel = parallel
        self.ds = self.open_nc()

    def to_csv(self,
               outfile: Optional[Union[str, Path]]="result.tab",
               subset: Optional[list]=None,
               time_in_row: Optional[bool]=False,
               ) -> pd.DataFrame:
        """
        Convert netCDF file contents into comma or tab separated file.

        Parameters
        ----------
        outfile: str
            Path to the output file.
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
        df = self.to_df(subset=subset, time_in_row=time_in_row)

        NCFile.df_to_csv(df, outfile)

        return df

    def to_df(self,
              subset: Optional[list]=None,
              time_in_row: Optional[bool]=False,
              ) -> pd.DataFrame:
        """
        Wrapper to ds_to_df static method. Convert xarray.Dataset into a
        pandas DataFrame.

        Parameters
        ----------
        subset: list
            List of variables to export from the Dataset.
        time_in_row: bool
            Whether time increases along row. Default is False.

        Returns
        -------
        df: pandas.DataFrame
            Dataframe with all colums specified in subset.
        """

        return NCFile.ds_to_df(self.ds, subset, time_in_row, self.parallel)

    def open_nc(self) -> xr.Dataset:
        """
        Loads netCDF file into xarray Dataset. It's basically a wrapper to
        xr.open_dataset to simplify the interface for pysd use case (loading
        simulation results).

        Returns
        -------
        xarray.Dataset

        """

        if self.parallel:
            return xr.open_dataset(self.ncfile, engine="netcdf4", chunks=-1)

        return xr.open_dataset(self.ncfile, engine="netcdf4")

    @staticmethod
    def _validate_nc_path(nc_path: Union[str, Path]) -> Path:
        """
        Checks validity of the nc_path passed by the user. We run these
        checks because xarray Exceptions are not very explicit.
        """

        if not isinstance(nc_path, (str, Path)):
            raise TypeError(f"Invalid file path type: {type(nc_path)}.\n"
                            "Please provide string or pathlib Path")

        if isinstance(nc_path, str):
            nc_path = Path(nc_path)

        if not nc_path.is_file():
            raise FileNotFoundError(f"{nc_path} could not be found.")

        if not nc_path.suffix == ".nc":
            raise TypeError("Input file must have nc extension.")

        return nc_path

    @staticmethod
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

    @staticmethod
    def _index_da_by_coord_labels(da: xr.DataArray, dims: list, coords: tuple
                                  ) -> tuple:
        """
        Generates variable names, combining the actual name of the variable
        with the coordinate names between brackets and separated by commas,
        and indexes the DataArray by the coordinate names specified in
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

    @staticmethod
    def _get_da_dims_coords(da: xr.DataArray, exclude_dim: str) -> tuple:
        """
        Returns the dimension names and coordinate labels in two separate lists.
        If a dimension name is in the exclude_dims list, the returned dims and
        coords will not include it.

        Parameters
        ----------
        exclude_dim: str
            Names of dimension to exclude.

        Returns
        -------
        dims: list
            List containing the names of the DataArray dimensions.
        coords: list
            List of lists of coordinates for each dimension.

        """
        dims, coords = [], []

        for dim in da.dims:
            if dim != exclude_dim:
                dims.append(dim)
                coords.append(da.coords[dim].values)

        return dims, coords

    @staticmethod
    def da_to_dict(da: xr.DataArray, index_dim: str) -> dict:
        """
        Splits a DataArray into a dictionary, with keys equal to the name
        of the variable plus all combinations of the cartesian product of
        coordinates within brackets, and values equal to the data
        corresponding to those coordinates along the index_dim dimension.

        Parameters
        ----------
        index_dim: str
            The coordinates of this dimension will not be fixed during
            indexing of the DataArray (i.e. the indexed data will be an
            a scalar or an array along this dimension).

        """
        dims, coords = NCFile._get_da_dims_coords(da, index_dim)

        l = []
        for coords_prod in itertools.product(*coords):
            l.append(NCFile._index_da_by_coord_labels(da, dims, coords_prod))

        return dict(l)

    @staticmethod
    def da_to_dict_delayed(da: xr.DataArray, index_dim: str) -> dict:

        """
        Same as da_to_dict, but using dask delayed and compute.
        This function runs much faster when da is a dask array (chunked).

        To use it on its own, you must first make the following imports:

        from dask import delayed, compute
        from dask.diagnostics import ProgressBar

        Parameters
        ----------
        index_dim: str
            The coordinates of this dimension will not be fixed during
            indexing (the indexed data will be an array along this dimension).

        """
        dims, coords = NCFile._get_da_dims_coords(da, index_dim)

        l = []
        for coords_prod in itertools.product(*coords):
            x = delayed(
                NCFile._index_da_by_coord_labels)(da, dims, coords_prod)
            l.append(x)

        with ProgressBar():
            res = compute(*l)

        return dict(res)

    @staticmethod
    def dict_to_df(d: dict, time_in_row: bool) -> pd.DataFrame:
        """
        Convert a dict to a pandas Dataframe.

        Parameters
        ----------
        d: dict
            Dictionary to convert to pandas DataFrame.
        time_in_row: bool
            Whether time increases along a column or a row.

        """
        # process time_in_row argument
        if not isinstance(time_in_row, bool):
            raise ValueError("time_in_row argument takes boolen values.")

        df = pd.DataFrame(d)

        if time_in_row:
            df = df.transpose()

        return df

    @staticmethod
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
        if out_fmt not in NCFile.valid_export_file_types:
            raise TypeError("Invalid output file format {out_fmt}\n"
                            "Supported formats are csv and tab.")

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

    @staticmethod
    def ds_to_df(ds: xr.Dataset,
                 subset: Optional[list]=None,
                 time_in_row: Optional[bool]=False,
                 parallel: Optional[bool]=False,
                 index_dim: Optional[str]="time"
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
        index_dim: str
            Name of dimensions to use as index of the resulting DataFrame
            (usually "time").

        Returns
        -------
        df: pandas.DataFrame
            Dataframe with all colums specified in subset.

        """
        subset = NCFile._validate_ds_subset(ds, subset)

        savedict = {}

        if parallel:
            global delayed, compute, ProgressBar
            from dask import delayed, compute
            from dask.diagnostics import ProgressBar

            for name in subset:
                print(f"\nProcessing variable {name}")
                da = ds[name]

                if da.dims:
                    savedict.update(
                        NCFile.da_to_dict_delayed(da, index_dim))
                else:
                    savedict.update({name: da.values.tolist()})
        else:
            for name in subset:
                print(f"\nProcessing variable {name}")
                da = ds[name]

                if da.dims:
                    savedict.update(NCFile.da_to_dict(da, index_dim))
                else:
                    savedict.update({name: da.values.tolist()})

        return NCFile.dict_to_df(savedict, time_in_row)



