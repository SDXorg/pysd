import warnings
import re
import random
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

from .utils import load_outputs


class Columns():
    """
    Class to save the read columns in data files
    """
    _files = {}

    @classmethod
    def read(cls, file_name, encoding=None):
        """
        Read the columns from the data file or return the previously read ones
        """
        if isinstance(file_name, str):
            file_name = Path(file_name)
        if file_name in cls._files:
            return cls._files[file_name]
        else:
            columns = cls.read_file(file_name, encoding)
            cls._files[file_name] = columns
            return columns

    @classmethod
    def read_file(cls, file_name, encoding=None):
        """
        Get the columns from an output csv or tab file.

        Parameters
        ----------
        file_name: str
            Output file to read. Must be csv or tab.

        encoding: str or None (optional)
            Encoding type to read output file. Needed if the file has special
            characters. Default is None.

        Returns
        -------
        out, transposed: set, bool
            The set of the columns in the output file and a boolean flag to
            indicate if the output file is transposed.

        """
        # in the most cases variables will be split per columns, then
        # read the first row to have all the column names
        out = cls.read_line(file_name, encoding)
        if out is None:
            raise ValueError(
                f"\nNot able to read '{str(file_name)}'. "
                + "Only '.csv', '.tab' files are accepted.")

        transpose = False

        try:
            # if we fail converting columns to float then they are
            # not numeric values, so current direction is okay
            [float(col) for col in random.sample(out, min(3, len(out)))]
            # we did not fail, read the first column to see if variables
            # are split per rows
            out = cls.read_col(file_name, encoding)
            transpose = True
            # if we still are able to transform values to float the
            # file is not valid
            [float(col) for col in random.sample(out, min(3, len(out)))]
        except ValueError:
            return out, transpose
        else:
            raise ValueError(
                f"Invalid file format '{str(file_name)}'... varible names "
                "should appear in the first row or in the first column...")

    @classmethod
    def read_line(cls, file_name, encoding=None):
        """
        Read the firts row and return a set of it.
        """
        if file_name.suffix.lower() == ".tab":
            return set(pd.read_table(file_name,
                                     nrows=0,
                                     encoding=encoding,
                                     dtype=str,
                                     header=0).iloc[:, 1:])
        elif file_name.suffix.lower() == ".csv":
            return set(pd.read_csv(file_name,
                                   nrows=0,
                                   encoding=encoding,
                                   dtype=str,
                                   header=0).iloc[:, 1:])
        else:
            return None

    @classmethod
    def read_col(cls, file_name, encoding=None):
        """
        Read the firts column and return a set of it.
        """
        if file_name.suffix.lower() == ".tab":
            return set(pd.read_table(file_name,
                                     usecols=[0],
                                     encoding=encoding,
                                     dtype=str).iloc[:, 0].to_list())
        elif file_name.suffix.lower() == ".csv":
            return set(pd.read_csv(file_name,
                                   usecols=[0],
                                   encoding=encoding,
                                   dtype=str).iloc[:, 0].to_list())

    @classmethod
    def get_columns(cls, file_name, vars=None, encoding=None):
        """
        Get columns names from a tab or csv file and return those that
        match with the given ones.

        Parameters
        ----------
        file_name: str
            Output file to read. Must be csv or tab.

        vars: list
            List of var names to find in the file.

        encoding: str or None (optional)
            Encoding type to read output file. Needed if the file has special
            characters. Default is None.

        Return
        ------
        columns, transpose: set, bool
            The set of columns as they are named in the input file and a
            boolean flag to indicate if the input file is transposed or
            not.

        """
        if vars is None:
            # Not var specified, return all available variables
            return cls.read(file_name, encoding)

        columns, transpose = cls.read(file_name, encoding)

        vars_extended = []
        for var in vars:
            vars_extended.append(var)
            if var.startswith('"') and var.endswith('"'):
                # the variables in "" are reded without " by pandas
                vars_extended.append(var[1:-1])

        outs = set()
        for var in columns:
            if var in vars_extended:
                # var is in vars_extended (no subscripts)
                outs.add(var)
                vars_extended.remove(var)
            else:
                for var1 in vars_extended:
                    if var.startswith(var1 + "["):
                        # var is subscripted
                        outs.add(var)

        return outs, transpose

    @classmethod
    def clean(cls):
        """
        Clean the dictionary of read files
        """
        cls._files = {}


class Data(object):
    # TODO add __init__ and use this class for used input pandas.Series
    # as Data
    # def __init__(self, data, coords, interp="interpolate"):

    def set_values(self, values):
        """Set new values from user input"""
        self.data = xr.DataArray(
            np.nan, self.final_coords, list(self.final_coords))

        if isinstance(values, pd.Series):
            index = list(values.index)
            index.sort()
            self.data = self.data.expand_dims(
                {'time': index}, axis=0).copy()

            for index, value in values.items():
                if isinstance(values.values[0], xr.DataArray):
                    self.data.loc[index].loc[value.coords] =\
                        value
                else:
                    self.data.loc[index] = value
        else:
            if isinstance(values, xr.DataArray):
                self.data.loc[values.coords] = values.values
            else:
                if self.final_coords:
                    self.data.loc[:] = values
                else:
                    self.data = values

    def __call__(self, time):
        try:
            if time in self.data['time'].values:
                outdata = self.data.sel(time=time)
            elif self.interp == "raw":
                return self.nan
            elif time > self.data['time'].values[-1]:
                warnings.warn(
                    self.py_name + "\n"
                    + "extrapolating data above the maximum value of the time")
                outdata = self.data[-1]
            elif time < self.data['time'].values[0]:
                warnings.warn(
                    self.py_name + "\n"
                    + "extrapolating data below the minimum value of the time")
                outdata = self.data[0]
            elif self.interp == "interpolate":
                outdata = self.data.interp(time=time)
            elif self.interp == 'look_forward':
                outdata = self.data.sel(time=time, method="backfill")
            elif self.interp == 'hold_backward':
                outdata = self.data.sel(time=time, method="pad")

            if self.is_float:
                # if data has no-coords return a float
                return float(outdata)
            else:
                # Remove time coord from the DataArray
                return outdata.reset_coords('time', drop=True)
        except (TypeError, KeyError):
            if self.data is None:
                raise ValueError(
                    self.py_name + "\n"
                    "Trying to interpolate data variable before loading"
                    " the data...")

            # this except catch the errors when a data has been
            # changed to a constant value by the user
            return self.data
        except Exception as err:
            raise err


class TabData(Data):
    """
    Data from tabular file tab/csv, it could be from Vensim output.
    """
    def __init__(self, real_name, py_name, coords, interp="interpolate"):
        self.real_name = real_name
        self.py_name = py_name
        self.coords = coords
        self.final_coords = coords
        self.interp = interp.replace(" ", "_") if interp else None
        self.is_float = not bool(coords)
        self.data = None

        if self.interp not in ["interpolate", "raw",
                               "look_forward", "hold_backward"]:
            raise ValueError(self.py_name + "\n"
                             + "The interpolation method (interp) must be "
                             + "'raw', 'interpolate', "
                             + "'look_forward' or 'hold_backward'")

    def load_data(self, file_names):
        """
        Load data values from files.

        Parameters
        ----------
        file_names: list or str
            Name of the files to search the variable in.

        Returns
        -------
        out: xarray.DataArray
            Resulting data array with the time in the first dimension.

        """
        if isinstance(file_names, (str, Path)):
            file_names = [file_names]

        for file_name in file_names:
            self.data = self._load_data(file_name)
            if self.data is not None:
                break

        if self.data is None:
            raise ValueError(
                f"_data_{self.py_name}\n"
                f"Data for {self.real_name} not found in "
                f"{', '.join([str(file_name) for file_name in file_names])}")

    def _load_data(self, file_name):
        """
        Load data values from output

        Parameters
        ----------
        file_name: str
            Name of the file to search the variable in.

        Returns
        -------
        out: xarray.DataArray or None
            Resulting data array with the time in the first dimension.

        """
        # TODO inlcude missing values managment as External objects
        # get columns to load variable
        columns, transpose = Columns.get_columns(
            file_name, vars=[self.real_name, self.py_name])

        if not columns:
            # the variable is not in the passed file
            return None

        if not self.coords:
            # 0 dimensional data
            self.nan = np.nan
            values = load_outputs(file_name, transpose, columns=columns)
            return xr.DataArray(
                values.iloc[:, 0].values,
                {'time': values.index.values},
                ['time'])

        # subscripted data
        dims = list(self.coords)

        values = load_outputs(file_name, transpose, columns=columns)

        self.nan = xr.DataArray(np.nan, self.coords, dims)
        out = xr.DataArray(
            np.nan,
            {'time': values.index.values, **self.coords},
            ['time'] + dims)

        for column in values.columns:
            coords = {
                dim: [coord]
                for (dim, coord)
                in zip(dims, re.split(r'\[|\]|\s*,\s*', column)[1:-1])
            }
            out.loc[coords] = np.expand_dims(
                values[column].values,
                axis=tuple(range(1, len(coords)+1))
            )
        return out
