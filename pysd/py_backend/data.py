import warnings
import re

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
        out = cls.read_line(file_name, encoding)
        if out is None:
            raise ValueError(
                f"\nNot able to read '{file_name}'. "
                + "Only '.csv', '.tab' files are accepted.")

        transpose = False

        try:
            [float(col) for col in out]
            out = cls.read_row(file_name, encoding)
            transpose = True
            [float(col) for col in out]
        except ValueError:
            return out, transpose
        else:
            raise ValueError(
                f"Invalid file format '{file_name}'... varible names "
                "should appear in the first row or in the first column...")

    @classmethod
    def read_line(cls, file_name, encoding=None):
        """
        Read the firts row and return a set of it.
        """
        # TODO add decode method if encoding is pased

        with open(file_name, 'r') as file:
            header = file.readline().rstrip()

        if file_name.lower().endswith(".tab"):
            return set(header.split("\t")[1:])
        elif file_name.lower().endswith(".csv"):
            # TODO improve like previous to go faster
            # splitting csv is not easy as , are in subscripts
            return set(pd.read_csv(file_name,
                                   nrows=0,
                                   encoding=encoding,
                                   dtype=str,
                                   header=0).iloc[:, 1:])
        else:
            return None

    @classmethod
    def read_row(cls, file_name, encoding=None):
        """
        Read the firts column and return a set of it.
        """
        if file_name.lower().endswith(".tab"):
            return set(pd.read_table(file_name,
                                     usecols=[0],
                                     encoding=encoding,
                                     dtype=str).iloc[:, 0].to_list())
        elif file_name.lower().endswith(".csv"):
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
            else:
                # the variable may have " on its name in the tab or csv file
                vars_extended.append('"' + var)
                vars_extended.append('"' + var + '"')

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

    def __call__(self, time):
        try:
            if time in self.data['time'].values:
                outdata = self.data.sel(time=time)
            elif self.interp == "raw":
                return np.nan
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
            elif self.interp == 'look forward':
                outdata = self.data.sel(time=time, method="backfill")
            elif self.interp == 'hold backward':
                outdata = self.data.sel(time=time, method="pad")

            if self.is_float:
                # if data has no-coords return a float
                return float(outdata)
            else:
                # Remove time coord from the DataArray
                return outdata.reset_coords('time', drop=True)
        except Exception as err:
            if self.data is None:
                raise ValueError(
                    self.py_name + "\n"
                    "Trying to interpolate data variable before loading"
                    " the data...")
            else:
                # raise any other possible error
                raise err


class TabData(Data):
    """
    Data from tabular file tab/cls, it could be from Vensim output.
    """
    def __init__(self, real_name, py_name, coords, interp="interpolate"):
        self.real_name = real_name
        self.py_name = py_name
        self.coords = coords
        self.interp = interp
        self.is_float = not bool(coords)
        self.data = None

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
        if isinstance(file_names, str):
            file_names = [file_names]

        for file_name in file_names:
            self.data = self._load_data(file_name)
            if self.data is not None:
                break

        if self.data is None:
            raise ValueError(
                f"_data_{self.py_name}\n"
                f"Data for {self.real_name} not found in "
                f"{', '.join(file_names)}")

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
            values = load_outputs(file_name, transpose, columns=columns)
            return xr.DataArray(
                values.iloc[:, 0].values,
                {'time': values.index.values},
                ['time'])

        # subscripted data
        dims = list(self.coords)

        values = load_outputs(file_name, transpose, columns=columns)

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
