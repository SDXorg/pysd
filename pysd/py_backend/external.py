"""
These classes are a collection of the needed tools to read external data.
The External type objects created by these classes are initialized before
the Stateful objects by functions.Model.initialize.
"""

import re
import os
import warnings
import pandas as pd  # TODO move to openpyxl
import numpy as np
import xarray as xr
from openpyxl import load_workbook
from . import utils


class Excels():
    """
    Class to save the read Excel files and thus avoid double reading
    """
    _Excels, _Excels_opyxl = {}, {}

    @classmethod
    def read(cls, file_name, sheet_name):
        """
        Read the Excel file or return the previously read one
        """
        if file_name + sheet_name in cls._Excels:
            return cls._Excels[file_name + sheet_name]
        else:
            excel = np.array([
                pd.to_numeric(ex, errors='coerce')
                for ex in
                pd.read_excel(file_name, sheet_name, header=None).values
                ])
            cls._Excels[file_name + sheet_name] = excel
            return excel

    @classmethod
    def read_opyxl(cls, file_name):
        """
        Read the Excel file using OpenPyXL or return the previously read one
        """
        if file_name in cls._Excels_opyxl:
            return cls._Excels_opyxl[file_name]
        else:
            excel = load_workbook(file_name, read_only=True, data_only=True)
            cls._Excels_opyxl[file_name] = excel
            return excel

    @classmethod
    def clean(cls):
        """
        Clean the dictionary of read files
        """
        for file in cls._Excels_opyxl.values():
            # close files open directly with openpyxls
            file.close()
            # files open with pandas are automatically closed
        cls._Excels, cls._Excels_opyxl = {}, {}


class External(object):
    """
    Main class of external objects

    Attributes
    ----------
    py_name: str
        The python name of the object
    missing: str ("warning", "error", "ignore", "keep")
        What to do with missing values. If "warning" (default)
        shows a warning message and interpolates the values.
        If "raise" raises an error. If "ignore" interpolates
        the values without showing anything. If "keep" it will keep
        the missing values, this option may cause the integration to
        fail, but it may be used to check the quality of the data.

    file: str
        File name from which the data is read.
    sheet: str
        Sheet name from which the data is read.

    """
    missing = "warning"

    def __init__(self, py_name):
        self.py_name = py_name
        self.file = None
        self.sheet = None

    def __str__(self):
        return self.py_name

    def _get_data_from_file(self, rows, cols):
        """
        Function to read data from excel file using rows and columns

        Parameters
        ----------
        rows: list of len 2
            first row and last row+1 to be read, starting from 0
        cols:  list of len 2
            first col and last col+1 to be read, starting from 0

        Returns
        -------
        data: pandas.DataFrame, pandas.Series or float
            depending on the shape of the requested data

        """
        # TODO move to openpyxl to avoid pandas dependency in this file.
        ext = os.path.splitext(self.file)[1].lower()
        if ext in ['.xls', '.xlsx']:
            # read data
            data = Excels.read(
                self.file,
                self.sheet)[rows[0]:rows[1], cols[0]:cols[1]].copy()

            shape = data.shape
            # if it is a single row remove its dimension
            if shape[1] == 1:
                data = data[:, 0]
            if shape[0] == 1:
                data = data[0]
            return data

        raise NotImplementedError(self.py_name + "\n"
                                  + "The files with extension "
                                  + ext + " are not implemented")

    def _get_data_from_file_opyxl(self, cellname):
        """
        Function to read data from excel file using cell range name

        Parameters
        ----------
        cellname: str
            the cell range name

        Returns
        -------
        data: numpy.ndarray or float
            depending on the shape of the requested data

        """
        # read data
        excel = Excels.read_opyxl(self.file)
        try:
            # Get the local id of the sheet
            # needed for searching in locals names
            # need to lower the sheetnames as Vensim has no case sensitivity
            sheetId = [sheetname_wb.lower() for sheetname_wb
                       in excel.sheetnames].index(self.sheet.lower())
        except ValueError:
            # Error if it is not able to get the localSheetId
            raise ValueError(self.py_name + "\n"
                             + "The sheet doesn't exist...\n"
                             + self._file_sheet)

        try:
            # Search for local and global names
            cellrange = excel.defined_names.get(cellname, sheetId)\
                        or excel.defined_names.get(cellname)
            coordinates = cellrange.destinations
            for sheet, cells in coordinates:
                if sheet.lower() == self.sheet.lower():
                    values = excel[sheet][cells]
                    try:
                        return np.array(
                            [[i.value if not isinstance(i.value, str)
                              else np.nan for i in j] for j in values],
                            dtype=float)
                    except TypeError:
                        return float(values.value)

            raise AttributeError

        except (KeyError, AttributeError):
            # key error if the cellrange doesn't exist in the file or sheet
            raise AttributeError(
              self.py_name + "\n"
              + "The cell range name:\t {}\n".format(cellname)
              + "Doesn't exist in:\n" + self._file_sheet
              )

    def _get_series_data(self, series_across, series_row_or_col, cell, size):
        """
        Function thar reads series and data from excel file for
        DATA and LOOKUPS.

        Parameters
        ----------
        series_across: "row", "column" or "name"
            The way to read series file.
        series_row_or_col: int or str
            If series_across is "row" the row number where the series data is.
            If series_across is "column" the column name where
              the series data is.
            If series_across is "name" the cell range name where
              the series data is.
        cell:
            If series_across is not "name, the top left cell where
              the data table starts.
            Else the name of the cell range where the data is.
        size:
            The size of the 2nd dimension of the data.

        Returns
        -------
        series, data: ndarray (1D), ndarray(1D/2D)
            The values of the series and data.

        """
        if series_across == "row":
            # Horizontal data (dimension values in a row)

            # get the dimension values
            first_row, first_col = self._split_excel_cell(cell)
            series = self._get_data_from_file(
                rows=[int(series_row_or_col)-1, int(series_row_or_col)],
                cols=[first_col, None])

            # read data
            data = self._get_data_from_file(
                rows=[first_row, first_row + size],
                cols=[first_col, None]).transpose()

        elif series_across == "column":
            # Vertical data (dimension values in a column)

            # get the dimension values
            first_row, first_col = self._split_excel_cell(cell)
            series_col = self._col_to_num(series_row_or_col)
            series = self._get_data_from_file(
                rows=[first_row, None],
                cols=[series_col, series_col+1])

            # read data
            data = self._get_data_from_file(
                rows=[first_row, None],
                cols=[first_col, first_col + size])

        else:
            # get series data
            series = self._get_data_from_file_opyxl(series_row_or_col)
            if isinstance(series, float):
                series = np.array([[series]])

            series_shape = series.shape

            if series_shape[0] == 1:
                # horizontal definition of lookup/time dimension
                series = series[0]
                transpose = True

            elif series_shape[1] == 1:
                # vertical definition of lookup/time dimension
                series = series[:, 0]
                transpose = False

            else:
                # Error if the lookup/time dimension is 2D
                raise ValueError(
                  self.py_name + "\n"
                  + "Dimension given in:\n"
                  + self._file_sheet
                  + "\tDimentime_missingsion name:"
                  + "\t{}\n".format(series_row_or_col)
                  + " is a table and not a vector"
                  )

            # get data
            data = self._get_data_from_file_opyxl(cell)

            if isinstance(data, float):
                data = np.array([[data]])

            if transpose:
                # transpose for horizontal definition of dimension
                data = data.transpose()

            if data.shape[0] != len(series):
                raise ValueError(
                  self.py_name + "\n"
                  + "Dimension and data given in:\n"
                  + self._file_sheet
                  + "\tDimension name:\t{}\n".format(series_row_or_col)
                  + "\tData name:\t{}\n".format(cell)
                  + " don't have the same length in the 1st dimension"
                  )

            if data.shape[1] != size:
                # Given coordinates length is different than
                # the lentgh of 2nd dimension
                raise ValueError(
                  self.py_name + "\n"
                  + "Data given in:\n"
                  + self._file_sheet
                  + "\tData name:\t{}\n".format(cell)
                  + " has not the same size as the given coordinates"
                  )

            if data.shape[1] == 1:
                # remove second dimension of data if its shape is (N, 1)
                data = data[:, 0]

        return series, data

    def _resolve_file(self, root):
        """
        Resolve input file path. Joining the file with the root and
        checking if it exists.

        Parameters
        ----------
        root: str
            The root path to the model file.

        Returns
        -------
        None

        """
        if self.file[0] == '?':
            # TODO add an option to include indirect references
            raise ValueError(
                self.py_name + "\n"
                + f"Indirect reference to file: {self.file}")

        self.file = os.path.join(root, self.file)

        if not os.path.isfile(self.file):
            raise FileNotFoundError(
                self.py_name + "\n"
                + f"File '{self.file}' not found.")

    def _initialize_data(self, element_type):
        """
        Initialize one element of DATA or LOOKUPS

        Parameters
        ----------
        element_type: str
            "lookup" for LOOKUPS, "data" for data.

        Returns
        -------
        data: xarray.DataArray
            Dataarray with the time or interpolation dimension
            as first dimension.

        """
        self._resolve_file(root=self.root)
        series_across = self._series_selector(self.x_row_or_col, self.cell)
        size = utils.compute_shape(self.coords, reshape_len=1,
                                   py_name=self.py_name)[0]

        series, data = self._get_series_data(
            series_across=series_across,
            series_row_or_col=self.x_row_or_col,
            cell=self.cell, size=size
        )

        # remove nan or missing values from dimension
        if series_across != "name":
            # Remove last nans only if the method is to read by row or col
            i = 0
            try:
                while np.isnan(series[i-1]):
                    i -= 1
            except IndexError:
                # series has len 0
                raise ValueError(
                    self.py_name + "\n"
                    + "Dimension given in:\n"
                    + self._file_sheet
                    + "\t{}:\t{}\n".format(series_across, self.x_row_or_col)
                    + " has length 0"
                    )

            if i != 0:
                series = series[:i]
                data = data[:i]

        # warning/error if missing data in the series
        if any(np.isnan(series)) and self.missing != "keep":
            valid_values = ~np.isnan(series)
            series = series[valid_values]
            data = data[valid_values]
            if self.missing == "warning":
                warnings.warn(
                  self.py_name + "\n"
                  + "Dimension value missing or non-valid in:\n"
                  + self._file_sheet
                  + "\t{}:\t{}\n".format(series_across, self.x_row_or_col)
                  + " the corresponding data value(s) to the "
                  + "missing/non-valid value(s) will be ignored\n\n"
                  )
            elif self.missing == "raise":
                raise ValueError(
                  self.py_name + "\n"
                  + "Dimension value missing or non-valid in:\n"
                  + self._file_sheet
                  + "\t{}:\t{}\n".format(series_across, self.x_row_or_col)
                  )

        # reorder data with increasing series
        if not np.all(np.diff(series) > 0) and self.missing != "keep":
            order = np.argsort(series)
            series = series[order]
            data = data[order]
            # Check if the lookup/time dimension is well defined
            if np.any(np.diff(series) == 0):
                raise ValueError(self.py_name + "\n"
                                 + "Dimension given in:\n"
                                 + self._file_sheet
                                 + "\t{}:\t{}\n".format(
                                    series_across, self.x_row_or_col)
                                 + " has repeated values")


        # Check for missing values in data
        if np.any(np.isnan(data)) and self.missing != "keep":
            if series_across == "name":
                cell_type = "Cellrange"
            else:
                cell_type = "Reference cell"

            if self.missing == "warning":
                # Fill missing values with the chosen interpolation method
                # what Vensim does during running for DATA
                warnings.warn(
                    self.py_name + "\n"
                    + "Data value missing or non-valid in:\n"
                    + self._file_sheet
                    + "\t{}:\t{}\n".format(cell_type, self.cell)
                    + " the corresponding value will be filled "
                    + "with the interpolation method of the object.\n\n"
                    )
            elif self.missing == "raise":
                raise ValueError(
                    self.py_name + "\n"
                    + "Data value missing or non-valid in:\n"
                    + self._file_sheet
                    + "\t{}:\t{}\n".format(cell_type, self.cell)
                    )
            # fill values
            self._fill_missing(series, data)

        reshape_dims = tuple([len(series)] + utils.compute_shape(self.coords))

        if len(reshape_dims) > 1:
            data = self._reshape(data, reshape_dims)

        if element_type == "lookup":
            dim_name = "lookup_dim"
        else:
            dim_name = "time"

        data = xr.DataArray(
            data=data,
            coords={dim_name: series, **self.coords},
            dims=[dim_name] + list(self.coords)
        )

        return data

    def _fill_missing(self, series, data):
        """
        Fills missing values in excel read data. Mutates the values in data.

        Parameters
        ----------
        series:
          the time series without missing values
        data:
          the data with missing values

        Returns
        -------
        None
        """
        # if data is 2dims we need to interpolate
        datanan = np.isnan(data)
        if len(data.shape) == 1:
            data[datanan] = self._interpolate_missing(
                series[datanan],
                series[~datanan],
                data[~datanan])
        else:
            for i, nanlist in enumerate(list(datanan.transpose())):
                data[nanlist, i] = self._interpolate_missing(
                    series[nanlist],
                    series[~nanlist],
                    data[~nanlist][:, i])

    def _interpolate_missing(self, x, xr, yr):
        """
        Interpolates a list of missing values from _fill_missing

        Parameters
        ----------
        x:
          list of missing values interpolate
        xr:
          non-missing x values
        yr:
          non-missing y values

        Returns
        -------
        y:
          Result after interpolating x with self.interp method

        """
        y = np.empty_like(x, dtype=float)
        for i, value in enumerate(x):
            if self.interp == "raw":
                y[i] = np.nan
            elif value >= xr[-1]:
                y[i] = yr[-1]
            elif value <= xr[0]:
                y[i] = yr[0]
            elif self.interp == 'look forward':
                y[i] = yr[xr >= value][0]
            elif self.interp == 'hold backward':
                y[i] = yr[xr <= value][-1]
            else:
                y[i] = np.interp(value, xr, yr)
        return y

    @property
    def _file_sheet(self):
        """
        Returns file and sheet name in a string
        """
        return "\tFile name:\t{}\n".format(self.file)\
               + "\tSheet name:\t{}\n".format(self.sheet)

    @staticmethod
    def _col_to_num(col):
        """
        Transforms the column name to int

        Parameters
        ----------
        col: str
          Column name

        Returns
        -------
        int
          Column number
        """
        if len(col) == 1:
            return ord(col.upper()) - ord('A')
        elif len(col) == 2:
            left = ord(col[0].upper()) - ord('A') + 1
            right = ord(col[1].upper()) - ord('A')
            return left * (ord('Z')-ord('A')+1) + right
        else:
            left = ord(col[0].upper()) - ord('A') + 1
            center = ord(col[1].upper()) - ord('A') + 1
            right = ord(col[2].upper()) - ord('A')
            return left * ((ord('Z')-ord('A')+1)**2)\
                + center * (ord('Z')-ord('A')+1)\
                + right

    def _split_excel_cell(self, cell):
        """
        Splits a cell value given in a string.
        Returns None for non-valid cell formats.

        Parameters
        ----------
        cell: str
          Cell like string, such as "A1", "b16", "AC19"...
          If it is not a cell like string will return None.

        Returns
        -------
        row number, column number: int, int
          If the cell input is valid. Both numbers are given in Python
          enumeration, i.e., first row and first column are 0.

        """
        split = re.findall(r'\d+|\D+', cell)
        try:
            # check that we only have two values [column, row]
            assert len(split) == 2
            # check that the column name has no special characters
            assert not re.compile('[^a-zA-Z]+').search(split[0])
            # check that row number is not 0
            assert int(split[1]) != 0
            # the column name has as maximum 3 letters
            assert len(split[0]) <= 3
            return int(split[1])-1, self._col_to_num(split[0])
        except AssertionError:
            return

    @staticmethod
    def _reshape(data, dims):
        """
        Reshapes an pandas.DataFrame, pandas.Series, xarray.DataArray
        or np.ndarray in the given dimensions.

        Parameters
        ----------
        data: xarray.DataArray/numpy.ndarray
          Data to be reshaped
        dims: tuple
          The dimensions to reshape.

        Returns
        -------
        numpy.ndarray
          reshaped array
        """
        try:
            data = data.values
        except AttributeError:
            pass

        return data.reshape(dims)

    def _series_selector(self, x_row_or_col, cell):
        """
        Selects if a series data (DATA/LOOKUPS), should be read by columns,
        rows or cellrange name.
        Based on the input format of x_row_or_col and cell.
        The format of the 2 variables must be consistent.

        Parameters
        ----------
        x_row_or_col: str
          String of a number if series is given in a row, letter if series is
          given in a column or name if the series is given by cellrange name.
        cell: str
          Cell identificator, such as "A1", or name if the data is given
          by cellrange name.

        Returns
        -------
        series_across: str
          "row" if series is given in a row
          "column" if series is given in a column
          "name" if series and data are given by range name

        """
        try:
            # if x_row_or_col is numeric the series must be a row
            int(x_row_or_col)
            return "row"

        except ValueError:
            if self._split_excel_cell(cell):
                # if the cell can be splitted means that the format is
                # "A1" like then the series must be a column
                return "column"
            else:
                return "name"


class ExtData(External):
    """
    Class for Vensim GET XLS DATA/GET DIRECT DATA
    """

    def __init__(self, file_name, sheet, time_row_or_col, cell,
                 interp, coords, root, py_name):
        super().__init__(py_name)
        self.files = [file_name]
        self.sheets = [sheet]
        self.time_row_or_cols = [time_row_or_col]
        self.cells = [cell]
        self.coordss = [coords]
        self.root = root
        self.interp = interp

        # check if the interpolation method is valid
        if not interp:
            self.interp = "interpolate"

        if self.interp not in ["interpolate", "raw",
                               "look forward", "hold backward"]:
            raise ValueError(self.py_name + "\n"
                             + " The interpolation method (interp) must be "
                             + "'raw', 'interpolate', "
                             + "'look forward' or 'hold backward")

    def add(self, file_name, sheet, time_row_or_col, cell,
            interp, coords):
        """
        Add information to retrieve new dimension in an already declared object
        """
        self.files.append(file_name)
        self.sheets.append(sheet)
        self.time_row_or_cols.append(time_row_or_col)
        self.cells.append(cell)
        self.coordss.append(coords)

        if not interp:
            interp = "interpolate"
        if interp != self.interp:
            raise ValueError(self.py_name + "\n"
                             + "Error matching interpolation method with "
                             + "previously defined one")

        if list(coords) != list(self.coordss[0]):
            raise ValueError(self.py_name + "\n"
                             + "Error matching dimensions with previous data")

    def initialize(self):
        """
        Initialize all elements and create the self.data xarray.DataArray
        """
        data = []
        zipped = zip(self.files, self.sheets, self.time_row_or_cols,
                     self.cells, self.coordss)
        for (self.file, self.sheet, self.x_row_or_col,
             self.cell, self.coords) in zipped:
            data.append(self._initialize_data("data"))
        self.data = utils.xrmerge(data)

    def __call__(self, time):

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

        if self.coordss[0]:
            # Remove time coord from the DataArray
            return outdata.reset_coords('time', drop=True)
        else:
            # if data has no-coords return a float
            return float(outdata)


class ExtLookup(External):
    """
    Class for Vensim GET XLS LOOKUPS/GET DIRECT LOOKUPS
    """

    def __init__(self, file_name, sheet, x_row_or_col, cell,
                 coords, root, py_name):
        super().__init__(py_name)
        self.files = [file_name]
        self.sheets = [sheet]
        self.x_row_or_cols = [x_row_or_col]
        self.cells = [cell]
        self.root = root
        self.coordss = [coords]
        self.interp = "interpolate"

    def add(self, file_name, sheet, x_row_or_col, cell, coords):
        """
        Add information to retrieve new dimension in an already declared object
        """
        self.files.append(file_name)
        self.sheets.append(sheet)
        self.x_row_or_cols.append(x_row_or_col)
        self.cells.append(cell)
        self.coordss.append(coords)

        if list(coords) != list(self.coordss[0]):
            raise ValueError(self.py_name + "\n"
                             + "Error matching dimensions with previous data")

    def initialize(self):
        """
        Initialize all elements and create the self.data xarray.DataArray
        """
        data = []
        zipped = zip(self.files, self.sheets, self.x_row_or_cols,
                     self.cells, self.coordss)
        for (self.file, self.sheet, self.x_row_or_col,
             self.cell, self.coords) in zipped:
            data.append(self._initialize_data("lookup"))
        self.data = utils.xrmerge(data)

    def __call__(self, x):
        return self._call(self.data, x)

    def _call(self, data, x):
        if isinstance(x, xr.DataArray):
            if not x.dims:
                # shape 0 xarrays
                return self._call(data, float(x))
            if np.all(x > data['lookup_dim'].values[-1]):
                outdata, _ = xr.broadcast(data[-1], x)
                warnings.warn(
                  self.py_name + "\n"
                  + "extrapolating data above the maximum value of the series")
            elif np.all(x < data['lookup_dim'].values[0]):
                outdata, _ = xr.broadcast(data[0], x)
                warnings.warn(
                  self.py_name + "\n"
                  + "extrapolating data below the minimum value of the series")
            else:
                data, _ = xr.broadcast(data, x)
                outdata = data[0].copy()
                for a in utils.xrsplit(x):
                    outdata.loc[a.coords] = self._call(
                        data.loc[a.coords],
                        float(a))
            # the output will be always an xarray
            return outdata.reset_coords('lookup_dim', drop=True)

        else:
            if x in data['lookup_dim'].values:
                outdata = data.sel(lookup_dim=x)
            elif x > data['lookup_dim'].values[-1]:
                outdata = data[-1]
                warnings.warn(
                  self.py_name + "\n"
                  + "extrapolating data above the maximum value of the series")
            elif x < data['lookup_dim'].values[0]:
                outdata = data[0]
                warnings.warn(
                  self.py_name + "\n"
                  + "extrapolating data below the minimum value of the series")
            else:
                outdata = data.interp(lookup_dim=x)

            # the output could be a float or an xarray
            if self.coordss[0]:
                # Remove lookup dimension coord from the DataArray
                return outdata.reset_coords('lookup_dim', drop=True)
            else:
                # if lookup has no-coords return a float
                return float(outdata)


class ExtConstant(External):
    """
    Class for Vensim GET XLS CONSTANTS/GET DIRECT CONSTANTS
    """

    def __init__(self, file_name, sheet, cell, coords, root, py_name):
        super().__init__(py_name)
        self.files = [file_name]
        self.sheets = [sheet]
        self.transposes = [cell[-1] == '*']
        self.cells = [cell.strip('*')]
        self.root = root
        self.coordss = [coords]

    def add(self, file_name, sheet, cell, coords):
        """
        Add information to retrieve new dimension in an already declared object
        """
        self.files.append(file_name)
        self.sheets.append(sheet)
        self.transposes.append(cell[-1] == '*')
        self.cells.append(cell.strip('*'))
        self.coordss.append(coords)

        if list(coords) != list(self.coordss[0]):
            raise ValueError(self.py_name + "\n"
                             + "Error matching dimensions with previous data")

    def initialize(self):
        """
        Initialize all elements and create the self.data xarray.DataArray
        """
        data = []
        zipped = zip(self.files, self.sheets, self.transposes,
                     self.cells, self.coordss)
        for (self.file, self.sheet, self.transpose,
             self.cell, self.coords) in zipped:
            data.append(self._initialize())
        self.data = utils.xrmerge(data)

    def _initialize(self):
        """
        Initialize one element
        """
        self._resolve_file(root=self.root)
        split = self._split_excel_cell(self.cell)
        if split:
            data_across = "cell"
            cell = split
        else:
            data_across = "name"
            cell = self.cell

        shape = utils.compute_shape(self.coords, reshape_len=2,
                                    py_name=self.py_name)

        if self.transpose:
            shape.reverse()

        data = self._get_constant_data(data_across, cell, shape)

        if self.transpose:
            data = data.transpose()

        if np.any(np.isnan(data)):
            # nan values in data
            if data_across == "name":
                cell_type = "Cellrange"
            else:
                cell_type = "Reference cell"

            if self.missing == "warning":
                warnings.warn(
                    self.py_name + "\n"
                    + "Constant value missing or non-valid in:\n"
                    + self._file_sheet
                    + "\t{}:\t{}\n".format(cell_type, self.cell)
                    )
            elif self.missing == "raise":
                raise ValueError(
                    self.py_name + "\n"
                    + "Constant value missing or non-valid in:\n"
                    + self._file_sheet
                    + "\t{}:\t{}\n".format(cell_type, self.cell)
                    )

        # Create only an xarray if the data is not 0 dimensional
        if len(self.coords) > 0:
            reshape_dims = tuple(utils.compute_shape(self.coords))

            if len(reshape_dims) > 1:
                data = self._reshape(data, reshape_dims)

            data = xr.DataArray(
                data=data, coords=self.coords, dims=list(self.coords)
            )

        return data

    def _get_constant_data(self, data_across, cell, shape):
        """
        Function thar reads data from excel file for CONSTANT

        Parameters
        ----------
        data_across: "cell" or "name"
            The way to read data file.
        cell: int or str
            If data_across is "cell" the lefttop split cell value where
            the data is.
            If data_across is "name" the cell range name where the data is.
        shape:
            The shape of the data in 2D.

        Returns
        -------
        data: float/ndarray(1D/2D)
            The values of the data.

        """
        if data_across == "cell":
            # read data from topleft cell name using pandas
            start_row, start_col = cell

            return self._get_data_from_file(
                rows=[start_row, start_row + shape[0]],
                cols=[start_col, start_col + shape[1]])

        else:
            # read data from cell range name using OpenPyXL
            data = self._get_data_from_file_opyxl(cell)

            try:
                # Remove length=1 axis
                data_shape = data.shape
                if data_shape[1] == 1:
                    data = data[:, 0]
                if data_shape[0] == 1:
                    data = data[0]
            except AttributeError:
                # Data is a float, nothing to do
                pass

            # Check data dims
            try:
                if shape[0] == 1 and shape[1] != 1:
                    assert shape[1] == len(data)
                elif shape[0] != 1 and shape[1] == 1:
                    assert shape[0] == len(data)
                elif shape[0] == 1 and shape[1] == 1:
                    assert isinstance(data, float)
                else:
                    assert tuple(shape) == data.shape
            except AssertionError:
                raise ValueError(self.py_name + "\n"
                                 + "Data given in:\n"
                                 + self._file_sheet
                                 + "\tData name:\t{}\n".format(cell)
                                 + " has not the same shape as the"
                                 + " given coordinates")

            return data

    def __call__(self):
        return self.data


class ExtSubscript(External):
    """
    Class for Vensim GET XLS SUBSCRIPT/GET DIRECT SUBSCRIPT
    """

    def __init__(self, file_name, sheet, firstcell, lastcell, prefix, root):
        super().__init__("Hardcoded external subscript")
        self.file = file_name
        self.sheet = sheet
        self._resolve_file(root=root)

        row_first, col_first = self._split_excel_cell(firstcell)
        row_last, col_last = self._split_excel_cell(lastcell)
        data = pd.read_excel(
            self.file, sheet,
            skiprows=row_first-1,
            nrows=row_last-row_first+1,
            usecols=np.arange(col_first, col_last+1)
            )

        self.subscript = [prefix + str(d) for d in data.values.flatten()]
