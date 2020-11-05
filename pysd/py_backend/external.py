import re
import os
import warnings
import pandas as pd
import numpy as np
import xarray as xr
from . import utils

try:
    # Optional dependency as openpyxl requires python 3.6 or greater
    # Used for reading data giving cell range names
    from openpyxl import load_workbook
except ModuleNotFoundError:
    warnings.warn("Not able to import openpyxl."
                  + "You will not be able to read Excel data"
                  + " by cell range names.")

class Excels():
    """
    Class to save the read Excel files and thus avoid double reading
    """
    _instance = None
    _Excels, _Excels_opyxl = {}, {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Excels, cls).__new__(cls)
        return cls._instance

    def read(cls, file_name):
        """
        Read the Excel file or return the previously read one
        """
        if file_name in cls._Excels:
            return cls._Excels[file_name]
        else:
            excel = pd.ExcelFile(file_name)
            cls._Excels[file_name] = excel
            return excel

    def read_opyxl(cls, file_name):
        """
        Read the Excel file using OpenPyXL or return the previously read one
        """
        if file_name in cls._Excels:
            return cls._Excels_opyxl[file_name]
        else:
            excel = load_workbook(file_name, read_only=True)
            cls._Excels_opyxl[file_name] = excel
            return excel

    def clean(cls):
        """
        Clean the dictionary of read files
        """
        cls._Excels, cls._Excels_opyxl = {}, {}

 
class External():
    """
    Main class of external objects
    """

    def _get_data_from_file(self, rows, cols):
        """
        Function to read data from excel file using rows and columns
        
        Parameters
        ----------
        rows: None, int, list or array
            if None, then will continue reading until no more data is available in the new row
            if int, then will start reading the table for the given row number
            if list, then will read between first and second row of the lis
            if array, then will read the given row numbers
        cols:  None, str, int, list or array
            if None, then will continue reading until no more data is available in the new column
            if str, then will read only the given column by its name
            if list, then will read between the given columns
            if array, then will read the given columns names/numbers

        Returns
        -------
        data: pandas.DataFrame, pandas.Series or float
            depending on the shape of the requested data
        """
        # TODO move to openpyxl to avoid pandas dependency in this file.
        # This must by a future implementation as openpyxl requires python 3.6 or greater
        ext = os.path.splitext(self.file)[1].lower()
        if ext in ['.xls', '.xlsx']:
            # rows not specified
            if rows is None:
                skip = nrows = None
            # rows is an int of the first value
            elif isinstance(rows, int):
                skip = rows
                nrows = None
            # rows my be a list of first and last value or a numpy.ndarray of valid values
            else:
                skip = rows[0] - 1
                nrows = rows[-1] - skip if rows[-1] is not None else None
            # cols is a list of first and last value
            if isinstance(cols, list):
                cols = [self.num_to_col(c) if isinstance(c, int) else c for c in cols]
                usecols = cols[0] + ":" + cols[1]
            # cols is a int/col_name or a np.ndarray of valid values
            else:
                usecols = cols

            # read data
            excel = Excels().read(self.file)
            
            data = excel.parse(sheet_name=self.tab, header=None, skiprows=skip,
                               nrows=nrows, usecols=usecols)

            # avoid the rows not passed in rows numpy.ndarray
            if isinstance(rows, np.ndarray):
                data = data.iloc[(rows - rows[0])]

            # if it is a single row remove its dimension
            if isinstance(rows, int) or\
               (isinstance(rows, list) and rows[0] == rows[1]):
                data = data.iloc[0]

            # if it is a single col remove its dimension
            if isinstance(cols, str) or\
               (isinstance(cols, list) and cols[0].lower() == cols[1].lower()):
                try:
                    # if there are multile rows
                    data = data.iloc[:, 0]
                except:
                    # if there are no rows
                    data = data.iloc[0]

            return data

        raise NotImplementedError("The files with extension "
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
        excel = Excels().read_opyxl(self.file)
        coordinates = excel.defined_names[cellname].destinations
        for sheet, cells in coordinates:
            if sheet == self.tab:
                values = excel[sheet][cells]
                try:
                    return np.array([[i.value for i in j] for j in values], dtype=float)
                except TypeError:
                    return values.value
            else:
                raise ValueError("The cell range name:\t {}".format(cellname)
                                 + "\n\t Doesn't exist in:"
                                 + "File name:\t{}\n".format(self.file)
                                 + "Sheet name:\t{}\n".format(self.tab))

    def _get_series_data(self, series_across, series_row_or_col, cell, size):
        """
        Function thar reads series and data from excel file for DATA and LOOKUPS
        
        Parameters
        ----------
        series_across: "row", "column" or "name"
            The way to read series file.
        series_row_or_col: int or str
            If series_across is "row" the row number where the series data is.
            If series_across is "column" the column name where the series data is.
            If series_across is "name" the cell range name where the series data is.
        cell:
            If series_across is not "name, the top left cell where the data table starts.
            Else the name of the cell range where the data is.
        size:
            The size of the 2nd dimension of the data.
        
        Returns
        -------
        series_data, data: ndarray (1D), ndarray(1D/2D)
            The values of the series and data.

        """
        if series_across == "row":
            # Horizontal data (dimension values in a row)

            # get the dimension values
            series_data = self._get_data_from_file(rows=int(series_row_or_col)-1, 
                                                   cols=None)

            first_data_row, first_col = self._split_excel_cell(cell)
    
            first_col = first_col.upper()
            first_col_float = self.col_to_num(first_col)
    
            # get a vector of the series index
            original_index = np.array(series_data.index)[first_col_float:]
            series_data = series_data[first_col_float:]
    
            # remove nan or missing values from dimension
            series_data = pd.to_numeric(series_data, errors='coerce')
            valid_values = ~np.isnan(series_data)
            original_index = original_index[valid_values]
            series_data = series_data[valid_values]

            # check if the series has no len 0
            if len(series_data) == 0:
                raise ValueError("Dimension given in:\n"
                                 + "File name:\t{}\n".format(self.file)
                                 + "Sheet name:\t{}\n".format(self.tab)
                                 + "Row number:\t{}\n".format(series_row_or_col)
                                 + " has length 0")
            
            last_col = self.num_to_col(original_index[-1])
            last_data_row = first_data_row + size - 1

            # Warning if there is missing value in the dimension
            if (np.diff(original_index) != 1).any():
                missing_index = np.arange(original_index[0], original_index[-1]+1)
                missing_index = np.setdiff1d(missing_index, original_index)
                cols = self.num_to_col(missing_index)
                cells = [col + series_row_or_col for col in cols]
                warnings.warn("\n\tDimension value missing or non-valid in:\n"
                              + "File name:\t{}\n".format(self.file)
                              + "Sheet name:\t{}\n".format(self.tab)
                              + "\n\tCell(s):\t{}\n".format(cells)
                              + "\tthe corresponding column(s) to the "
                              + "missing/non-valid value(s) will be ignored\n\n")

            # read data
            data = self._get_data_from_file(
                rows=[first_data_row, last_data_row],
                cols=original_index)
            data = data.transpose()

        elif series_across == "column":
            # Vertical data (dimension values in a column)

            # get the dimension values
            first_row, first_col = self._split_excel_cell(cell)
            series_data = self._get_data_from_file(rows=[first_row, None],
                                                   cols=series_row_or_col)

            # get a vector of the series index
            original_index = np.array(series_data.index)
            series_data = pd.to_numeric(series_data, errors='coerce')

            # remove nan or missing values from dimension
            valid_values = ~np.isnan(series_data)
            original_index = original_index[valid_values]
            series_data = series_data[valid_values]
 
            # check if the series has no len 0
            if len(series_data) == 0:
                raise ValueError("Dimension given in:\n"
                                 + "File name:\t{}\n".format(self.file)
                                 + "Sheet name:\t{}\n".format(self.tab)
                                 + "Column:\t{}\n".format(series_row_or_col)
                                 + " has length 0")

            last_row = original_index[-1] + first_row
            last_col = self.num_to_col(self.col_to_num(first_col) + size - 1)
 
            # Warning if there is missing value in the dimension
            if (np.diff(original_index) != 1).any():
                missing_index = np.arange(original_index[0], original_index[-1]+1)
                missing_index = np.setdiff1d(missing_index, original_index) + first_row
                cells = [series_row_or_col + str(row) for row in missing_index]
                warnings.warn("\n\tDimension value missing or non-valid in:\n"
                              + "File name:\t{}\n".format(self.file)
                              + "Sheet name:\t{}\n".format(self.tab)
                              + "\n\tCell(s):\t{}\n".format(cells)
                              + "\tthe corresponding column(s) to the "
                              + "missing/non-valid value(s) will be ignored\n\n")

            # read data
            data = self._get_data_from_file(
                rows=first_row+original_index,
                cols=[first_col, last_col])

        else:
            # get series data
            series_data = self._get_data_from_file_opyxl(series_row_or_col)
    
            try:
                series_shape = series_data.shape
            except AttributeError:
                # Error if the lookup/time dimension has len 0 or 1
                raise ValueError("\n\tDimension given in:\n"
                                 + "File name:\t{}\n".format(self.file)
                                 + "Sheet name:\t{}\n".format(self.tab)
                                 + "Dimension name:\t{}\n".format(series_row_or_col)
                                 + " is not a vector")
    
            if series_shape[0] == 1:
                # horizontal definition of lookup/time dimension
                series_data = series_data[0]
                transpose = True
    
            elif series_shape[1] == 1:
                # vertical definition of lookup/time dimension
                series_data = series_data[:, 0]
                transpose = False
    
            else:
                # Error if the lookup/time dimension is 2D
                raise ValueError("\n\tDimension given in:\n"
                                 + "File name:\t{}\n".format(self.file)
                                 + "Sheet name:\t{}\n".format(self.tab)
                                 + "Dimension name:\t{}\n".format(series_row_or_col)
                                 + " is a table and not a vector")
    
            # Substract missing values in the series
            nan_index = np.isnan(series_data)
            
            if nan_index.any():
                series_data = series_data[~nan_index]
                warnings.warn("\n\tDimension value missing or non-valid in:\n"
                              + "File name:\t{}\n".format(self.file)
                              + "Sheet name:\t{}\n".format(self.tab)
                              + "Dimension name:\t{}\n".format(series_row_or_col)
                              + "\tthe corresponding data value(s) to the "
                              + "missing/non-valid value(s) will be ignored\n\n")
            # get data
            data = _get_data_from_file_opyxl(cell)
            
            if transpose:
                # transpose for horizontal definition of dimension
                data = data.transpose()
    
            if data.shape[1] != size:
                # Given coordinates length is different than lentgh of 2nd dimension
                raise ValueError("\n\tData given in:\n"
                                  + "File name:\t{}\n".format(self.file)
                                  + "Sheet name:\t{}\n".format(self.tab)
                                  + "Data name:\t{}\n".format(cell)
                                  + " has not the same size as the given coordinates")
    
            if data.shape[1] == 1:
                # remove second dimension of data if its shape is (N, 1)
                data = data[:, 0]
            
            try:
                # substract missing values from series and check 1st dimension
                data = data[~nan_index]
            except IndexError:
                raise ValueError("\n\tDimension and data given in:\n"
                                  + "File name:\t{}\n".format(self.file)
                                  + "Sheet name:\t{}\n".format(self.tab)
                                  + "Dimension name:\t{}\n".format(series_row_or_col)
                                  + "Data name:\t{}\n".format(cell)
                                  + " don't have the same length in the 1st dimension")

        # TODO manage data NA and missing values
        return series_data, data

    def _resolve_file(self, root=None, possible_ext=None):

        possible_ext = possible_ext or ['', '.xls', '.xlsx', '.odt', '.txt', '.tab']
 
        if self.file[0] == '?':
            self.file = os.path.join(root, self.file[1:])

        if not os.path.isfile(self.file):
            for ext in possible_ext:
                if os.path.isfile(self.file + ext):
                    self.file = self.file + ext
                    return
        
            raise FileNotFoundError(self.file)

        else:
             return

    def _initialize_data(self, dim_name):
        """
        Initialize one element of DATA or LOOKUPS

        Parameters
        ----------
        dim_name: str
            Dimension name.
            "lookup_dim" for LOOKUPS, "time" for DATA.

        Returns
        -------
        data: xarray.DataArray
            Dataarray with the time or interpolation dimension
            as first dimension.
        """
        self._resolve_file(root=self.root)
        series_across = self._series_selector(self.x_row_or_col, self.cell):
        size = _compute_shape(self.coords, self.dim, reshape_len=1)[0]

        series, data = self._get_series_data(
            series_across=series_across,
            series_row_or_col=self.x_row_or_col,
            cell=self.cell, size=size
        )

        # Check if the lookup/time dimension is strictly monotonous
        series_diff = np.diff(series.values)
        if np.any(series_diff >= 0) and np.any(series_diff <= 0):
            raise ValueError("Dimension given in:\n"
                             + "\tFile name:\t{}\n".format(self.file)
                             + "\tSheet name:\t{}\n".format(self.tab)
                             + "\t{}:\t{}\n".format(series_across, self.x_row_or_col)
                             + " is not strictly monotonous")

        reshape_dims = tuple([len(series)] + _compute_shape(self.coords, self.dims))
        if len(reshape_dims) > 1:
            data = self.reshape(data, reshape_dims)

        data = xr.DataArray(
            data=data,
            coords={dim_name: x_data, **self.coords},
            dims=[dim_name] + self.dim
        )

        return data


    @staticmethod
    def col_to_num(col):
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
        else:
            left = ord(col[0].upper()) - ord('A') + 1
            right = ord(col[1].upper()) - ord('A')
            return left * (ord('Z')-ord('A')+1) + right

    @staticmethod
    def num_to_col(num):
        """
        Transforms the column number to name. Also working with lists.
        
        Parameters
        ----------
        col_v: int or list of ints
          Column number(s)

        Returns
        -------
        int/list
          Column name(s)
        """
        def _to_ABC(x):
           if x < 26:
               return chr(ord('A')+x)
           return _to_ABC(int(x/26)-1) + _to_ABC(int(x%26))
    
        try:
            len(num)
            return [_to_ABC(col) for col in num]
        except TypeError:
            return _to_ABC(num)
 
    @staticmethod
    def _split_excel_cell(cell):
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
        row number, column name: int, str
          If the cell input is valid.
      
        """
        split = re.findall('\d+|\D+', cell)
        try:
            # check that we only have two values [column, row]
            assert len(split) == 2
            # check that the column name has no special characters
            assert not re.compile('[^a-zA-Z]+').search(split[0])
            # check that row number is not 0
            assert int(split[1]) != 0
            return int(split[1]), split[0]
        except AssertionError:
            return

    @staticmethod
    def reshape(data, dims):
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

    @staticmethod
    def _compute_shape(coords, dims, reshape_len=None):
        """
        Computes the 'shape' of a coords dictionary.
        Function used to rearange data in xarrays and
        to compute the number of rows/columns to be read in a file.
        
        Parameters
        ----------
        coords: dict 
          Dictionary of the dimension names as a keys with their
          values.
        dims: list
          Ordered list of the dimensions.
        reshape_len: int (optional)
          Number of dimensions of the output shape.
          The shape will ony compute the corresponent table
          dimensions to read from Excel, then, the dimensions 
          with length one will be ignored at first.
          Lately, it will complete with 1 on the left of the shape
          if the reshape_len value is bigger than the length of shape.
          Will raise a ValueError if we try to reshape to a reshape_len
          smaller than the initial shape.
        
        Returns
        -------
        shape: list
          Shape of the ordered dictionary or of the desired table or vector
          
        Note
        ----
        Dictionaries in Python >= 3.7 are ordered, which means that
        we could remove dims if there is a not backward compatible
        version of the library which only works in Python 3.7+. For now,
        the dimensions list is passed to make it work properly for all the users.
        """
        if not reshape_len:
            return [len(coords[dim]) for dim in dims]
    
        shape = [len(coords[dim]) for dim in dims if len(coords[dim]) > 1]
    
        shape_len = len(shape)
        
        if shape_len > reshape_len:
            raise ValueError("The shape of the coords to read in a "
                             + " external file must be at most " 
                             + "{} dimensional".format(reshape_len))
            
        return [1]*(reshape_len-shape_len) + shape
    
    @staticmethod
    def _series_selector(x_row_or_col, cell):
        """
        Selects if a series data (DATA/LOOKUPS), should be read by columns, rows or cell name.
        Based on the input format of x_row_or_col and cell.
        The format of the 2 variables must be consistent.
        
        Parameters
        ----------
        x_row_or_col: str
          String of a number if series is given in a row, letter if series is given in a column
          or name if the series is given by cell range name.
        cell: str
          Cell identificator, such as "A1", or name if the data is given by cell range name.
    
        Returns
        -------
        series_across: str
          "row" if series is given in a row
          "column" if series is given in a column
          "name" if series and data are given by range name
    
        """
        if x_row_or_col.isnumeric():
            # if x_row_or_col is numeric the series must be a row
            return "row"
    
        else:
            if _split_excel_cell(cell):
                # if the cell can be splitted means that the format is "A1" like
                # then the series must be a column
                return "column"
            else:
                return "name"


class ExtData(External):
    """
    Class for Vensim GET XLS DATA/GET DIRECT DATA
    """
    def __init__(self, file_name, tab, time_row_or_col, cell, interp, root, coords, dims=None):
        self.files = [file_name]
        self.tabs = [tab]
        self.time_row_or_cols = [time_row_or_col]
        self.cells = [cell]
        self.roots = [root]
        self.coordss = [coords]
        #TODO replace by dims readed ones to make compatible with Python < 3.7
        self.dims = [list(coords)]

        # This value should be unique
        self.interp = interp

    def add(self, file_name, tab, time_row_or_col, cell, interp, root, coords, dims=None):
        """
        Add information to retrieve new dimension in an already declared object
        """
        self.files.append(file_name)
        self.tabs.append(tab)
        self.time_row_or_cols.append(time_row_or_col)
        self.cells.append(cell)
        self.roots.append(root)
        self.coordss.append(coords)
        #TODO replace by dims readed ones to make compatible with Python < 3.7
        self.dims.append(list(coords))

    def initialize(self):
        """
        Initialize all elements and create the self.data xarray.DataArray
        """
        data = []
        zipped = zip(self.files, self.tabs, self.time_row_or_cols,
                     self.cells, self.roots, self.coordss, self.dims)
        for (self.file, self.tab, self.x_row_or_col,
             self.cell, self.root, self.coords, self.dim)\
          in zipped:
            data.append(self._initialize_data("time"))
        self.data = utils.xrmerge(data)

    def __call__(self, time):

        if time > self.data['time'].values[-1]:
            outdata = self.data[-1]
        elif time < self.data['time'].values[0]:
            outdata = self.data[0]
        elif self.interp == 'interpolate' or self.interp is None:  # 'interpolate' is the default
            outdata = self.data.interp(time=time)
        elif self.interp == 'look forward':
            next_t = self.data['time'][self.data['time'] >= time][0]
            outdata = self.data.sel(time=next_t)
        elif self.interp == 'hold backward':
            last_t = self.data['time'][self.data['time'] <= time][-1]
            outdata = self.data.sel(time=last_t)
        else:
            # For :raw: (or actually any other/invalid) keyword directives
            try:
                outdata = self.data.sel(time=time)
            except KeyError:
                return np.nan

        # if output from the lookup is a float return as float and not xarray
        try:
            return float(outdata)
        except TypeError:
            return outdata


class ExtLookup(External):
    """
    Class for Vensim GET XLS LOOKUPS/GET DIRECT LOOKUPS
    """
    def __init__(self, file_name, tab, x_row_or_col, cell, root, coords, dims=None):
        self.files = [file_name]
        self.tabs = [tab]
        self.x_row_or_cols = [x_row_or_col]
        self.cells = [cell]
        self.roots = [root]
        self.coordss = [coords]
        #TODO replace by dims readed ones to make compatible with Python < 3.7
        self.dims = [list(coords)]

    def add(self, file_name, tab, x_row_or_col, cell, root, coords, dims=None):
        """
        Add information to retrieve new dimension in an already declared object
        """
        self.files.append(file_name)
        self.tabs.append(tab)
        self.x_row_or_cols.append(x_row_or_col)
        self.cells.append(cell)
        self.roots.append(root)
        self.coordss.append(coords)
        #TODO replace by dims readed ones to make compatible with Python < 3.7
        self.dims.append(list(coords))

    def initialize(self):
        """
        Initialize all elements and create the self.data xarray.DataArray
        """
        data = []
        zipped = zip(self.files, self.tabs, self.x_row_or_cols,
                     self.cells, self.roots, self.coordss, self.dims)
        for (self.file, self.tab, self.x_row_or_col,
             self.cell, self.root, self.coords, self.dim)\
          in zipped:
            data.append(self._initialize_data("lookup_dim"))
        self.data = utils.xrmerge(data)

    def __call__(self, x):
        return self._call(x)

    def _call(self, x):        
        if isinstance(x, xr.DataArray):
            return xr.DataArray(data=self._call(x.values), coords=x.coords, dims=x.dims)
        
        elif isinstance(x, np.ndarray):
            return np.array([self._call(i) for i in x])

        if x > self.data['lookup_dim'].values[-1]:
            outdata = self.data[-1]
        elif x < self.data['lookup_dim'].values[0]:
            outdata = self.data[0]
        else: 
            outdata = self.data.interp(lookup_dim=x)

        # if output from the lookup is a float return as float and not xarray
        try:
            return float(outdata)
        except TypeError:
            return outdata


class ExtConstant(External):
    """
    Class for Vensim GET XLS CONSTANT/GET DIRECT CONSTANT
    """
    def __init__(self, file_name, tab, cell, root, coords, dim=None):
        self.files = [file_name]
        self.tabs = [tab]
        self.transposes = [cell[-1] == '*']
        self.cells = [cell.strip('*')]
        self.roots = [root]
        self.coordss = [coords]
        #TODO replace by dims readed ones to make compatible with Python < 3.7
        self.dims = [list(coords)]

    def add(self, file_name, tab, cell, root, coords, dim=None):
        """
        Add information to retrieve new dimension in an already declared object
        """
        self.files.append(file_name)
        self.tabs.append(tab)
        self.transposes.append(cell[-1] == '*')
        self.cells.append(cell.strip('*'))
        self.roots.append(root)
        self.coordss.append(coords)
        #TODO replace by dims readed ones to make compatible with Python < 3.7
        self.dims.append(list(coords))

    def initialize(self):
        """
        Initialize all elements and create the self.data xarray.DataArray
        """
        data = []
        zipped = zip(self.files, self.tabs, self.transposes,
                     self.cells, self.roots, self.coordss, self.dims)
        for (self.file, self.tab, self.transpose,
            self.cell, self.root, self.coords, self.dim)\
          in zipped:
            data.append(self._initialize())
        self.data = utils.xrmerge(data)

    def _initialize(self):
        """
        Initialize one element
        """
        self._resolve_file(root=self.root)
        split = _split_excel_cell(self.cell)
        if split:
            series_across = "cell"
            cell = split
        else:
            series_across = "name"
            cell = self.cell
        
        shape = _compute_shape(self.coords, self.dims, reshape_len=2)
        
        if self.transpose:
            shape.reverse()
        
        _get_constant_data(data_across, cell, shape)
        
        if self.transpose:
            data = data.transpose()


        # Create only an xarray if the data is not 0 dimensional
        if len(self.dims) > 0:
            reshape_dims = tuple(_compute_shape(self.coords, self.dims))
        
            if len(reshape_dims) > 1: data = self.reshape(data, reshape_dims) 

            data = xr.DataArray(
                data=data, coords=self.coords, dims=self.dims
            )

        return data

def _get_constant_data(self, data_across, cell, shape):
    if data_across == "cell":
        
        start_row, start_col = cell
        
        end_row = start_row + size[0] - 1
        end_col = self.num_to_col(self.col_to_num(start_col) + size[1] - 1)

        return self._get_data_from_file(rows=[start_row, end_row], cols=[start_col, end_col])
            
    else:
        data = self._get_data_from_file_opyxl(cell)
        
        # TODO check dims
        
        return data
    
    def __call__(self):
        return self.data


class ExtSubscript(External):
    """
    Class for Vensim GET XLS SUBSCRIPT/GET DIRECT SUBSCRIPT
    """
    def __init__(self, file_name, tab, firstcell, lastcell, prefix):
        self.file_name = file_name
        self.tab = tab

        row_first, col_first = self._split_excel_cell(firstcell)
        row_last, col_last = self._split_excel_cell(lastcell)
        data = self._get_data_from_file(rows=[row_first, row_last],
                                        cols=[col_first, col_last])

        self.subscript = [prefix + str(d) for d in data.values.flatten()]

