import re
import os
import warnings
import pandas as pd
import numpy as np
import xarray as xr
from . import utils

class Excels():
    """
    Class to save the read Excel files and thus avoud double reading
    """
    _instance = None
    _Excels = {}

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

    def clean(cls):
        """
        Clean the dictionary of read files
        """
        cls._Excels = {}
 
class External():
    """
    Main class of external objects
    """

    def _get_data_from_file(self, rows, cols, axis="columns", dropna=False):
        """
        Function thar reads data from excel file
        """
        # TODO document well 
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

 
            if dropna:
                data = data.dropna(how="all", axis=axis)
 
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

        raise NotImplementedError("The files with extension" + ext + "are not implemented")

    def _get_series_data(self, series_across, series_row_or_col, cell, size):
        """
        Function thar reads series data from excel file
        """
        # TODO document well 
        if series_across:
            # Horizontal data (dimension values in a row)

            # get the dimension values
            series_data = self._get_data_from_file(rows=int(series_row_or_col)-1, 
                                                   cols=None, dropna=False)

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
                cols=original_index, dropna=False
            )
            data = data.transpose()

        else:
            # Vertical data (dimension values in a column)

            # get the dimension values
            first_row, first_col = self._split_excel_cell(cell)
            series_data = self._get_data_from_file(rows=[first_row, None],
                                                   cols=series_row_or_col,
                                                   axis="rows", 
                                                   dropna=False)

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
                cols=[first_col, last_col],
                dropna=False)

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
        x_across = self.x_row_or_col.isnumeric()
        size = int(np.product([len(v) for v in self.coords.values()]))

        x_data, data = self._get_series_data(
            series_across=x_across,
            series_row_or_col=self.x_row_or_col,
            cell=self.cell, size=size
        )

        # Check if the lookup/time dimension is strictly monotonous
        x_data_diff = np.diff(x_data.values)
        if np.any(x_data_diff >= 0) and np.any(x_data_diff <= 0):
            row_or_col = "Row number" if x_across else "Column"
            print(x_data)
            raise ValueError("Dimension given in:\n"
                             + "\tFile name:\t{}\n".format(self.file)
                             + "\tSheet name:\t{}\n".format(self.tab)
                             + "\t{}:\t{}\n".format(row_or_col, self.x_row_or_col)
                             + " is not strictly monotonous")

        reshape_dims = tuple([len(x_data)]+ [len(i) for i in self.coords.values()])
        if len(reshape_dims) > 1:
            data = self.reshape(data, reshape_dims)

        data = xr.DataArray(
            data=data,
            coords={dim_name: x_data, **self.coords},
            dims=[dim_name] + list(self.coords)
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
        Splits cell number and letter.
        
        Parameters
        ----------
        cell: str
          Cell name such as "A1"
        
        Returns
        -------
        tuple (int, str)
          Cell number, Cell letter
        """
        return int(re.sub("[a-zA-Z]+", "", cell)), re.sub("[^a-zA-Z]+", "", cell)

    @staticmethod
    def reshape(data, dims):
        """
        Reshapes an xarray.DataArray or np.ndarray in the given dimensions.

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


class ExtData(External):
    """
    Class for Vensim GET XLS DATA/GET DIRECT DATA
    """
    def __init__(self, file_name, tab, time_row_or_col, cell, interp, root, coords):
        self.files = [file_name]
        self.tabs = [tab]
        self.time_row_or_cols = [time_row_or_col]
        self.cells = [cell]
        self.roots = [root]
        self.coordss = [coords]

        # This value should be unique
        self.interp = interp

    def add(self, file_name, tab, time_row_or_col, cell, interp, root, coords):
        """
        Add information to retrieve new dimension in an already declared object
        """
        self.files.append(file_name)
        self.tabs.append(tab)
        self.time_row_or_cols.append(time_row_or_col)
        self.cells.append(cell)
        self.roots.append(root)
        self.coordss.append(coords)

    def initialize(self):
        """
        Initialize all elements and create the self.data xarray.DataArray
        """
        data = []
        zipped = zip(self.files, self.tabs, self.time_row_or_cols,
                     self.cells, self.roots, self.coordss)
        for self.file, self.tab, self.x_row_or_col,\
            self.cell, self.root, self.coords in zipped:
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
    def __init__(self, file_name, tab, x_row_or_col, cell, root, coords):
        self.files = [file_name]
        self.tabs = [tab]
        self.x_row_or_cols = [x_row_or_col]
        self.cells = [cell]
        self.roots = [root]
        self.coordss = [coords]

    def add(self, file_name, tab, x_row_or_col, cell, root, coords):
        """
        Add information to retrieve new dimension in an already declared object
        """
        self.files.append(file_name)
        self.tabs.append(tab)
        self.x_row_or_cols.append(x_row_or_col)
        self.cells.append(cell)
        self.roots.append(root)
        self.coordss.append(coords)

    def initialize(self):
        """
        Initialize all elements and create the self.data xarray.DataArray
        """
        data = []
        zipped = zip(self.files, self.tabs, self.x_row_or_cols,
                     self.cells, self.roots, self.coordss)
        for self.file, self.tab, self.x_row_or_col,\
            self.cell, self.root, self.coords in zipped:
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
    def __init__(self, file_name, tab, cell, root, coords):
        self.files = [file_name]
        self.tabs = [tab]
        self.transposes = [cell[-1] == '*']
        self.cells = [cell.strip('*')]
        self.roots = [root]
        self.coordss = [coords]

    def add(self, file_name, tab, cell, root, coords):
        """
        Add information to retrieve new dimension in an already declared object
        """
        self.files.append(file_name)
        self.tabs.append(tab)
        self.transposes.append(cell[-1] == '*')
        self.cells.append(cell.strip('*'))
        self.roots.append(root)
        self.coordss.append(coords)

    def initialize(self):
        """
        Initialize all elements and create the self.data xarray.DataArray
        """
        data = []
        zipped = zip(self.files, self.tabs, self.transposes,
                     self.cells, self.roots, self.coordss)
        for self.file, self.tab, self.transpose,\
            self.cell, self.root, self.coords in zipped:
            data.append(self._initialize())
        self.data = utils.xrmerge(data)

    def _initialize(self):
        """
        Initialize one element
        """
        self._resolve_file(root=self.root)
        dims = list(self.coords)
        start_row, start_col = self._split_excel_cell(self.cell)
        end_row = start_row
        end_col = start_col
        if dims:
            if self.transpose:
                end_row = start_row + len(self.coords[dims[-1]]) - 1
            else:
                end_col = self.num_to_col(self.col_to_num(start_col) + len(self.coords[dims[-1]]) -1)

            if len(dims) >= 2:
                if self.transpose:
                    end_col = self.num_to_col(self.col_to_num(start_col) + len(self.coords[dims[-2]]))
                else:
                    end_row = start_row + len(self.coords[dims[-2]]) - 1

        data = self._get_data_from_file(rows=[start_row, end_row], cols=[start_col, end_col])
        if self.transpose:
            data = data.transpose()    

        # Create only an xarray if the data is not 0 dimensional
        if len(self.coords.values()) > 0:
            reshape_dims = tuple([len(i) for i in self.coords.values()])
        
            if len(reshape_dims) > 1: data = self.reshape(data, reshape_dims) 

            data = xr.DataArray(
                data=data, coords=self.coords, dims=list(self.coords)
            )

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


