import re

class Excels():

    _instance = None
    _Excels = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Excels, cls).__new__(cls)
        return cls._instance

    def read(cls, file_name):
        if file_name in cls._Excels:
            return cls._Excels[file_name]
        else: 
            excel = pd.ExcelFile(file_name)
            cls._Excels[file_name] = excel
           return excel
 
class External():

        self.file = file_name
        self.tab = tab
    def _get_data_from_file(rows, cols, axis="columns", dropna=False):
        ext = os.path.splitext(self.file)[1].lower()
        if ext in ['.xls', '.xlsx']:
            if rows is None:
                skip = nrows = None
            elif isinstance(rows, int):
                skip = rows
                nrows = None
            else:
                skip = rows[0] - 1
                nrows = rows[1] - skip if rows[1] is not None else None
            if isinstance(cols, list):
                cols = [self.num_to_col(c) if isinstance(c, int) else c for c in cols]
                usecols = cols[0] + ":" + cols[1]
            else:
                usecols = cols

            excel = cls.read(self.file)
            
            data = excel.parse(sheet_name=self.tab, header=None, skiprows=skip,
                               nrows=nrows, usecols=usecols)
            
            if dropna:
                data = data.dropna(how="all", axis=axis)
            
            if isinstance(rows, int) or\
               (isinstance(rows, list) and rows[0] == rows[1]):
                data = data.iloc[0]
            if isinstance(cols, str) or\
               (isinstance(cols, list) and cols[0].lower() == cols[1].lower()):
                if isinstance(data, pd.DataFrame):
                    data = data.iloc[:, 0]
                elif isinstance(data, pd.Series):
                    data.index = range(data.size)
                    data = data[0]
                else: 
                    raise NotImplementedError
            return data

        raise NotImplementedError(ext)

    def _get_series_data(self, series_across, series_row_or_col, cell, size):
        
        if series_across:
            # Get serie from 1 to size
        
            series_data = _get_data_from_file(rows=int(series_row_or_col)-1, 
                                              cols=None, dropna=False)
            first_data_row, first_col = self._split_excel_cell(cell)
    
            first_col = first_col.upper()
            first_col_float = self.col_to_num(first_col)
    
            original_index = np.array(series_data.index)[first_col_float:]
            series_data = series_data[first_col_float:]
    
            series_data = pd.to_numeric(series_data, errors='coerce')
            valid_values = ~np.isnan(series_data)
            original_index = original_index[valid_values]
            series_data = series_data[valid_values]
    
            if len(series_data) == 0:
                sys.exit("Dimension given in:\n"
                         + "File name:\t{}\n".format(self.file)
                         + "Sheet name:\t{}\n".format(self.tab)
                         + "Row number:\t{}\n".format(series_row_or_col)
                         + " has length 0")
            
            last_col = self.num_to_col(original_index[-1])
            last_data_row = first_data_row + size - 1
     
            if (np.diff(original_index) != 1).any():
                missing_index = np.arange(original_index[0], original_index[-1]+1)
                missing_index = np.setdiff1d(missing_index, original_index)
                cells = self.num_to_col(missing_index)
                cells = [cell + series_row_or_col for cell in cells]
                warnings.warn("\n\tDimension value missing or non-valid in:\n"
                              + "File name:\t{}\n".format(self.file)
                              + "Sheet name:\t{}\n".format(self.tab)
                              + "\n\tCell(s):\t{}\n".format(cells)
                              + "\tthe corresponding column(s) to the "
                              + "missing/non-valid value(s) will be ignored\n\n")
                          
            data = _get_data_from_file(rows=[first_data_row, last_data_row],
                                       cols=original_index, dropna=False)
            data = data.transpose()
    
        else:
            # TODO improve the lookup as before to remove/Warning when missing values
            first_row, first_col = self._split_excel_cell(cell)
            series_data = _get_data_from_file(rows=[first_row, None],
                                              cols=series_row_or_col,
                                              axis="rows", 
                                              dropna=True)
    
            last_row = first_row + series_data.size - 1
            cols = [first_col, self.col_to_num(first_col) + size]
            data = _get_data_from_file(self, rows=[first_row, last_row], cols=cols)
    
        return series_data, data

    def _resolve_file(self, root=None, possible_ext=None):

        possible_ext = possible_ext or ['.xls', '.xlsx', '.odt', '.txt', '.tab']
 
        if self.file[0] == '?':
            self.file = os.path.join(root, self.file[1:])

        if not os.path.isfile(self.file):
            for ext in possible_ext:
                if os.path.isfile(self.file + ext):
                    self.file = self.file + ext
                    return
 
        raise FileNotFoundError(self.file)

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
        Transforms the column number to name. Also working with lists
        
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
        Splits cell number and letter
        
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


class Data(External):
    def __init__(self, file_name, tab, time_row_or_col, cell, interp, time, root, coords):
        super(Data, self).__init__(file_name, tab)
        self.time_row_or_col = time_row_or_col
        self.cell = cell
        self.time_func = time
        self.interp = interp
        self.coords = coords

    def initialize(self):
        self._resolve_file(root=root)
        time_across = self.time_row_or_col.isnumeric()
        size = int(np.product([len(v) for v in self.coords.values()]))

        time_data, data = _get_series_data(series_across=time_across,
                                           series_row_or_col=self.time_row_or_col,
            cell=self.cell, size=size
        )

        reshape_dims = tuple( [len(i) for i in self.coords.values()] + [len(time_data)] )
        if len(reshape_dims) > 1:
            data = reshape(data, reshape_dims)

        self.state = xr.DataArray(
            data=data, coords={**self.coords, 'time': time_data}, dims=list(self.coords)+['time']
        )

    def __call__(self):

        time = self.time_func()
        if time > self.state['time'][-1]:
            return self.state['time'][-1]
        elif time < self.state['time'][0]:
            return self.state['time'][0]

        if self.interp == 'interpolate' or self.interp is None:  # 'interpolate' is the default
            return self.state.interp(time=time)
        elif self.interp == 'look forward':
            next_t = self.state['time'][self.state['time'] >= time][0]
            return self.state.sel(time=next_t)
        elif self.interp == 'hold backward':
            last_t = self.state['time'][self.state['time'] <= time][-1]
            return self.state.sel(time=last_t)

        # For :raw: (or actually any other/invalid) keyword directives
        try:
            return self.state.sel(time=time)
        except KeyError:
            return np.nan

class ExtLookup(External):
    def __init__(self, file_name, tab, x_row_or_col, cell, root, coords):
        super(ExtConstant, self).__init__(file_name, tab)
        self.x_row_or_col = x_row_or_col
        self.cell = cell
        self.coords = coords
        self.initialize()

    def initialize(self):
        x_across = self.x_row_or_col.isnumeric()
        size = int(np.product([len(v) for v in self.coords.values()]))

        x_data, data = _get_series_data(series_across=x_across,
                                        series_row_or_col=self.x_row_or_col,
                                        cell=self.cell, size=size)

        reshape_dims = tuple( [len(x_data)] + [len(i) for i in self.coords.values()] )

        if len(reshape_dims) > 1:
            data = reshape(data, reshape_dims)

        self.state = xr.DataArray(
            data=data, coords={'x': x_data, **self.coords},
            dims=['x'] + list(self.coords))
        # TODO add interpolation to missing values

    def __call__(self, x):
        return self._call(x)

    def _call(self, x):        
        if isinstance(x, xr.DataArray):
            return xr.DataArray(data=self._call(x.values), coords=x.coords, dims=x.dims)
        
        elif isinstance(x, np.ndarray):
            return np.array([self._call(i) for i in x])

        
        if x > self.state['x'].values[-1]:
            return self.state.values[-1]
        elif x < self.state['x'].values[0]:
            return self.state.values[0]
        return self.state.interp(x=x).values

class ExtConstant(External):
    def __init__(self, file_name, tab, cell, root, coords):
        super(ExtConstant, self).__init__(file_name, tab)
        self.transpose = cell[-1] == '*'
        self.cell = cell.strip('*')
        self.coords = coords
        self.initialize()

    def initialize(self):
        dims = list(self.coords)
        start_row, start_col = self._split_excel_cell(self.cell)
        end_row = start_row
        end_col = start_col
        if dims:
            if self.transpose:
                end_row = start_row + len(self.coords[dims[-1]]) - 1
            else:
                end_col = self.num_to_col(col_to_num(start_col) + len(self.coords[dims[-1]]))

            if len(dims) >= 2:
                if self.transpose:
                    end_col = self.num_to_col(col_to_num(start_col) + len(self.coords[dims[-2]]))
                else:
                    end_row = start_row + len(self.coords[dims[-2]]) - 1

        data = _get_data_from_file(self.file, tab=self.tab, rows=[start_row, end_row], cols=[start_col, end_col])
        if self.transpose:
            data = data.transpose()    

        if len(self.coords.values()) > 0:
            reshape_dims = tuple([len(i) for i in self.coords.values()])
        
            if len(reshape_dims) > 1: data = reshape(data, reshape_dims) 

            self.value = xr.DataArray(
                data=data, coords=self.coords, dims=list(self.coords)
            )
        else:
            self.value = data

    def __call__(self):
        return self.value

def get_direct_subscript(file, tab, firstcell, lastcell, prefix):
    file = _resolve_file(file)

    row_first, col_first = self._split_excel_cell(firstcell)
    row_last, col_last = self._split_excel_cell(lastcell)
    data = _get_data_from_file(
        file, tab,
        rows=[row_first, row_last],
        cols=[col_first, col_last]
    )
    return [prefix + str(d) for d in data.flatten()]


get_xls_subscript = get_direct_subscript

def _isFloat(n):
    try:
        if (n == "nan"): return False
        float(n)
        return True
    except ValueError:
        return False

def reshape(data, dims):
    
    if isinstance(data, (np.int64, np.float64)):
        return np.array(data).reshape(dims)
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        return data.values.reshape(dims)
    elif isinstance(data, pd.DataFrame):
        return data

