"""
ModelOutput class is used to build different output objects based on
user input. For now, available output types are pandas DataFrame or
netCDF4 Dataset.
The OutputHandlerInterface class is an interface for the creation of handlers
for other output object types.
"""
import abc
import time as t

from csv import QUOTE_NONE
from pathlib import Path

import regex as re

import numpy as np
import xarray as xr
import pandas as pd

from pysd._version import __version__
from pysd.tools.ncfiles import NCFile

from . utils import xrsplit


class ModelOutput():
    """
    Manages outputs from simulations. Handles different types of outputs by
    dispatchinging the tasks to adequate object handlers.

    Parameters
    ----------
    out_file: str or pathlib.Path
        Path to the file where the results will be written.

    """
    valid_output_files = [".nc", ".csv", ".tab"]

    def __init__(self, out_file=None):

        if out_file:
            ModelOutput.check_output_file_path(out_file)

        # Add any other handlers that you write here, in the order you
        # want them to run (DataFrameHandler runs first)
        self.handler = DataFrameHandler(DatasetHandler(None)).handle(out_file)

    def set_capture_elements(self, capture_elements):
        self.handler.capture_elements_step = capture_elements["step"] + \
            ["time"]
        self.handler.capture_elements_run = capture_elements["run"]

    def initialize(self, model):
        """ Delegating the creation of the results object and its elements to
        the appropriate handler."""
        self.handler.initialize(model)

    def update(self, model):
        """ Delegating the update of the results object and its elements to the
        appropriate handler."""
        self.handler.update(model)

    def postprocess(self, **kwargs):
        """ Delegating the postprocessing of the results object to the
        appropriate handler."""
        return self.handler.postprocess(**kwargs)

    def add_run_elements(self, model):
        """ Delegating the addition of results with run cache in the output
        object to the appropriate handler."""
        self.handler.add_run_elements(model)

    @staticmethod
    def check_output_file_path(output_file):

        if not isinstance(output_file, (str, Path)):
            raise TypeError(
                    "Paths must be strings or pathlib Path objects.")

        output_file = Path(output_file)

        file_extension = output_file.suffix
        if file_extension not in ModelOutput.valid_output_files:
            raise ValueError(
                    f"Unsupported output file format {file_extension}")

        return output_file

    @staticmethod
    def collect(model, flatten_output=True):
        """
        Collect results after one or more simulation steps, and save to
        desired output format (DataFrame, csv, tab or netCDF).

        Parameters
        ----------
        model: pysd.py_backend.model.Model
            PySD Model object.


        flatten_output: bool (optional)
            If True, once the output dataframe has been formatted will
            split the xarrays in new columns following Vensim's naming
            to make a totally flat output. Default is True.
            This argument will be ignored when passing a netCDF4 file
            path in the output_file argument.

        """

        del model._dependencies["OUTPUTS"]

        model.output.add_run_elements(model)

        model._remove_constant_cache()

        return model.output.postprocess(
            return_addresses=model.return_addresses, flatten=flatten_output)


class OutputHandlerInterface(metaclass=abc.ABCMeta):
    """
    Interface for the creation of different output handlers.
    """
    def __init__(self, next=None):
        self._next = next

    def handle(self, out_file):
        """
        If the concrete handler can write on the output file type passed by the
        user, it returns the handler itself, else it goes to the next handler.

        Parameters
        ----------
        out_file: str or pathlib.Path
            Path to the file where the results will be written.

        Returns
        -------
        handler

        """
        handler = self.process_output(out_file)

        if handler is not None:  # the handler can write the out_file type.
            return handler
        else:
            return self._next.handle(out_file)

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'process_output') and
                callable(subclass.process_output) and
                hasattr(subclass, 'initialize') and
                callable(subclass.initialize) and
                hasattr(subclass, 'update') and
                callable(subclass.update) and
                hasattr(subclass, 'postprocess') and
                callable(subclass.postprocess) and
                hasattr(subclass, 'add_run_elements') and
                callable(subclass.add_run_elements) or
                NotImplemented)

    @abc.abstractmethod
    def process_output(self, out_file):
        """
        If concrete handler can process out_file, returns it, else returns
        None.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def initialize(self, model):
        """
        Create the results object and its elements based on capture_elemetns.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, model):
        """
        Update the results object at each iteration at which resutls are
        stored.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def postprocess(self, **kwargs):
        """
        Perform different tasks at the time of returning the results object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def add_run_elements(self, model):
        """
        Add elements with run cache to the results object.
        """
        raise NotImplementedError


class DatasetHandler(OutputHandlerInterface):
    """
    Manages simulation results stored as netCDF4 Dataset.
    """

    def __init__(self, next):
        super().__init__(next)
        self.out_file = None
        self.ds = None
        self._step = 0
        self.nc = __import__("netCDF4")

    @property
    def step(self):
        """
        Used as time index for the output Dataset. Increases by one at each
        iteration.
        """
        return self._step

    def __update_step(self):
        """
        Increases the _step attribute by 1 at each model iteration.
        """
        self._step = self.step + 1

    def process_output(self, out_file):
        """
        If out_file can be handled by this concrete handler, it returns the
        handler instance, else it returns None.

        Parameters
        ----------
        out_file: str or pathlib.Path
            Path to the file where the results will be written.

        Returns
        -------
        None or DatasetHandler instance

        """
        if out_file:
            out_file = Path(out_file)
            if out_file.suffix == ".nc":
                self.out_file = out_file
                return self

    def initialize(self, model):
        """
        Creates a netCDF4 Dataset and adds model dimensions and variables
        present in the capture elements to it.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object

        Returns
        -------
        None

        """
        self.ds = self.nc.Dataset(self.out_file, "w")

        # defining global attributes
        self.ds.description = "Results for simulation run on " \
            f"{t.ctime(t.time())} using PySD version {__version__}"
        self.ds.model_file = model.py_model_file or model.mdl_file
        self.ds.timestep = f"{model.time.time_step()}" if model.cache_type[
            "time_step"] == "run" else "Variable"
        self.ds.initial_time = f"{model.time.initial_time()}"
        self.ds.final_time = f"{model.time.final_time()}" if model.cache_type[
            "final_time"] == "run" else "Variable"

        # creating variables for all model dimensions
        for dim_name, coords in model.subscripts.items():
            coords = np.array(coords)
            # create dimension
            self.ds.createDimension(dim_name, len(coords))
            # length of the longest string in the coords
            max_str_len = len(max(coords, key=len))
            # create variable for the dimension
            var = self.ds.createVariable(
                dim_name, f"S{max_str_len}", (dim_name,))
            # assigning coords to dimension
            var[:] = coords

        # creating the time dimension as unlimited
        self.ds.createDimension("time", None)
        # creating variables
        self.__create_ds_vars(model, self.capture_elements_step)

    def update(self, model):
        """
        Writes values of cache step variables from the capture_elements
        list in the netCDF4 Dataset.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object

        Returns
        -------
        None

        """
        for key in self.capture_elements_step:
            comp = model[key]
            if isinstance(comp, xr.DataArray):
                self.ds[key][self.step, :] = comp.values
            else:
                self.ds[key][self.step] = comp

        self.__update_step()

    def __update_run_elements(self, model):
        """
        Writes values of cache run elements from the cature_elements set
        in the netCDF4 Dataset.
        Cache run elements do not have the time dimension.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object

        Returns
        -------
        None

        """
        for key in self.capture_elements_run:
            comp = model[key]
            if isinstance(comp, xr.DataArray):
                self.ds[key][:] = comp.values
            else:
                self.ds[key][:] = comp

    def postprocess(self, **kwargs):
        """
        Closes netCDF4 Dataset.

        Returns
        -------
        None
        """
        self.ds.close()
        print(f"Results stored in {self.out_file}")

    def add_run_elements(self, model):
        """
        Adds constant elements to netCDF4 Dataset.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object

        Returns
        -------
        None

        """
        # creating variables in capture_elements
        self.__create_ds_vars(model, self.capture_elements_run, time_dim=False)
        self.__update_run_elements(model)

    def __create_ds_vars(self, model, capture_elements, time_dim=True):
        """
        Create new variables in a netCDF4 Dataset from the capture_elements.
        Data is zlib compressed by default for netCDF4 1.6.0 and above.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object.
        capture_elements: list
            List of variable or parameter names to include as variables in the
            dataset.
        time_dim: bool
            Whether to add time as the first dimension for the variable.

        Returns
        -------
        None

        """
        kwargs = dict()

        if tuple(self.nc.__version__.split(".")) >= ('1', '6', '0'):
            kwargs["compression"] = "zlib"

        for key in capture_elements:
            comp = model[key]

            dims = tuple()
            if isinstance(comp, xr.DataArray):
                dims = tuple(comp.dims)
            if time_dim:
                dims = ("time",) + dims

            var = self.ds.createVariable(key, "f8", dims, **kwargs)
            # adding metadata for each var from the model.doc
            for col in model.doc.columns:
                if col in ["Subscripts", "Limits"]:
                    # pass those that cannot be saved as attributes
                    continue
                var.setncattr(
                    col,
                    model.doc.loc[model.doc["Py Name"] == key, col].values[0]
                    or "Missing"
                    )


class DataFrameHandler(OutputHandlerInterface):
    """
    Manages simulation results stored as pandas DataFrame.
    """
    def __init__(self, next):
        super().__init__(next)
        self.ds = None
        self.out_file = None

    def process_output(self, out_file):
        """
        If this handler can process out_file, it returns True, else False.
        DataFrameHandler handles outputs to be saved as *.csv or *.tab files,
        and is the default handler when no output file is passed by the user.

        Parameters
        ----------
        out_file: str or pathlib.Path
            Path to the file where the results will be written.

        Returns
        -------
        None or DataFrameHandler instance

        """

        if not out_file:
            self.out_file = None
            return self

        self.out_file = Path(out_file)

        if self.out_file.suffix in [".csv", ".tab"]:
            return self

    def initialize(self, model):
        """
        Creates a pandas DataFrame and adds model variables as columns.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object

        Returns
        -------
        None

        """
        self.ds = pd.DataFrame(columns=self.capture_elements_step)

    def update(self, model):
        """
        Add a row to the results pandas DataFrame with the values of the
        variables listed in capture_elements.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object

        Returns
        -------
        None

        """
        self.ds.loc[model.time.round()] = [
            getattr(model.components, key)()
            for key in self.capture_elements_step]

    def postprocess(self, **kwargs):
        """
        Delete time column from the pandas DataFrame and flatten xarrays if
        required.

        Returns
        -------
        ds: pandas.DataFrame
            Simulation results stored as a pandas DataFrame.

        """
        # delete time column as it was created only for avoiding errors
        # of appending data. See previous TODO.
        del self.ds["time"]

        # enforce flattening if df is to be saved to csv or tab file
        flatten = True if self.out_file else kwargs.get("flatten", None)

        df = DataFrameHandler.make_flat_df(
            self.ds, kwargs["return_addresses"], flatten
            )
        if self.out_file:
            NCFile.df_to_text_file(df, self.out_file)

        return df

    def add_run_elements(self, model):
        """
        Adds constant elements to a dataframe.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object

        Returns
        -------
        None

        """
        nx = len(self.ds.index)
        for element in self.capture_elements_run:
            self.ds[element] = [getattr(model.components, element)()] * nx

    @staticmethod
    def make_flat_df(df, return_addresses, flatten=False):
        """
        Takes a dataframe from the outputs of the integration processes,
        renames the columns as the given return_adresses and splits xarrays
        if needed.

        Parameters
        ----------
        df: pandas.DataFrame
            Dataframe to process.

        return_addresses: dict
            Keys will be column names of the resulting dataframe, and are what
            the user passed in as 'return_columns'. Values are a tuple:
            (py_name, {coords dictionary}) which tells us where to look for the
            value to put in that specific column.

        flatten: bool (optional)
                If True, once the output dataframe has been formatted will
                split the xarrays in new columns following vensim's naming
                to make a totally flat output. Default is False.

        Returns
        -------
        new_df: pandas.DataFrame
            Formatted dataframe.

        """
        new_df = {}
        for real_name, (pyname, address) in return_addresses.items():
            if address:
                # subset the specific address
                values = [x.loc[address] for x in df[pyname].values]
            else:
                # get the full column
                values = df[pyname].to_list()

            is_dataarray = len(values) != 0 and isinstance(
                values[0], xr.DataArray)

            if is_dataarray and values[0].size == 1:
                # some elements are returned as 0-d arrays, convert
                # them to float
                values = [float(x) for x in values]
                is_dataarray = False

            if flatten and is_dataarray:
                DataFrameHandler.__add_flat(new_df, real_name, values)
            else:
                new_df[real_name] = values

        return pd.DataFrame(index=df.index, data=new_df)

    @staticmethod
    def __add_flat(savedict, name, values):
        """
        Add float lists from a list of xarrays to a provided dictionary.

        Parameters
        ----------
        savedict: dict
            Dictionary to save the data on.

        name: str
            The base name of the variable to save the data.

        values: list
            List of xarrays to convert to split in floats.

        Returns
        -------
        None

        """
        # remove subscripts from name if given
        name = re.sub(r'\[.*\]', '', name)
        dims = values[0].dims

        # split values in xarray.DataArray
        lval = [xrsplit(val) for val in values]
        for i, ar in enumerate(lval[0]):
            vals = [float(v[i]) for v in lval]
            subs = '[' + ','.join([str(ar.coords[dim].values)
                                   for dim in dims]) + ']'
            savedict[name+subs] = vals
