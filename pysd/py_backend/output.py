"""
ModelOutput class is used to build different output objects based on
user input. For now, available output types are pandas DataFrame or
netCDF4 Dataset.
The OutputHandlerInterface class is an interface for the creation of handlers
for other output object types.
"""
import abc
import warnings
import time as t

from csv import QUOTE_NONE

import regex as re

import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc

from pysd._version import __version__

from . utils import xrsplit


class ModelOutput():
    """
    Handles different types of outputs by dispatchinging the tasks to adequate
    object handlers.

    Parameters
    ----------
    model: pysd.Model
        PySD Model object
    capture_elements: set
        Which model elements to capture - uses pysafe names.
    out_file: str or pathlib.Path
        Path to the file where the results will be written.
    """
    valid_output_files = [".nc", ".csv", ".tab"]

    def __init__(self, model, capture_elements, out_file=None):

        self.handler = self.__handle(out_file)

        capture_elements.add("time")
        self.capture_elements = capture_elements

        self.initialize(model)

    def __handle(self, out_file):
        # TODO improve the handler to avoid if then else statements
        if out_file:
            if out_file.suffix == ".nc":
                return DatasetHandler(out_file)
        # when the users expects a csv or tab output file, it defaults to the
        # DataFrame path
        return DataFrameHandler(out_file)

    def initialize(self, model):
        """ Delegating the creation of the results object and its elements to
        the appropriate handler."""
        self.handler.initialize(model, self.capture_elements)

    def update(self, model):
        """ Delegating the update of the results object and its elements to the
        appropriate handler."""
        self.handler.update(model, self.capture_elements)

    def postprocess(self, **kwargs):
        """ Delegating the postprocessing of the results object to the
        appropriate handler."""
        return self.handler.postprocess(**kwargs)

    def add_run_elements(self, model, run_elements):
        """ Delegating the addition of results with run cache in the output
        object to the appropriate handler."""
        self.handler.add_run_elements(model, run_elements)


class OutputHandlerInterface(metaclass=abc.ABCMeta):
    """
    Interface for the creation of different output type handlers.
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'initialize') and
                callable(subclass.initialize) and
                hasattr(subclass, 'update') and
                callable(subclass.update) and
                hasattr(subclass, 'postprocess') and
                callable(subclass.postprocess) and
                hasattr(subclass, 'add_run_elements') and
                callable(subclass.add_run_elements) or
                NotImplemented)

    @abc.abstractmethod
    def initialize(self, model, capture_elements):
        """
        Create the results object and its elements based on capture_elemetns.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, model, capture_elements):
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
    def add_run_elements(self, model, capture_elements):
        """
        Add elements with run cache to the results object.
        """
        raise NotImplementedError


class DatasetHandler(OutputHandlerInterface):
    """
    Manages simulation results stored as netCDF4 Dataset.
    """

    def __init__(self, out_file):
        self.out_file = out_file
        self.ds = None
        self._step = 0

    @property
    def step(self):
        return self._step

    def __update_step(self):
        self._step = self.step + 1

    def initialize(self, model, capture_elements):
        """
        Creates a netCDF4 Dataset and adds model dimensions and variables
        present in the capture elements to it.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object
        capture_elements: set
            Which model elements to capture - uses pysafe names.

        Returns
        -------
        None

        """
        self.ds = nc.Dataset(self.out_file, "w")

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
        self.__create_ds_vars(model, capture_elements)

    def update(self, model, capture_elements):
        """
        Writes values of variables in capture_elements in the netCDF4 Dataset.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object
        capture_elements: set
            Which model elements to capture - uses pysafe names.

        Returns
        -------
        None
        """
        for key in capture_elements:

            comp = model[key]

            if "time" in self.ds[key].dimensions:
                if isinstance(comp, xr.DataArray):
                    self.ds[key][self.step, :] = comp.values
                else:
                    self.ds[key][self.step] = comp
            else:
                try:  # this issue can arise with external objects
                    if isinstance(comp, xr.DataArray):
                        self.ds[key][:] = comp.values
                    else:
                        self.ds[key][:] = comp
                except ValueError:
                    warnings.warn(
                        f"The dimensions of {key} in the results "
                        "do not match the declared dimensions for this "
                        "variable. The resulting values will not be "
                        "included in the results file.")

        self.__update_step()

    def postprocess(self, **kwargs):
        """
        Closes netCDF4 Dataset.

        Returns
        -------
        None
        """

        # close Dataset
        self.ds.close()

        print(f"Results stored in {self.out_file}")

    def add_run_elements(self, model, capture_elements):
        """
        Adds constant elements to netCDF4 Dataset.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object
        capture_elements: list
            List of constant elements

        Returns
        -------
        None
        """
        # creating variables in capture_elements
        self.__create_ds_vars(model, capture_elements, time_dim=False)

        self.update(model, capture_elements)

    def __create_ds_vars(self, model, capture_elements, time_dim=True):
        """
        Create new variables in a netCDF4 Dataset from the capture_elements.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object
        capture_elements: set
            Which model elements to capture - uses pysafe names.
        time_dim: bool
            Whether to add time as the first dimension for the variable.

        Returns
        -------
        None

        """
        for key in capture_elements:
            comp = model[key]

            dims = ()

            if isinstance(comp, xr.DataArray):
                dims = tuple(comp.dims)

            if time_dim:
                dims = ("time",) + dims

            var = self.ds.createVariable(key, "f8", dims, compression="zlib")

            # adding metadata for each var from the model.doc
            for col in model.doc.columns:
                var.setncattr(
                    col,
                    model.doc.loc[model.doc["Py Name"] == key, col].values[0] \
                        or "Missing"
                    )


class DataFrameHandler(OutputHandlerInterface):
    """
    Manages simulation results stored as pandas DataFrame.
    """
    def __init__(self, out_file):
        self.ds = None
        self.output_file = out_file

    def initialize(self, model, capture_elements):
        """
        Creates a pandas DataFrame and adds model variables as columns.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object
        capture_elements: set
            Which model elements to capture - uses pysafe names.

        Returns
        -------
        None

        """
        self.ds = pd.DataFrame(columns=capture_elements)

    def update(self, model, capture_elements):
        """
        Add a row to the results pandas DataFrame with the values of the
        variables listed in capture_elements.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object
        capture_elements: set
            Which model elements to capture - uses pysafe names.

        Returns
        -------
        None
        """

        self.ds.at[model.time.round()] = [
            getattr(model.components, key)()
            for key in capture_elements]

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
        flatten = True if self.output_file else kwargs.get("flatten", None)

        df = DataFrameHandler.make_flat_df(
            self.ds, kwargs["return_addresses"], flatten
            )
        if self.output_file:
            self.__save_to_file(df)

        return df

    def __save_to_file(self, output):
        """
        Saves models output.

        Paramters
        ---------
        output: pandas.DataFrame

        options: argparse.Namespace

        Returns
        -------
        None

        """

        if self.output_file.suffix == ".tab":
            sep = "\t"
        else:
            sep = ","

        # QUOTE_NONE used to print the csv/tab files as vensim does with
        # special characterse, e.g.: "my-var"[Dimension]
        output.to_csv(
            self.output_file, sep, index_label="Time", quoting=QUOTE_NONE
            )

        print(f"Data saved in '{self.output_file}'")

    def add_run_elements(self, model, capture_elements):
        """
        Adds constant elements to a dataframe.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object
        capture_elements: set
            Which model elements to capture - uses pysafe names.

        Returns
        -------
        None
        """
        nx = len(self.ds.index)
        for element in capture_elements:
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
