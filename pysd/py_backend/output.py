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

import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc

from pysd._version import __version__

from . import utils


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
    valid_output_files = [".nc"]

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
            raise ValueError(
                    f"Unsupported output file format {out_file.suffix}")
        return DataFrameHandler()

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

    def update(self, model, capture_elements):
        """
        Update the results object at each iteration at which resutls are
        stored.
        """
        raise NotImplementedError

    def postprocess(self, **kwargs):
        """
        Perform different tasks at the time of returning the results object.
        """
        raise NotImplementedError

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
        self.step = 0
        self.ds = None

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
        self.ds.description = "Results for simulation run on" \
            f"{t.ctime(t.time())} using PySD version {__version__}"
        self.ds.model_file = model.py_model_file
        self.ds.timestep = f"{model.components.time_step()}"
        self.ds.initial_time = f"{model.components.initial_time()}"
        self.ds.final_time = f"{model.components.final_time()}"

        # creating variables for all model dimensions
        for dim_name, coords in model.subscripts.items():
            coords = np.array(coords)
            # create dimension
            self.ds.createDimension(dim_name, len(coords))
            # length of the longest string in the coords
            max_str_len = len(max(coords, key=len))

            # create variable
            # TODO: check if the type could be defined otherwise)
            var = self.ds.createVariable(dim_name, f"S{max_str_len}",
                  (dim_name,))
            # assigning values to variable
            var[:] = coords

        # creating the time dimension as unlimited
        self.ds.createDimension("time", None)

        # creating variables in capture_elements
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

            comp = getattr(model.components, key)
            comp_vals = comp()

            if "time" in self.ds[key].dimensions:
                if isinstance(comp_vals, xr.DataArray):
                    self.ds[key][self.step, :] = comp_vals.values
                elif isinstance(comp_vals, np.ndarray):
                    self.ds[key][self.step, :] = comp_vals
                else:
                    self.ds[key][self.step] = comp_vals
            else:
                try: # this issue can arise with external objects
                    if isinstance(comp_vals, xr.DataArray):
                        self.ds[key][:] = comp_vals.values
                    elif isinstance(comp_vals, np.ndarray):
                        if comp_vals.size == 1:
                            self.ds[key][:] = comp_vals
                        else:
                            self.ds[key][:] = comp_vals
                    else:
                        self.ds[key][:] = comp_vals
                except ValueError:
                    warnings.warn(f"The dimensions of {key} in the results "
                    "do not match the declared dimensions for this "
                    "variable. The resulting values will not be "
                    "included in the results file.")

        self.step += 1

    def postprocess(self, **kwargs):
        """
        Closes netCDF4 Dataset.

        Returns
        -------
        None
        """

        # close Dataset
        self.ds.close()

        if kwargs.get("flatten"):
            warnings.warn("DataArrays stored in netCDF4 will not be flattened")

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
        # TODO we are looping through all capture elements twice. This
        # could be avoided
        self.__create_ds_vars(model, capture_elements, add_time=False)

        self.update(model, capture_elements)

    def __create_ds_vars(self, model, capture_elements, add_time=True):
        """
        Create new variables in a netCDF4 Dataset from the capture_elements.

        Parameters
        ----------
        model: pysd.Model
            PySD Model object
        capture_elements: set
            Which model elements to capture - uses pysafe names.
        add_time: bool
            Whether to add a time as the first dimension for the variables.

        Returns
        -------
        None

        """

        for key in capture_elements:
            comp = getattr(model.components, key)
            comp_vals = comp()

            dims = ()

            if isinstance(comp_vals, (xr.DataArray, np.ndarray)):
                if comp.subscripts:
                    dims = tuple(comp.subscripts)

            if add_time:
                dims = ("time",) + dims

            self.ds.createVariable(key, "f8", dims, compression="zlib")

            # adding units and description as metadata for each var
            self.ds[key].units = model.doc.loc[
                        model.doc["Py Name"] == key,
                        "Units"].values[0] or "Missing"
            self.ds[key].description = model.doc.loc[
                        model.doc["Py Name"] == key,
                        "Comment"].values[0] or "Missing"


class DataFrameHandler(OutputHandlerInterface):
    """
    Manages simulation results stored as pandas DataFrame.
    """
    def __init__(self):
        self.ds = None

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
        self.ds =  pd.DataFrame(columns=capture_elements)

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

        return utils.make_flat_df(self.ds,
                                  kwargs["return_addresses"],
                                  kwargs["flatten"])

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
