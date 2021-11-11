"""
Model components and time managing classes.
"""

import os
import warnings
import re
import random
from importlib.machinery import SourceFileLoader

import numpy as np
import xarray as xr

from pysd._version import __version__
from .utils import load_outputs, get_columns_to_load


class Components(object):
    """
    Workaround class to let the user do:
        model.components.var = value
    """
    def __init__(self, py_model_file, set_components):
        object.__setattr__(self, "_components", self._load(py_model_file))
        object.__setattr__(self, "_set_components", set_components)

    def _load(self, py_model_file):
        """
        Load model components.

        Parameters
        ----------
        py_model_file: str
            Model file to be loaded.

        Returns
        -------
        components: module
            The imported file content.

        """
        # need a unique identifier for the imported module.
        module_name = os.path.splitext(py_model_file)[0]\
            + str(random.randint(0, 1000000))
        try:
            return SourceFileLoader(
                module_name, py_model_file).load_module()
        except TypeError:
            raise ImportError(
                "\n\nNot able to import the model. "
                + "This may be because the model was compiled with an "
                + "earlier version of PySD, you can check on the top of "
                + " the model file you are trying to load."
                + "\nThe current version of PySd is :"
                + "\n\tPySD " + __version__ + "\n\n"
                + "Please translate again the model with the function"
                + " read_vensim or read_xmile.")

    def __getattribute__(self, name):
        """
        Get attribute from the class. Try Except vlock is used to load directly
        model components in order to avoid making the model slower during the
        integration.
        """
        try:
            return getattr(object.__getattribute__(self, "_components"), name)
        except AttributeError:
            if name in ["_components", "_set_components",
                        "_set_component", "_load"]:
                # The attribute is from the class Components
                return object.__getattribute__(self, name)
            else:
                raise NameError(f"Component '{name}' not found in the model.")

    def __setattr__(self, name, value):
        """
        Workaround calling the Macro._set_components method
        """
        self._set_components({name: value})

    def _set_component(self, name, value):
        """
        Replaces the previous setter.
        """
        setattr(
            object.__getattribute__(self, "_components"),
            name,
            value
        )


class Time(object):
    def __init__(self):
        self._time = None
        self.stage = None
        self.return_timestamps = None

    def __call__(self):
        return self._time

    def set_control_vars(self, **kwargs):
        """
        Set the control variables valies

        Parameters
        ----------
        **kwards:
            initial_time: float, callable or None
                Initial time.
            final_time: float, callable or None
                Final time.
            time_step: float, callable or None
                Time step.
            saveper: float, callable or None
                Saveper.

        """
        def _convert_value(value):
            # this function is necessary to avoid copying the pointer in the
            # lambda function.
            if callable(value):
                return value
            else:
                return lambda: value

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, _convert_value(value))

        if "initial_time" in kwargs:
            self._initial_time = self.initial_time()
            self._time = self.initial_time()

    def in_bounds(self):
        """
        Check if time is smaller than current final time value.

        Returns
        -------
        bool:
            True if time is smaller than final time. Otherwise, returns Fase.

        """
        return self._time < self.final_time()

    def in_return(self):
        """ Check if current time should be returned """
        if self.return_timestamps is not None:
            return self._time in self.return_timestamps

        time_delay = self._time - self._initial_time
        save_per = self.saveper()
        prec = self.time_step() * 1e-10
        return time_delay % save_per < prec or -time_delay % save_per < prec

    def add_return_timestamps(self, return_timestamps):
        """ Add return timestamps """
        if return_timestamps is None or hasattr(return_timestamps, '__len__'):
            self.return_timestamps = return_timestamps
        else:
            self.return_timestamps = [return_timestamps]

    def update(self, value):
        """ Update current time value """
        self._time = value

    def reset(self):
        """ Reset time value to the initial """
        self._time = self._initial_time


class Data(object):
    def __init__(self, data, coords, interp="interpolate"):
        self.data = data
        self.interp = "interpolate"
        self.is_float = not bool(coords)

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

        if self.is_float:
            # if data has no-coords return a float
            return float(outdata)
        else:
            # Remove time coord from the DataArray
            return outdata.reset_coords('time', drop=True)


class RegData(Data):
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
        file_names: list
            Name of the files to search the variable in.

        Returns
        -------
        out: xarray.DataArray
            Resulting data array with the time in the first dimension.

        """

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
        # get columns to load variable
        columns = get_columns_to_load(
            file_name, False, vars=[self.real_name, self.py_name])

        if not columns:
            # try reading transposed file
            columns = get_columns_to_load(
                file_name, True, vars=[self.real_name, self.py_name])

            if not columns:
                # variable not found
                return None
            else:
                # variable must be read from a transposed file
                transpose = True
        else:
            # variable found
            transpose = False

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
