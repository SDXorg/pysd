import warnings

import pandas as pd
import numpy as np
import xarray as xr

from . import utils


class Lookups(object):
    def set_values(self, values):
        """Set new values from user input"""
        self.data = xr.DataArray(
            np.nan, self.final_coords, list(self.final_coords))

        if isinstance(values, pd.Series):
            index = list(values.index)
            index.sort()
            self.data = self.data.expand_dims(
                {'lookup_dim': index}, axis=0).copy()

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

    def __call__(self, x, final_subs=None):
        try:
            return self._call(self.data, x, final_subs)
        except (TypeError, KeyError):
            # this except catch the errors when a lookups has been
            # changed to a constant value by the user
            if final_subs and isinstance(self.data, xr.DataArray):
                # self.data is an array, reshape it
                outdata = xr.DataArray(np.nan, final_subs, list(final_subs))
                return xr.broadcast(outdata, self.data)[1]
            elif final_subs:
                # self.data is a float, create an array
                return xr.DataArray(self.data, final_subs, list(final_subs))
            else:
                return self.data

    def _call(self, data, x, final_subs=None):
        if isinstance(x, xr.DataArray):
            if not x.dims:
                # shape 0 xarrays
                return self._call(data, float(x))

            outdata = xr.DataArray(np.nan, final_subs, list(final_subs))

            if self.interp != "extrapolate" and\
               np.all(x > data['lookup_dim'].values[-1]):
                outdata_ext = data[-1]
                warnings.warn(
                  self.py_name + "\n"
                  + "extrapolating data above the maximum value of the series")
            elif self.interp != "extrapolate" and\
              np.all(x < data['lookup_dim'].values[0]):
                outdata_ext = data[0]
                warnings.warn(
                  self.py_name + "\n"
                  + "extrapolating data below the minimum value of the series")
            else:
                data = xr.broadcast(data, x)[0]
                for a in utils.xrsplit(x):
                    outdata.loc[a.coords] = self._call(
                        data.loc[a.coords],
                        float(a))
                return outdata

            # return the final array in the specified dimensions order
            return xr.broadcast(
                outdata, outdata_ext.reset_coords('lookup_dim', drop=True))[1]

        else:
            if x in data['lookup_dim'].values:
                outdata = data.sel(lookup_dim=x)
            elif x > data['lookup_dim'].values[-1]:
                if self.interp == "extrapolate":
                    # extrapolate method for xmile models
                    k = (data[-1]-data[-2])\
                        / (data['lookup_dim'].values[-1]
                           - data['lookup_dim'].values[-2])
                    outdata = data[-1] + k*(x - data['lookup_dim'].values[-1])
                else:
                    outdata = data[-1]
                warnings.warn(
                  self.py_name + "\n"
                  + "extrapolating data above the maximum value of the series")
            elif x < data['lookup_dim'].values[0]:
                if self.interp == "extrapolate":
                    # extrapolate method for xmile models
                    k = (data[1]-data[0])\
                        / (data['lookup_dim'].values[1]
                           - data['lookup_dim'].values[0])
                    outdata = data[0] + k*(x - data['lookup_dim'].values[0])
                else:
                    outdata = data[0]
                warnings.warn(
                  self.py_name + "\n"
                  + "extrapolating data below the minimum value of the series")
            elif self.interp == 'hold_backward':
                outdata = data.sel(lookup_dim=x, method="pad")
            else:
                outdata = data.interp(lookup_dim=x)

            # the output could be a float or an xarray
            if self.is_float:
                # if lookup has no-coords return a float
                return float(outdata)
            else:
                # Remove lookup dimension coord from the DataArray
                return outdata.reset_coords('lookup_dim', drop=True)


class HardcodedLookups(Lookups):
    """Class for lookups defined in the file"""

    def __init__(self, x, y, coords, interp, final_coords, py_name):
        # TODO: avoid add and merge all declarations in one definition
        self.is_float = not bool(coords)
        self.py_name = py_name
        self.final_coords = final_coords
        self.values = [(x, y, coords)]
        self.interp = interp

    def add(self, x, y, coords):
        self.values.append((x, y, coords))

    def initialize(self):
        """
        Initialize all elements and create the self.data xarray.DataArray
        """
        if len(self.values) == 1:
            # Just loag one value (no add)
            for x, y, coords in self.values:
                if len(x) != len(set(x)):
                    raise ValueError(
                        self.py_name + "\n"
                        "x dimension has repeated values..."
                    )
                try:
                    y = np.array(y).reshape((len(x),) + (1,)*len(coords))
                except ValueError:
                    raise ValueError(
                        self.py_name + "\n"
                        "x and y dimensions have different length..."
                    )
                self.data = xr.DataArray(
                    np.tile(y, [1] + utils.compute_shape(coords)),
                    {"lookup_dim": x, **coords},
                    ["lookup_dim"] + list(coords)
                )
        else:
            # Load in several lines (add)
            self.data = xr.DataArray(
                np.nan, self.final_coords, list(self.final_coords))

            for x, y, coords in self.values:
                if "lookup_dim" not in self.data.dims:
                    # include lookup_dim dimension in the final array
                    self.data = self.data.expand_dims(
                        {"lookup_dim": x}, axis=0).copy()
                else:
                    # add new coordinates (if needed) to lookup_dim
                    x_old = list(self.data.lookup_dim.values)
                    x_new = list(set(x).difference(x_old))
                    self.data = self.data.reindex(lookup_dim=x_old+x_new)

                # reshape y value and assign it to self.data
                y = np.array(y).reshape((len(x),) + (1,)*len(coords))
                self.data.loc[{"lookup_dim": x, **coords}] =\
                    np.tile(y, [1] + utils.compute_shape(coords))

        # sort data
        self.data = self.data.sortby("lookup_dim")

        if np.any(np.isnan(self.data)):
            # fill missing values of different input lookup_dim values
            values = self.data.values
            self._fill_missing(self.data.lookup_dim.values, values)
            self.data = xr.DataArray(values, self.data.coords, self.data.dims)

    def _fill_missing(self, series,  data):
        """
        Fills missing values in lookups to have a common series.
        Mutates the values in data.

        Returns
        -------
        None

        """
        if len(data.shape) > 1:
            # break the data array until arrive to a vector
            for i in range(data.shape[1]):
                if np.any(np.isnan(data[:, i])):
                    self._fill_missing(series, data[:, i])
        elif not np.all(np.isnan(data)):
            # interpolate missing values
            data[np.isnan(data)] = np.interp(
                series[np.isnan(data)],
                series[~np.isnan(data)],
                data[~np.isnan(data)])
