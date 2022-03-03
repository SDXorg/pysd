import warnings

import numpy as np
import xarray as xr

from . import utils


class Lookups(object):
    # TODO add __init__ and use this class for used input pandas.Series
    # as Lookups
    # def __init__(self, data, coords, interp="interpolate"):

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
            if self.is_float:
                # if lookup has no-coords return a float
                return float(outdata)
            else:
                # Remove lookup dimension coord from the DataArray
                return outdata.reset_coords('lookup_dim', drop=True)


class HardcodedLookups(Lookups):
    """Class for lookups defined in the file"""

    def __init__(self, x, y, coords, py_name):
        # TODO: avoid add and merge all declarations in one definition
        self.is_float = not bool(coords)
        self.py_name = py_name
        y = np.array(y).reshape((len(x),) + (1,)*len(coords))
        self.data = xr.DataArray(
            np.tile(y, [1] + utils.compute_shape(coords)),
            {"lookup_dim": x, **coords},
            ["lookup_dim"] + list(coords)
        )
        self.x = set(x)

    def add(self, x, y, coords):
        y = np.array(y).reshape((len(x),) + (1,)*len(coords))
        self.data = self.data.combine_first(
            xr.DataArray(
                np.tile(y, [1] + utils.compute_shape(coords)),
                {"lookup_dim": x, **coords},
                ["lookup_dim"] + list(coords)
            ))
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
