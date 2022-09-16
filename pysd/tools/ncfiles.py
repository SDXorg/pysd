from csv import QUOTE_NONE
import xarray as xr
import pandas as pd
import itertools
from pysd.py_backend.utils import xrsplit

def nc_to_csv(ncfile, outfile="result.csv", subset=None):

    ds = xr.open_dataset(ncfile)

    savedict = {}

    for name, da in ds.data_vars.items():
        # TODO only if in subset
        #if name in subset:
        dims = da.dims

        if not da.dims:
            savedict[name] = da.values.tolist()
        else:
            # TODO make a function that returns coordinates of all elements
            # instead of using xrsplit, because in the end we just use it to
            # index the DataArray directly
            split_array = xrsplit(da)
            for elem in split_array:
                coords = {dim: str(elem.coords[dim].values) for dim in dims if dim != "time"}
                subs = "[" + ",".join(coords.values()) + "]"
                savedict[name + subs] = da.loc[coords].values

    df = pd.DataFrame(savedict)

    if outfile.endswith(".csv"):
        sep = ","
    elif outfile.endswith(".tab"):
        sep = "\t"
    else:
        raise ValueError("Wrong export file type")

    df.to_csv(outfile, sep, index_label="Time", quoting=QUOTE_NONE, escapechar="\\")




