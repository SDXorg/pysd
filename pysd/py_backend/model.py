"""
Macro and Model classes are the main classes for loading and interacting
with a PySD model. Model class allows loading and running a PySD model.
Several methods and propierties are inherited from Macro class, which
allows integrating a model or a Macro expression (set of functions in
a separate file).
"""
import time
import warnings
import inspect
import pickle
from pathlib import Path
from copy import deepcopy
from typing import Union

import numpy as np
import xarray as xr
import pandas as pd

from pysd._version import __version__

from . import utils
from .statefuls import DynamicStateful, Stateful
from .external import External, Excels, ExtLookup, ExtData

from .cache import Cache, constant_cache
from .data import TabData
from .lookups import HardcodedLookups
from .components import Components, Time
from .output import ModelOutput


class Macro(DynamicStateful):
    """
    The Macro class implements a stateful representation of the system,
    and contains the majority of methods for accessing and modifying
    components.

    When the instance in question also serves as the root model object
    (as opposed to a macro or submodel within another model) it will have
    added methods to facilitate execution.

    The Macro object will be created with components drawn from a
    translated Python model file.

    Parameters
    ----------
    py_model_file: str or pathlib.Path
        Filename of a model or macro which has already been converted
        into a Python format.
    params: dict or None (optional)
        Dictionary of the macro parameters. Default is None.
    return_func: str or None (optional)
        The name of the function to return from the macro. Default is None.
    time: components.Time or None (optional)
        Time object for integration. If None a new time object will
        be generated (for models), if passed the time object will be
        used (for macros). Default is None.
    time_initialization: callable or None
        Time to set at the begginning of the Macro. Default is None.
    data_files: dict or list or str or None
        The dictionary with keys the name of file and variables to
        load the data from. Or the list of names or name of the file
        to search the data in. Only works for TabData type object
        and it is neccessary to provide it. Default is None.
    py_name: str or None
        The name of the Macro object. Default is None.

    See also
    --------
    :class:`pysd.py_backend.model.Model`

    """
    def __init__(self, py_model_file, params=None, return_func=None,
                 time=None, time_initialization=None, data_files=None,
                 py_name=None):
        super().__init__()
        self.time = time
        self.time_initialization = time_initialization
        # Initialize the cache object
        self.cache = Cache()
        # Python name of the object (for Macros)
        self.py_name = py_name
        # Booleans to avoid loading again external data or lookups
        self.external_loaded = False
        self.lookups_loaded = False
        # Functions with constant cache
        self._constant_funcs = set()
        # Attributes that are set later
        self.stateful_initial_dependencies = None
        self.initialize_order = None
        self.cache_type = None
        self._components_setter_tracker = {}
        # Load model/macro from file and save in components
        self.components = Components(str(py_model_file), self.set_components)

        if __version__.split(".")[0]\
           != self.get_pysd_compiler_version().split(".")[0]:
            raise ImportError(
                "\n\nNot able to import the model. "
                + "The model was translated with a "
                + "not compatible version of PySD:"
                + "\n\tPySD " + self.get_pysd_compiler_version()
                + "\n\nThe current version of PySd is:"
                + "\n\tPySD " + __version__ + "\n\n"
                + "Please translate again the model with the function"
                + " read_vensim or read_xmile.")

        # Assing some protected attributes for easier access
        self._namespace = self.components._components.component.namespace
        self._dependencies =\
            self.components._components.component.dependencies.copy()
        self._subscript_dict = getattr(
            self.components._components, "_subscript_dict", {})
        self._modules = getattr(
            self.components._components, "_modules", {})

        self._doc = self._build_doc()

        if params is not None:
            # add params to namespace
            self._namespace.update(self.components._components._params)
            # create new components with the params
            self._set_components(params, new=True)
            # update dependencies
            for param in params:
                self._dependencies[
                    self._namespace[param]] = {"time"}

        # Get the collections of stateful elements and external elements
        self._stateful_elements = {
            name: getattr(self.components, name)
            for name in dir(self.components)
            if isinstance(getattr(self.components, name), Stateful)
        }
        self._dynamicstateful_elements = [
            getattr(self.components, name) for name in dir(self.components)
            if isinstance(getattr(self.components, name), DynamicStateful)
        ]
        self._external_elements = [
            getattr(self.components, name) for name in dir(self.components)
            if isinstance(getattr(self.components, name), External)
        ]
        self._macro_elements = [
            getattr(self.components, name) for name in dir(self.components)
            if isinstance(getattr(self.components, name), Macro)
        ]

        self._data_elements = [
            getattr(self.components, name) for name in dir(self.components)
            if isinstance(getattr(self.components, name), TabData)
        ]

        self._lookup_elements = [
            getattr(self.components, name) for name in dir(self.components)
            if isinstance(getattr(self.components, name), HardcodedLookups)
        ]

        # Load data files
        if data_files:
            self._get_data(data_files)

        # Assign the cache type to each variable
        self._assign_cache_type()
        # Get the initialization order of Stateful elements
        self._get_initialize_order()

        if return_func is not None:
            # Assign the return value of Macros
            self.return_func = getattr(self.components, return_func)
        else:
            self.return_func = lambda: 0

        self.py_model_file = str(py_model_file)

    def __call__(self):
        return self.return_func()

    @property
    def doc(self) -> pd.DataFrame:
        """
        The documentation of the model.
        """
        return self._doc.copy()

    @property
    def namespace(self) -> dict:
        """
        The namespace dictionary of the model.
        """
        return self._namespace.copy()

    @property
    def dependencies(self) -> dict:
        """
        The dependencies dictionary of the model.
        """
        return self._dependencies.copy()

    @property
    def subscripts(self) -> dict:
        """
        The subscripts dictionary of the model.
        """
        return self._subscript_dict.copy()

    @property
    def modules(self) -> Union[dict, None]:
        """
        The dictionary of modules of the model. If the model is not
        split by modules it returns None.
        """
        return self._modules.copy() or None

    def clean_caches(self):
        """
        Clean the cache of the object and the macros objects that it
        contains
        """
        self.cache.clean()
        # if nested macros
        [macro.clean_caches() for macro in self._macro_elements]

    def _get_data(self, data_files):
        if isinstance(data_files, dict):
            for data_file, vars in data_files.items():
                for var in vars:
                    found = False
                    for element in self._data_elements:
                        if var in [element.py_name, element.real_name]:
                            element.load_data(data_file)
                            found = True
                            break
                    if not found:
                        raise ValueError(
                            f"'{var}' not found as model data variable")

        else:
            for element in self._data_elements:
                element.load_data(data_files)

    def _get_initialize_order(self):
        """
        Get the initialization order of the stateful elements
        and their the full dependencies.
        """
        # get the full set of dependencies to initialize an stateful object
        # includying all levels
        self.stateful_initial_dependencies = {
            ext: set()
            for ext in self._dependencies
            if (ext.startswith("_") and not ext.startswith("_active_initial_"))
        }
        for element in self.stateful_initial_dependencies:
            self._get_full_dependencies(
                element, self.stateful_initial_dependencies[element],
                "initial")

        # get the full dependencies of stateful objects taking into account
        # only other objects
        current_deps = {
            element: [
                dep for dep in deps
                if dep in self.stateful_initial_dependencies
            ] for element, deps in self.stateful_initial_dependencies.items()
        }

        # get initialization order of the stateful elements
        self.initialize_order = []
        delete = True
        while delete:
            delete = []
            for element in current_deps:
                if not current_deps[element]:
                    # if stateful element has no deps on others
                    # add to the queue to initialize
                    self.initialize_order.append(element)
                    delete.append(element)
                    for element2 in current_deps:
                        # remove dependency on the initialized element
                        if element in current_deps[element2]:
                            current_deps[element2].remove(element)
            # delete visited elements
            for element in delete:
                del current_deps[element]

        if current_deps:
            # if current_deps is not an empty set there is a circular
            # reference between stateful objects
            raise ValueError(
                'Circular initialization...\n'
                + 'Not able to initialize the following objects:\n\t'
                + '\n\t'.join(current_deps))

    def _get_full_dependencies(self, element, dep_set, stateful_deps):
        """
        Get all dependencies of an element, i.e., also get the dependencies
        of the dependencies. When finding an stateful element only dependencies
        for initialization are considered.

        Parameters
        ----------
        element: str
            Element to get the full dependencies.
        dep_set: set
            Set to include the dependencies of the element.
        stateful_deps: "initial" or "step"
            The type of dependencies to take in the case of stateful objects.

        Returns
        -------
        None

        """
        deps = self._dependencies[element]
        if element.startswith("_"):
            deps = deps[stateful_deps]
        for dep in deps:
            if dep not in dep_set and not dep.startswith("__")\
               and dep != "time":
                dep_set.add(dep)
                self._get_full_dependencies(dep, dep_set, stateful_deps)

    def _add_constant_cache(self):
        for element, cache_type in self.cache_type.items():
            if cache_type == "run":
                self.components._set_component(
                    element,
                    constant_cache(getattr(self.components, element))
                )
                self._constant_funcs.add(element)

    def _remove_constant_cache(self):
        for element in self._constant_funcs:
            self.components._set_component(
                element,
                getattr(self.components, element).function)
            # remove attributes added with constant cache
            delattr(getattr(self.components, element), 'function')
            delattr(getattr(self.components, element), 'value')
        self._constant_funcs.clear()

    def _assign_cache_type(self):
        """
        Assigns the cache type to all the elements from the namespace.
        """
        self.cache_type = {"time": None}

        for element in self._namespace.values():
            if element not in self.cache_type\
               and element in self._dependencies:
                self._assign_cache(element)

        for element, cache_type in self.cache_type.items():
            if cache_type is not None:
                if element not in self.cache.cached_funcs\
                   and self._count_calls(element) > 1:
                    self.components._set_component(
                        element,
                        self.cache(getattr(self.components, element)))
                    self.cache.cached_funcs.add(element)

    def _count_calls(self, element):
        n_calls = 0
        for subelement in self._dependencies:
            if subelement.startswith("_") and\
               element in self._dependencies[subelement]["step"]:
                if element in\
                   self._dependencies[subelement]["initial"]:
                    n_calls +=\
                        2*self._dependencies[subelement]["step"][element]
                else:
                    n_calls +=\
                        self._dependencies[subelement]["step"][element]
            elif (not subelement.startswith("_") and
                  element in self._dependencies[subelement]):
                n_calls +=\
                    self._dependencies[subelement][element]

        return n_calls

    def _assign_cache(self, element):
        """
        Assigns the cache type to the given element and its dependencies if
        needed.

        Parameters
        ----------
        element: str
            Element name.

        Returns
        -------
        None

        """
        if not self._dependencies[element]:
            self.cache_type[element] = "run"
        elif "__lookup__" in self._dependencies[element]:
            self.cache_type[element] = None
        elif self._isdynamic(self._dependencies[element]):
            self.cache_type[element] = "step"
        else:
            self.cache_type[element] = "run"
            for subelement in self._dependencies[element]:
                if subelement.startswith("_initial_")\
                   or subelement.startswith("__"):
                    continue
                if subelement not in self.cache_type:
                    self._assign_cache(subelement)
                if self.cache_type[subelement] == "step":
                    self.cache_type[element] = "step"
                    break

    def _isdynamic(self, dependencies):
        """

        Parameters
        ----------
        dependencies: iterable
            List of dependencies.

        Returns
        -------
        isdynamic: bool
            True if 'time' or a dynamic stateful objects is in dependencies.

        """
        if "time" in dependencies:
            return True
        for dep in dependencies:
            if dep.startswith("_") and not dep.startswith("_initial_")\
               and not dep.startswith("__"):
                return True
        return False

    def get_pysd_compiler_version(self):
        """
        Returns the version of pysd complier that used for generating
        this model
        """
        return self.components.__pysd_version__

    def initialize(self):
        """
        This function initializes the external objects and stateful objects
        in the given order.
        """
        # Initialize time
        if self.time is None:
            self.time = self.time_initialization()

        # Reset time to the initial one
        self.time.reset()
        self.cache.clean()

        self.components._init_outer_references({
            'scope': self,
            'time': self.time
        })

        if not self.lookups_loaded:
            # Initialize HardcodedLookups elements
            for element in self._lookup_elements:
                element.initialize()

            self.lookups_loaded = True

        if not self.external_loaded:
            # Initialize external elements
            self.initialize_external_data()

        # Initialize stateful objects
        for element_name in self.initialize_order:
            self._stateful_elements[element_name].initialize()

    def ddt(self):
        return np.array([component.ddt() for component
                         in self._dynamicstateful_elements], dtype=object)

    @property
    def state(self):
        return np.array([component.state for component
                         in self._dynamicstateful_elements], dtype=object)

    @state.setter
    def state(self, new_value):
        [component.update(val) for component, val
         in zip(self._dynamicstateful_elements, new_value)]

    def initialize_external_data(self, externals=None):

        """
        Initializes external data.

        If a path to a netCDF file containing serialized values of some or
        all of the model external data is passed in the external argument,
        those will be loaded from the file.

        To get the full performance gain of loading the externals from a netCDF
        file, the model should be loaded with initialize=False first.

        Examples of usage are available at
        `Advanced Usage <https://pysd.readthedocs.io/en/master/advanced_usage.html#initializing-external-data-from-netcdf-file>`__.

        Parameters
        ----------
        externals: str or pathlib.Path (optional)
            Path to the netCDF file that contains the model external objects.

        Returns
        -------
        None

        See also
        --------
        :func:`pysd.py_backend.model.Macro.serialize_externals`

        Note
        ----
        To load externals from a netCDF file you need to have installed
        the optional dependency `netCDF4`.

        """

        if not externals:
            for ext in self._external_elements:
                ext.initialize()

            # Remove Excel data from memory
            Excels.clean()

            self.external_loaded = True

            return

        externals = Path(externals)

        if not externals.is_file():
            raise FileNotFoundError(f"Invalid file path ({str(externals)})")

        try:
            ds = xr.open_dataset(externals)
        except ValueError:  # pragma: no cover
            raise ModuleNotFoundError("No module named 'netCDF4'")

        for ext in self._external_elements:
            if ext.py_name in ds.data_vars.keys():
                # Initialize external from nc file
                da = ds.data_vars[ext.py_name]
                if isinstance(ext, ExtData):
                    # Rename again time dimension
                    time_dim = [dim for dim in da.dims if dim.startswith(
                        "time_#")][0]
                    da = da.rename({time_dim: "time"})
                elif isinstance(ext, ExtLookup):
                    # Rename again lookup_dim dimension
                    lookup_dim = [dim for dim in da.dims if dim.startswith(
                        "lookup_dim_#")][0]
                    da = da.rename({lookup_dim: "lookup_dim"})
                # Assign the value
                if da.dims:
                    ext.data = da
                else:
                    ext.data = float(da.data)
            else:
                # Initialize external from original file
                ext.initialize()

        Excels.clean()

        self.external_loaded = True

    def serialize_externals(self, export_path="externals.nc",
                            include_externals="all", exclude_externals=None):
        """
        Stores a netCDF file with the data and metadata for all model external
        objects.

        This method is useful for models with lots of external inputs, which
        are slow to load into memory. Once exported, the resulting netCDF file
        can be passed as argument to the initialize_external_data method.

        Names of variables should be those in the model (python safe,
        without the _ext_type_ string in front).

        Examples of usage are available at
        `Advanced Usage <https://pysd.readthedocs.io/en/master/advanced_usage.html#initializing-external-data-from-netcdf-file>`__.

        Parameters
        ----------
        export_path: str or pathlib.Path (optional)
            Path of the resulting *.nc* file.
        include_externals: list or str (optional)
            External objects to export to netCDF.
            If 'all', then all externals are exported to the *.nc* file.
            The argument also accepts a list containing spreadsheet file names,
            external variable names or a combination of both. If a spreadsheet
            file path is passed, all external objects defined in it will be
            included in the *.nc* file. Spreadsheet tab names are not currently
            supported, because the same may be used in different files.
            Better customisation can be achieved by combining the
            include_externals and exclude_externals (see description below)
            arguments.
        exclude_externals: list or None (optional)
            Exclude external objects from being included in the exported nc
            file. It accepts either variable names, spreadsheet files or a
            combination of both.

        Returns
        -------
        None

        See also
        --------
        :func:`pysd.py_backend.model.Macro.initialize_external_data`

        Note
        ----
        To run this function you need to have installed the optional
        dependency `netCDF4`.

        """
        data = {}
        metadata = {}

        lookup_dims = utils.UniqueDims("lookup_dim")
        data_dims = utils.UniqueDims("time")

        if isinstance(export_path, str):
            export_path = Path(export_path)

        if not include_externals:
            raise ValueError("include_externals argument must not be None.")

        # Generate a Dataframe to make simpler the search of external
        # objects to include
        # TODO include also checking the original name
        py_names = []
        externals_dict = {"py_var_name": [], "file": [], "ext": []}
        for ext in self._external_elements:
            py_names.append(ext.py_name)
            externals_dict["py_var_name"].append(
                self.__get_varname_from_ext_name(ext.py_name))
            externals_dict["file"].append(set(ext.files))
            externals_dict["ext"].append(ext)
        exts_df = pd.DataFrame(index=py_names, data=externals_dict)

        if include_externals != "all":
            if not isinstance(include_externals, (list, set)):
                raise TypeError(
                    "include_externals must be 'all', or a list, or a set.")

            # subset only to the externals to include
            exts_df = exts_df[[
                name in include_externals
                or var_name in include_externals
                or bool(file.intersection(include_externals))
                for name, (var_name, file)
                in exts_df[["py_var_name", "file"]].iterrows()
            ]]

        if exclude_externals:
            if not isinstance(exclude_externals, (list, set)):
                raise TypeError("exclude_externals must be a list or a set.")

            # subset only to the externals to include
            exts_df = exts_df[[
                name not in exclude_externals
                and var_name not in exclude_externals
                and not bool(file.intersection(exclude_externals))
                for name, (var_name, file)
                in exts_df[["py_var_name", "file"]].iterrows()
            ]]

        for _, (ext, var_name) in exts_df[["ext", "py_var_name"]].iterrows():
            self.__include_for_serialization(
                ext, var_name, data, metadata, lookup_dims, data_dims
            )

        # create description to be used as global attribute of the dataset
        description = {
            "description": f"External objects for {self.py_model_file} "
                           f"exported on {time.ctime(time.time())} "
                           f"using PySD version {__version__}"
        }

        # create Dataset
        ds = xr.Dataset(data_vars=data, attrs=description)

        # add data_vars attributes
        for key, values in metadata.items():
            ds[key].attrs = values

        try:
            ds.to_netcdf(export_path)
        except KeyError:  # pragma: no cover
            raise ModuleNotFoundError("No module named 'netCDF4'")

    def __include_for_serialization(self, ext, py_name_clean, data, metadata,
                                    lookup_dims, data_dims):
        """
        Initialize the external object and get the data and metadata for
        inclusion in the netCDF.

        This function updates the metadata dict with the metadata corresponding
        to each external object, which is collected from model.doc. It also
        updates the data dict, with the data from the external object (ext).

        It renames the "time" dimension of ExtData types by appending _#
        followed by a unique number at the end of it. For large models, this
        prevents having an unnecessary large time dimension, which then causes
        all ExtData objects to have many nans when stored in a xarray Dataset.
        It does the same for the "lookup_dim" of all ExtLookup objects.

        Note
        ----
        Though subscripts can be read from Excel, they are hardcoded
        during the model building process. Therefore they will not be
        serialized.

        Parameters
        ----------
        ext: pysd.py_backend.externals.External
            External object. It can be any of the External subclasses
            (ExtConstant, ExtData, ExtLookup)
        py_name_clean: str
            Name of the variable without _ext_[constant|data|lookup] prefix.
        data: dict
            Collects all the data for each external, which is later used to
            build the xarray Dataset.
        metadata: dict
            Collects the metadata for each external, which is later included
            as data_vars attributes in the xarray Dataset.
        lookup_dims: utils.UniqueDims
            UniqueDims object for "lookup_dim" dimension.
        data_dims: utils.UniqueDims
            UniqueDims object for "time" dimension.

        Returns
        -------
        None

        """
        ext.initialize()

        # collecting variable metadata from model._doc
        var_meta = {
            col:
            self._doc.loc[self._doc["Py Name"] == py_name_clean, col].values[0]
            or "Missing"
            for col in self._doc.columns
        }
        var_meta["files"] = ";".join(ext.files)
        var_meta["tabs"] = ";".join(ext.tabs)
        var_meta["cells"] = ";".join(ext.cells)
        # TODO: add also time_row_or_cols

        da = ext.data

        # Renaming shared dims by all ExtData ("time") and ExtLookup
        # ("lookup_dim") external objects
        if isinstance(ext, ExtData):
            new_name = data_dims.name_new_dim("time", da.coords["time"].values)
            da = da.rename({"time": new_name})
        if isinstance(ext, ExtLookup):
            new_name = lookup_dims.name_new_dim("lookup_dim",
                                                da.coords["lookup_dim"].values)
            da = da.rename({"lookup_dim": new_name})

        metadata.update({ext.py_name: var_meta})
        data.update({ext.py_name: da})

        # TODO use a logger
        print(f"Finished processing variable {py_name_clean}.")

    def __get_varname_from_ext_name(self, varname):
        """
        Returns the name of the variable that depends on the external object
        named varname. If that is not possible (see warning in the code to
        understand when that may happen), it gets the name by removing the
        _ext_[constant|data|lookup] prefix from the varname.

        Parameters
        ----------
        varname: str
            Variable name to which the External object is assigned.

        Returns
        -------
        var: str
            Name of the variable that calls the variable with name varname.

        """
        for var, deps in self._dependencies.items():
            for _, ext_name in deps.items():
                if varname == ext_name:
                    return var

        warnings.warn(
            f"No variable depends upon '{varname}'. This is likely due "
            f"to the fact that '{varname}' is defined using a mix of "
            "DATA and CONSTANT. Though Vensim allows it, it is "
            "not recommended."
        )
        return "_".join(varname.split("_")[3:])

    def get_args(self, param):
        """
        Returns the arguments of a model element.

        Parameters
        ----------
        param: str or func
            The model element name or function.

        Returns
        -------
        args: list
            List of arguments of the function.

        Examples
        --------
        >>> model.get_args('birth_rate')
        >>> model.get_args('Birth Rate')

        See also
        --------
        :func:`pysd.py_backend.model.Macro.get_coords`

        """
        if isinstance(param, str):
            func_name = utils.get_key_and_value_by_insensitive_key_or_value(
                param,
                self._namespace)[1] or param

            func = getattr(self.components, func_name)
        else:
            func = param

        if hasattr(func, 'args'):
            # cached functions
            return func.args
        else:
            # regular functions
            args = inspect.getfullargspec(func)[0]
            if 'self' in args:
                args.remove('self')
            return args

    def get_coords(self, param):
        """
        Returns the coordinates and dims of a model element.

        Parameters
        ----------
        param: str or func
            The model element name or function.

        Returns
        -------
        (coords, dims) or None: (dict, list) or None
            The coords and the dimensions of the element if it has.
            Otherwise, returns None.

        Examples
        --------
        >>> model.get_coords('birth_rate')
        >>> model.get_coords('Birth Rate')

        See also
        --------
        :func:`pysd.py_backend.model.Macro.get_args`

        """
        if isinstance(param, str):
            func_name = utils.get_key_and_value_by_insensitive_key_or_value(
                param,
                self._namespace)[1] or param

            func = getattr(self.components, func_name)

        else:
            func = param

        if hasattr(func, "subscripts"):
            dims = func.subscripts
            if not dims:
                return None
            coords = {dim: self.components._subscript_dict[dim]
                      for dim in dims}
            return coords, dims
        elif hasattr(func, "state") and isinstance(func.state, xr.DataArray):
            value = func()
        else:
            return None

        dims = list(value.dims)
        coords = {coord: list(value.coords[coord].values)
                  for coord in value.coords}
        return coords, dims

    def __getitem__(self, param):
        """
        Returns the current value of a model component.

        Parameters
        ----------
        param: str or func
            The model element name.

        Returns
        -------
        value: float or xarray.DataArray
            The value of the model component.

        Examples
        --------
        >>> model['birth_rate']
        >>> model['Birth Rate']

        Note
        ----
        It will crash if the model component takes arguments.

        See also
        --------
        :func:`pysd.py_backend.model.Macro.get_series_data`

        """
        func_name = utils.get_key_and_value_by_insensitive_key_or_value(
            param,
            self._namespace)[1] or param

        if self.get_args(getattr(self.components, func_name)):
            raise ValueError(
                "Trying to get the current value of a lookup "
                "to get all the values with the series data use "
                "model.get_series_data(param)\n\n")

        return getattr(self.components, func_name)()

    def get_series_data(self, param):
        """
        Returns the original values of a model lookup/data component.

        Parameters
        ----------
        param: str
            The model lookup/data element name.

        Returns
        -------
        value: xarray.DataArray
            Array with the value of the interpolating series
            in the first dimension.

        Examples
        --------
        >>> model['room_temperature']
        >>> model['Room temperature']

        """
        func_name = utils.get_key_and_value_by_insensitive_key_or_value(
            param,
            self._namespace)[1] or param

        if func_name.startswith("_ext_"):
            return getattr(self.components, func_name).data
        elif "__data__" in self._dependencies[func_name]:
            return getattr(
                self.components,
                self._dependencies[func_name]["__data__"]
            ).data
        elif "__lookup__" in self._dependencies[func_name]:
            return getattr(
                self.components,
                self._dependencies[func_name]["__lookup__"]
            ).data
        else:
            raise ValueError(
                "Trying to get the values of a constant variable. "
                "'model.get_series_data' only works lookups/data objects.\n\n")

    def set_components(self, params):
        """
        Set the value of exogenous model elements.
        Element values should be passed with a dictionary in the
        function call. Values can be numeric type or pandas Series.
        Series will be interpolated by integrator.

        Parameters
        ----------
        params: dict
            Dictionary with the name of the elements to modify and the
            value that they would take. If the passed value is a
            :class:`float` or a :class:`xarray.DataArray` (must be
            compatible with the dimensions of the variable). In this
            case, the variable will have a constant value, returning
            the past value and broadcasting to all dimensions, if
            necessary. If a :class:`pandas.Series` is passed, the
            variable will be of type data and will use the time of the
            model to interpolate the result having as reference the
            indexes of the series. In this case, the series 'data' can
            also be a :class:`float` or a :class:`xarray.DataArray`,
            as with constant values. In the case of the target using
            a :class:`pysd.py_backend.lookups.Lookup` object, it will
            modify the object values to use the original arguments when
            being call. More detailed information and examples of usage
            are available at
            `Getting Started <https://pysd.readthedocs.io/en/master/getting_started.html#setting-parameter-values>`__.

            To write more complex relationships, which may or may not
            include other model variables, a callable, e.g. a function,
            can be passed that takes the same arguments as the original
            function and returns a :class:`float` or a
            :class:`xarray.DataArray` with exactly the same dimensions
            as the original function. More detailed information and
            examples of usage are available at
            `Advanced Usage <https://pysd.readthedocs.io/en/master/advanced_usage.html#replacing-model-components-with-more-complex-objects>`__.


        Note
        ----
        This function is to modify the value or equations of the
        variables, it won't work properly with Stateful objects, e.g.
        Integ, DelayFixed... In order to modify them it will be
        necessary to do it manually, if you have other inputs it is
        recommended to modify these ones. To change their initial
        value :func:`pysd.py_backend.model.Model.set_initial_condition`
        method could be used.


        Examples
        --------
        >>> model.set_components({'birth_rate': 10})
        >>> model.set_components({'Birth Rate': 10})

        >>> br = pandas.Series(index=range(30), data=np.sin(range(30))
        >>> model.set_components({'birth_rate': br})

        See also
        --------
        :func:`pysd.py_backend.model.Model.set_initial_condition`
        :func:`pysd.py_backend.model.Macro.get_coords`
        :func:`pysd.py_backend.model.Macro.get_args`

        """
        self._components_setter_tracker.update(params)
        self._set_components(params, new=False)

    def _set_components(self, params, new):
        """
        Set the value of exogenous model elements, giving the option to
        set new components (used in Macros).
        """

        for key, value in params.items():
            func_name = utils.get_key_and_value_by_insensitive_key_or_value(
                key,
                self._namespace)[1]

            if isinstance(value, np.ndarray) or isinstance(value, list):
                raise TypeError(
                    'When setting ' + key + '\n'
                    'Setting subscripted must be done using a xarray.DataArray'
                    ' with the correct dimensions or a constant value '
                    '(https://pysd.readthedocs.io/en/master/'
                    'getting_started.html)')

            if func_name is None:
                raise NameError(
                    "\n'%s' is not recognized as a model component."
                    % key)

            if new:
                func = None
                dims = None
            else:
                func = getattr(self.components, func_name)
                _, dims = self.get_coords(func) or (None, None)

            # if the variable is a lookup or a data we perform the change in
            # the object they call
            func_type = getattr(func, "type", None)
            if func_type in ["Lookup", "Data"]:
                # getting the object from original dependencies
                obj = self._dependencies[func_name][f"__{func_type.lower()}__"]
                getattr(
                    self.components,
                    obj
                ).set_values(value)

                if not isinstance(value, pd.Series):
                    warnings.warn(
                        "Replacing interpolation data with constant values.")

                # Update dependencies
                if func_type == "Data":
                    if isinstance(value, pd.Series):
                        self._dependencies[func_name] = {
                            "time": 1, "__data__": obj
                        }
                    else:
                        self._dependencies[func_name] = {"__data__": obj}

                continue

            if func_type == "Stateful":
                warnings.warn(
                    "Replacing the value of Stateful variable with "
                    "an expression. To set initial conditions use "
                    "`set_initial_condition` instead..."
                )

            if isinstance(value, pd.Series):
                if func_type == "Constant":
                    warnings.warn(
                        "Replacing a constant value with a "
                        "time-dependent value. The value will be "
                        "interpolated over time."
                    )
                new_function, deps = self._timeseries_component(
                    value, dims)
                self._dependencies[func_name] = deps
            elif callable(value):
                if func_type == "Constant":
                    warnings.warn(
                        "Replacing a constant value with a callable. "
                        "The value may not be constant anymore."
                    )
                new_function = value
                # Using step cache adding time as dependency
                # TODO it would be better if we can parse the content
                # of the function to get all the dependencies
                self._dependencies[func_name] = {"time": 1}

            else:
                if func_type != "Constant":
                    warnings.warn("Replacing a variable by a constant value.")
                new_function = self._constant_component(value, dims)
                self._dependencies[func_name] = {}

            # copy attributes from the original object to proper working
            # of internal functions
            new_function.__name__ = func_name
            new_function.__dict__.update(getattr(func, "__dict__", {}))
            # set the new function
            self.components._set_component(func_name, new_function)
            if func_name in self.cache.cached_funcs:
                self.cache.cached_funcs.remove(func_name)

    def _timeseries_component(self, series, dims):
        """ Internal function for creating a timeseries model element """
        # this is only called if the set_component function recognizes a
        # pandas series
        # TODO: raise a warning if extrapolating from the end of the series.
        # TODO: data type variables should be creted using a Data object
        # lookup type variables should be created using a Lookup object

        if isinstance(series.values[0], xr.DataArray):
            # the interpolation will be time dependent
            return lambda: utils.rearrange(xr.concat(
                series.values,
                series.index).interp(concat_dim=self.time()).reset_coords(
                'concat_dim', drop=True),
                dims, self._subscript_dict), {'time': 1}

        elif dims:
            # the interpolation will be time dependent
            return lambda: utils.rearrange(
                float(np.interp(self.time(), series.index, series.values)),
                dims, self._subscript_dict), {'time': 1}

        else:
            # the interpolation will be time dependent
            return lambda: float(np.interp(
                self.time(), series.index, series.values
                )), {'time': 1}

    def _constant_component(self, value, dims):
        """ Internal function for creating a constant model element """
        if dims:
            return lambda: utils.rearrange(
                value, dims, self._subscript_dict)

        else:
            return lambda: value

    def set_initial_value(self, time, initial_value):
        """
        Set the system initial value.

        Parameters
        ----------
        time : float or int
            The system intial time to be set.

        initial_value : dict
            A (possibly partial) dictionary of the system initial values.
            The keys to this dictionary may be either pysafe names or
            original model file names.

        See also
        --------
        :func:`pysd.py_backend.model.Model.set_initial_condition`

        """
        self.time.set_control_vars(initial_time=time)
        stateful_name = "_NONE"
        modified_statefuls = set()

        for key, value in initial_value.items():
            component_name =\
                utils.get_key_and_value_by_insensitive_key_or_value(
                    key, self._namespace)[1]
            if component_name is not None:
                if self._dependencies[component_name]:
                    deps = list(self._dependencies[component_name])
                    if len(deps) == 1 and deps[0] in self.initialize_order:
                        stateful_name = deps[0]
            else:
                component_name = key
                stateful_name = key

            try:
                _, dims = self.get_coords(component_name)
            except TypeError:
                dims = None

            if isinstance(value, xr.DataArray)\
               and not set(value.dims).issubset(set(dims)):
                raise ValueError(
                    f"\nInvalid dimensions for {component_name}."
                    f"It should be a subset of {dims}, "
                    f"but passed value has {list(value.dims)}")

            if isinstance(value, np.ndarray) or isinstance(value, list):
                raise TypeError(
                    'When setting ' + key + '\n'
                    'Setting subscripted must be done using a xarray.DataArray'
                    ' with the correct dimensions or a constant value '
                    '(https://pysd.readthedocs.io/en/master/'
                    'getting_started.html)')

            # Try to update stateful component
            try:
                element = getattr(self.components, stateful_name)
                if dims:
                    value = utils.rearrange(
                        value, dims,
                        self._subscript_dict)
                element.initialize(value)
                modified_statefuls.add(stateful_name)
            except NameError:
                # Try to override component
                raise ValueError(
                    f"\nUnrecognized stateful '{component_name}'. If you want"
                    " to set a value of a regular component. Use params={"
                    f"'{component_name}': {value}" + "} instead.")

        self.clean_caches()

        # get the elements to initialize
        elements_to_initialize =\
            self._get_elements_to_initialize(modified_statefuls)

        # Initialize remaining stateful objects
        for element_name in self.initialize_order:
            if element_name in elements_to_initialize:
                self._stateful_elements[element_name].initialize()

    def _get_elements_to_initialize(self, modified_statefuls):
        elements_to_initialize = set()
        for stateful, deps in self.stateful_initial_dependencies.items():
            if stateful in modified_statefuls:
                # if elements initial conditions have been modified
                # we should not modify it
                continue
            for modified_sateteful in modified_statefuls:
                if modified_sateteful in deps:
                    # if element has dependencies on a modified element
                    # we should re-initialize it
                    elements_to_initialize.add(stateful)
                    continue

        return elements_to_initialize

    def export(self):
        """Exports stateful values to a dictionary."""
        return {
            name: element.export()
            for name, element in self._stateful_elements.items()
        }

    def _set_stateful(self, stateful_dict):
        """
        Set stateful values.

        Parameters
        ----------
        stateful_dict: dict
          Dictionary of the stateful elements and the attributes to change.

        """
        for element, attrs in stateful_dict.items():
            component = getattr(self.components, element)
            if hasattr(component, '_set_stateful'):
                component._set_stateful(attrs)
            else:
                [
                    setattr(component, attr, value)
                    for attr, value in attrs.items()
                ]

    def _build_doc(self):
        """
        Formats a table of documentation strings to help users remember
        variable names, and understand how they are translated into
        Python safe names.

        Returns
        -------
        docs_df: pandas dataframe
            Dataframe with columns for the model components:
                - Real names
                - Python safe identifiers (as used in model.components)
                - Units string
                - Documentation strings from the original model file
        """
        collector = []
        for name, pyname in self._namespace.items():
            element = getattr(self.components, pyname)
            collector.append({
                'Real Name': name,
                'Py Name': pyname,
                'Subscripts': element.subscripts,
                'Units': element.units,
                'Limits': element.limits,
                'Type': element.type,
                'Subtype': element.subtype,
                'Comment': element.__doc__.strip().strip("\n").strip()
                if element.__doc__ else None
            })

        return pd.DataFrame(
            collector
        ).sort_values(by="Real Name").reset_index(drop=True)

    def __str__(self):
        """ Return model source files """

        # JT: Might be helpful to return not only the source file, but
        # also how the instance differs from that source file. This
        # would give a more accurate view of the current model.
        string = 'Translated Model File: ' + self.py_model_file
        if hasattr(self, 'mdl_file'):
            string += '\n Original Model File: ' + self.mdl_file

        return string


class Model(Macro):
    """
    The Model class implements a stateful representation of the system.
    It inherits methods from the Macro class to integrate the model and
    access and modify model components. It also contains the main
    methods for running the model.

    The Model object will be created with components drawn from a
    translated Python model file.

    Parameters
    ----------
    py_model_file: str or pathlib.Path
        Filename of a model which has already been converted into a
        Python format.
    data_files: dict or list or str or None
        The dictionary with keys the name of file and variables to
        load the data from there. Or the list of names or name of the
        file to search the data in. Only works for TabData type object
        and it is neccessary to provide it. Default is None.
    initialize: bool
        If False, the model will not be initialize when it is loaded.
        Default is True.
    missing_values : str ("warning", "error", "ignore", "keep") (optional)
        What to do with missing values. If "warning" (default)
        shows a warning message and interpolates the values.
        If "raise" raises an error. If "ignore" interpolates
        the values without showing anything. If "keep" it will keep
        the missing values, this option may cause the integration to
        fail, but it may be used to check the quality of the data.

    See also
    --------
    :class:`pysd.py_backend.model.Macro`

    """
    def __init__(self, py_model_file, data_files, initialize, missing_values):
        """ Sets up the Python objects """
        super().__init__(py_model_file, None, None, Time(),
                         data_files=data_files)
        self.data_files = data_files
        self.missing_values = missing_values
        # set time component
        self.time.stage = 'Load'
        # set control var privately to do not change it when copying
        self.time._set_control_vars(**self.components._control_vars)
        # Attributes that are set later
        self.progress = None
        self.output = None
        self.capture_elements = None
        self.return_addresses = None
        self._stepper_mode = None
        self._submodel_tracker = {}

        if initialize:
            self.initialize()

    def initialize(self):
        """
        Initializes the simulation model.

        See also
        --------
        :func:`pysd.py_backend.model.Macro.initialize`
        :func:`pysd.py_backend.model.Model.reload`

        """
        self.time.stage = 'Initialization'
        External.missing = self.missing_values
        super().initialize()

    def run(self, params=None, return_columns=None, return_timestamps=None,
            initial_condition='original', final_time=None, time_step=None,
            saveper=None, reload=False, progress=False, flatten_output=True,
            cache_output=True, output_file=None):
        """
        Simulate the model's behavior over time.
        Return a pandas dataframe with timestamps as rows and model
        elements as columns. More detailed information and examples
        of usage are available at
        `Getting Started <https://pysd.readthedocs.io/en/master/getting_started.html#running-the-model>`__.

        Parameters
        ----------
        params: dict (optional)
            Keys are strings of model component names.
            Values are numeric or pandas Series.
            Numeric values represent constants over the model integration.
            Timeseries will be interpolated to give time-varying input.
            For more information, check the documentation of
            :func:`pysd.py_backend.model.Macro.set_components`.

        return_timestamps: list, numeric, ndarray (1D) (optional)
            Timestamps in model execution at which to return state information.
            Defaults to model-file specified timesteps.

        return_columns: list, 'step' or None (optional)
            List of string model component names, returned dataframe
            will have corresponding columns. If 'step' only variables with
            cache step will be returned. If None, variables with cache step
            and run will be returned. Default is None.

        initial_condition: str or (float, dict) (optional)
            The starting time, and the state of the system (the values of
            all the stocks) at that starting time. 'original' or 'o' uses
            model-file specified initial condition. 'current' or 'c' uses
            the state of the model after the previous execution. Other str
            objects, loads initial conditions from the pickle file with the
            given name.(float, dict) tuple lets the user specify a starting
            time (float) and (possibly partial) dictionary of initial values
            for stock (stateful) objects. Default is 'original'.
            For more information, check the documentation of
            :func:`pysd.py_backend.model.Model.set_initial_condition`

        final_time: float or None
            Final time of the simulation. If float, the given value will be
            used to compute the return_timestamps (if not given) and as a
            final time. If None the last value of return_timestamps will be
            used as a final time. Default is None.

        time_step: float or None
            Time step of the simulation. If float, the given value will be
            used to compute the return_timestamps (if not given) and
            euler time series. If None the default value from components
            will be used. Default is None.

        saveper: float or None
            Saving step of the simulation. If float, the given value will be
            used to compute the return_timestamps (if not given). If None
            the default value from components will be used. Default is None.

        reload : bool (optional)
            If True, reloads the model from the translated model file
            before making changes. Default is False.

        progress : bool (optional)
            If True, a progressbar will be shown during integration.
            Default is False.

        flatten_output: bool (optional)
            If True, once the output dataframe has been formatted will
            split the xarrays in new columns following Vensim's naming
            to make a totally flat output. Default is True.
            This argument will be ignored when passing a netCDF4 file
            path in the output_file argument.

        cache_output: bool (optional)
           If True, the number of calls of outputs variables will be increased
           in 1. This helps caching output variables if they are called only
           once. For performance reasons, if time step = saveper it is
           recommended to activate this feature, if time step << saveper
           it is recommended to deactivate it. Default is True.

        output_file: str, pathlib.Path or None (optional)
           Path of the file in which to save simulation results.
           Currently, csv, tab and nc (netCDF4) files are supported.


        Examples
        --------
        >>> model.run(params={'exogenous_constant': 42})
        >>> model.run(params={'exogenous_variable': timeseries_input})
        >>> model.run(return_timestamps=[1, 2, 3, 4, 10])
        >>> model.run(return_timestamps=10)
        >>> model.run(return_timestamps=np.linspace(1, 10, 20))
        >>> model.run(output_file="results.nc")

        See also
        --------
        :func:`pysd.py_backend.model.Macro.set_components`
        :func:`pysd.py_backend.model.Model.set_initial_condition`
        :func:`pysd.py_backend.model.Model.reload`

        """
        self._stepper_mode = False

        if reload:
            self.reload()

        self._config_simulation(params, return_columns, return_timestamps,
                                initial_condition, final_time, time_step,
                                saveper, cache_output, progress=progress)

        # instante output object
        self.output = ModelOutput(output_file)
        self.output.set_capture_elements(self.capture_elements)
        self.output.initialize(self)

        self._integrate()

        return self.output.collect(self, flatten_output)

    def set_stepper(self, output_obj, params=None, step_vars=[],
                    return_columns=None, return_timestamps=None,
                    initial_condition='original', final_time=None,
                    time_step=None, saveper=None, cache_output=True):
        """
        Configure the model stepping behavior. Examples of usage are
        available at
        `Advanced Usage <https://pysd.readthedocs.io/en/master/advanced_usage.html#running-models-one-or-more-step-s-at-a-time>`__.

        Parameters
        ----------
        output_obj: ModelOutput
            Instance of ModelOutput where the simulation results will be
            stored.

        params: dict (optional)
            Keys are strings of model component names.
            Values are numeric or pandas Series.
            Numeric values represent constants over the model integration.
            Timeseries will be interpolated to give time-varying input.

        step_vars: list
            List of variable or parameter names whose values might be
            updated after one or more simulation steps.

        return_columns: list, 'step' or None (optional)
            List of string model component names, returned dataframe
            will have corresponding columns. If 'step' only variables
            with cache step will be returned. If None, variables with
            cache step and run will be returned. Default is None.

        return_timestamps: list, numeric, ndarray (1D) (optional)
            Timestamps in model execution at which to return state
            information. Defaults to model-file specified timesteps.

        initial_condition: str or (float, dict) (optional)
            The starting time, and the state of the system (the values
            of all the stocks) at that starting time. 'original' or 'o'
            uses model-file specified initial condition. 'current' or
            'c' uses the state of the model after the previous
            execution. Other str objects, loads initial conditions from
            the pickle file with the given name.(float, dict) tuple lets
            the user specify a starting time (float) and (possibly
            partial) dictionary of initial values for stock (stateful)
            objects. Default is 'original'.

        final_time: float or None
            Final time of the simulation. If float, the given value will
            be used to compute the return_timestamps (if not given) and
            as a final time. If None the last value of return_timestamps
            will be used as a final time. Default is None.

        time_step: float or None
            Time step of the simulation. If float, the given value will
            be used to compute the return_timestamps (if not given) and
            euler time series. If None the default value from components
            will be used. Default is None.

        saveper: float or None
            Saving step of the simulation. If float, the given value will
            be used to compute the return_timestamps (if not given). If None
            the default value from components will be used. Default is None.

        cache_output: bool (optional)
           If True, the number of calls of outputs variables will be increased
           in 1. This helps caching output variables if they are called only
           once. For performance reasons, if time step = saveper it is
           recommended to activate this feature, if time step << saveper
           it is recommended to deactivate it. Default is True.

        See also
        --------
        :func:`pysd.py_backend.model.Model.step`

        """
        self.output = output_obj

        self._stepper_mode = True

        self._config_simulation(params, return_columns, return_timestamps,
                                initial_condition, final_time, time_step,
                                saveper, cache_output, step_vars=step_vars)

        self.output.set_capture_elements(self.capture_elements)
        self.output.initialize(self)
        self.output.update(self)

    def step(self, num_steps=1, step_vars={}):
        """
        Run a model step. Updates model variables first (optional), and then
        runs any number of model steps. To collect the outputs after one or
        more steps, use the collect method of the ModelOutput class.
        Examples of usage are available at
        `Advanced Usage <https://pysd.readthedocs.io/en/master/advanced_usage.html#running-models-one-or-more-step-s-at-a-time>`__

        Parameters
        ----------
        num_steps: int
            Number of steps that the iterator should run with the values
            of variables defined in step_vars argument.

        step_vars: dict
            Varibale names that should be updated before running the step
            as keys, and the actual values of the variables as values.

        Returns
        -------
        None

        See also
        --------
        :func:`pysd.py_backend.model.Model.set_stepper`

        """
        # TODO warn the user if we exceeded the final_time??
        self._set_components(step_vars, new=False)

        for _ in range(num_steps):
            self._integrate_step()
            if self.time.in_return():
                self.output.update(self)

    def _config_simulation(self, params, return_columns, return_timestamps,
                           initial_condition, final_time, time_step,
                           saveper, cache_output, **kwargs):
        """
        Internal method to set all simulation config parameters. Arguments
        to this function are those of the run and set_stepper methods.
        """
        # set control var at the beginning in case they are needed to
        # initialize any object
        self._set_control_vars(return_timestamps, final_time, time_step,
                               saveper)

        if params:
            self.set_components(params)

        if self._stepper_mode:
            for step_var in kwargs["step_vars"]:
                self._dependencies[step_var]["time"] = 1

        # update cache types after setting params
        self._assign_cache_type()

        # set initial conditions
        self.set_initial_condition(initial_condition)
        # set control vars again in case a pickle has been used
        self._set_control_vars(return_timestamps, final_time, time_step,
                               saveper)

        # progressbar only makes sense when not running step by step
        if not self._stepper_mode:
            self.progress = self._set_progressbar(kwargs["progress"])

        self.capture_elements = self._set_capture_elements(return_columns)

        # include outputs in cache if needed
        self._dependencies["OUTPUTS"] = {
            element: 1 for element in self.capture_elements["step"]
        }

        if cache_output:
            # udate the cache type taking into account the outputs
            self._assign_cache_type()

        # add constant cache to thosa variable that are constants
        self._add_constant_cache()

        # set Run mode
        self.time.stage = 'Run'

        # need to clean cache to remove the values from active_initial
        self.clean_caches()

    def _set_capture_elements(self, return_columns):
        """
        Define which variables will be stored in the output object, according
        to the return_columns passed by the user. The list is stored in the
        capture_elements attribute, which is a dictionary with keys "run" and
        "step".

        Parameters
        ----------

        return_columns:list, 'step' or None (optional)
            List of string model component names, returned dataframe
            will have corresponding columns. If 'step' only variables with
            cache step will be returned. If None, variables with cache step
            and run will be returned. Default is None.

        Returns
        -------
        capture_elements: dict
            Dictionary of list with keywords step and run.

        """
        if return_columns is None or isinstance(return_columns, str):
            return_columns = self._default_return_columns(return_columns)

        capture_elements, self.return_addresses = utils.get_return_elements(
            return_columns, self._namespace)

        # create a dictionary splitting run cached and others
        capture_elements = self._split_capture_elements(capture_elements)

        return capture_elements

    def _set_progressbar(self, progress):
        """
        Configures the progressbar, according to the user provided argument.
        If final_time or time_step are functions, then the progressbar is
        automatically disabled, regardless of the value of the progress
        argument.

        Parameters
        ----------
        progress: bool

        """
        if progress and (self.cache_type["final_time"] == "step" or
                         self.cache_type["time_step"] == "step"):
            warnings.warn(
                "The progressbar is not compatible with dynamic "
                "final time or time step. Both variables must be "
                "constants to prompt progress."
            )
            progress = False

        return progress

    def _set_control_vars(self, return_timestamps, final_time, time_step,
                          saveper):
        self.time.add_return_timestamps(return_timestamps)
        if self.time.return_timestamps is not None and not final_time:
            # if not final time given the model will end in the list
            # return timestamp (the list is reversed for popping)
            if self.time.return_timestamps:
                final_time = self.time.return_timestamps[0]
            else:
                final_time = self.time._next_return

        self.time.set_control_vars(
            final_time=final_time, time_step=time_step, saveper=saveper)

    def select_submodel(self, vars=[], modules=[], exogenous_components={},
                        inplace=True):
        """
        Select a submodel from the original model. After selecting a submodel
        only the necessary stateful objects for integrating this submodel will
        be computed. Examples of usage are available at
        `Advanced Usage <https://pysd.readthedocs.io/en/master/advanced_usage.html#selecting-and-running-a-submodel>`__.

        Parameters
        ----------
        vars: set or list of strings (optional)
            Variables to include in the new submodel.
            It can be an empty list if the submodel is only selected by
            module names. Default is an empty list.

        modules: set or list of strings (optional)
            Modules to include in the new submodel.
            It can be an empty list if the submodel is only selected by
            variable names. Default is an empty list. Can select a full
            module or a submodule by passing the path without the .py, e.g.:
            "view_1/submodule1".

        exogenous_components: dictionary of parameters (optional)
            Exogenous value to fix to the model variables that are needed
            to run the selected submodel. The exogenous_components should
            be passed as a dictionary in the same way it is done for
            set_components method. By default it is an empty dict and
            the needed exogenous components will be set to a numpy.nan value.

        inplace: bool (optional)
            If True it will modify current object and will return None.
            If False it will create a copy of the model and return it
            keeping the original model unchange. Default is True.

        Returns
        -------
        None or pysd.py_backend.model.Model
            If inplace=False it will return a modified copy of the
            original model.

        Note
        ----
        modules can be only passed when the model has been split in
        different files during translation.

        Examples
        --------
        >>> model.select_submodel(
        ...     vars=["Room Temperature", "Teacup temperature"])
        UserWarning: Selecting submodel, to run the full model again use model.reload()

        >>> model.select_submodel(
        ...     modules=["view_1", "view_2/subview_1"])
        UserWarning: Selecting submodel, to run the full model again use model.reload()
        UserWarning: Exogenous components for the following variables are necessary but not given:
            initial_value_stock1, stock3

        >>> model.select_submodel(
        ...     vars=["stock3"],
        ...     modules=["view_1", "view_2/subview_1"])
        UserWarning: Selecting submodel, to run the full model again use model.reload()
        UserWarning: Exogenous components for the following variables are necessary but not given:
            initial_value_stock1, initial_value_stock3
        Please, set them before running the model using set_components method...

        >>> model.select_submodel(
        ...     vars=["stock3"],
        ...     modules=["view_1", "view_2/subview_1"],
        ...     exogenous_components={
        ...         "initial_value_stock1": 3,
        ...         "initial_value_stock3": 5})
        UserWarning: Selecting submodel, to run the full model again use model.reload()

        See also
        --------
        :func:`pysd.py_backend.model.Model.get_vars_in_module`
        :func:`pysd.py_backend.model.Model.get_dependencies`

        """
        if inplace:
            self._select_submodel(vars, modules, exogenous_components)
        else:
            return self.copy()._select_submodel(
                vars, modules, exogenous_components)

    def _select_submodel(self, vars, modules, exogenous_components={}):
        self._submodel_tracker = {
            "vars": vars,
            "modules": modules
        }
        deps = self.get_dependencies(vars, modules)
        warnings.warn(
            "Selecting submodel, "
            "to run the full model again use model.reload()")

        # get set of all dependencies and all variables to select
        all_deps = deps.d_deps["initial"].copy()
        all_deps.update(deps.d_deps["step"])
        all_deps.update(deps.d_deps["lookup"])

        all_vars = all_deps.copy()
        all_vars.update(deps.c_vars)

        # clean dependendies and namespace dictionaries, and remove
        # the rows from the documentation
        for real_name, py_name in self._namespace.copy().items():
            if py_name not in all_vars:
                del self._namespace[real_name]
                del self._dependencies[py_name]
                self._doc.drop(
                    self._doc.index[self._doc["Real Name"] == real_name],
                    inplace=True
                )

        for py_name in self._dependencies.copy().keys():
            if py_name.startswith("_") and py_name not in deps.s_deps:
                del self._dependencies[py_name]

        # remove active initial from s_deps as they are "fake" objects
        # in dependencies
        deps.s_deps = {
            dep for dep in deps.s_deps if not dep.startswith("_active_initial")
        }

        # reassing the dictionary and lists of needed stateful objects
        self._stateful_elements = {
            name: getattr(self.components, name)
            for name in deps.s_deps
            if isinstance(getattr(self.components, name), Stateful)
        }
        self._dynamicstateful_elements = [
            getattr(self.components, name) for name in deps.s_deps
            if isinstance(getattr(self.components, name), DynamicStateful)
        ]
        self._macro_elements = [
            getattr(self.components, name) for name in deps.s_deps
            if isinstance(getattr(self.components, name), Macro)
        ]

        # keeping only needed external objects
        ext_deps = set()
        for values in self._dependencies.values():
            if "__external__" in values:
                ext_deps.add(values["__external__"])
        self._external_elements = [
            getattr(self.components, name) for name in ext_deps
            if isinstance(getattr(self.components, name), External)
        ]

        # set all exogenous values to np.nan by default
        new_components = {element: np.nan for element in all_deps}
        # update exogenous values with the user input
        [new_components.update(
            {
                utils.get_key_and_value_by_insensitive_key_or_value(
                    key,
                    self._namespace)[1]: value
            }) for key, value in exogenous_components.items()]

        self.set_components(new_components)

        # show a warning message if exogenous values are needed for a
        # dependency
        new_components = [
            key for key, value in new_components.items() if value is np.nan]
        if new_components:
            warnings.warn(
                "Exogenous components for the following variables are "
                f"necessary but not given:\n\t{', '.join(new_components)}"
                "\n\n Please, set them before running the model using "
                "set_components method...")

        # re-assign the cache_type and initialization order
        self._assign_cache_type()
        self._get_initialize_order()
        return self

    def get_dependencies(self, vars=[], modules=[]):
        """
        Get the dependencies of a set of variables or modules.

        Parameters
        ----------
        vars: set or list of strings (optional)
            Variables to get the dependencies from.
            It can be an empty list if the dependencies are computed only
            using modules. Default is an empty list.
        modules: set or list of strings (optional)
            Modules to get the dependencies from.
            It can be an empty list if the dependencies are computed only
            using variables. Default is an empty list. Can select a full
            module or a submodule by passing the path without the .py, e.g.:
            "view_1/submodule1".

        Returns
        -------
        dependencies: pysd.py_backend.utils.Dependencies
            Dependencies data object.

        Note
        ----
        modules can be only passed when the model has been split in
        different files during translation.

        Examples
        --------
        >>> print(model.get_dependencies(
        ...     vars=["Room Temperature", "Teacup temperature"]))
        Selected variables (total 1):
            room_temperature, teacup_temperature
        Stateful objects integrated with the selected variables (total 1):
            _integ_teacup_temperature

        >>> print(model.get_dependencies(
        ...     modules=["view_1", "view_2/subview_1"]))
        Selected variables (total 4):
            var1, var2, stock1, delay1
        Dependencies for initialization only (total 1):
            initial_value_stock1
        Dependencies that may change over time (total 2):
            stock3
        Stateful objects integrated with the selected variables (total 1):
            _integ_stock1, _delay_fixed_delay1

        >>> print(model.get_dependencies(
        ...     vars=["stock3"],
        ...     modules=["view_1", "view_2/subview_1"]))
        Selected variables (total 4):
            var1, var2, stock1, stock3, delay1
        Dependencies for initialization only (total 1):
            initial_value_stock1, initial_value_stock3
        Stateful objects integrated with the selected variables (total 1):
            _integ_stock1, _integ_stock3, _delay_fixed_delay1

        See also
        --------
        :func:`pysd.py_backend.model.Model.get_vars_in_module`

        """
        def check_dep(deps_obj, deps, initial=False):
            for dep in deps:
                if dep in deps_obj.c_vars or dep.startswith("__"):
                    pass
                elif dep.startswith("_"):
                    deps_obj.s_deps.add(dep)
                    dep = self._dependencies[dep]
                    check_dep(deps_obj, dep["initial"], True)
                    check_dep(deps_obj, dep["step"])
                else:
                    if initial and dep not in deps_obj.d_deps["step"]\
                       and dep not in deps_obj.d_deps["lookup"]:
                        deps_obj.d_deps["initial"].add(dep)
                    else:
                        if dep in deps_obj.d_deps["initial"]:
                            deps_obj.d_deps["initial"].remove(dep)
                        if self.get_args(dep):
                            deps_obj.d_deps["lookup"].add(dep)
                        else:
                            deps_obj.d_deps["step"].add(dep)

        dependencies = utils.Dependencies(
            {"time", "time_step", "initial_time", "final_time", "saveper"},
            {"initial": set(), "step": set(), "lookup": set()},
            set()
        )
        for var in vars:
            py_name = utils.get_key_and_value_by_insensitive_key_or_value(
                    var,
                    self._namespace)[1]
            dependencies.c_vars.add(py_name)
        for module in modules:
            dependencies.c_vars.update(self.get_vars_in_module(module))

        for var in dependencies.c_vars:
            if var == "time":
                continue
            check_dep(dependencies, self._dependencies[var])

        return dependencies

    def get_vars_in_module(self, module):
        """
        Return the name of Python vars in a module.

        Parameters
        ----------
        module: str
            Name of the module to search in.

        Returns
        -------
        vars: set
            Set of varible names in the given module.

        See also
        --------
        :func:`pysd.py_backend.model.Model.get_dependencies`

        """
        if self._modules:
            module_content = self._modules.copy()
        else:
            raise ValueError(
                "Trying to get a module from a non-modularized model")

        try:
            # get the module or the submodule content
            for submodule in module.split("/"):
                module_content = module_content[submodule]
            module_content = [module_content]
        except KeyError:
            raise NameError(
                f"Module or submodule '{submodule}' not found...\n")

        vars, new_content = set(), []

        while module_content:
            # find the vars in the module or the submodule
            for content in module_content:
                if isinstance(content, list):
                    vars.update(content)
                else:
                    [new_content.append(value) for value in content.values()]

            module_content, new_content = new_content, []

        return vars

    def copy(self, reload=False):
        """
        Create a copy of the current model.

        Parameters
        ----------
        reload: bool (optional)
            If True the model will be copied without applying to it any
            change, the copy will simply load the model again from the
            translated file. This would be equivalent to doing
            :py:func:`pysd.load` with the same arguments. Otherwise, it
            will apply the same changes that have been applied to the
            original model and update the states (faithful copy).
            Default is False.

        Warning
        -------
        The copy function will load a new model from the file and apply
        the same changes to it. If any of these changes have replaced a
        variable with a function that references other variables in the
        model, the copy will not work properly since the function will
        still reference the variables in the original model, in which
        case the function should be redefined.

        See also
        --------
        :func:`pysd.py_backend.model.Model.reload`

        """
        # initialize the new model?
        initialize = self.time.stage != 'Load'
        # create a new model
        new_model = type(self)(
            py_model_file=deepcopy(self.py_model_file),
            data_files=deepcopy(self.data_files),
            initialize=initialize,
            missing_values=deepcopy(self.missing_values)
        )
        if reload:
            # return reloaded copy
            return new_model
        # copy the values of the stateful objects
        if initialize:
            new_model._set_stateful(deepcopy(super().export()))
        # copy time object values
        new_model.time._set_time(deepcopy(self.time.export()))
        # set other components
        with warnings.catch_warnings():
            # filter warnings that have been already shown in original model
            warnings.simplefilter("ignore")
            # substract submodel
            if self._submodel_tracker:
                new_model._select_submodel(**self._submodel_tracker)
            # copy modified parameters
            new_model.set_components(self._components_setter_tracker)

        return new_model

    def reload(self):
        """
        Reloads the model from the translated model file, so that all the
        parameters are back to their original value.

        See also
        --------
        :func:`pysd.py_backend.model.Model.copy`
        :func:`pysd.py_backend.model.Model.initialize`

        """
        self.__init__(self.py_model_file, data_files=self.data_files,
                      initialize=True,
                      missing_values=self.missing_values)

    def _default_return_columns(self, which):
        """
        Return a list of the model elements tha change on time that
        does not include lookup other functions that take parameters
        or run-cached functions.

        Parameters
        ----------
        which: str or None
            If it is 'step' only cache step elements will be returned.
            Else cache 'step' and 'run' elements will be returned.
            Default is None.

        Returns
        -------
        return_columns: list
            List of columns to return

        """
        if which == 'step':
            types = ['step']
        else:
            types = ['step', 'run']

        return_columns = []

        for key, pykey in self._namespace.items():
            if pykey in self.cache_type and self.cache_type[pykey] in types\
               and not self.get_args(pykey):

                return_columns.append(key)

        return return_columns

    def _split_capture_elements(self, capture_elements):
        """
        Splits the capture elements list between those with run cache
        and others.

        Parameters
        ----------
        capture_elements: list
            Captured elements list

        Returns
        -------
        capture_dict: dict
            Dictionary of list with keywords step and run.

        """
        capture_dict = {'step': [], 'run': [], None: []}
        [capture_dict[self.cache_type[element]].append(element)
         for element in capture_elements]
        return capture_dict

    def set_initial_condition(self, initial_condition):
        """ Set the initial conditions of the integration.

        Parameters
        ----------
        initial_condition : str or (float, dict) or pathlib.Path
            The starting time, and the state of the system (the values of
            all the stocks) at that starting time. 'original' or 'o'uses
            model-file specified initial condition. 'current' or 'c' uses
            the state of the model after the previous execution. Other str
            objects, loads initial conditions from the pickle file with the
            given name.(float, dict) tuple lets the user specify a starting
            time (float) and (possibly partial) dictionary of initial values
            for stock (stateful) objects.

        Examples
        --------
        >>> model.set_initial_condition('original')
        >>> model.set_initial_condition('current')
        >>> model.set_initial_condition('exported_pickle.pic')
        >>> model.set_initial_condition((10, {'teacup_temperature': 50}))

        See also
        --------
        :func:`pysd.py_backend.model.Macro.set_initial_value`

        """
        if isinstance(initial_condition, str)\
           and initial_condition.lower() not in ["original", "o",
                                                 "current", "c"]:
            initial_condition = Path(initial_condition)

        if isinstance(initial_condition, tuple):
            self.initialize()
            self.set_initial_value(*initial_condition)
        elif isinstance(initial_condition, Path):
            self.import_pickle(initial_condition)
            self.time.set_control_vars(initial_time=self.time())
        elif isinstance(initial_condition, str):
            if initial_condition.lower() in ["original", "o"]:
                self.time.set_control_vars(
                    initial_time=self.components._control_vars["initial_time"])
                self.initialize()
        else:
            raise TypeError(
                "Invalid initial conditions. "
                + "Check documentation for valid entries or use "
                + "'help(model.set_initial_condition)'.")

    def _euler_step(self, dt):
        """
        Performs a single step in the euler integration,
        updating stateful components

        Parameters
        ----------
        dt : float
            This is the amount to increase time by this step

        """
        self.state = self.state + self.ddt() * dt

    def _integrate(self):
        """
        Performs euler integration and writes results to the out_obj.

        Returns
        -------
        None

        """

        if self.progress:
            # initialize progress bar
            progressbar = utils.ProgressBar(
                int((self.time.final_time()-self.time())/self.time.time_step())
            )
        else:
            # when None is used the update will do nothing
            progressbar = utils.ProgressBar(None)

        # performs the time stepping
        while self.time.in_bounds():
            if self.time.in_return():
                self.output.update(self)

            self._integrate_step()
            progressbar.update()

        # need to add one more time step, because we run only the state
        # updates in the previous loop and thus may be one short.
        if self.time.in_return():
            self.output.update(self)

        progressbar.finish()

    def _integrate_step(self):
        self._euler_step(self.time.time_step())
        self.time.update(self.time()+self.time.time_step())
        self.clean_caches()

    def export(self, file_name):
        """
        Export stateful values to pickle file.

        Parameters
        ----------
        file_name: str or pathlib.Path
          Name of the file to export the values.

        See also
        --------
        :func:`pysd.py_backend.model.Model.import_pickle`

        """
        warnings.warn(
            "\nCompatibility of exported states could be broken between"
            " different versions of PySD or xarray, current versions:\n"
            f"\tPySD {__version__}\n\txarray {xr.__version__}\n"
        )
        with open(file_name, 'wb') as file:
            pickle.dump(
                (self.time.export(),
                 super().export(),
                 {'pysd': __version__, 'xarray': xr.__version__}
                 ), file)

    def import_pickle(self, file_name):
        """
        Import stateful values from pickle file.

        Parameters
        ----------
        file_name: str or pathlib.Path
          Name of the file to import the values from.

        See also
        --------
        :func:`pysd.py_backend.model.Model.export_pickle`

        """
        with open(file_name, 'rb') as file:
            time_dict, stateful_dict, metadata = pickle.load(file)

        if __version__ != metadata['pysd']\
           or xr.__version__ != metadata['xarray']:  # pragma: no cover
            warnings.warn(
                "\nCompatibility of exported states could be broken between"
                " different versions of PySD or xarray. Current versions:\n"
                f"\tPySD {__version__}\n\txarray {xr.__version__}\n"
                "Loaded versions:\n"
                f"\tPySD {metadata['pysd']}\n\txarray {metadata['xarray']}\n"
                )

        self.time._set_time(time_dict)
        self._set_stateful(stateful_dict)
