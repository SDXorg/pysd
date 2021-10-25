"""
The stateful objects are used and updated each time step with an update
method. This include from basic Integ class objects until the Model
class objects.
"""

import inspect
import re
import pickle
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from . import utils
from .functions import zidz, if_then_else
from .external import External, Excels
from .decorators import Cache, constant_cache
from .components import Components, Time

from pysd._version import __version__


small_vensim = 1e-6  # What is considered zero according to Vensim Help


class Stateful(object):
    # the integrator needs to be able to 'get' the current state of the object,
    # and get the derivative. It calculates the new state, and updates it.
    # The state can be any object which is subject to basic (element-wise)
    # algebraic operations
    def __init__(self):
        self._state = None
        self.shape_info = None
        self.py_name = ""

    def __call__(self, *args, **kwargs):
        return self.state

    @property
    def state(self):
        if self._state is None:
            raise AttributeError(
                self.py_name
                + "\nAttempt to call stateful element"
                + " before it is initialized.")
        return self._state

    @state.setter
    def state(self, new_value):
        if self.shape_info:
            self._state = xr.DataArray(data=new_value, **self.shape_info)
        else:
            self._state = new_value


class DynamicStateful(Stateful):

    def __init__(self):
        super().__init__()

    def update(self, state):
        try:
            self.state = state
        except Exception as err:
            raise ValueError(err.args[0] + "\n\n"
                             + "Could not update the value of "
                             + self.py_name)


class Integ(DynamicStateful):
    """
    Implements INTEG function
    """
    def __init__(self, ddt, initial_value, py_name):
        """

        Parameters
        ----------
        ddt: function
          This will become an attribute of the object
        initial_value: function
          Initial value
        py_name: str
          Python name to identify the object
        """
        super().__init__()
        self.init_func = initial_value
        self.ddt = ddt
        self.shape_info = None
        self.py_name = py_name

    def initialize(self, init_val=None):
        if init_val is None:
            self.state = self.init_func()
        else:
            self.state = init_val
        if isinstance(self.state, xr.DataArray):
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}

    def export(self):
        return {'state': self.state, 'shape_info': self.shape_info}


class Delay(DynamicStateful):
    """
    Implements DELAY function
    """
    # note that we could have put the `delay_input` argument as a parameter to
    # the `__call__` function, and more closely mirrored the vensim syntax.
    # However, people may get confused this way in thinking that they need
    # only one delay object and can call it with various arguments to delay
    # whatever is convenient. This method forces them to acknowledge that
    # additional structure is being created in the delay object.

    def __init__(self, delay_input, delay_time, initial_value, order, tstep,
                 py_name):
        """

        Parameters
        ----------
        delay_input: function
        delay_time: function
        initial_value: function
        order: function
        py_name: str
          Python name to identify the object
        """
        super().__init__()
        self.init_func = initial_value
        self.delay_time_func = delay_time
        self.input_func = delay_input
        self.order_func = order
        self.order = None
        self.tstep = tstep
        self.shape_info = None
        self.py_name = py_name

    def initialize(self, init_val=None):
        order = self.order_func()

        if order != int(order):
            warnings.warn(self.py_name + '\n' +
                          'Casting delay order '
                          + f'from {order} to {int(order)}')

        self.order = int(order)  # The order can only be set once
        if self.order*self.tstep() > np.min(self.delay_time_func()):
            while self.order*self.tstep() > np.min(self.delay_time_func()):
                self.order -= 1
            warnings.warn(self.py_name + '\n' +
                          'Delay time very small, casting delay order '
                          + f'from {int(order)} to {self.order}')

        if init_val is None:
            init_state_value = self.init_func() * self.delay_time_func()
        else:
            init_state_value = init_val * self.delay_time_func()

        if isinstance(init_state_value, xr.DataArray):
            # broadcast self.state
            self.state = init_state_value.expand_dims({
                '_delay': np.arange(self.order)}, axis=0)
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}
        else:
            self.state = np.array([init_state_value] * self.order)

    def __call__(self):
        if self.shape_info:
            return self.state[-1].reset_coords('_delay', drop=True)\
                   / self.delay_time_func()
        else:
            return self.state[-1] / self.delay_time_func()

    def ddt(self):
        outflows = self.state / self.delay_time_func()
        inflows = np.roll(outflows, 1, axis=0)
        if self.shape_info:
            inflows[0] = self.input_func().values
        else:
            inflows[0] = self.input_func()
        return (inflows - outflows) * self.order

    def export(self):
        return {'state': self.state, 'shape_info': self.shape_info}


class DelayN(DynamicStateful):
    """
    Implements DELAY N function
    """
    # note that we could have put the `delay_input` argument as a parameter to
    # the `__call__` function, and more closely mirrored the vensim syntax.
    # However, people may get confused this way in thinking that they need
    # only one delay object and can call it with various arguments to delay
    # whatever is convenient. This method forces them to acknowledge that
    # additional structure is being created in the delay object.

    def __init__(self, delay_input, delay_time, initial_value, order, tstep,
                 py_name):
        """

        Parameters
        ----------
        delay_input: function
        delay_time: function
        initial_value: function
        order: function
        py_name: str
          Python name to identify the object
        """
        super().__init__()
        self.init_func = initial_value
        self.delay_time_func = delay_time
        self.input_func = delay_input
        self.order_func = order
        self.order = None
        self.times = None
        self.tstep = tstep
        self.shape_info = None
        self.py_name = py_name

    def initialize(self, init_val=None):
        order = self.order_func()

        if order != int(order):
            warnings.warn(self.py_name + '\n' +
                          'Casting delay order '
                          + f'from {order} to {int(order)}')

        self.order = int(order)  # The order can only be set once
        if self.order*self.tstep() > np.min(self.delay_time_func()):
            while self.order*self.tstep() > np.min(self.delay_time_func()):
                self.order -= 1
            warnings.warn(self.py_name + '\n' +
                          'Delay time very small, casting delay order '
                          + f'from {int(order)} to {self.order}')

        if init_val is None:
            init_state_value = self.init_func() * self.delay_time_func()
        else:
            init_state_value = init_val * self.delay_time_func()

        if isinstance(init_state_value, xr.DataArray):
            # broadcast self.state
            self.state = init_state_value.expand_dims({
                '_delay': np.arange(self.order)}, axis=0)
            self.times = self.delay_time_func().expand_dims({
                '_delay': np.arange(self.order)}, axis=0)
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}
        else:
            self.state = np.array([init_state_value] * self.order)
            self.times = np.array([self.delay_time_func()] * self.order)

    def __call__(self):
        if self.shape_info:
            return self.state[-1].reset_coords('_delay', drop=True)\
                   / self.times[0].reset_coords('_delay', drop=True)
        else:
            return self.state[-1] / self.times[0]

    def ddt(self):
        if self.shape_info:
            # if is xarray need to preserve coords
            self.times = self.times.roll({'_delay': 1}, False)
            self.times[0] = self.delay_time_func()
            outflows = self.state / self.times
            inflows = outflows.roll({'_delay': 1}, False)
        else:
            # if is float use numpy.roll
            self.times = np.roll(self.times, 1, axis=0)
            self.times[0] = self.delay_time_func()
            outflows = self.state / self.times
            inflows = np.roll(outflows, 1, axis=0)

        inflows[0] = self.input_func()
        return (inflows - outflows)*self.order

    def export(self):
        return {'state': self.state, 'times': self.times,
                'shape_info': self.shape_info}


class DelayFixed(DynamicStateful):
    """
    Implements DELAY FIXED function
    """

    def __init__(self, delay_input, delay_time, initial_value, tstep,
                 py_name):
        """

        Parameters
        ----------
        delay_input: function
        delay_time: function
        initial_value: function
        order: function
        py_name: str
          Python name to identify the object
        """
        super().__init__()
        self.init_func = initial_value
        self.delay_time_func = delay_time
        self.input_func = delay_input
        self.tstep = tstep
        self.order = None
        self.pointer = 0
        self.py_name = py_name

    def initialize(self, init_val=None):
        order = max(self.delay_time_func()/self.tstep(), 1)

        if order != int(order):
            warnings.warn(
                self.py_name + '\n'
                + 'Casting delay order from %f to %i' % (
                    order, round(order + small_vensim)))

        # need to add a small decimal to ensure that 0.5 is rounded to 1
        # The order can only be set once
        self.order = round(order + small_vensim)

        if init_val is None:
            init_state_value = self.init_func()
        else:
            init_state_value = init_val

        self.state = init_state_value
        self.pipe = [init_state_value] * self.order

    def __call__(self):
        return self.state

    def ddt(self):
        return np.nan

    def update(self, state):
        self.pipe[self.pointer] = self.input_func()
        self.pointer = (self.pointer + 1) % self.order
        self.state = self.pipe[self.pointer]

    def export(self):
        return {'state': self.state, 'pointer': self.pointer,
                'pipe': self.pipe}


class Forecast(DynamicStateful):
    """
    Implements FORECAST function
    """
    def __init__(self, forecast_input, average_time, horizon, py_name):
        """

        Parameters
        ----------
        forecast_input: function
        average_time: function
        horizon: function
        py_name: str
          Python name to identify the object
        """

        super().__init__()
        self.horizon = horizon
        self.average_time = average_time
        self.input = forecast_input
        self.py_name = py_name

    def initialize(self, init_val=None):

        # self.state = AV in the vensim docs
        if init_val is None:
            self.state = self.input()
        else:
            self.state = init_val

        if isinstance(self.state, xr.DataArray):
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}

    def __call__(self):
        return self.input() * (
            1 + zidz(self.input() - self.state,
                     self.average_time() * self.state
                     )*self.horizon()
        )

    def ddt(self):
        return (self.input() - self.state) / self.average_time()

    def export(self):
        return {'state': self.state, 'shape_info': self.shape_info}


class Smooth(DynamicStateful):
    """
    Implements SMOOTH function
    """
    def __init__(self, smooth_input, smooth_time, initial_value, order,
                 py_name):
        """

        Parameters
        ----------
        smooth_input: function
        smooth_time: function
        initial_value: function
        order: function
        py_name: str
          Python name to identify the object
        """
        super().__init__()
        self.init_func = initial_value
        self.smooth_time_func = smooth_time
        self.input_func = smooth_input
        self.order_func = order
        self.order = None
        self.shape_info = None
        self.py_name = py_name

    def initialize(self, init_val=None):
        self.order = self.order_func()  # The order can only be set once

        if init_val is None:
            init_state_value = self.init_func()
        else:
            init_state_value = init_val

        if isinstance(init_state_value, xr.DataArray):
            # broadcast self.state
            self.state = init_state_value.expand_dims({
                '_smooth': np.arange(self.order)}, axis=0)
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}
        else:
            self.state = np.array([init_state_value] * self.order)

    def __call__(self):
        if self.shape_info:
            return self.state[-1].reset_coords('_smooth', drop=True)
        else:
            return self.state[-1]

    def ddt(self):
        targets = np.roll(self.state, 1, axis=0)
        if self.shape_info:
            targets[0] = self.input_func().values
        else:
            targets[0] = self.input_func()
        return (targets - self.state) * self.order / self.smooth_time_func()

    def export(self):
        return {'state': self.state, 'shape_info': self.shape_info}


class Trend(DynamicStateful):
    """
    Implements TREND function
    """
    def __init__(self, trend_input, average_time, initial_trend, py_name):
        """

        Parameters
        ----------
        trend_input: function
        average_time: function
        initial_trend: function
        py_name: str
          Python name to identify the object
        """

        super().__init__()
        self.init_func = initial_trend
        self.average_time_function = average_time
        self.input_func = trend_input
        self.py_name = py_name

    def initialize(self, init_val=None):
        if init_val is None:
            self.state = self.input_func()\
                / (1 + self.init_func()*self.average_time_function())
        else:
            self.state = self.input_func()\
                / (1 + init_val*self.average_time_function())

        if isinstance(self.state, xr.DataArray):
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}

    def __call__(self):
        return zidz(self.input_func() - self.state,
                    self.average_time_function() * np.abs(self.state))

    def ddt(self):
        return (self.input_func() - self.state) / self.average_time_function()

    def export(self):
        return {'state': self.state, 'shape_info': self.shape_info}


class SampleIfTrue(DynamicStateful):
    def __init__(self, condition, actual_value, initial_value, py_name):
        """

        Parameters
        ----------
        condition: function
        actual_value: function
        initial_value: function
        py_name: str
          Python name to identify the object
        """
        super().__init__()
        self.condition = condition
        self.actual_value = actual_value
        self.init_func = initial_value
        self.py_name = py_name

    def initialize(self, init_val=None):
        if init_val is None:
            self.state = self.init_func()
        else:
            self.state = init_val
        if isinstance(self.state, xr.DataArray):
            self.shape_info = {'dims': self.state.dims,
                               'coords': self.state.coords}

    def __call__(self):
        return if_then_else(self.condition(),
                            self.actual_value,
                            lambda: self.state)

    def ddt(self):
        return np.nan

    def update(self, state):
        self.state = self.state*0 + if_then_else(self.condition(),
                                                 self.actual_value,
                                                 lambda: self.state)

    def export(self):
        return {'state': self.state, 'shape_info': self.shape_info}


class Initial(Stateful):
    """
    Implements INITIAL function
    """
    def __init__(self, initial_value, py_name):
        """

        Parameters
        ----------
        initial_value: function
        py_name: str
          Python name to identify the object
        """
        super().__init__()
        self.init_func = initial_value
        self.py_name = py_name

    def initialize(self, init_val=None):
        if init_val is None:
            self.state = self.init_func()
        else:
            self.state = init_val

    def export(self):
        return {'state': self.state}


class Macro(DynamicStateful):
    """
    The Model class implements a stateful representation of the system,
    and contains the majority of methods for accessing and modifying model
    components.

    When the instance in question also serves as the root model object
    (as opposed to a macro or submodel within another model) it will have
    added methods to facilitate execution.
    """

    def __init__(self, py_model_file, params=None, return_func=None,
                 time=None, time_initialization=None, py_name=None):
        """
        The model object will be created with components drawn from a
        translated python model file.

        Parameters
        ----------
        py_model_file : <string>
            Filename of a model which has already been converted into a
            python format.
        get_time:
            needs to be a function that returns a time object
        params
        return_func
        """
        super().__init__()
        self.time = time
        self.time_initialization = time_initialization
        self.cache = Cache()
        self.py_name = py_name
        self.external_loaded = False
        self.components = Components(py_model_file, self.set_components)

        if __version__.split(".")[0]\
           != self.get_pysd_compiler_version().split(".")[0]:
            raise ImportError(
                "\n\nNot able to import the model. "
                + "The model was compiled with a "
                + "not compatible version of PySD:"
                + "\n\tPySD " + self.get_pysd_compiler_version()
                + "\n\nThe current version of PySd is:"
                + "\n\tPySD " + __version__ + "\n\n"
                + "Please translate again the model with the function"
                + " read_vensim or read_xmile.")

        if params is not None:
            self.set_components(params, new=True)
            for param in params:
                self.components._dependencies[
                    self.components._namespace[param]] = {"time"}

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

        self._assign_cache_type()
        self._get_initialize_order()

        if return_func is not None:
            self.return_func = getattr(self.components, return_func)
        else:
            self.return_func = lambda: 0

        self.py_model_file = py_model_file

    def __call__(self):
        return self.return_func()

    def clean_caches(self):
        self.cache.clean()
        # if nested macros
        [macro.clean_caches() for macro in self._macro_elements]

    def _get_initialize_order(self):
        """
        Get the initialization order of the stateful elements
        and their the full dependencies.
        """
        # get the full set of dependencies to initialize an stateful object
        # includying all levels
        self.stateful_initial_dependencies = {
            ext: set()
            for ext in self.components._dependencies
            if ext.startswith("_")
        }
        for element in self.stateful_initial_dependencies:
            self._get_full_dependencies(
                element, self.stateful_initial_dependencies[element],
                "initial")

        # get the full dependencies of stateful objects taking into account
        # only other objects
        current_deps = {
            element: [dep for dep in deps if dep.startswith("_")]
            for element, deps in self.stateful_initial_dependencies.items()
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
        deps = self.components._dependencies[element]
        if element.startswith("_"):
            deps = deps[stateful_deps]
        for dep in deps:
            if dep not in dep_set and not dep.startswith("__")\
               and dep != "time":
                dep_set.add(dep)
                self._get_full_dependencies(dep, dep_set, stateful_deps)

    def _add_constant_cache(self):
        self.constant_funcs = set()
        for element, cache_type in self.cache_type.items():
            if cache_type == "run":
                if self.get_args(element):
                    self.components._set_component(
                        element,
                        constant_cache(getattr(self.components, element), None)
                    )
                else:
                    self.components._set_component(
                        element,
                        constant_cache(getattr(self.components, element))
                    )
                self.constant_funcs.add(element)

    def _remove_constant_cache(self):
        for element in self.constant_funcs:
            self.components._set_component(
                element,
                getattr(self.components, element).function)
        self.constant_funcs = set()

    def _assign_cache_type(self):
        """
        Assigns the cache type to all the elements from the namespace.
        """
        self.cache_type = {"time": None}

        for element in self.components._namespace.values():
            if element not in self.cache_type\
               and element in self.components._dependencies:
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
        for subelement in self.components._dependencies:
            if subelement.startswith("_") and\
               element in self.components._dependencies[subelement]["step"]:
                if element in\
                   self.components._dependencies[subelement]["initial"]:
                    n_calls +=\
                        2*self.components._dependencies[subelement][
                            "step"][element]
                else:
                    n_calls +=\
                        self.components._dependencies[subelement][
                            "step"][element]
            elif (not subelement.startswith("_") and
                  element in self.components._dependencies[subelement]):
                n_calls +=\
                    self.components._dependencies[subelement][element]

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
        if not self.components._dependencies[element]:
            self.cache_type[element] = "run"
        elif "__lookup__" in self.components._dependencies[element]:
            self.cache_type[element] = None
        elif self._isdynamic(self.components._dependencies[element]):
            self.cache_type[element] = "step"
        else:
            self.cache_type[element] = "run"
            for subelement in self.components._dependencies[element]:
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

        if not self.external_loaded:
            # Initialize external elements
            for element in self._external_elements:
                element.initialize()

            # Remove Excel data from memory
            Excels.clean()

            self.external_loaded = True

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

    def export(self, file_name):
        """
        Export stateful values to pickle file.

        Parameters
        ----------
        file_name: str
          Name of the file to export the values.

        """
        warnings.warn(
            "\nCompatibility of exported states could be broken between"
            " different versions of PySD or xarray, current versions:\n"
            f"\tPySD {__version__}\n\txarray {xr.__version__}\n"
        )
        stateful_elements = {
            name: element.export()
            for name, element in self._stateful_elements.items()
        }

        with open(file_name, 'wb') as file:
            pickle.dump(
                (self.time(),
                 stateful_elements,
                 {'pysd': __version__, 'xarray': xr.__version__}
                 ), file)

    def import_pickle(self, file_name):
        """
        Import stateful values from pickle file.

        Parameters
        ----------
        file_name: str
          Name of the file to import the values from.

        """
        with open(file_name, 'rb') as file:
            time, stateful_dict, metadata = pickle.load(file)

        if __version__ != metadata['pysd']\
           or xr.__version__ != metadata['xarray']:  # pragma: no cover
            warnings.warn(
                "\nCompatibility of exported states could be broken between"
                " different versions of PySD or xarray. Current versions:\n"
                f"\tPySD {__version__}\n\txarray {xr.__version__}\n"
                "Loaded versions:\n"
                f"\tPySD {metadata['pysd']}\n\txarray {metadata['xarray']}\n"
                )

        self.set_stateful(stateful_dict)
        self.time.set_control_vars(initial_time=time)

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

        """
        if isinstance(param, str):
            func_name = utils.get_value_by_insensitive_key_or_value(
                param,
                self.components._namespace) or param

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

        """
        if isinstance(param, str):
            func_name = utils.get_value_by_insensitive_key_or_value(
                param,
                self.components._namespace) or param

            func = getattr(self.components, func_name)

        else:
            func = param

        if not self.get_args(func):
            value = func()
        else:
            value = func(0)

        if isinstance(value, xr.DataArray):
            dims = list(value.dims)
            coords = {coord: list(value.coords[coord].values)
                      for coord in value.coords}
            return coords, dims
        else:
            return None

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

        """
        func_name = utils.get_value_by_insensitive_key_or_value(
            param,
            self.components._namespace) or param

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
        func_name = utils.get_value_by_insensitive_key_or_value(
            param,
            self.components._namespace) or param

        try:
            if func_name.startswith("_ext_"):
                return getattr(self.components, func_name).data
            elif self.get_args(getattr(self.components, func_name)):
                return getattr(self.components,
                               "_ext_lookup_" + func_name).data
            else:
                return getattr(self.components,
                               "_ext_data_" + func_name).data
        except NameError:
            raise ValueError(
                "Trying to get the values of a hardcoded lookup/data or "
                "other type of variable. 'model.get_series_data' only works "
                "with external lookups/data objects.\n\n")

    def set_components(self, params, new=False):
        """ Set the value of exogenous model elements.
        Element values can be passed as keyword=value pairs in the
        function call. Values can be numeric type or pandas Series.
        Series will be interpolated by integrator.

        Examples
        --------
        >>> model.set_components({'birth_rate': 10})
        >>> model.set_components({'Birth Rate': 10})

        >>> br = pandas.Series(index=range(30), values=np.sin(range(30))
        >>> model.set_components({'birth_rate': br})


        """
        # TODO: allow the params argument to take a pandas dataframe, where
        # column names are variable names. However some variables may be
        # constant or have no values for some index. This should be processed.
        # TODO: make this compatible with loading outputs from other files

        for key, value in params.items():
            func_name = utils.get_value_by_insensitive_key_or_value(
                key,
                self.components._namespace)

            if isinstance(value, np.ndarray) or isinstance(value, list):
                raise TypeError(
                    'When setting ' + key + '\n'
                    'Setting subscripted must be done using a xarray.DataArray'
                    ' with the correct dimensions or a constant value '
                    '(https://pysd.readthedocs.io/en/master/basic_usage.html)')

            if func_name is None:
                raise NameError(
                    "\n'%s' is not recognized as a model component."
                    % key)

            if new:
                dims, args = None, None
            else:
                func = getattr(self.components, func_name)
                _, dims = self.get_coords(func) or (None, None)
                args = self.get_args(func)

            if isinstance(value, pd.Series):
                new_function, deps = self._timeseries_component(
                    value, dims, args)
                self.components._dependencies[func_name] = deps
            elif callable(value):
                new_function = value
                args = self.get_args(value)
                if args:
                    # user function needs arguments, add it as a lookup
                    # to avoud caching it
                    self.components._dependencies[func_name] =\
                        {"__lookup__": None}
                else:
                    # TODO it would be better if we can parse the content
                    # of the function to get all the dependencies
                    # user function takes no arguments, using step cache
                    # adding time as dependency
                    self.components._dependencies[func_name] = {"time": 1}

            else:
                new_function = self._constant_component(value, dims, args)
                self.components._dependencies[func_name] = {}

            # this won't handle other statefuls...
            if '_integ_' + func_name in dir(self.components):
                warnings.warn("Replacing the equation of stock"
                              + "{} with params".format(key),
                              stacklevel=2)

            new_function.__name__ = func_name
            self.components._set_component(func_name, new_function)
            if func_name in self.cache.cached_funcs:
                self.cache.cached_funcs.remove(func_name)

    def _timeseries_component(self, series, dims, args=[]):
        """ Internal function for creating a timeseries model element """
        # this is only called if the set_component function recognizes a
        # pandas series
        # TODO: raise a warning if extrapolating from the end of the series.
        if isinstance(series.values[0], xr.DataArray) and args:
            # the argument is already given in the model when the model
            # is called
            return lambda x: utils.rearrange(xr.concat(
                series.values,
                series.index).interp(concat_dim=x).reset_coords(
                'concat_dim', drop=True),
                dims, self.components._subscript_dict), {'__lookup__': None}

        elif isinstance(series.values[0], xr.DataArray):
            # the interpolation will be time dependent
            return lambda: utils.rearrange(xr.concat(
                series.values,
                series.index).interp(concat_dim=self.time()).reset_coords(
                'concat_dim', drop=True),
                dims, self.components._subscript_dict), {'time': 1}

        elif args and dims:
            # the argument is already given in the model when the model
            # is called
            return lambda x: utils.rearrange(
                np.interp(x, series.index, series.values),
                dims, self.components._subscript_dict), {'__lookup__': None}

        elif args:
            # the argument is already given in the model when the model
            # is called
            return lambda x:\
                np.interp(x, series.index, series.values), {'__lookup__': None}

        elif dims:
            # the interpolation will be time dependent
            return lambda: utils.rearrange(
                np.interp(self.time(), series.index, series.values),
                dims, self.components._subscript_dict), {'time': 1}

        else:
            # the interpolation will be time dependent
            return lambda:\
                np.interp(self.time(), series.index, series.values),\
                {'time': 1}

    def _constant_component(self, value, dims, args=[]):
        """ Internal function for creating a constant model element """
        if args and dims:
            # need to pass an argument to keep consistency with the calls
            # to the function
            return lambda x: utils.rearrange(
                value, dims, self.components._subscript_dict)

        elif args:
            # need to pass an argument to keep consistency with the calls
            # to the function
            return lambda x: value

        elif dims:
            return lambda: utils.rearrange(
                value, dims, self.components._subscript_dict)

        else:
            return lambda: value

    def set_initial_value(self, t, initial_value):
        """ Set the system initial value.

        Parameters
        ----------
        t : numeric
            The system time

        initial_value : dict
            A (possibly partial) dictionary of the system initial values.
            The keys to this dictionary may be either pysafe names or
            original model file names

        """
        self.time.set_control_vars(initial_time=t)
        stateful_name = "_NONE"
        modified_statefuls = set()

        for key, value in initial_value.items():
            component_name = utils.get_value_by_insensitive_key_or_value(
                key, self.components._namespace)
            if component_name is not None:
                if self.components._dependencies[component_name]:
                    deps = list(self.components._dependencies[component_name])
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
                    '(https://pysd.readthedocs.io/en/master/basic_usage.html)')

            # Try to update stateful component
            try:
                element = getattr(self.components, stateful_name)
                if dims:
                    value = utils.rearrange(
                        value, dims,
                        self.components._subscript_dict)
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

    def set_stateful(self, stateful_dict):
        """
        Set stateful values.

        Parameters
        ----------
        stateful_dict: dict
          Dictionary of the stateful elements and the attributes to change.

        """
        for element, attrs in stateful_dict.items():
            for attr, value in attrs.items():
                setattr(getattr(self.components, element), attr, value)

    def doc(self):
        """
        Formats a table of documentation strings to help users remember
        variable names, and understand how they are translated into
        python safe names.

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
        for name, varname in self.components._namespace.items():
            try:
                # TODO correct this when Original Eqn is in several lines
                docstring = getattr(self.components, varname).__doc__
                lines = docstring.split('\n')

                for unit_line in range(3, 9):
                    # this loop detects where Units: starts as
                    # sometimes eqn could be split in several lines
                    if re.findall('Units:', lines[unit_line]):
                        break
                if unit_line == 3:
                    eqn = lines[2].replace("Original Eqn:", "").strip()
                else:
                    eqn = '; '.join(
                        [line.strip() for line in lines[3:unit_line]])

                collector.append(
                    {'Real Name': name,
                     'Py Name': varname,
                     'Eqn': eqn,
                     'Unit': lines[unit_line].replace("Units:", "").strip(),
                     'Lims': lines[unit_line+1].replace("Limits:", "").strip(),
                     'Type': lines[unit_line+2].replace("Type:", "").strip(),
                     'Subs': lines[unit_line+3].replace("Subs:", "").strip(),
                     'Comment': '\n'.join(lines[(unit_line+4):]).strip()})
            except Exception:
                pass

        docs_df = pd.DataFrame(collector)
        docs_df.fillna('None', inplace=True)

        order = ['Real Name', 'Py Name', 'Unit', 'Lims',
                 'Type', 'Subs', 'Eqn', 'Comment']
        return docs_df[order].sort_values(
            by='Real Name').reset_index(drop=True)

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
    def __init__(self, py_model_file, initialize, missing_values):
        """ Sets up the python objects """
        super().__init__(py_model_file, None, None, Time())
        self.time.stage = 'Load'
        self.time.set_control_vars(**self.components._control_vars)
        self.missing_values = missing_values
        if initialize:
            self.initialize()

    def initialize(self):
        """ Initializes the simulation model """
        self.time.stage = 'Initialization'
        External.missing = self.missing_values
        super().initialize()

    def run(self, params=None, return_columns=None, return_timestamps=None,
            initial_condition='original', final_time=None, time_step=None,
            saveper=None, reload=False, progress=False, flatten_output=False,
            cache_output=True):
        """
        Simulate the model's behavior over time.
        Return a pandas dataframe with timestamps as rows,
        model elements as columns.

        Parameters
        ----------
        params: dict (optional)
            Keys are strings of model component names.
            Values are numeric or pandas Series.
            Numeric values represent constants over the model integration.
            Timeseries will be interpolated to give time-varying input.

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
            all the stocks) at that starting time. 'original' or 'o'uses
            model-file specified initial condition. 'current' or 'c' uses
            the state of the model after the previous execution. Other str
            objects, loads initial conditions from the pickle file with the
            given name.(float, dict) tuple lets the user specify a starting
            time (float) and (possibly partial) dictionary of initial values
            for stock (stateful) objects. Default is 'original'.

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
            split the xarrays in new columns following vensim's naming
            to make a totally flat output. Default is False.

        cache_output: bool (optional)
           If True, the number of calls of outputs variables will be increased
           in 1. This helps caching output variables if they are called only
           once. For performance reasons, if time step = saveper it is
           recommended to activate this feature, if time step << saveper
           it is recommended to deactivate it. Default is True.

        Examples
        --------
        >>> model.run(params={'exogenous_constant': 42})
        >>> model.run(params={'exogenous_variable': timeseries_input})
        >>> model.run(return_timestamps=[1, 2, 3.1415, 4, 10])
        >>> model.run(return_timestamps=10)
        >>> model.run(return_timestamps=np.linspace(1, 10, 20))

        See Also
        --------
        pysd.set_components : handles setting model parameters
        pysd.set_initial_condition : handles setting initial conditions

        """
        if reload:
            self.reload()

        self.progress = progress

        self.time.add_return_timestamps(return_timestamps)
        if self.time.return_timestamps is not None and not final_time:
            final_time = self.time.return_timestamps[-1]

        self.time.set_control_vars(
            final_time=final_time, time_step=time_step, saveper=saveper)

        if params:
            self.set_components(params)

        # update cache types after setting params
        self._assign_cache_type()

        self.set_initial_condition(initial_condition)

        if return_columns is None or isinstance(return_columns, str):
            return_columns = self._default_return_columns(return_columns)

        capture_elements, return_addresses = utils.get_return_elements(
            return_columns, self.components._namespace)

        # create a dictionary splitting run cached and others
        capture_elements = self._split_capture_elements(capture_elements)

        # include outputs in cache if needed
        self.components._dependencies["OUTPUTS"] = {
            element: 1 for element in capture_elements["step"]
        }
        if cache_output:
            self._assign_cache_type()
            self._add_constant_cache()

        # Run the model
        self.time.stage = 'Run'
        # need to clean cache to remove the values from active_initial
        self.clean_caches()

        res = self._integrate(capture_elements['step'])

        del self.components._dependencies["OUTPUTS"]

        self._add_run_elements(res, capture_elements['run'])
        self._remove_constant_cache()

        return_df = utils.make_flat_df(res, return_addresses, flatten_output)

        return return_df

    def reload(self):
        """
        Reloads the model from the translated model file, so that all the
        parameters are back to their original value.
        """
        self.__init__(self.py_model_file, initialize=True,
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

        for key, pykey in self.components._namespace.items():
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
            Dictionary of sets with keywords step and run.

        """
        capture_dict = {'step': set(), 'run': set(), None: set()}
        [capture_dict[self.cache_type[element]].add(element)
         for element in capture_elements]
        return capture_dict

    def set_initial_condition(self, initial_condition):
        """ Set the initial conditions of the integration.

        Parameters
        ----------
        initial_condition : str or (float, dict)
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

        See Also
        --------
        model.set_initial_value()

        """

        if isinstance(initial_condition, tuple):
            self.initialize()
            self.set_initial_value(*initial_condition)
        elif isinstance(initial_condition, str):
            if initial_condition.lower() in ["original", "o"]:
                self.time.set_control_vars(
                    initial_time=self.components._control_vars["initial_time"])
                self.initialize()
            elif initial_condition.lower() in ["current", "c"]:
                pass
            else:
                self.import_pickle(initial_condition)
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

    def _integrate(self, capture_elements):
        """
        Performs euler integration.

        Parameters
        ----------
        capture_elements: set
            Which model elements to capture - uses pysafe names.

        Returns
        -------
        outputs: pandas.DataFrame
            Output capture_elements data.

        """
        # necessary to have always a non-xaray object for appending objects
        # to the DataFrame time will always be a model element and not saved
        # TODO: find a better way of saving outputs
        capture_elements.add("time")
        outputs = pd.DataFrame(columns=capture_elements)

        if self.progress:
            # initialize progress bar
            progressbar = utils.ProgressBar(
                int((self.time.final_time()-self.time())/self.time.time_step())
            )
        else:
            # when None is used the update will do nothing
            progressbar = utils.ProgressBar(None)

        while self.time.in_bounds():
            if self.time.in_return():
                outputs.at[self.time()] = [getattr(self.components, key)()
                                           for key in capture_elements]
            self._euler_step(self.time.time_step())
            self.time.update(self.time()+self.time.time_step())
            self.clean_caches()
            progressbar.update()

        # need to add one more time step, because we run only the state
        # updates in the previous loop and thus may be one short.
        if self.time.in_return():
            outputs.at[self.time()] = [getattr(self.components, key)()
                                       for key in capture_elements]

        progressbar.finish()

        # delete time column as it was created only for avoiding errors
        # of appending data. See previous TODO.
        del outputs["time"]
        return outputs

    def _add_run_elements(self, df, capture_elements):
        """
        Adds constant elements to a dataframe.

        Parameters
        ----------
        df: pandas.DataFrame
            Dataframe to add elements.

        capture_elements: list
            List of constant elements

        Returns
        -------
        None

        """
        nt = len(df.index.values)
        for element in capture_elements:
            df[element] = [getattr(self.components, element)()] * nt
