"""
These functions have no direct analog in the standard python data analytics
stack, or require information about the internal state of the system beyond
what is present in the function call. We provide them in a structure that
makes it easy for the model elements to call.
"""

import inspect
import os
import re
import pickle
import random
import warnings
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
import xarray as xr

from . import utils
from .functions import zidz, if_then_else
from .external import External, Excels
from .decorators import Cache, constant_cache

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

    def __call__(self, *args, **kwargs):
        return self.state

    @property
    def state(self):
        if self._state is None:
            raise AttributeError('Attempt to call stateful element'
                                 + ' before it is initialized.')
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
    def __init__(self, ddt, initial_value, py_name="Integ object"):
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
        return {self.py_name: {
            'state': self.state,
            'shape_info': self.shape_info}}


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

    def __init__(self, delay_input, delay_time, initial_value, order,
                 tstep=lambda: 0, py_name="Delay object"):
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
        return {self.py_name: {
            'state': self.state,
            'shape_info': self.shape_info}}


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

    def __init__(self, delay_input, delay_time, initial_value, order,
                 tstep, py_name):
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
        return {self.py_name: {
            'state': self.state,
            'times': self.times,
            'shape_info': self.shape_info}}


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
        return {self.py_name: {
            'state': self.state,
            'pointer': self.pointer,
            'pipe': self.pipe}}


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
        return {self.py_name: {
            'state': self.state,
            'shape_info': self.shape_info}}


class Smooth(DynamicStateful):
    """
    Implements SMOOTH function
    """
    def __init__(self, smooth_input, smooth_time, initial_value, order,
                 py_name="Smooth object"):
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
        return {self.py_name: {
            'state': self.state,
            'shape_info': self.shape_info}}


class Trend(DynamicStateful):
    """
    Implements TREND function
    """
    def __init__(self, trend_input, average_time, initial_trend,
                 py_name="Trend object"):
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
        return {self.py_name: {
            'state': self.state,
            'shape_info': self.shape_info}}


class SampleIfTrue(DynamicStateful):
    def __init__(self, condition, actual_value, initial_value,
                 py_name="SampleIfTrue object"):
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
        return {self.py_name: {
            'state': self.state,
            'shape_info': self.shape_info}}


class Initial(Stateful):
    """
    Implements INITIAL function
    """
    def __init__(self, initial_value, py_name="Initial object"):
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
        return {self.py_name: {
            'state': self.state}}


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

        # need a unique identifier for the imported module.
        module_name = os.path.splitext(py_model_file)[0]\
            + str(random.randint(0, 1000000))
        try:
            self.components = SourceFileLoader(module_name,
                                               py_model_file).load_module()
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
            self.set_components(params)
            for param in params:
                self.components._dependencies[
                    self.components._namespace[param]] = {"time"}

        # Get the collections of stateful elements and external elements
        self._stateful_elements = [
            getattr(self.components, name) for name in dir(self.components)
            if isinstance(getattr(self.components, name), Stateful)
        ]
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
                    setattr(
                        self.components, element,
                        constant_cache(getattr(self.components, element), None)
                    )
                else:
                    setattr(
                        self.components, element,
                        constant_cache(getattr(self.components, element))
                    )
                self.constant_funcs.add(element)

    def _remove_constant_cache(self):
        for element in self.constant_funcs:
            setattr(
                self.components, element,
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
                    setattr(
                        self.components, element,
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
                if subelement.startswith("_initial_") or subelement.startswith("__"):
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
            if dep.startswith("_") and not dep.startswith("_initial_") and not dep.startswith("__"):
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

        self.cache.clean()

        self.components._init_outer_references({
            'scope': self,
            'time': self.time
        })

        # Initialize external elements
        for element in self._external_elements:
            element.initialize()

        Excels.clean()

        # Initialize stateful objects
        remaining = set(self._stateful_elements)
        for element_name in self.initialize_order:
            for element in remaining:
                if element.py_name == element_name:
                    element.initialize()
                    break
            remaining.remove(element)

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
        stateful_elements = {}
        [stateful_elements.update(component.export()) for component
         in self._stateful_elements]

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
           or xr.__version__ != metadata['xarray']:
            warnings.warn(
                "\nCompatibility of exported states could be broken between"
                " different versions of PySD or xarray. Current versions:\n"
                f"\tPySD {__version__}\n\txarray {xr.__version__}\n"
                "Loaded versions:\n"
                f"\tPySD {metadata['pysd']}\n\txarray {metadata['xarray']}\n"
                )
        self.set_stateful(stateful_dict)

        self.time.update(time)

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

            if hasattr(self.components, func_name):
                func = getattr(self.components, func_name)
            else:
                NameError(
                    "\n'%s' is not recognized as a model component."
                    % param)
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

            if hasattr(self.components, func_name):
                func = getattr(self.components, func_name)
            else:
                NameError(
                    "\n'%s' is not recognized as a model component."
                    % param)
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

        if func_name.startswith("_ext_"):
            if hasattr(self.components, func_name):
                return getattr(self.components, func_name).data
        elif self.get_args(getattr(self.components, func_name)):
            if hasattr(self.components, "_ext_lookup_" + func_name):
                return getattr(self.components,
                               "_ext_lookup_" + func_name).data
        else:
            if hasattr(self.components, "_ext_data_" + func_name):
                return getattr(self.components,
                               "_ext_data_" + func_name).data

        raise ValueError(
            "Trying to get the values of a hardcoded lookup/data or "
            "other type of variable. 'model.get_series_data' only works "
            "with external lookups/data objects.\n\n")

    def set_components(self, params):
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

            try:
                func = getattr(self.components, func_name)
                _, dims = self.get_coords(func) or (None, None)
                args = self.get_args(func)
            except (AttributeError, TypeError):
                dims, args = None, None

            if isinstance(value, pd.Series):
                new_function, deps = self._timeseries_component(
                    value, dims, args)
                self.components._dependencies[func_name] = deps
            elif callable(value):
                new_function = value
            else:
                new_function = self._constant_component(value, dims, args)
                self.components._dependencies[func_name] = {}

            # this won't handle other statefuls...
            if '_integ_' + func_name in dir(self.components):
                warnings.warn("Replacing the equation of stock"
                              + "{} with params".format(key),
                              stacklevel=2)

            new_function.__name__ = func_name
            setattr(self.components, func_name, new_function)
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
        self.time.update(t)
        self.clean_caches()
        stateful_name = "_NONE"
        # TODO make this more solid, link with builder or next TODO?
        stateful_init = [
            "_integ_", "_delay_", "_delayfixed_", "_delayn_",
            "_sample_if_true_", "_smooth_", "_trend_", "_initial_"]

        for key, value in initial_value.items():
            component_name = utils.get_value_by_insensitive_key_or_value(
                key, self.components._namespace)
            if component_name is not None:
                for element in self._stateful_elements:
                    # TODO make this more solid, add link between stateful
                    # objects and model vars
                    for init in stateful_init:
                        if init + component_name == element.py_name:
                            stateful_name = element.py_name
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
            if hasattr(self.components, stateful_name):
                element = getattr(self.components, stateful_name)
                if dims:
                    value = utils.rearrange(
                        value, dims,
                        self.components._subscript_dict)
                element.initialize(value)
                # TODO clean only necessary cache
                self.cache.clean()
            else:
                # Try to override component
                raise ValueError(
                    f"\nSetting {component_name} to a constant value with "
                    "initial_conditions. Use params={"
                    f"'{component_name}': {value}"+"} instead.")

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


class Time(object):
    def __init__(self, t=None, dt=None):
        self._t = t
        self._step = dt
        self.stage = None

    def __call__(self):
        return self._t

    def step(self):
        return self._step

    def update(self, value):
        if self._t is not None:
            self._step = value - self._t

        self._t = value


class Model(Macro):
    def __init__(self, py_model_file, initialize, missing_values):
        """ Sets up the python objects """
        super().__init__(py_model_file, None, None, Time())
        self.time.stage = 'Load'
        self.missing_values = missing_values
        if initialize:
            self.initialize()

    def initialize(self):
        """ Initializes the simulation model """
        self.time.update(self.components.initial_time())
        self.time.stage = 'Initialization'
        External.missing = self.missing_values
        super().initialize()

    def _build_euler_timeseries(self, return_timestamps=None, final_time=None):
        """
        - The integration steps need to include the return values.
        - There is no point running the model past the last return value.
        - The last timestep will be the last in that requested for return
        - Spacing should be at maximum what is specified by the integration
          time step.
        - The initial time should be the one specified by the model file,
          OR it should be the initial condition.
        - This function needs to be called AFTER the model is set in its
          initial state
        Parameters
        ----------
        return_timestamps: numpy array
            Must be specified by user or built from model file before this
            function is called.

        final_time: float or None
            Final time of the simulation. If float, the given final time
            will be used. If None, the last return_timestamps will be used.
            Default is None.

        Returns
        -------
        ts: numpy array
            The times that the integrator will use to compute time history

        """
        t_0 = self.time()
        try:
            t_f = return_timestamps[-1]
        except IndexError:
            # return_timestamps is an empty list
            # model default final time or passed argument value
            t_f = self.components.final_time()

        if final_time is not None:
            t_f = max(final_time, t_f)

        ts = np.arange(
            t_0,
            t_f+self.components.time_step()/2,
            self.components.time_step(),
            dtype=np.float64
        )

        # Add the returned time series into the integration array.
        # Best we can do for now. This does change the integration ever
        # so slightly, but for well-specified models there shouldn't be
        # sensitivity to a finer integration time step.
        return np.sort(np.unique(np.append(ts, return_timestamps)))

    def _format_return_timestamps(self, return_timestamps=None):
        """
        Format the passed in return timestamps value as a numpy array.
        If no value is passed, build up array of timestamps based upon
        model start and end times, and the 'saveper' value.

        Parameters
        ----------
        return_timestamps: float, iterable of floats or None (optional)
          Iterable of timestamps to return or None. Default is None.

        Returns
        -------
        ndarray (float)

        """
        if return_timestamps is None:
            # Build based upon model file Start, Stop times and Saveper
            # Vensim's standard is to expect that the data set includes
            # the `final time`, so we have to add an extra period to
            # make sure we get that value in what numpy's `arange` gives us.
            return np.arange(
                self.time(),
                self.components.final_time() + self.components.saveper()/2,
                self.components.saveper(), dtype=float
            )

        try:
            return np.array(return_timestamps, ndmin=1, dtype=float)
        except Exception:
            raise TypeError(
                '`return_timestamps` expects an iterable of numeric values'
                ' or a single numeric value')

    def run(self, params=None, return_columns=None, return_timestamps=None,
            initial_condition='original', final_time=None, time_step=None,
            saveper=None, reload=False, progress=False, flatten_output=False):
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

        # TODO move control variables to a class
        if params is None:
            params = {}
        if final_time:
            params['final_time'] = final_time
        elif return_timestamps is not None:
            params['final_time'] =\
                self._format_return_timestamps(return_timestamps)[-1]
        if time_step:
            params['time_step'] = time_step
        if saveper:
            params['saveper'] = saveper
        # END TODO

        if params:
            self.set_components(params)

        # update cache types after setting params
        self._assign_cache_type()

        self.set_initial_condition(initial_condition)
        # TODO move control variables to a class
        # save control variables
        replace = {
            'initial_time': self.time()
        }
        # END TODO

        self._add_constant_cache()

        return_timestamps = self._format_return_timestamps(return_timestamps)

        t_series = self._build_euler_timeseries(return_timestamps, final_time)

        if return_columns is None or isinstance(return_columns, str):
            return_columns = self._default_return_columns(return_columns)

        self.time.stage = 'Run'
        # need to clean cache to remove the values from active_initial
        self.cache.clean()

        capture_elements, return_addresses = utils.get_return_elements(
            return_columns, self.components._namespace)

        # create a dictionary splitting run cached and others
        capture_elements = self._split_capture_elements(capture_elements)

        res = self._integrate(t_series, capture_elements['step'],
                              return_timestamps)

        self._add_run_elements(res, capture_elements['run'], replace=replace)
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
        PySD.set_initial_value()

        """

        if isinstance(initial_condition, tuple):
            self.initialize()
            self.set_initial_value(*initial_condition)
        elif isinstance(initial_condition, str):
            if initial_condition.lower() in ['original', 'o']:
                self.initialize()
            elif initial_condition.lower() in ['current', 'c']:
                pass
            else:
                self.import_pickle(initial_condition)
        else:
            raise TypeError('Check documentation for valid entries')

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

    def _integrate(self, time_steps, capture_elements, return_timestamps):
        """
        Performs euler integration

        Parameters
        ----------
        time_steps: iterable
            the time steps that the integrator progresses over
        capture_elements: set
            which model elements to capture - uses pysafe names
        return_timestamps:
            which subset of 'timesteps' should be values be returned?

        Returns
        -------
        outputs: list of dictionaries

        """
        # necessary to have always a non-xaray object for appending objects
        # to the DataFrame time will always be a model element and not saved
        # TODO: find a better way of saving outputs
        capture_elements.add("time")
        outputs = pd.DataFrame(columns=capture_elements)

        if self.progress:
            # initialize progress bar
            progressbar = utils.ProgressBar(len(time_steps)-1)
        else:
            # when None is used the update will do nothing
            progressbar = utils.ProgressBar(None)

        for t2 in time_steps[1:]:
            if self.time() in return_timestamps:
                outputs.at[self.time()] = [getattr(self.components, key)()
                                           for key in capture_elements]
            self._euler_step(t2 - self.time())
            self.time.update(t2)
            self.clean_caches()
            progressbar.update()
            # TODO move control variables to a class and automatically stop
            # when updating time
            if self.time() >= self.components.final_time():
                break

        # need to add one more time step, because we run only the state
        # updates in the previous loop and thus may be one short.
        if self.time() in return_timestamps:
            outputs.at[self.time()] = [getattr(self.components, key)()
                                       for key in capture_elements]

        progressbar.finish()

        # delete time column as it was created only for avoiding errors
        # of appending data. See previous TODO.
        del outputs["time"]
        return outputs

    def _add_run_elements(self, df, capture_elements, replace={}):
        """
        Adds constant elements to a dataframe.

        Parameters
        ----------
        df: pandas.DataFrame
            Dataframe to add elements.

        capture_elements: list
            List of constant elements

        replace: dict
            Ouputs values to replace.
            TODO: move control variables to a class and avoid this.

        Returns
        -------
        None

        """
        nt = len(df.index.values)
        for element in capture_elements:
            df[element] = [getattr(self.components, element)()] * nt

        # TODO: move control variables to a class and avoid this.
        # update initial time values in df (necessary if initial_conditions)
        for it, value in replace.items():
            if it in df:
                df[it] = value
            elif it.upper() in df:
                df[it.upper()] = value
            elif it.replace('_', ' ') in df:
                df[it.replace('_', ' ')] = value
            elif it.replace('_', ' ').upper() in df:
                df[it.replace('_', ' ').upper()] = value
