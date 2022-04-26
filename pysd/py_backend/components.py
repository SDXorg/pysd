"""
Model components and time managing classes.
"""

import os
import random
import inspect
from importlib.machinery import SourceFileLoader

import numpy as np

from pysd._version import __version__


class Component(object):

    def __init__(self):
        self.namespace = {}
        self.dependencies = {}

    def add(self, name, units=None, limits=(np.nan, np.nan),
            subscripts=None, comp_type=None, comp_subtype=None,
            depends_on={}, other_deps={}):
        """
        This decorators allows assigning metadata to a function.
        """
        def decorator(function):
            function.name = name
            function.units = units
            function.limits = limits
            function.subscripts = subscripts
            function.type = comp_type
            function.subtype = comp_subtype
            function.args = inspect.getfullargspec(function)[0]

            # include component in namespace and dependencies
            self.namespace[name] = function.__name__
            if function.__name__ != "time":
                self.dependencies[function.__name__] = depends_on
                self.dependencies.update(other_deps)

            return function

        return decorator


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
    rprec = 1e-10  # relative precission for final time and saving time

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
        return self._time + self.time_step()*self.rprec < self.final_time()

    def in_return(self):
        """ Check if current time should be returned """
        if self.return_timestamps is not None:
            return self._time in self.return_timestamps

        time_delay = self._time - self._initial_time
        save_per = self.saveper()
        prec = self.time_step() * self.rprec
        return time_delay % save_per < prec or -time_delay % save_per < prec

    def round(self):
        """ Return rounded time to outputs to avoid float precission error"""
        return np.round(
            self._time,
            -int(np.log10(self.time_step()*self.rprec)))

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
