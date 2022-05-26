import numpy as np

from pysd.py_backend.components import Time, Components
from pysd import read_vensim


class TestComponents():
    def test_load_components(self, _root):
        test_model = _root.joinpath("test-models/samples/teacup/teacup.mdl")
        test_model_py = _root.joinpath("test-models/samples/teacup/teacup.py")

        read_vensim(test_model)

        # set function for testing
        executed = []

        def set_component(input_dict):
            executed.append(("SET", input_dict))

        # create object
        components = Components(test_model_py, set_component)

        # main attributes of the class
        assert hasattr(components, "_components")
        assert hasattr(components, "_set_components")

        # check getting elements
        assert components.room_temperature() == 70

        # check setting elements
        components.room_temperature = 5

        assert ("SET", {"room_temperature": 5}) in executed

        def temperature():
            return 34

        components.teacup_temperature = temperature

        assert ("SET", {"teacup_temperature": temperature}) in executed


class TestTime():
    def test_set_control_vars(self):
        time = Time()

        def initial_time():
            return 2

        def final_time():
            return 23

        def time_step():
            return 0.25

        def saveper():
            return 0.75

        time.set_control_vars(
            initial_time=initial_time, final_time=final_time,
            saveper=saveper, time_step=time_step)

        assert time() == 2
        assert time.initial_time() == 2
        assert time.final_time() == 23
        assert time.time_step() == 0.25
        assert time.saveper() == 0.75

        time.update(10)

        assert time() == 10
        assert time.initial_time() == 2
        assert time.final_time() == 23
        assert time.time_step() == 0.25
        assert time.saveper() == 0.75

        time.reset()

        assert time() == 2
        assert time.initial_time() == 2
        assert time.final_time() == 23
        assert time.time_step() == 0.25
        assert time.saveper() == 0.75

        time.set_control_vars(
            saveper=lambda: 2, time_step=lambda: 1)

        assert time() == 2
        assert time.initial_time() == 2
        assert time.final_time() == 23
        assert time.time_step() == 1
        assert time.saveper() == 2

    def test_set_control_vars_with_constants(self):
        time = Time()
        time.set_control_vars(
            initial_time=2, final_time=23, saveper=0.75, time_step=0.25)

        assert time() == 2
        assert time.initial_time() == 2
        assert time.final_time() == 23
        assert time.time_step() == 0.25
        assert time.saveper() == 0.75

        time.set_control_vars(
            initial_time=6)

        assert time() == 6
        assert time.initial_time() == 6
        assert time.final_time() == 23
        assert time.time_step() == 0.25
        assert time.saveper() == 0.75

        time.set_control_vars(
            final_time=50, saveper=4, time_step=1)

        assert time() == 6
        assert time.initial_time() == 6
        assert time.final_time() == 50
        assert time.time_step() == 1
        assert time.saveper() == 4

    def test_in_bounds(self):
        time = Time()
        time.set_control_vars(
            initial_time=2, final_time=23, saveper=0.75, time_step=0.25)

        assert time.in_bounds()
        time.update(21)
        assert time.in_bounds()
        time.update(23)
        assert not time.in_bounds()
        time.update(24)
        assert not time.in_bounds()

        my_time = {"final_time": 30}

        def final_time():
            return my_time["final_time"]

        time.set_control_vars(
            initial_time=2, final_time=final_time,
            saveper=0.75, time_step=0.25)

        # dynamic final_time time
        assert time.in_bounds()
        time.update(23)
        assert time.in_bounds()
        my_time["final_time"] = 20
        assert not time.in_bounds()
        my_time["final_time"] = 50
        assert time.in_bounds()

    def test_in_return_saveperper(self):
        time = Time()
        time.set_control_vars(
            initial_time=2, final_time=100, saveper=0.75, time_step=0.25)

        assert time.in_return()
        time.update(2.25)
        assert not time.in_return()
        time.update(2.75)
        assert time.in_return()
        time.update(77)
        assert time.in_return()

        # dynamical initial_time
        my_time = {"initial_time": 2}

        def initial_time():
            return my_time["initial_time"]

        time.set_control_vars(
            initial_time=initial_time, final_time=100,
            saveper=0.75, time_step=0.25)

        assert time.in_return()
        time.update(2.25)
        assert not time.in_return()
        time.update(2.75)
        assert time.in_return()
        time.update(77)
        assert time.in_return()

        # changing initial_time time var during run must no affect saving time
        my_time["initial_time"] = 2.25

        time.reset()
        assert time.initial_time() == 2.25
        assert time.in_return()
        time.update(2.25)
        assert not time.in_return()
        time.update(2.75)
        assert time.in_return()
        time.update(77)
        assert time.in_return()

        # dynamical saveperper
        my_time["saveper"] = 0.75

        def saveper():
            return my_time["saveper"]

        time.set_control_vars(
            initial_time=2, final_time=100, saveper=saveper, time_step=0.25)

        assert time.in_return()
        time.update(2.25)
        assert not time.in_return()
        time.update(2.75)
        assert time.in_return()
        time.update(3)
        assert not time.in_return()

        my_time["saveper"] = 1

        time.reset()
        assert time.in_return()
        time.update(2.25)
        assert not time.in_return()
        time.update(2.75)
        assert not time.in_return()
        time.update(3)
        assert time.in_return()

    def test_in_return_timestamps(self):
        time = Time()
        time.set_control_vars(
            initial_time=2, final_time=100, saveper=1, time_step=0.25)

        assert time.in_return()
        time.update(4)
        assert time.in_return()
        time.update(10)
        assert time.in_return()
        time.update(12)
        assert time.in_return()
        time.update(37)
        assert time.in_return()

        time.reset()
        time.add_return_timestamps([2, 10, 37])
        assert time.in_return()
        time.update(4)
        assert not time.in_return()
        time.update(10)
        assert time.in_return()
        time.update(12)
        assert not time.in_return()
        time.update(37)
        assert time.in_return()

        time.reset()
        time.add_return_timestamps(np.array([4, 12]))
        assert not time.in_return()
        time.update(4)
        assert time.in_return()
        time.update(10)
        assert not time.in_return()
        time.update(12)
        assert time.in_return()
        time.update(37)
        assert not time.in_return()

        time.reset()
        time.add_return_timestamps(37)
        assert not time.in_return()
        time.update(4)
        assert not time.in_return()
        time.update(10)
        assert not time.in_return()
        time.update(12)
        assert not time.in_return()
        time.update(37)
        assert time.in_return()

        time.reset()
        time.add_return_timestamps(None)
        assert time.in_return()
        time.update(4)
        assert time.in_return()
        time.update(10)
        assert time.in_return()
        time.update(12)
        assert time.in_return()
        time.update(37)
        assert time.in_return()
