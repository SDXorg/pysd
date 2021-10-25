import unittest

import numpy as np

from pysd.py_backend.components import Time


class TestTime(unittest.TestCase):
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

        self.assertEqual(time(), 2)
        self.assertEqual(time.initial_time(), 2)
        self.assertEqual(time.final_time(), 23)
        self.assertEqual(time.time_step(), 0.25)
        self.assertEqual(time.saveper(), 0.75)

        time.update(10)

        self.assertEqual(time(), 10)
        self.assertEqual(time.initial_time(), 2)
        self.assertEqual(time.final_time(), 23)
        self.assertEqual(time.time_step(), 0.25)
        self.assertEqual(time.saveper(), 0.75)

        time.reset()

        self.assertEqual(time(), 2)
        self.assertEqual(time.initial_time(), 2)
        self.assertEqual(time.final_time(), 23)
        self.assertEqual(time.time_step(), 0.25)
        self.assertEqual(time.saveper(), 0.75)

        time.set_control_vars(
            saveper=lambda: 2, time_step=lambda: 1)

        self.assertEqual(time(), 2)
        self.assertEqual(time.initial_time(), 2)
        self.assertEqual(time.final_time(), 23)
        self.assertEqual(time.time_step(), 1)
        self.assertEqual(time.saveper(), 2)

    def test_set_control_vars_with_constants(self):
        time = Time()
        time.set_control_vars(
            initial_time=2, final_time=23, saveper=0.75, time_step=0.25)

        self.assertEqual(time(), 2)
        self.assertEqual(time.initial_time(), 2)
        self.assertEqual(time.final_time(), 23)
        self.assertEqual(time.time_step(), 0.25)
        self.assertEqual(time.saveper(), 0.75)

        time.set_control_vars(
            initial_time=6)

        self.assertEqual(time(), 6)
        self.assertEqual(time.initial_time(), 6)
        self.assertEqual(time.final_time(), 23)
        self.assertEqual(time.time_step(), 0.25)
        self.assertEqual(time.saveper(), 0.75)

        time.set_control_vars(
            final_time=50, saveper=4, time_step=1)

        self.assertEqual(time(), 6)
        self.assertEqual(time.initial_time(), 6)
        self.assertEqual(time.final_time(), 50)
        self.assertEqual(time.time_step(), 1)
        self.assertEqual(time.saveper(), 4)

    def test_in_bounds(self):
        time = Time()
        time.set_control_vars(
            initial_time=2, final_time=23, saveper=0.75, time_step=0.25)

        self.assertTrue(time.in_bounds())
        time.update(21)
        self.assertTrue(time.in_bounds())
        time.update(23)
        self.assertFalse(time.in_bounds())
        time.update(24)
        self.assertFalse(time.in_bounds())

        my_time = {"final_time": 30}

        def final_time():
            return my_time["final_time"]

        time.set_control_vars(
            initial_time=2, final_time=final_time,
            saveper=0.75, time_step=0.25)

        # dynamic final_time time
        self.assertTrue(time.in_bounds())
        time.update(23)
        self.assertTrue(time.in_bounds())
        my_time["final_time"] = 20
        self.assertFalse(time.in_bounds())
        my_time["final_time"] = 50
        self.assertTrue(time.in_bounds())

    def test_in_return_saveperper(self):
        time = Time()
        time.set_control_vars(
            initial_time=2, final_time=100, saveper=0.75, time_step=0.25)

        self.assertTrue(time.in_return())
        time.update(2.25)
        self.assertFalse(time.in_return())
        time.update(2.75)
        self.assertTrue(time.in_return())
        time.update(77)
        self.assertTrue(time.in_return())

        # dynamical initial_time
        my_time = {"initial_time": 2}

        def initial_time():
            return my_time["initial_time"]

        time.set_control_vars(
            initial_time=initial_time, final_time=100,
            saveper=0.75, time_step=0.25)

        self.assertTrue(time.in_return())
        time.update(2.25)
        self.assertFalse(time.in_return())
        time.update(2.75)
        self.assertTrue(time.in_return())
        time.update(77)
        self.assertTrue(time.in_return())

        # changing initial_time time var during run must no affect saving time
        my_time["initial_time"] = 2.25

        time.reset()
        self.assertEqual(time.initial_time(), 2.25)
        self.assertTrue(time.in_return())
        time.update(2.25)
        self.assertFalse(time.in_return())
        time.update(2.75)
        self.assertTrue(time.in_return())
        time.update(77)
        self.assertTrue(time.in_return())

        # dynamical saveperper
        my_time["saveper"] = 0.75

        def saveper():
            return my_time["saveper"]

        time.set_control_vars(
            initial_time=2, final_time=100, saveper=saveper, time_step=0.25)

        self.assertTrue(time.in_return())
        time.update(2.25)
        self.assertFalse(time.in_return())
        time.update(2.75)
        self.assertTrue(time.in_return())
        time.update(3)
        self.assertFalse(time.in_return())

        my_time["saveper"] = 1

        time.reset()
        self.assertTrue(time.in_return())
        time.update(2.25)
        self.assertFalse(time.in_return())
        time.update(2.75)
        self.assertFalse(time.in_return())
        time.update(3)
        self.assertTrue(time.in_return())

    def test_in_return_timestamps(self):
        time = Time()
        time.set_control_vars(
            initial_time=2, final_time=100, saveper=1, time_step=0.25)

        self.assertTrue(time.in_return())
        time.update(4)
        self.assertTrue(time.in_return())
        time.update(10)
        self.assertTrue(time.in_return())
        time.update(12)
        self.assertTrue(time.in_return())
        time.update(37)
        self.assertTrue(time.in_return())

        time.reset()
        time.add_return_timestamps([2, 10, 37])
        self.assertTrue(time.in_return())
        time.update(4)
        self.assertFalse(time.in_return())
        time.update(10)
        self.assertTrue(time.in_return())
        time.update(12)
        self.assertFalse(time.in_return())
        time.update(37)
        self.assertTrue(time.in_return())

        time.reset()
        time.add_return_timestamps(np.array([4, 12]))
        self.assertFalse(time.in_return())
        time.update(4)
        self.assertTrue(time.in_return())
        time.update(10)
        self.assertFalse(time.in_return())
        time.update(12)
        self.assertTrue(time.in_return())
        time.update(37)
        self.assertFalse(time.in_return())

        time.reset()
        time.add_return_timestamps(37)
        self.assertFalse(time.in_return())
        time.update(4)
        self.assertFalse(time.in_return())
        time.update(10)
        self.assertFalse(time.in_return())
        time.update(12)
        self.assertFalse(time.in_return())
        time.update(37)
        self.assertTrue(time.in_return())

        time.reset()
        time.add_return_timestamps(None)
        self.assertTrue(time.in_return())
        time.update(4)
        self.assertTrue(time.in_return())
        time.update(10)
        self.assertTrue(time.in_return())
        time.update(12)
        self.assertTrue(time.in_return())
        time.update(37)
        self.assertTrue(time.in_return())
