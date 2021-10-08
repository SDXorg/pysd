import unittest

import numpy as np

from pysd.py_backend.time import Time


class TestTime(unittest.TestCase):
    def test_set_control_vars(self):
        time = Time()

        def initial():
            return 2

        def final():
            return 23

        def step():
            return 0.25

        def save():
            return 0.75

        time.set_control_vars(
            initial=initial, final=final, save=save, step=step)

        self.assertEqual(time(), 2)
        self.assertEqual(time.initial(), 2)
        self.assertEqual(time.final(), 23)
        self.assertEqual(time.step(), 0.25)
        self.assertEqual(time.save(), 0.75)

        time.update(10)

        self.assertEqual(time(), 10)
        self.assertEqual(time.initial(), 2)
        self.assertEqual(time.final(), 23)
        self.assertEqual(time.step(), 0.25)
        self.assertEqual(time.save(), 0.75)

        time.reset()

        self.assertEqual(time(), 2)
        self.assertEqual(time.initial(), 2)
        self.assertEqual(time.final(), 23)
        self.assertEqual(time.step(), 0.25)
        self.assertEqual(time.save(), 0.75)

        time.set_control_vars(
            save=lambda: 2, step=lambda: 1)

        self.assertEqual(time(), 2)
        self.assertEqual(time.initial(), 2)
        self.assertEqual(time.final(), 23)
        self.assertEqual(time.step(), 1)
        self.assertEqual(time.save(), 2)

    def test_set_control_vars_with_constants(self):
        time = Time()
        time.set_control_vars(
            initial=2, final=23, save=0.75, step=0.25)

        self.assertEqual(time(), 2)
        self.assertEqual(time.initial(), 2)
        self.assertEqual(time.final(), 23)
        self.assertEqual(time.step(), 0.25)
        self.assertEqual(time.save(), 0.75)

        time.set_control_vars(
            initial=6)

        self.assertEqual(time(), 6)
        self.assertEqual(time.initial(), 6)
        self.assertEqual(time.final(), 23)
        self.assertEqual(time.step(), 0.25)
        self.assertEqual(time.save(), 0.75)

        time.set_control_vars(
            final=50, save=4, step=1)

        self.assertEqual(time(), 6)
        self.assertEqual(time.initial(), 6)
        self.assertEqual(time.final(), 50)
        self.assertEqual(time.step(), 1)
        self.assertEqual(time.save(), 4)

    def test_in_bounds(self):
        time = Time()
        time.set_control_vars(
            initial=2, final=23, save=0.75, step=0.25)

        self.assertTrue(time.in_bounds())
        time.update(21)
        self.assertTrue(time.in_bounds())
        time.update(23)
        self.assertFalse(time.in_bounds())
        time.update(24)
        self.assertFalse(time.in_bounds())

        my_time = {"final": 30}

        def final():
            return my_time["final"]

        time.set_control_vars(
            initial=2, final=final, save=0.75, step=0.25)

        # dynamic final time
        self.assertTrue(time.in_bounds())
        time.update(23)
        self.assertTrue(time.in_bounds())
        my_time["final"] = 20
        self.assertFalse(time.in_bounds())
        my_time["final"] = 50
        self.assertTrue(time.in_bounds())

    def test_in_return_saveper(self):
        time = Time()
        time.set_control_vars(
            initial=2, final=100, save=0.75, step=0.25)

        self.assertTrue(time.in_return())
        time.update(2.25)
        self.assertFalse(time.in_return())
        time.update(2.75)
        self.assertTrue(time.in_return())
        time.update(77)
        self.assertTrue(time.in_return())

        # dynamical initial
        my_time = {"initial": 2}

        def initial():
            return my_time["initial"]

        time.set_control_vars(
            initial=initial, final=100, save=0.75, step=0.25)

        self.assertTrue(time.in_return())
        time.update(2.25)
        self.assertFalse(time.in_return())
        time.update(2.75)
        self.assertTrue(time.in_return())
        time.update(77)
        self.assertTrue(time.in_return())

        # changing initial time var during run must no affect saving time
        my_time["initial"] = 2.25

        time.reset()
        self.assertEqual(time.initial(), 2.25)
        self.assertTrue(time.in_return())
        time.update(2.25)
        self.assertFalse(time.in_return())
        time.update(2.75)
        self.assertTrue(time.in_return())
        time.update(77)
        self.assertTrue(time.in_return())

        # dynamical saveper
        my_time["save"] = 0.75

        def save():
            return my_time["save"]

        time.set_control_vars(
            initial=2, final=100, save=save, step=0.25)

        self.assertTrue(time.in_return())
        time.update(2.25)
        self.assertFalse(time.in_return())
        time.update(2.75)
        self.assertTrue(time.in_return())
        time.update(3)
        self.assertFalse(time.in_return())

        my_time["save"] = 1

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
            initial=2, final=100, save=1, step=0.25)

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
