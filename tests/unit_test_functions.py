import unittest

import numpy as np
import xarray as xr


class TestInputFunctions(unittest.TestCase):
    def test_ramp(self):
        from pysd.py_backend.functions import ramp

        self.assertEqual(ramp(lambda: 14, .5, 10, 18), 2)

        self.assertEqual(ramp(lambda: 4, .5, 10, 18), 0)

        self.assertEqual(ramp(lambda: 24, .5, 10, 18), 4)

        self.assertEqual(ramp(lambda: 24, .5, 10, -1), 7)

    def test_step(self):
        from pysd.py_backend.functions import step

        self.assertEqual(step(lambda: 5, 1, 10), 0)

        self.assertEqual(step(lambda: 15, 1, 10), 1)

        self.assertEqual(step(lambda: 10, 1, 10), 1)

    def test_pulse(self):
        from pysd.py_backend.functions import pulse

        self.assertEqual(pulse(lambda: 0, 1, 3), 0)

        self.assertEqual(pulse(lambda: 1, 1, 3), 1)

        self.assertEqual(pulse(lambda: 2, 1, 3), 1)

        self.assertEqual(pulse(lambda: 4, 1, 3), 0)

        self.assertEqual(pulse(lambda: 5, 1, 3), 0)

    def test_pulse_chain(self):
        from pysd.py_backend.functions import pulse_train
        # before train starts
        self.assertEqual(pulse_train(lambda: 0, 1, 3, 5, 12), 0)
        # on train start
        self.assertEqual(pulse_train(lambda: 1, 1, 3, 5, 12), 1)
        # within first pulse
        self.assertEqual(pulse_train(lambda: 2, 1, 3, 5, 12), 1)
        # end of first pulse
        self.assertEqual(pulse_train(lambda: 4, 1, 3, 5, 12), 0)
        # after first pulse before second
        self.assertEqual(pulse_train(lambda: 5, 1, 3, 5, 12), 0)
        # on start of second pulse
        self.assertEqual(pulse_train(lambda: 6, 1, 3, 5, 12), 1)
        # within second pulse
        self.assertEqual(pulse_train(lambda: 7, 1, 3, 5, 12), 1)
        # after second pulse
        self.assertEqual(pulse_train(lambda: 10, 1, 3, 5, 12), 0)
        # on third pulse
        self.assertEqual(pulse_train(lambda: 11, 1, 3, 5, 12), 1)
        # on train end
        self.assertEqual(pulse_train(lambda: 12, 1, 3, 5, 12), 0)
        # after train
        self.assertEqual(pulse_train(lambda: 15, 1, 3, 5, 13), 0)

    def test_pulse_magnitude(self):
        from pysd.py_backend.functions import pulse_magnitude
        from pysd.py_backend.statefuls import Time

        # Pulse function with repeat time
        # before first impulse
        t = Time()
        t.set_control_vars(initial=0, step=1)
        self.assertEqual(pulse_magnitude(t, 10, 2, 5), 0)
        # first impulse
        t.update(2)
        self.assertEqual(pulse_magnitude(t, 10, 2, 5), 10)
        # after first impulse and before second
        t.update(4)
        self.assertEqual(pulse_magnitude(t, 10, 2, 5), 0)
        # second impulse
        t.update(7)
        self.assertEqual(pulse_magnitude(t, 10, 2, 5), 10)
        # after second and before third impulse
        t.update(9)
        self.assertEqual(pulse_magnitude(t, 10, 2, 5), 0)
        # third impulse
        t.update(12)
        self.assertEqual(pulse_magnitude(t, 10, 2, 5), 10)
        # after third impulse
        t.update(14)
        self.assertEqual(pulse_magnitude(t, 10, 2, 5), 0)

        # Pulse function without repeat time
        # before first impulse
        t = Time()
        t.set_control_vars(initial=0, step=1)
        self.assertEqual(pulse_magnitude(t, 10, 2), 0)
        # first impulse
        t.update(2)
        self.assertEqual(pulse_magnitude(t, 10, 2), 10)
        # after first impulse and before second
        t.update(4)
        self.assertEqual(pulse_magnitude(t, 10, 2), 0)
        # second impulse
        t.update(7)
        self.assertEqual(pulse_magnitude(t, 10, 2), 0)
        # after second and before third impulse
        t.update(9)
        self.assertEqual(pulse_magnitude(t, 10, 2), 0)

    def test_xidz(self):
        from pysd.py_backend.functions import xidz
        self.assertEqual(xidz(1, -0.00000001, 5), 5)
        self.assertEqual(xidz(1, 0, 5), 5)
        self.assertEqual(xidz(1, 8, 5), 0.125)

    def test_zidz(self):
        from pysd.py_backend.functions import zidz
        self.assertEqual(zidz(1, -0.00000001), 0)
        self.assertEqual(zidz(1, 0), 0)
        self.assertEqual(zidz(1, 8), 0.125)


class TestStatsFunctions(unittest.TestCase):
    def test_bounded_normal(self):
        from pysd.py_backend.functions import bounded_normal
        min_val = -4
        max_val = .2
        mean = -1
        std = .05
        seed = 1
        results = np.array(
            [bounded_normal(min_val, max_val, mean, std, seed)
             for _ in range(1000)])

        self.assertGreaterEqual(results.min(), min_val)
        self.assertLessEqual(results.max(), max_val)
        self.assertAlmostEqual(results.mean(), mean, delta=std)
        self.assertAlmostEqual(results.std(), std, delta=std)
        self.assertGreater(len(np.unique(results)), 100)


class TestLogicFunctions(unittest.TestCase):
    def test_if_then_else_basic(self):
        from pysd.py_backend.functions import if_then_else
        self.assertEqual(if_then_else(True, lambda: 1, lambda: 0), 1)
        self.assertEqual(if_then_else(False, lambda: 1, lambda: 0), 0)

        # Ensure lazzy evaluation
        self.assertEqual(if_then_else(True, lambda: 1, lambda: 1/0), 1)
        self.assertEqual(if_then_else(False, lambda: 1/0, lambda: 0), 0)

        with self.assertRaises(ZeroDivisionError):
            if_then_else(True, lambda: 1/0, lambda: 0)
        with self.assertRaises(ZeroDivisionError):
            if_then_else(False, lambda: 1, lambda: 1/0)

    def test_if_then_else_with_subscripted(self):
        # this test only test the lazzy evaluation and basics
        # subscripted_if_then_else test all the possibilities

        from pysd.py_backend.functions import if_then_else

        coords = {'dim1': [0, 1], 'dim2': [0, 1]}
        dims = list(coords)

        xr_true = xr.DataArray([[True, True], [True, True]], coords, dims)
        xr_false = xr.DataArray([[False, False], [False, False]], coords, dims)
        xr_mixed = xr.DataArray([[True, False], [False, True]], coords, dims)

        out_mixed = xr.DataArray([[1, 0], [0, 1]], coords, dims)

        self.assertEqual(if_then_else(xr_true, lambda: 1, lambda: 0), 1)
        self.assertEqual(if_then_else(xr_false, lambda: 1, lambda: 0), 0)
        self.assertTrue(
            if_then_else(xr_mixed, lambda: 1, lambda: 0).equals(out_mixed))

        # Ensure lazzy evaluation
        self.assertEqual(if_then_else(xr_true, lambda: 1, lambda: 1/0), 1)
        self.assertEqual(if_then_else(xr_false, lambda: 1/0, lambda: 0), 0)

        with self.assertRaises(ZeroDivisionError):
            if_then_else(xr_true, lambda: 1/0, lambda: 0)
        with self.assertRaises(ZeroDivisionError):
            if_then_else(xr_false, lambda: 1, lambda: 1/0)
        with self.assertRaises(ZeroDivisionError):
            if_then_else(xr_mixed, lambda: 1/0, lambda: 0)
        with self.assertRaises(ZeroDivisionError):
            if_then_else(xr_mixed, lambda: 1, lambda: 1/0)


class TestLookup(unittest.TestCase):
    def test_lookup(self):
        from pysd.py_backend.functions import lookup

        xpts = [0, 1, 2, 3,  5,  6, 7, 8]
        ypts = [0, 0, 1, 1, -1, -1, 0, 0]

        for x, y in zip(xpts, ypts):
            self.assertEqual(
                y,
                lookup(x, xpts, ypts),
                "Wrong result at X=" + str(x))

    def test_lookup_extrapolation_inbounds(self):
        from pysd.py_backend.functions import lookup_extrapolation

        xpts = [0, 1, 2, 3,  5,  6, 7, 8]
        ypts = [0, 0, 1, 1, -1, -1, 0, 0]

        expected_xpts = np.arange(-0.5, 8.6, 0.5)
        expected_ypts = [
            0,
            0, 0,
            0, 0.5,
            1, 1,
            1, 0.5, 0, -0.5, -1,
            -1, -1, -0.5,
            0, 0, 0, 0
        ]

        for x, y in zip(expected_xpts, expected_ypts):
            self.assertEqual(
                y,
                lookup_extrapolation(x, xpts, ypts),
                "Wrong result at X=" + str(x))

    def test_lookup_extrapolation_two_points(self):
        from pysd.py_backend.functions import lookup_extrapolation

        xpts = [0, 1]
        ypts = [0, 1]

        expected_xpts = np.arange(-0.5, 1.6, 0.5)
        expected_ypts = [-0.5, 0.0, 0.5, 1.0, 1.5]

        for x, y in zip(expected_xpts, expected_ypts):
            self.assertEqual(
                y,
                lookup_extrapolation(x, xpts, ypts),
                "Wrong result at X=" + str(x))

    def test_lookup_extrapolation_outbounds(self):
        from pysd.py_backend.functions import lookup_extrapolation

        xpts = [0, 1, 2, 3]
        ypts = [0, 1, 1, 0]

        expected_xpts = np.arange(-0.5, 3.6, 0.5)
        expected_ypts = [
            -0.5,
            0.0, 0.5, 1.0,
            1.0, 1.0,
            0.5, 0,
            -0.5
        ]

        for x, y in zip(expected_xpts, expected_ypts):
            self.assertEqual(
                y,
                lookup_extrapolation(x, xpts, ypts),
                "Wrong result at X=" + str(x))

    def test_lookup_discrete(self):
        from pysd.py_backend.functions import lookup_discrete

        xpts = [0, 1, 2, 3,  5,  6, 7, 8]
        ypts = [0, 0, 1, 1, -1, -1, 0, 0]

        expected_xpts = np.arange(-0.5, 8.6, 0.5)
        expected_ypts = [
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1,
            -1, -1, -1, -1,
            0, 0, 0, 0
        ]

        for x, y in zip(expected_xpts, expected_ypts):
            self.assertEqual(
                y,
                lookup_discrete(x, xpts, ypts),
                "Wrong result at X=" + str(x))


class TestFunctions(unittest.TestCase):

    def test_sum(self):
        """
        Test for sum function
        """
        from pysd.py_backend.functions import sum

        coords = {'d1': [9, 1], 'd2': [2, 4]}
        coords_d1, coords_d2 = {'d1': [9, 1]}, {'d2': [2, 4]}
        dims = ['d1', 'd2']

        data = xr.DataArray([[1, 2], [3, 4]], coords, dims)

        self.assertTrue(sum(
            data,
            dim=['d1']).equals(xr.DataArray([4, 6], coords_d2, ['d2'])))
        self.assertTrue(sum(
            data,
            dim=['d2']).equals(xr.DataArray([3, 7], coords_d1, ['d1'])))
        self.assertEqual(sum(data, dim=['d1', 'd2']), 10)
        self.assertEqual(sum(data), 10)

    def test_prod(self):
        """
        Test for sum function
        """
        from pysd.py_backend.functions import prod

        coords = {'d1': [9, 1], 'd2': [2, 4]}
        coords_d1, coords_d2 = {'d1': [9, 1]}, {'d2': [2, 4]}
        dims = ['d1', 'd2']

        data = xr.DataArray([[1, 2], [3, 4]], coords, dims)

        self.assertTrue(prod(
            data,
            dim=['d1']).equals(xr.DataArray([3, 8], coords_d2, ['d2'])))
        self.assertTrue(prod(
            data,
            dim=['d2']).equals(xr.DataArray([2, 12], coords_d1, ['d1'])))
        self.assertEqual(prod(data, dim=['d1', 'd2']), 24)
        self.assertEqual(prod(data), 24)

    def test_vmin(self):
        """
        Test for vmin function
        """
        from pysd.py_backend.functions import vmin

        coords = {'d1': [9, 1], 'd2': [2, 4]}
        coords_d1, coords_d2 = {'d1': [9, 1]}, {'d2': [2, 4]}
        dims = ['d1', 'd2']

        data = xr.DataArray([[1, 2], [3, 4]], coords, dims)

        self.assertTrue(vmin(
            data,
            dim=['d1']).equals(xr.DataArray([1, 2], coords_d2, ['d2'])))
        self.assertTrue(vmin(
            data,
            dim=['d2']).equals(xr.DataArray([1, 3], coords_d1, ['d1'])))
        self.assertEqual(vmin(data, dim=['d1', 'd2']), 1)
        self.assertEqual(vmin(data), 1)

    def test_vmax(self):
        """
        Test for vmax function
        """
        from pysd.py_backend.functions import vmax

        coords = {'d1': [9, 1], 'd2': [2, 4]}
        coords_d1, coords_d2 = {'d1': [9, 1]}, {'d2': [2, 4]}
        dims = ['d1', 'd2']

        data = xr.DataArray([[1, 2], [3, 4]], coords, dims)

        self.assertTrue(vmax(
            data,
            dim=['d1']).equals(xr.DataArray([3, 4], coords_d2, ['d2'])))
        self.assertTrue(vmax(
            data,
            dim=['d2']).equals(xr.DataArray([2, 4], coords_d1, ['d1'])))
        self.assertEqual(vmax(data, dim=['d1', 'd2']), 4)
        self.assertEqual(vmax(data), 4)

    def test_invert_matrix(self):
        """
        Test for invert_matrix function
        """
        from pysd.py_backend.functions import invert_matrix

        coords1 = {'d1': ['a', 'b'], 'd2': ['a', 'b']}
        coords2 = {'d0': ['1', '2'], 'd1': ['a', 'b'], 'd2': ['a', 'b']}
        coords3 = {'d0': ['1', '2'],
                   'd1': ['a', 'b', 'c'],
                   'd2': ['a', 'b', 'c']}

        data1 = xr.DataArray([[1, 2], [3, 4]], coords1, ['d1', 'd2'])
        data2 = xr.DataArray([[[1, 2], [3, 4]], [[-1, 2], [5, 4]]],
                             coords2,
                             ['d0', 'd1', 'd2'])
        data3 = xr.DataArray([[[1, 2, 3], [3, 7, 2], [3, 4, 6]],
                              [[-1, 2, 3], [4, 7, 3], [5, 4, 6]]],
                             coords3,
                             ['d0', 'd1', 'd2'])

        for data in [data1, data2, data3]:
            datai = invert_matrix(data)
            self.assertEqual(data.dims, datai.dims)

            if len(data.shape) == 2:
                # two dimensions xarrays
                self.assertTrue((
                    abs(np.dot(data, datai) - np.dot(datai, data))
                    < 1e-14
                    ).all())
                self.assertTrue((
                    abs(np.dot(data, datai) - np.identity(data.shape[-1]))
                    < 1e-14
                    ).all())
            else:
                # three dimensions xarrays
                for i in range(data.shape[0]):
                    self.assertTrue((
                        abs(np.dot(data[i], datai[i])
                            - np.dot(datai[i], data[i]))
                        < 1e-14
                        ).all())
                    self.assertTrue((
                        abs(np.dot(data[i], datai[i])
                            - np.identity(data.shape[-1]))
                        < 1e-14
                        ).all())

    def test_incomplete(self):
        from pysd.py_backend.functions import incomplete
        from warnings import catch_warnings

        with catch_warnings(record=True) as w:
            incomplete()
            self.assertEqual(len(w), 1)
            self.assertTrue('Call to undefined function' in str(w[-1].message))

    def test_not_implemented_function(self):
        from pysd.py_backend.functions import not_implemented_function

        with self.assertRaises(NotImplementedError):
            not_implemented_function("NIF")
