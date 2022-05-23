import pytest
import numpy as np
import xarray as xr

from pysd.py_backend.components import Time
from pysd.py_backend.functions import\
    ramp, step, pulse, xidz, zidz, if_then_else, sum, prod, vmin, vmax,\
    invert_matrix


class TestInputFunctions():
    def test_ramp(self):
        assert ramp(lambda: 14, .5, 10, 18) == 2

        assert ramp(lambda: 4, .5, 10, 18) == 0

        assert ramp(lambda: 24, .5, 10, 18) == 4

        assert ramp(lambda: 24, .5, 10) == 7

        assert ramp(lambda: 50, .5, 10) == 20

    def test_step(self):
        assert step(lambda: 5, 1, 10) == 0

        assert step(lambda: 15, 1, 10) == 1

        assert step(lambda: 10, 1, 10) == 1

    def test_pulse(self):
        assert pulse(lambda: 0, 1, width=3) == 0

        assert pulse(lambda: 1, 1, width=3) == 1

        assert pulse(lambda: 2, 1, width=3) == 1

        assert pulse(lambda: 4, 1, width=3) == 0

        assert pulse(lambda: 5, 1, width=3) == 0

    def test_pulse_chain(self):
        # before train starts
        assert pulse(lambda: 0, 1, repeat_time=5, width=3, end=12) == 0
        # on train start
        assert pulse(lambda: 1, 1, repeat_time=5, width=3, end=12) == 1
        # within first pulse
        assert pulse(lambda: 2, 1, repeat_time=5, width=3, end=12) == 1
        # end of first pulse
        assert pulse(lambda: 4, 1, repeat_time=5, width=3, end=12) == 0
        # after first pulse before second
        assert pulse(lambda: 5, 1, repeat_time=5, width=3, end=12) == 0
        # on start of second pulse
        assert pulse(lambda: 6, 1, repeat_time=5, width=3, end=12) == 1
        # within second pulse
        assert pulse(lambda: 7, 1, repeat_time=5, width=3, end=12) == 1
        # after second pulse
        assert pulse(lambda: 10, 1, repeat_time=5, width=3, end=12) == 0
        # on third pulse
        assert pulse(lambda: 11, 1, repeat_time=5, width=3, end=12) == 1
        # on train end
        assert pulse(lambda: 12, 1, repeat_time=5, width=3, end=12) == 0
        # after train
        assert pulse(lambda: 15, 1, repeat_time=5, width=3, end=13) == 0

    def test_pulse_magnitude(self):
        # Pulse function with repeat time
        # before first impulse
        t = Time()
        t.set_control_vars(initial_time=0, time_step=1)
        assert pulse(t, 2, repeat_time=5, magnitude=10) == 0
        # first impulse
        t.update(2)
        assert pulse(t, 2, repeat_time=5, magnitude=10) == 10
        # after first impulse and before second
        t.update(4)
        assert pulse(t, 2, repeat_time=5, magnitude=10) == 0
        # second impulse
        t.update(7)
        assert pulse(t, 2, repeat_time=5, magnitude=10) == 10
        # after second and before third impulse
        t.update(9)
        assert pulse(t, 2, repeat_time=5, magnitude=10) == 0
        # third impulse
        t.update(12)
        assert pulse(t, 2, repeat_time=5, magnitude=10) == 10
        # after third impulse
        t.update(14)
        assert pulse(t, 2, repeat_time=5, magnitude=10) == 0

        t = Time()
        t.set_control_vars(initial_time=0, time_step=0.2)
        assert pulse(t, 3, repeat_time=5, magnitude=5) == 0
        # first impulse
        t.update(2)
        assert pulse(t, 3, repeat_time=5, magnitude=5) == 0
        # after first impulse and before second
        t.update(3)
        assert pulse(t, 3, repeat_time=5, magnitude=5) == 25
        # second impulse
        t.update(7)
        assert pulse(t, 3, repeat_time=5, magnitude=5) == 0
        # after second and before third impulse
        t.update(8)
        assert pulse(t, 3, repeat_time=5, magnitude=5) == 25
        # third impulse
        t.update(12)
        assert pulse(t, 3, repeat_time=5, magnitude=5) == 0
        # after third impulse
        t.update(13)
        assert pulse(t, 3, repeat_time=5, magnitude=5) == 25

        # Pulse function without repeat time
        # before first impulse
        t = Time()
        t.set_control_vars(initial_time=0, time_step=1)
        assert pulse(t, 2, magnitude=10) == 0
        # first impulse
        t.update(2)
        assert pulse(t, 2, magnitude=10) == 10
        # after first impulse and before second
        t.update(4)
        assert pulse(t, 2, magnitude=10) == 0
        # second impulse
        t.update(7)
        assert pulse(t, 2, magnitude=10) == 0
        # after second and before third impulse
        t.update(9)
        assert pulse(t, 2, magnitude=10) == 0

        t = Time()
        t.set_control_vars(initial_time=0, time_step=0.1)
        assert pulse(t, 4, magnitude=10) == 0
        # first impulse
        t.update(2)
        assert pulse(t, 4, magnitude=10) == 0
        # after first impulse and before second
        t.update(4)
        assert pulse(t, 4, magnitude=10) == 100
        # second impulse
        t.update(7)
        assert pulse(t, 4, magnitude=10) == 0
        # after second and before third impulse
        t.update(9)
        assert pulse(t, 4, magnitude=10) == 0

    def test_numeric_error(self):
        time = Time()
        time.set_control_vars(initial_time=0, time_step=0.1, final_time=10)
        err = 4e-16

        # upper numeric error
        time.update(3 + err)
        assert 3 != time(), "there is no numeric error included"

        assert pulse(time, 3) == 1
        assert pulse(time, 1, repeat_time=2) == 1

        # lower numeric error
        time.update(3 - err)
        assert 3 != time(), "there is no numeric error included"

        assert pulse(time, 3) == 1
        assert pulse(time, 1, repeat_time=2) == 1

    def test_xidz(self):
        assert xidz(1, -0.00000001, 5) == 5
        assert xidz(1, 0, 5) == 5
        assert xidz(1, 8, 5) == 0.125

    def test_zidz(self):
        assert zidz(1, -0.00000001) == 0
        assert zidz(1, 0) == 0
        assert zidz(1, 8) == 0.125


class TestLogicFunctions():
    def test_if_then_else_basic(self):
        assert if_then_else(True, lambda: 1, lambda: 0) == 1
        assert if_then_else(False, lambda: 1, lambda: 0) == 0

        # Ensure lazzy evaluation
        assert if_then_else(True, lambda: 1, lambda: 1/0) == 1
        assert if_then_else(False, lambda: 1/0, lambda: 0) == 0

        with pytest.raises(ZeroDivisionError):
            if_then_else(True, lambda: 1/0, lambda: 0)
        with pytest.raises(ZeroDivisionError):
            if_then_else(False, lambda: 1, lambda: 1/0)

    def test_if_then_else_with_subscripted(self):
        # this test only test the lazzy evaluation and basics
        # subscripted_if_then_else test all the possibilities
        coords = {'dim1': [0, 1], 'dim2': [0, 1]}
        dims = list(coords)

        xr_true = xr.DataArray([[True, True], [True, True]], coords, dims)
        xr_false = xr.DataArray([[False, False], [False, False]], coords, dims)
        xr_mixed = xr.DataArray([[True, False], [False, True]], coords, dims)

        out_mixed = xr.DataArray([[1, 0], [0, 1]], coords, dims)

        assert if_then_else(xr_true, lambda: 1, lambda: 0) == 1
        assert if_then_else(xr_false, lambda: 1, lambda: 0) == 0
        assert if_then_else(xr_mixed, lambda: 1, lambda: 0).equals(out_mixed)

        # Ensure lazzy evaluation
        assert if_then_else(xr_true, lambda: 1, lambda: 1/0) == 1
        assert if_then_else(xr_false, lambda: 1/0, lambda: 0) == 0

        with pytest.raises(ZeroDivisionError):
            if_then_else(xr_true, lambda: 1/0, lambda: 0)
        with pytest.raises(ZeroDivisionError):
            if_then_else(xr_false, lambda: 1, lambda: 1/0)
        with pytest.raises(ZeroDivisionError):
            if_then_else(xr_mixed, lambda: 1/0, lambda: 0)
        with pytest.raises(ZeroDivisionError):
            if_then_else(xr_mixed, lambda: 1, lambda: 1/0)


class TestFunctions():

    def test_sum(self):
        """
        Test for sum function
        """
        coords = {'d1': [9, 1], 'd2': [2, 4]}
        coords_d1, coords_d2 = {'d1': [9, 1]}, {'d2': [2, 4]}
        dims = ['d1', 'd2']

        data = xr.DataArray([[1, 2], [3, 4]], coords, dims)

        assert sum(data, dim=['d1']).equals(
            xr.DataArray([4, 6], coords_d2, ['d2']))
        assert sum(data, dim=['d2']).equals(
            xr.DataArray([3, 7], coords_d1, ['d1']))
        assert sum(data, dim=['d1', 'd2']) == 10
        assert sum(data) == 10

    def test_prod(self):
        """
        Test for sum function
        """
        coords = {'d1': [9, 1], 'd2': [2, 4]}
        coords_d1, coords_d2 = {'d1': [9, 1]}, {'d2': [2, 4]}
        dims = ['d1', 'd2']

        data = xr.DataArray([[1, 2], [3, 4]], coords, dims)

        assert prod(data, dim=['d1']).equals(
            xr.DataArray([3, 8], coords_d2, ['d2']))
        assert prod(data, dim=['d2']).equals(
            xr.DataArray([2, 12], coords_d1, ['d1']))
        assert prod(data, dim=['d1', 'd2']) == 24
        assert prod(data) == 24

    def test_vmin(self):
        """
        Test for vmin function
        """
        coords = {'d1': [9, 1], 'd2': [2, 4]}
        coords_d1, coords_d2 = {'d1': [9, 1]}, {'d2': [2, 4]}
        dims = ['d1', 'd2']

        data = xr.DataArray([[1, 2], [3, 4]], coords, dims)

        assert vmin(data, dim=['d1']).equals(
            xr.DataArray([1, 2], coords_d2, ['d2']))
        assert vmin(data, dim=['d2']).equals(
            xr.DataArray([1, 3], coords_d1, ['d1']))
        assert vmin(data, dim=['d1', 'd2']) == 1
        assert vmin(data) == 1

    def test_vmax(self):
        """
        Test for vmax function
        """
        coords = {'d1': [9, 1], 'd2': [2, 4]}
        coords_d1, coords_d2 = {'d1': [9, 1]}, {'d2': [2, 4]}
        dims = ['d1', 'd2']

        data = xr.DataArray([[1, 2], [3, 4]], coords, dims)

        assert vmax(data, dim=['d1']).equals(
            xr.DataArray([3, 4], coords_d2, ['d2']))
        assert vmax(data, dim=['d2']).equals(
            xr.DataArray([2, 4], coords_d1, ['d1']))
        assert vmax(data, dim=['d1', 'd2']) == 4
        assert vmax(data) == 4

    def test_invert_matrix(self):
        """
        Test for invert_matrix function
        """
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
            assert data.dims == datai.dims

            if len(data.shape) == 2:
                # two dimensions xarrays
                assert (
                    abs(np.dot(data, datai) - np.dot(datai, data))
                    < 1e-14
                    ).all()
                assert (
                    abs(np.dot(data, datai) - np.identity(data.shape[-1]))
                    < 1e-14
                    ).all()
            else:
                # three dimensions xarrays
                for i in range(data.shape[0]):
                    assert (
                        abs(np.dot(data[i], datai[i])
                            - np.dot(datai[i], data[i]))
                        < 1e-14
                        ).all()
                    assert (
                        abs(np.dot(data[i], datai[i])
                            - np.identity(data.shape[-1]))
                        < 1e-14
                        ).all()

    def test_incomplete(self):
        from pysd.py_backend.functions import incomplete
        from warnings import catch_warnings

        with catch_warnings(record=True) as w:
            incomplete()
            assert len(w) == 1
            assert 'Call to undefined function' in str(w[-1].message)

    def test_not_implemented_function(self):
        from pysd.py_backend.functions import not_implemented_function

        with pytest.raises(NotImplementedError):
            not_implemented_function("NIF")
