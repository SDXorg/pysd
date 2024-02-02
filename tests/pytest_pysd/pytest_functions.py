from datetime import datetime

import pytest
import numpy as np
import xarray as xr

from pysd.py_backend.components import Time
from pysd.py_backend.functions import\
    ramp, step, pulse, xidz, zidz, if_then_else, sum, prod, vmin, vmax,\
    invert_matrix, get_time_value, vector_select


class TestInputFunctions():
    def test_ramp(self):
        t = Time()
        t.set_control_vars(time_step=1)

        t.update(4)
        assert ramp(t, .5, 10, 18) == 0

        t.update(14)
        assert ramp(t, .5, 10, 18) == 2

        t.update(24)
        assert ramp(t, .5, 10, 18) == 4

        assert ramp(t, .5, 10) == 7

        t.update(50)
        assert ramp(t, .5, 10) == 20

        # arrays start
        coords = {"dim1": ["A", "B"], "dim2": ["X", "Y", "Z"]}
        start_array = xr.DataArray(
            [[8., 7., 9.], [4., 5., 6.]],
            coords
        )

        t.update(4)
        assert ramp(t, .5, start_array, 18).equals(
            xr.DataArray([[0., 0., 0.], [0., 0., 0.]], coords))

        t.update(14)
        assert ramp(t, .5, start_array, 18).equals(
            xr.DataArray([[3., 3.5, 2.5], [5., 4.5, 4.]], coords))

        t.update(24)
        assert ramp(t, .5, start_array, 18).equals(
            xr.DataArray([[5., 5.5, 4.5], [7., 6.5, 6.]], coords))

        assert ramp(t, .5, start_array).equals(
            xr.DataArray([[8., 8.5, 7.5], [10., 9.5, 9.]], coords))

        # arrays finish
        finish_array = xr.DataArray(
            [[10., 15., 18.], [20., 16., 12.]],
            coords
        )

        t.update(14)
        assert ramp(t, .5, 10, finish_array).equals(
            xr.DataArray([[0., 2., 2.], [2., 2., 1.]], coords))

        t.update(25)
        assert ramp(t, .5, 10, finish_array).equals(
            xr.DataArray([[0., 2.5, 4.], [5., 3., 1.]], coords))

        # arrays slope
        slope_array = xr.DataArray(
            [[0.5, .1, 1.], [2., -1., -0.5]],
            coords
        )

        t.update(4)
        assert ramp(t, slope_array, 10, 18).equals(
            xr.DataArray([[0., 0., 0.], [0., 0., 0.]], coords))

        t.update(14)
        assert ramp(t, slope_array, 10, 18).equals(
            xr.DataArray([[2., .4, 4.], [8., -4, -2.]], coords))

        t.update(24)
        assert ramp(t, slope_array, 10, 18).equals(
            xr.DataArray([[4., .8, 8.], [16., -8., -4.]], coords))

        t.update(50)
        assert ramp(t, slope_array, 10).equals(
            xr.DataArray([[20., 4., 40.], [80., -40., -20.]], coords))

        # arrays all
        t.update(4)
        assert ramp(t, slope_array, start_array, finish_array).equals(
            xr.DataArray([[0., 0., 0.], [0., 0., 0.]], coords))

        t.update(12)
        assert ramp(t, slope_array, start_array, finish_array).equals(
            xr.DataArray([[1.,  0.5,  3.], [16., -7., -3.]], coords))

        t.update(26)
        assert ramp(t, slope_array, start_array, finish_array).equals(
            xr.DataArray([[1., .8, 9.], [32., -11., -3.]], coords))

        t.update(27)
        assert ramp(t, slope_array, start_array).equals(
            xr.DataArray([[9.5, 2., 18.], [46., -22., -10.5]], coords))

    def test_step(self):
        t = Time()
        t.set_control_vars(time_step=1)

        t.update(5)
        assert step(t, 1, 10) == 0

        t.update(10)
        assert step(t, 1, 10) == 1

        t.update(15)
        assert step(t, 1, 10) == 1

        # arrays value
        coords = {"dim1": ["A", "B"], "dim2": ["X", "Y", "Z"]}
        value_array = xr.DataArray(
            [[0.5, .1, 1.], [2., -1., -0.5]],
            coords
        )

        t.update(5)
        assert step(t, value_array, 10).equals(
            xr.zeros_like(value_array))

        t.update(10)
        assert step(t, value_array, 10).equals(value_array)

        t.update(15)
        assert step(t, value_array, 10).equals(value_array)

        # arrays tstep
        tstep_array = xr.DataArray(
            [[8., 10., 6.], [20., 11., 15.]],
            coords
        )

        t.update(5)
        assert step(t, 1, tstep_array).equals(
            xr.zeros_like(value_array))

        t.update(10)
        assert step(t, 1, tstep_array).equals(tstep_array <= 10)

        t.update(15)
        assert step(t, 1, tstep_array).equals(tstep_array <= 15)

        # arrays all
        t.update(5)
        assert step(t, value_array, tstep_array).equals(
            xr.zeros_like(value_array))

        t.update(10)
        assert step(t, value_array, tstep_array).equals(
            value_array*(tstep_array <= 10))

        t.update(15)
        assert step(t, value_array, tstep_array).equals(
            value_array*(tstep_array <= 15))

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

    @pytest.mark.parametrize(
        "year,quarter,month,day,hour,minute,second,microsecond",
        [
            (2020, 1, 1, 25, 23, 45, 5, 233),
            (2020, 1, 3, 4, 15, 10, 23, 3323),
            (195, 2, 4, 4, 15, 10, 23, 3323),
            (195, 2, 6, 4, 15, 10, 23, 3323),
            (2045, 3, 7, 31, 0, 15, 55, 33233),
            (1330, 3, 9, 30, 0, 15, 55, 33233),
            (3000, 4, 10, 1, 13, 15, 55, 33233),
            (1995, 4, 12, 1, 13, 15, 55, 33233),
        ]
    )
    def test_get_time_value_machine(self, mocker, year, quarter, month, day,
                                    hour, minute, second, microsecond):
        """Test get_time_value with machine time reltiveto=2"""

        mock_time = datetime(
            year, month, day, hour, minute, second, microsecond)

        mocker.patch(
            "pysd.py_backend.utils.get_current_computer_time",
            return_value=mock_time)

        assert get_time_value(None, 2, np.random.randint(-100, 100), 1)\
            == year

        assert get_time_value(None, 2, np.random.randint(-100, 100), 2)\
            == quarter

        assert get_time_value(None, 2, np.random.randint(-100, 100), 3)\
            == month

        assert get_time_value(None, 2, np.random.randint(-100, 100), 4)\
            == day

        assert get_time_value(None, 2, np.random.randint(-100, 100), 5)\
            == mock_time.weekday()

        assert get_time_value(None, 2, np.random.randint(-100, 100), 6)\
            == (mock_time - datetime(1, 1, 1)).days

        assert get_time_value(None, 2, np.random.randint(-100, 100), 7)\
            == hour

        assert get_time_value(None, 2, np.random.randint(-100, 100), 8)\
            == minute

        assert get_time_value(None, 2, np.random.randint(-100, 100), 9)\
            == second + 1e-6*microsecond

        assert get_time_value(None, 2, np.random.randint(-100, 100), 10)\
            == (mock_time - datetime(1, 1, 1)).seconds % 500000

    @pytest.mark.parametrize(
        "measure,relativeto,raise_type,error_message",
        [
            (  # relativeto=1
                0,
                1,
                NotImplementedError,
                r"'relativeto=1' not implemented\.\.\."
            ),
            (  # relativeto=3
                0,
                3,
                ValueError,
                r"Invalid argument value 'relativeto=3'\. "
                r"'relativeto' must be 0, 1 or 2\."
            ),
            (  # measure=0;relativeto=2
                0,
                2,
                ValueError,
                r"Invalid argument 'measure=0' with 'relativeto=2'\."
            ),
            (  # measure=11
                11,
                2,
                ValueError,
                r"Invalid argument value 'measure=11'\. "
                r"'measure' must be 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 or 10\."
            ),
            (  # relativeto=0;measure=2
                2,
                0,
                NotImplementedError,
                r"The case 'relativeto=0' and 'measure=2' "
                r"is not implemented\.\.\."
            ),
        ],
        ids=[
            "relativeto=1", "relativeto=3", "measure=0;relativeto=2",
            "measure=11", "relativeto=0;measure=2"
        ]
    )
    def test_get_time_value_errors(self, measure, relativeto,
                                   raise_type, error_message):

        with pytest.raises(raise_type, match=error_message):
            get_time_value(
                lambda: 0, relativeto, np.random.randint(-100, 100), measure)

    def test_vector_select(self):
        warn_message =\
            r"Vensim's help says that numerical_action=5 computes the "\
            r"product of selection_array \^ expression_array\. But, in fact,"\
            r" Vensim is computing the product of expression_array \^ "\
            r" selection array\. The output of this function behaves as "\
            r"Vensim, expression_array \^ selection_array\."

        array = xr.DataArray([3, 10, 2], {'dim': ["A", "B", "C"]})
        sarray = xr.DataArray([1, 0, 2], {'dim': ["A", "B", "C"]})

        with pytest.warns(UserWarning, match=warn_message):
            assert vector_select(sarray, array, ["dim"], np.nan, 5, 1)\
                == 12

        sarray = xr.DataArray([0, 0, 0], {'dim': ["A", "B", "C"]})
        assert vector_select(sarray, array, ["dim"], 123, 0, 2) == 123

    @pytest.mark.parametrize(
        "selection_array,expression_array,dim,numerical_action,"
        "error_action,raise_type,error_message",
        [
            (  # error_action=1
                xr.DataArray([0, 0], {'dim': ["A", "B"]}),
                xr.DataArray([1, 2], {'dim': ["A", "B"]}),
                ["dim"],
                0,
                1,
                FloatingPointError,
                r"All the values of selection_array are 0\.\.\."
            ),
            (  # error_action=2
                xr.DataArray([1, 1], {'dim': ["A", "B"]}),
                xr.DataArray([1, 2], {'dim': ["A", "B"]}),
                ["dim"],
                0,
                2,
                FloatingPointError,
                r"More than one non-zero values in selection_array\.\.\."
            ),
            (  # error_action=3a
                xr.DataArray([0, 0], {'dim': ["A", "B"]}),
                xr.DataArray([1, 2], {'dim': ["A", "B"]}),
                ["dim"],
                0,
                3,
                FloatingPointError,
                r"All the values of selection_array are 0\.\.\."
            ),
            (  # error_action=3b
                xr.DataArray([1, 1], {'dim': ["A", "B"]}),
                xr.DataArray([1, 2], {'dim': ["A", "B"]}),
                ["dim"],
                0,
                3,
                FloatingPointError,
                r"More than one non-zero values in selection_array\.\.\."
            ),
            (  # numerical_action=11
                xr.DataArray([1, 1], {'dim': ["A", "B"]}),
                xr.DataArray([1, 2], {'dim': ["A", "B"]}),
                ["dim"],
                11,
                0,
                ValueError,
                r"Invalid argument value 'numerical_action=11'\. "
                r"'numerical_action' must be 0, 1, 2, 3, 4, 5, 6, "
                r"7, 8, 9 or 10\."
            ),
        ],
        ids=[
            "error_action=1", "error_action=2", "error_action=3a",
            "error_action=3b", "numerical_action=11"
        ]
    )
    def test_vector_select_errors(self, selection_array, expression_array,
                                  dim, numerical_action, error_action,
                                  raise_type, error_message):

        with pytest.raises(raise_type, match=error_message):
            vector_select(
                selection_array, expression_array, dim,  0,
                numerical_action, error_action)

    def test_incomplete(self):
        from pysd.py_backend.functions import incomplete

        with pytest.warns(RuntimeWarning, match='Call to undefined function'):
            incomplete()

    def test_not_implemented_function(self):
        from pysd.py_backend.functions import not_implemented_function

        with pytest.raises(NotImplementedError):
            not_implemented_function("NIF")
