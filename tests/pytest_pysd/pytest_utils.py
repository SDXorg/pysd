import doctest

import pytest
import pandas as pd
import xarray as xr

from pysd.tools.benchmarking import assert_frames_close
import pysd
from pysd.py_backend.utils import compute_shape, rearrange, load_outputs,\
    ProgressBar


class TestUtils():

    def test_xrsplit(self):
        array1d = xr.DataArray([0.5, 0., 1.],
                               {'ABC': ['A', 'B', 'C']},
                               ['ABC'])
        array2d = xr.DataArray([[0.5, -1.5],
                                [-1., -0.5],
                                [-0.75, 0.]],
                               {'ABC': ['A', 'B', 'C'],
                                'XY': ['X', 'Y']},
                               ['ABC', 'XY'])
        array3d = xr.DataArray([[[0.5, 4.], [-1.5, 3.]],
                                [[-1., 2.], [-0.5, 5.5]],
                                [[-0.75, 0.75], [0., -1.]]],
                               {'ABC': ['A', 'B', 'C'],
                                'XY': ['X', 'Y'],
                                'FG': ['F', 'G']},
                               ['ABC', 'XY', 'FG'])
        s1d = pysd.utils.xrsplit(array1d)
        s2d = pysd.utils.xrsplit(array2d)
        s3d = pysd.utils.xrsplit(array3d)

        # check length
        assert len(s1d) == 3
        assert len(s2d) == 6
        assert len(s3d) == 12

        # check all values for 1d
        assert xr.DataArray(0.5, {'ABC': ['A']}, ['ABC']) in s1d
        assert xr.DataArray(0., {'ABC': ['B']}, ['ABC']) in s1d
        assert xr.DataArray(1., {'ABC': ['C']}, ['ABC']) in s1d
        # check some values for 2d and 3d
        assert xr.DataArray(
            0.5, {'ABC': ['A'], 'XY': ['X']}, ['ABC', 'XY']
            ) in s2d
        assert xr.DataArray(
            -0.5, {'ABC': ['B'], 'XY': ['Y']}, ['ABC', 'XY']
            ) in s2d
        assert xr.DataArray(
            -0.5, {'ABC': ['B'], 'XY': ['Y'], 'FG': ['F']}, ['ABC', 'XY', 'FG']
            ) in s3d
        assert xr.DataArray(
            0.75, {'ABC': ['C'], 'XY': ['X'], 'FG': ['G']}, ['ABC', 'XY', 'FG']
            ) in s3d

    def test_get_return_elements_subscirpts(self):
        assert pysd.utils.get_return_elements(
                ["Inflow A[Entry 1,Column 1]",
                 "Inflow A[Entry 1,Column 2]"],
                {'Inflow A': 'inflow_a'})\
            == (
            ['inflow_a'],
            {'Inflow A[Entry 1,Column 1]': ('inflow_a',
                                            ('Entry 1', 'Column 1')),
             'Inflow A[Entry 1,Column 2]': ('inflow_a',
                                            ('Entry 1', 'Column 2'))}
            )

    def test_get_return_elements_realnames(self):
        assert pysd.utils.get_return_elements(
                ["Inflow A", "Inflow B"],
                {'Inflow A': 'inflow_a', 'Inflow B': 'inflow_b'})\
            == (
            ['inflow_a', 'inflow_b'],
            {'Inflow A': ('inflow_a', None),
             'Inflow B': ('inflow_b', None)}
            )

    def test_get_return_elements_pysafe_names(self):
        assert pysd.utils.get_return_elements(
                ["inflow_a", "inflow_b"],
                {'Inflow A': 'inflow_a', 'Inflow B': 'inflow_b'})\
            == (
            ['inflow_a', 'inflow_b'],
            {'inflow_a': ('inflow_a', None),
             'inflow_b': ('inflow_b', None)}
            )

    def test_get_return_elements_not_found_error(self):
        """"
        Test for not found element
        """

        with pytest.raises(KeyError):
            pysd.utils.get_return_elements(
                ["inflow_a", "inflow_b", "inflow_c"],
                {'Inflow A': 'inflow_a', 'Inflow B': 'inflow_b'})

    def test_doctests(self):
        doctest.DocTestSuite(pysd.utils)

    def test_compute_shape(self):
        """"
        Test for computing the shape of an array giving coordinates dictionary
        and ordered dimensions.
        """
        coords = [
          {},
          {'ABC': ['A', 'B', 'C'],
           'XY': ['X'],
           'val': [1, 2, 3, 4, 5, 6, 7, 8]},
          {'val': [1, 2, 3, 4, 5],
           'ABC': ['A', 'B', 'C'],
           'XY': ['X']},
          {'ABC': ['A', 'B', 'C'],
           'val': [1, 2, 3, 4, 5, 6, 7, 8],
           'XY': ['X', 'Y']},
          {'val': [1, 2, 3, 4, 5, 6, 7, 8],
           'ABC': ['A', 'B', 'C']},
          {'val': [1, 2, 3, 4, 5, 6, 7, 8]}
         ]

        shapes = [[], [3, 1, 8], [5, 3, 1], [3, 8, 2], [8, 3], [8]]

        for c, s in zip(coords, shapes):
            assert compute_shape(c) == s

    def test_compute_shape_reshape(self):
        """"
        Test for computing the shape of an array giving coordinates dictionary
        and ordered dimensions with reshape.
        """
        coords = [
          {'ABC': ['A', 'B', 'C'],
           'XY': ['X'],
           'val': [1, 2, 3, 4, 5, 6, 7, 8]},
          {'val': [1, 2, 3, 4, 5],
           'ABC': ['A', 'B', 'C'],
           'XY': ['X']},
          {'ABC': ['A', 'B', 'C'],
           'val': [1, 2, 3, 4, 5, 6, 7, 8],
           'XY': ['X', 'Y']},
          {'val': [1, 2, 3, 4, 5, 6, 7, 8],
           'ABC': ['A', 'B', 'C']},
          {'val': [1, 2, 3, 4, 5, 6, 7, 8]}
         ]

        # reshapes list for 1, 2 and 3
        shapes123 = [
          [None, None, None, None, [8]],
          [[3, 8], [5, 3], None, [8, 3], [1, 8]],
          [[1, 3, 8], [1, 5, 3], [3, 8, 2], [1, 8, 3], [1, 1, 8]]
        ]

        for i, shapes in enumerate(shapes123):
            for c, s in zip(coords, shapes):
                if s:
                    assert compute_shape(c, i+1) == s
                else:
                    with pytest.raises(ValueError):
                        compute_shape(c, i+1)

    def test_rearrange(self):
        # simple cases are tested, complex cases are tested with test-models
        _subscript_dict = {
            'd1': ['a', 'b', 'c'],
            'd2': ['b', 'c'],
            'd3': ['b', 'c'],
            'd4': ['e', 'f']
        }

        xr_input_subdim = xr.DataArray([1, 4, 2],
                                       {'d1': ['a', 'b', 'c']}, ['d1'])
        xr_input_subdim2 = xr.DataArray([[1, 4, 2], [3, 4, 5]],
                                        {'d1': ['a', 'b', 'c'],
                                         'd4': ['e', 'f']},
                                        ['d4', 'd1'])
        xr_input_updim = xr.DataArray([1, 4],
                                      {'d2': ['b', 'c']}, ['d2'])
        xr_input_switch = xr.DataArray([[1, 4], [8, 5]],
                                       {'d2': ['b', 'c'], 'd3': ['b', 'c']},
                                       ['d2', 'd3'])
        xr_input_float = 3.

        xr_out_subdim = xr.DataArray([4, 2],
                                     {'d2': ['b', 'c']}, ['d2'])
        xr_out_subdim2 = xr.DataArray([[4, 2], [4, 5]],
                                      {'d2': ['b', 'c'],
                                       'd4': ['e', 'f']},
                                      ['d4', 'd2'])
        xr_out_updim = xr.DataArray([[1, 1], [4, 4]],
                                    {'d2': ['b', 'c'], 'd3': ['b', 'c']},
                                    ['d2', 'd3'])
        xr_out_switch = xr.DataArray([[1, 4], [8, 5]],
                                     {'d2': ['b', 'c'], 'd3': ['b', 'c']},
                                     ['d3', 'd2'])
        xr_out_float = xr.DataArray(3.,
                                    {'d2': ['b', 'c']}, ['d2'])

        assert xr_out_subdim.equals(
            rearrange(xr_input_subdim, ['d2'], _subscript_dict))

        assert xr_out_subdim2.equals(
            rearrange(xr_input_subdim2, ['d4', 'd2'], _subscript_dict))

        assert xr_out_updim.equals(
            rearrange(xr_input_updim, ['d2', 'd3'], _subscript_dict))

        assert xr_out_switch.equals(
            rearrange(xr_input_switch, ['d3', 'd2'], _subscript_dict))

        assert xr_out_float.equals(
            rearrange(xr_input_float, ['d2'], _subscript_dict))

        assert rearrange(None, ['d2'], _subscript_dict) is None


class TestLoadOutputs():
    def test_non_valid_outputs(self, _root):
        outputs = _root.joinpath("more-tests/not_vensim/test_not_vensim.txt")

        error_message = r"Not able to read '.*%s'\." % outputs.name
        with pytest.raises(ValueError, match=error_message):
            load_outputs(outputs)

    def test_transposed_frame(self, _root):
        assert_frames_close(
            load_outputs(_root.joinpath("data/out_teacup.csv")),
            load_outputs(
                _root.joinpath("data/out_teacup_transposed.csv"),
                transpose=True))

    def test_load_columns(self, _root):
        out0 = load_outputs(_root.joinpath("data/out_teacup.csv"))

        out1 = load_outputs(
            _root.joinpath("data/out_teacup.csv"),
            columns=["Room Temperature", "Teacup Temperature"])

        out2 = load_outputs(
            _root.joinpath("data/out_teacup_transposed.csv"),
            transpose=True,
            columns=["Heat Loss to Room"])

        assert set(out1.columns)\
            == set(["Room Temperature", "Teacup Temperature"])

        assert set(out2.columns)\
            == set(["Heat Loss to Room"])

        assert (out0.index == out1.index).all()
        assert (out0.index == out2.index).all()


class TestProgressbar():
    def test_progressbar(self):
        pbar = ProgressBar(10)

        for i in range(10):
            assert pbar.counter == i
            pbar.update()

        pbar.finish()

        pbar = ProgressBar()
        assert not hasattr(pbar, 'counter')
        pbar.update()
        pbar.finish()
