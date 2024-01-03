from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc

from pysd.tools.benchmarking import assert_frames_close
from pysd.py_backend.output import OutputHandlerInterface, DatasetHandler, \
    DataFrameHandler, ModelOutput


test_model_look = Path(
    "test-models/tests/get_lookups_subscripted_args/"
    "test_get_lookups_subscripted_args.mdl"
)
test_model_constants = Path(
    "test-models/tests/get_constants_subranges/"
    "test_get_constants_subranges.mdl"
)
test_model_numeric_coords = Path(
    "test-models/tests/subscript_1d_arrays/"
    "test_subscript_1d_arrays.mdl"
)
test_variable_step = Path(
    "test-models/tests/control_vars/"
    "test_control_vars.mdl"
)
test_partial_definitions = Path(
    "test-models/tests/partial_range_definitions/"
    "test_partial_range_definitions.mdl"
)


class TestOutput():

    def test_output_handler_interface(self):
        # when the class does not inherit from OutputHandlerInterface, it must
        # implement all the interface to be a subclass of
        # OutputHandlerInterface.
        # Add any additional Handler here.
        assert issubclass(DatasetHandler, OutputHandlerInterface)
        assert issubclass(DataFrameHandler, OutputHandlerInterface)

        class ThatFollowsInterface:
            """
            This class does not inherit from OutputHandlerInterface, but it
            overrides all its methods (it follows the interface).
            """
            def initialize(self, model):
                pass

            def update(self, model):
                pass

            def postprocess(self, **kwargs):
                pass

            def add_run_elements(self):
                pass

        # eventhough it does not inherit from OutputHandlerInterface, it is
        # considered a subclass, because it follows the interface
        assert issubclass(ThatFollowsInterface, OutputHandlerInterface)

        class IncompleteHandler:
            """
            Class that does not follow the full interface
            (add_run_elements is missing).
            """
            def initialize(self, model, capture_elements):
                pass

            def update(self, model, capture_elements):
                pass

            def postprocess(self, **kwargs):
                pass

        # It does not inherit from OutputHandlerInterface and does not fulfill
        # its interface
        assert not issubclass(IncompleteHandler, OutputHandlerInterface)

        class EmptyHandler(OutputHandlerInterface):
            """
            When the class DOES inherit from OutputHandlerInterface, but does
            not override all its abstract methods, then it cannot be
            instantiated
            """
            pass

        # it is a subclass because it inherits from it
        assert issubclass(EmptyHandler, OutputHandlerInterface)

        # it cannot be instantiated because it does not override all abstract
        # methods
        with pytest.raises(TypeError):
            EmptyHandler()

        # calling methods that are not overriden returns NotImplementedError
        # this should never happen, because these methods are instance methods,
        # therefore the class needs to be instantiated first
        with pytest.raises(NotImplementedError):
            EmptyHandler.initialize(EmptyHandler, "model")

        with pytest.raises(NotImplementedError):
            EmptyHandler.update(EmptyHandler, "model")

        with pytest.raises(NotImplementedError):
            EmptyHandler.postprocess(EmptyHandler)

        with pytest.raises(NotImplementedError):
            EmptyHandler.add_run_elements(
                EmptyHandler, "model")

    @pytest.mark.parametrize("model_path", [test_model_look])
    def test_invalid_output_file(self, model):
        error_message = ".* str .* os.PathLike object .*, not .*int.*"
        with pytest.raises(TypeError, match=error_message):
            model.run(output_file=1234)

        error_message = "Unsupported output file format .txt"
        with pytest.raises(ValueError, match=error_message):
            model.run(output_file="file.txt")

    @pytest.mark.parametrize(
        "model_path,dims,values",
        [
            (
                test_model_look,
                {
                    "Rows": 2,
                    "Dim": 2,
                    "time": 61
                },
                {
                    "lookup_1d_time": (("time",), None),
                    "d2d": (("time", "Rows", "Dim"), None),
                    "initial_time": (tuple(), 0),
                    "final_time": (tuple(), 30),
                    "saveper": (tuple(), 0.5),
                    "time_step": (tuple(), 0.5)
                }

            ),
            (
                test_model_constants,
                {
                    "dim1": 5,
                    "dim1a": 2,
                    "dim1c": 3,
                    'time': 2
                },
                {
                    "constant": (
                        ("dim1",),
                        np.array([0.,  0.,  1., 15., 50.])
                    )
                }
            ),
            (
                test_model_numeric_coords,
                {
                    "One Dimensional Subscript": 3,
                    'time': 101
                },
                {
                    "rate_a": (
                        ("One Dimensional Subscript",),
                        np.array([0.01, 0.02, 0.03])),
                    "stock_a": (
                        ("time", "One Dimensional Subscript"),
                        np.array([
                            np.arange(0, 1.0001, 0.01),
                            np.arange(0, 2.0001, 0.02),
                            np.arange(0, 3.0001, 0.03)],
                            dtype=float).transpose()),
                    "time": (("time",), np.arange(0.0, 101.0, 1.0))
                }
            ),
            (
                test_variable_step,
                {
                    "time": 25
                },
                {
                    "final_time": (
                        ("time",),
                        np.array([
                            10., 10., 10., 10., 10., 10.,
                            50., 50., 50., 50., 50., 50.,
                            50., 50., 50., 50., 50., 50.,
                            50., 50., 50., 50., 50., 50., 50.
                        ])),
                    "initial_time": (
                        ("time",),
                        np.array([
                            0., 0., 0., 0.2, 0.2, 0.2,
                            0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                            0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                            0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                        ])),
                    "time_step": (
                        ("time",),
                        np.array([
                            1., 1., 1., 1., 0.5, 0.5,
                            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                        ])),
                    "saveper": (
                        ("time",),
                        np.array([
                            1., 1., 1., 1., 0.5, 0.5,
                            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                            0.5, 0.5, 0.5, 0.5, 5., 5.,
                            5., 5., 5., 5., 5., 5., 5.]))
                }
            ),
            (
                test_partial_definitions,
                {
                    "my range": 5,
                    "time": 11
                },
                {
                    "partial_data": (("time", "my range"), None),
                    "partial_constants": (("my range",), None)
                }
            )
        ],
        ids=["lookups", "constants", "numeric_coords", "variable_step",
             "partial_definitions"]
    )
    @pytest.mark.filterwarnings("ignore")
    def test_output_nc(self, tmp_path, model, dims, values):

        out_file = tmp_path.joinpath("results.nc")

        model.run(output_file=out_file)

        with nc.Dataset(out_file, "r") as ds:
            assert ds.ncattrs() == [
                'description', 'model_file', 'timestep', 'initial_time',
                'final_time']
            assert list(ds.dimensions) == list(dims)
            # dimensions are stored as variables
            for dim, n in dims.items():
                # check dimension size
                assert ds[dim].size == n
                assert dim in ds.variables.keys()
                # check dimension type
                if dim != "time":
                    assert ds[dim].dtype in ["S1", str]
                else:
                    assert ds[dim].dtype == float

            for var, (dim, val) in values.items():
                # check variable dimensions
                assert ds[var].dimensions == dim
                if val is not None:
                    # check variable values if given
                    assert np.all(np.isclose(ds[var][:].data, val))

            # Check variable attributes
            doc = model.doc
            doc.set_index("Py Name", drop=False, inplace=True)
            doc.drop(columns=["Subscripts", "Limits"], inplace=True)

            for var in doc["Py Name"]:
                if doc.loc[var, "Type"] == "Lookup":
                    continue
                for key in doc.columns:
                    assert getattr(ds[var], key) == (doc.loc[var, key]
                                                     or "Missing")

    @pytest.mark.parametrize(
        "model_path,fmt,sep",
        [
            (test_model_look, "csv", ","),
            (test_model_look, "tab", "\t")])
    @pytest.mark.filterwarnings("ignore")
    def test_output_csv(self, fmt, sep, capsys, model, tmp_path):
        out_file = tmp_path.joinpath("results." + fmt)

        model.run(output_file=out_file)

        captured = capsys.readouterr()  # capture stdout
        assert f"Data saved in '{out_file}'" in captured.out

        df = pd.read_csv(out_file, sep=sep)

        assert df["Time"].iloc[-1] == model["final_time"]
        assert df["Time"].iloc[0] == model["initial_time"]
        assert df.shape == (61, 51)
        assert not df.isnull().values.any()
        assert "lookup 3d time[B;Row1]" in df.columns or \
            "lookup 3d time[B,Row1]" in df.columns

    @pytest.mark.parametrize("model_path", [test_model_look])
    def test_dataset_handler_step_setter(self, tmp_path, model):
        capture_elements = {"run": [], "step": []}
        results = tmp_path.joinpath("results.nc")
        output = ModelOutput(results)
        output.set_capture_elements(capture_elements)
        output.initialize(model)

    def test_make_flat_df(self):

        df = pd.DataFrame(index=[1], columns=['elem1'])
        df.loc[1] = [xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                  {'Dim1': ['A', 'B', 'C'],
                                   'Dim2': ['D', 'E', 'F']},
                                  dims=['Dim1', 'Dim2'])]

        expected = pd.DataFrame(index=[1], data={'Elem1[B,F]': 6.})

        return_addresses = {
            'Elem1[B,F]': ('elem1', {'Dim1': ['B'], 'Dim2': ['F']})}

        actual = DataFrameHandler.make_flat_df(df, return_addresses)

        # check all columns are in the DataFrame
        assert set(actual.columns) == set(expected.columns)
        assert_frames_close(actual, expected, rtol=1e-8, atol=1e-8)

    def test_make_flat_df_0dxarray(self):

        df = pd.DataFrame(index=[1], columns=['elem1'])
        df.loc[1] = [xr.DataArray(5)]

        expected = pd.DataFrame(index=[1], data={'Elem1': 5.})

        return_addresses = {'Elem1': ('elem1', {})}

        actual = DataFrameHandler.make_flat_df(
            df, return_addresses, flatten=True)

        # check all columns are in the DataFrame
        assert set(actual.columns) == set(expected.columns)
        assert_frames_close(actual, expected, rtol=1e-8, atol=1e-8)

    def test_make_flat_df_nosubs(self):

        df = pd.DataFrame(index=[1], columns=['elem1', 'elem2'])
        df.loc[1] = [25, 13]

        expected = pd.DataFrame(index=[1], columns=['Elem1', 'Elem2'])
        expected.loc[1] = [25, 13]

        return_addresses = {'Elem1': ('elem1', {}),
                            'Elem2': ('elem2', {})}

        actual = DataFrameHandler.make_flat_df(df, return_addresses)

        # check all columns are in the DataFrame
        assert set(actual.columns) == set(expected.columns)
        assert all(actual['Elem1'] == expected['Elem1'])
        assert all(actual['Elem2'] == expected['Elem2'])

    def test_make_flat_df_return_array(self):
        """ There could be cases where we want to
        return a whole section of an array - ie, by passing in only part of
        the simulation dictionary. in this case, we can't force to float..."""

        df = pd.DataFrame(index=[1], columns=['elem1', 'elem2'])
        df.loc[1] = [xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                  {'Dim1': ['A', 'B', 'C'],
                                   'Dim2': ['D', 'E', 'F']},
                                  dims=['Dim1', 'Dim2']),
                     xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                  {'Dim1': ['A', 'B', 'C'],
                                   'Dim2': ['D', 'E', 'F']},
                                  dims=['Dim1', 'Dim2'])]

        expected = pd.DataFrame(index=[1], columns=['Elem1[A, Dim2]', 'Elem2'])
        expected.loc[1] = [xr.DataArray([[1, 2, 3]],
                                        {'Dim1': ['A'],
                                         'Dim2': ['D', 'E', 'F']},
                                        dims=['Dim1', 'Dim2']),
                           xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                        {'Dim1': ['A', 'B', 'C'],
                                         'Dim2': ['D', 'E', 'F']},
                                        dims=['Dim1', 'Dim2'])]

        return_addresses = {
            'Elem1[A, Dim2]': ('elem1', {'Dim1': ['A'],
                                         'Dim2': ['D', 'E', 'F']}),
            'Elem2': ('elem2', {})}

        actual = DataFrameHandler.make_flat_df(df, return_addresses)

        # check all columns are in the DataFrame
        assert set(actual.columns) == set(expected.columns)
        # need to assert one by one as they are xarrays
        assert actual.loc[1, 'Elem1[A, Dim2]'].equals(
                expected.loc[1, 'Elem1[A, Dim2]'])
        assert actual.loc[1, 'Elem2'].equals(expected.loc[1, 'Elem2'])

    def test_make_flat_df_flatten(self):

        df = pd.DataFrame(index=[1], columns=['elem1', 'elem2'])
        df.loc[1] = [xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                  {'Dim1': ['A', 'B', 'C'],
                                   'Dim2': ['D', 'E', 'F']},
                                  dims=['Dim1', 'Dim2']),
                     xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                  {'Dim1': ['A', 'B', 'C'],
                                   'Dim2': ['D', 'E', 'F']},
                                  dims=['Dim1', 'Dim2'])]

        expected = pd.DataFrame(index=[1], columns=[
            'Elem1[A,D]',
            'Elem1[A,E]',
            'Elem1[A,F]',
            'Elem2[A,D]',
            'Elem2[A,E]',
            'Elem2[A,F]',
            'Elem2[B,D]',
            'Elem2[B,E]',
            'Elem2[B,F]',
            'Elem2[C,D]',
            'Elem2[C,E]',
            'Elem2[C,F]'])

        expected.loc[1] = [1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        return_addresses = {
            'Elem1[A,Dim2]': ('elem1', {'Dim1': ['A'],
                                        'Dim2': ['D', 'E', 'F']}),
            'Elem2': ('elem2', {})}

        actual = DataFrameHandler.make_flat_df(
            df, return_addresses, flatten=True)

        # check all columns are in the DataFrame
        assert set(actual.columns) == set(expected.columns)
        # need to assert one by one as they are xarrays
        for col in set(expected.columns):
            assert actual.loc[:, col].values == expected.loc[:, col].values

    def test_make_flat_df_flatten_transposed(self):

        df = pd.DataFrame(index=[1], columns=['elem2'])
        df.loc[1] = [
            xr.DataArray(
                [[1, 4, 7], [2, 5, 8], [3, 6, 9]],
                {'Dim2': ['D', 'E', 'F'], 'Dim1': ['A', 'B', 'C']},
                ['Dim2', 'Dim1']
            ).transpose("Dim1", "Dim2")
        ]

        expected = pd.DataFrame(index=[1], columns=[
            'Elem2[A,D]',
            'Elem2[A,E]',
            'Elem2[A,F]',
            'Elem2[B,D]',
            'Elem2[B,E]',
            'Elem2[B,F]',
            'Elem2[C,D]',
            'Elem2[C,E]',
            'Elem2[C,F]'])

        expected.loc[1] = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        return_addresses = {
            'Elem2': ('elem2', {})}

        actual = DataFrameHandler.make_flat_df(
            df, return_addresses, flatten=True)

        # check all columns are in the DataFrame
        assert set(actual.columns) == set(expected.columns)
        # need to assert one by one as they are xarrays
        for col in set(expected.columns):
            assert actual.loc[:, col].values == expected.loc[:, col].values

    def test_make_flat_df_times(self):

        df = pd.DataFrame(index=[1, 2], columns=['elem1'])
        df['elem1'] = [xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                    {'Dim1': ['A', 'B', 'C'],
                                    'Dim2': ['D', 'E', 'F']},
                                    dims=['Dim1', 'Dim2']),
                       xr.DataArray([[2, 4, 6], [8, 10, 12], [14, 16, 19]],
                                    {'Dim1': ['A', 'B', 'C'],
                                     'Dim2': ['D', 'E', 'F']},
                                    dims=['Dim1', 'Dim2'])]

        expected = pd.DataFrame([{'Elem1[B,F]': 6}, {'Elem1[B,F]': 12}])
        expected.index = [1, 2]

        return_addresses = {'Elem1[B,F]': ('elem1', {'Dim1': ['B'],
                                                     'Dim2': ['F']})}
        actual = DataFrameHandler.make_flat_df(df, return_addresses)

        # check all columns are in the DataFrame
        assert set(actual.columns) == set(expected.columns)
        assert set(actual.index) == set(expected.index)
        assert all(actual['Elem1[B,F]'] == expected['Elem1[B,F]'])
