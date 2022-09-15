from pathlib import Path
from warnings import simplefilter, catch_warnings

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc

from pysd.tools.benchmarking import assert_frames_close
from pysd.py_backend.output import OutputHandlerInterface, DatasetHandler, \
    DataFrameHandler, ModelOutput

import pysd


_root = Path(__file__).parent.parent

test_model_look = _root.joinpath(
    "test-models/tests/get_lookups_subscripted_args/"
    + "test_get_lookups_subscripted_args.mdl")
test_model_constants = _root.joinpath(
    "test-models/tests/get_constants_subranges/"
    "test_get_constants_subranges.mdl"
)
test_model_numeric_coords = _root.joinpath(
    "test-models/tests/subscript_1d_arrays/"
    "test_subscript_1d_arrays.mdl"
)
test_variable_step = _root.joinpath(
    "test-models/tests/control_vars/"
    "test_control_vars.mdl"
)
test_partial_definitions = _root.joinpath(
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
            def process_output(self, out_file):
                pass

            def initialize(self, model, capture_elements):
                pass

            def update(self, model, capture_elements):
                pass

            def postprocess(self, **kwargs):
                pass

            def add_run_elements(self, capture_elemetns):
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
            EmptyHandler.initialize(EmptyHandler, "model", "capture")

        with pytest.raises(NotImplementedError):
            EmptyHandler.process_output(EmptyHandler, "out_file")

        with pytest.raises(NotImplementedError):
            EmptyHandler.update(EmptyHandler, "model", "capture")

        with pytest.raises(NotImplementedError):
            EmptyHandler.postprocess(EmptyHandler)

        with pytest.raises(NotImplementedError):
            EmptyHandler.add_run_elements(
                EmptyHandler, "model", "capture")

    def test_output_nc(self, shared_tmpdir):
        model = pysd.read_vensim(test_model_look)
        model.progress = False

        out_file = shared_tmpdir.joinpath("results.nc")

        with catch_warnings(record=True) as w:
            simplefilter("always")
            model.run(output_file=out_file)

        with nc.Dataset(out_file, "r") as ds:
            assert ds.ncattrs() == [
                'description', 'model_file', 'timestep', 'initial_time',
                'final_time']
            assert list(ds.dimensions.keys()) == ["Rows", "Dim", "time"]
            # dimensions are stored as variables
            assert ds["Rows"].size == 2
            assert "Rows" in ds.variables.keys()
            assert "time" in ds.variables.keys()
            # scalars do not have the time dimension
            assert ds["initial_time"].size == 1
            # cache step variables have the "time" dimension
            assert ds["lookup_1d_time"].dimensions == ("time",)

            assert ds["d2d"].dimensions == ("time", "Rows", "Dim")

            with catch_warnings(record=True) as w:
                simplefilter("always")
                assert ds["d2d"].Comment == "Missing"
                assert ds["d2d"].Units == "Missing"

        # test cache run variables with dimensions
        model2 = pysd.read_vensim(test_model_constants)
        model2.progress = False

        out_file2 = shared_tmpdir.joinpath("results2.nc")

        with catch_warnings(record=True) as w:
            simplefilter("always")
            model2.run(output_file=out_file2)

        with nc.Dataset(out_file2, "r") as ds:
            assert ds["time_step"].size == 1
            assert "constant" in list(ds.variables.keys())
            assert ds["constant"].dimensions == ("dim1",)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                assert ds["dim1"][:].data.dtype == "S1"

        # dimension with numeric coords
        model3 = pysd.read_vensim(test_model_numeric_coords)
        model3.progress = False

        out_file3 = shared_tmpdir.joinpath("results3.nc")

        with catch_warnings(record=True) as w:
            simplefilter("always")
            model3.run(output_file=out_file3)

        # using xarray instead of netCDF4 to load the dataset

        with catch_warnings(record=True) as w:
            simplefilter("always")
            ds = xr.open_dataset(out_file3, engine="netcdf4")

        assert "time" in ds.dims
        assert ds["rate_a"].dims == ("One Dimensional Subscript",)
        assert ds["stock_a"].dims == ("time", "One Dimensional Subscript")

        # coordinates get dtype=object when their length is different
        assert ds["One Dimensional Subscript"].data.dtype == "O"

        # check data
        assert np.array_equal(
                ds["time"].data, np.arange(0.0, 101.0, 1.0))
        assert np.allclose(
                ds["stock_a"][0, :].data, np.array([0.0, 0.0, 0.0]))
        assert np.allclose(
                ds["stock_a"][-1, :].data, np.array([1.0, 2.0, 3.0]))
        assert ds["rate_a"].shape == (3,)

        # variable attributes
        assert list(model.doc.columns) == list(ds["stock_a"].attrs.keys())

        # global attributes
        assert list(ds.attrs.keys()) == [
            'description', 'model_file', 'timestep', 'initial_time',
            'final_time']

        model4 = pysd.read_vensim(test_variable_step)
        model4.progress = False

        out_file4 = shared_tmpdir.joinpath("results4.nc")

        with catch_warnings(record=True) as w:
            simplefilter("always")
            model4.run(output_file=out_file4)

        with catch_warnings(record=True) as w:
            simplefilter("always")
            ds = xr.open_dataset(out_file4, engine="netcdf4")

        # global attributes for variable timestep
        assert ds.attrs["timestep"] == "Variable"
        assert ds.attrs["final_time"] == "Variable"

        # saveper final_time and time_step have time dimension
        assert ds["saveper"].dims == ("time",)
        assert ds["time_step"].dims == ("time",)
        assert ds["final_time"].dims == ("time",)

        assert np.unique(ds["time_step"]).size == 2

    def test_output_nc2(self, shared_tmpdir):
        # dimension with numeric coords
        with catch_warnings(record=True) as w:
            simplefilter("always")
            model5 = pysd.read_vensim(test_partial_definitions)
        model5.progress = False

        out_file5 = shared_tmpdir.joinpath("results5.nc")

        with catch_warnings(record=True) as w:
            simplefilter("always")
            model5.run(output_file=out_file5)

        # using xarray instead of netCDF4 to load the dataset

        with catch_warnings(record=True) as w:
            simplefilter("always")
            ds = xr.open_dataset(out_file5, engine="netcdf4")

        print(ds)

    @pytest.mark.parametrize("fmt,sep", [("csv", ","), ("tab", "\t")])
    def test_output_csv(self, fmt, sep, capsys, shared_tmpdir):
        model = pysd.read_vensim(test_model_look)
        model.progress = False

        out_file = shared_tmpdir.joinpath("results." + fmt)

        with catch_warnings(record=True) as w:
            simplefilter("always")
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

    def test_dataset_handler_step_setter(self, shared_tmpdir):
        model = pysd.read_vensim(test_model_look)
        capture_elements = set()
        results = shared_tmpdir.joinpath("results.nc")
        output = ModelOutput(model, capture_elements, results)

        # Dataset handler step cannot be modified from the outside
        with pytest.raises(AttributeError):
            output.handler.step = 5

        with pytest.raises(AttributeError):
            output.handler.__update_step()

        assert output.handler.step == 0

    def test_make_flat_df(self):

        df = pd.DataFrame(index=[1], columns=['elem1'])
        df.at[1] = [xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
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
        df.at[1] = [xr.DataArray(5)]

        expected = pd.DataFrame(index=[1], data={'Elem1': 5.})

        return_addresses = {'Elem1': ('elem1', {})}

        actual = DataFrameHandler.make_flat_df(
            df, return_addresses, flatten=True)

        # check all columns are in the DataFrame
        assert set(actual.columns) == set(expected.columns)
        assert_frames_close(actual, expected, rtol=1e-8, atol=1e-8)

    def test_make_flat_df_nosubs(self):

        df = pd.DataFrame(index=[1], columns=['elem1', 'elem2'])
        df.at[1] = [25, 13]

        expected = pd.DataFrame(index=[1], columns=['Elem1', 'Elem2'])
        expected.at[1] = [25, 13]

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
        df.at[1] = [xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                 {'Dim1': ['A', 'B', 'C'],
                                  'Dim2': ['D', 'E', 'F']},
                                 dims=['Dim1', 'Dim2']),
                    xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                 {'Dim1': ['A', 'B', 'C'],
                                  'Dim2': ['D', 'E', 'F']},
                                 dims=['Dim1', 'Dim2'])]

        expected = pd.DataFrame(index=[1], columns=['Elem1[A, Dim2]', 'Elem2'])
        expected.at[1] = [xr.DataArray([[1, 2, 3]],
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
        df.at[1] = [xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
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

        expected.at[1] = [1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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
        df.at[1] = [
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

        expected.at[1] = [1, 2, 3, 4, 5, 6, 7, 8, 9]

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
