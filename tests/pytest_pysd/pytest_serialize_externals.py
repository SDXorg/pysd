import pytest

from pathlib import Path
import xarray as xr
import numpy as np

import pysd

data_ext = Path(
    "test-models/tests/get_data_args_3d_xls/test_get_data_args_3d_xls.mdl")
constant_ext = Path("test-models/tests/get_constants_subranges/"
                    "test_get_constants_subranges.mdl")
lookup_ext = Path("test-models/tests/get_lookups_data_3d_xls/"
                  "test_get_lookups_data_3d_xls.mdl")
lookup_subranges = Path("test-models/tests/get_lookups_subset/"
                        "test_get_lookups_subset.mdl")
mixed_definitions = Path("test-models/tests/get_mixed_definitions/"
                         "test_get_mixed_definitions.mdl")
subscript_ext = Path("test-models/tests/get_subscript_3d_arrays_xls/"
                     "test_get_subscript_3d_arrays_xls.mdl")
multiple_excels = Path("test-models/tests/get_lookups_data_3d_xls/"
                       "test_get_lookups_data_3d_xls.mdl")
incomplete_constant_def = Path("test-models/tests/"
                               "get_constants_incomplete_subscript/"
                               "test_get_constants_incomplete_subscript.mdl")


class TestSerialization():

    @pytest.mark.parametrize("model_path", [data_ext])
    def test_serialize_externals_file_content(self, tmp_path, model):

        result_path = tmp_path / "externals.nc"

        model.serialize_externals(export_path=result_path,
                                  include_externals=["input.xls"],
                                  exclude_externals=["data_backward"])

        ds = xr.open_dataset(result_path)

        assert "_ext_data_data_backward" not in ds.data_vars.keys()
        assert "_ext_data_data_forward" in ds.data_vars.keys()
        assert np.allclose(
            ds.data_vars["_ext_data_data_forward"].loc[
                {"time_#1": 10.0}].data,
            np.array([[10., 5.], [-5., 1.]]), equal_nan=True)
        assert np.allclose(ds.coords["time_#1"].values,
                           np.array([0.,  5., 10., 15., 20., 25., 30.]),
                           equal_nan=True)
        assert "description" in ds.attrs.keys()
        assert all(map(
            lambda x: x in ds.data_vars["_ext_data_data_forward"].attrs.keys(),
            ["Py Name", "sheets", "files"]))

    @pytest.mark.parametrize(
        "model_path,include,exclude",
        [
            (data_ext, ["data_forward"], None),
            (data_ext, "all", ["data_backward"]),
            (lookup_ext, ["lookup_function"], None),
            (lookup_subranges, "all", None),
            (constant_ext, ["constant"], None),
        ],
        ids=["data", "data_exclude", "lookup", "lookup_subranges", "constant"])
    def test_serialize_externals_different_types(self, model,
                                                 include, exclude):

        # external subscripts are not included because they are hardcoded in
        # the Python model.
        model_path = Path(model.py_model_file)
        externals_path = model_path.with_suffix(".nc")

        if include == "all":
            if exclude:
                varnames_included = {
                    ext.py_name: "_".join(ext.py_name.split("_")[3:])
                    for ext in model._external_elements
                    if not "_".join(ext.py_name.split("_")[3:]) in exclude
                }
            else:
                varnames_included = {
                    ext.py_name: "_".join(ext.py_name.split("_")[3:])
                    for ext in model._external_elements
                }
        else:
            if exclude:
                varnames_included = {
                    model._dependencies[var]["__external__"]: var
                    for var in include if var not in exclude
                }
            else:
                varnames_included = {
                    model._dependencies[var]["__external__"]: var
                    for var in include
                }

        model.serialize_externals(export_path=externals_path,
                                  include_externals=include,
                                  exclude_externals=exclude)

        ds = xr.open_dataset(externals_path)

        # all expected variables are in the dataset
        assert all([x in ds.data_vars.keys() for x in varnames_included])

        if exclude:
            assert not any([x in ds.data_vars.keys() for x in exclude])

        # the values in the dataset are the same than in the model
        assert all(np.allclose(
            ds.data_vars[py_name].values,
            getattr(model.components, py_name).data.values, equal_nan=True)
                for py_name in varnames_included.keys())

        model = pysd.load(model_path, initialize=False)
        model.initialize_external_data(externals=externals_path)

        # the values in the dataset are the same than in the model
        # reinitialized
        assert all(np.allclose(
            ds.data_vars[py_name].values,
            getattr(model.components, py_name).data.values, equal_nan=True)
                for py_name in varnames_included.keys())

    @pytest.mark.parametrize("model_path", [mixed_definitions])
    def test_serialize_mixed_definitions(self, model):

        model_path = Path(model.py_model_file)
        externals_path = model_path.with_suffix(".nc")

        varname = "_ext_constant_variable"
        warn_message = f"No variable depends upon {varname}. This is likely " \
                       f"due to the fact that {varname} is defined using a " \
                       "mix of DATA and CONSTANT. Though Vensim allows it, " \
                       "it is not recommended."
        with pytest.warns(UserWarning, match=warn_message):
            model.serialize_externals(export_path=externals_path,
                                      include_externals="all",
                                      exclude_externals=None)

        ds = xr.open_dataset(externals_path)

        assert list(ds.coords) == [
            'dim', 'adim', 'Type', 'DimAB', 'time_#1', 'time_#2']
        assert ds.data_vars["_ext_constant_const_var_1"].dims == (
            'dim',)

        # There will be nans stored in the dataset if the dimension
        # does not have values in all dimensions (incomplete definition)
        assert any(ds.data_vars["_ext_constant_const_var_1"].isnull())
        np.testing.assert_array_equal(
            model["_ext_constant_const_var_1"].data,
            ds.data_vars["_ext_constant_const_var_1"].data)

        # re stands for reinitialized loading from external data file
        model_re = pysd.load(model_path, initialize=False)
        model_re.initialize_external_data(externals=externals_path)
        assert model_re.external_loaded

        # Once the model is re initialized loading external data, it should not
        # have the nans that are in the dataset, even if it is an incompletely
        # defined index
        np.testing.assert_array_equal(
            model_re["_ext_constant_const_var_1"].data,
            model["_ext_constant_const_var_1"].data)
        np.testing.assert_array_equal(
            model_re["_ext_constant_const_var_2"].data,
            model["_ext_constant_const_var_2"].data)
        # this one has nans when initializing from excel
        np.testing.assert_array_equal(
            model_re["_ext_constant_const_var_3"].data,
            model["_ext_constant_const_var_3"].data)
        np.testing.assert_array_equal(
            model_re["_ext_constant_variable"].data,
            model["_ext_constant_variable"].data)
        np.testing.assert_array_equal(
            model_re.get_series_data("_ext_data_data_var_1").data,
            model.get_series_data("_ext_data_data_var_1").data)
        np.testing.assert_array_equal(
            model_re.get_series_data("_ext_data_data_var_2").data,
            model.get_series_data("_ext_data_data_var_2").data)
        np.testing.assert_array_equal(
            model_re.get_series_data("_ext_data_data_var_3").data,
            model.get_series_data("_ext_data_data_var_3").data)
        np.testing.assert_array_equal(
            model_re.get_series_data("_ext_data_variable").data,
            model.get_series_data("_ext_data_variable").data)
        np.testing.assert_array_equal(
            model_re.get_series_data("_ext_data_variable").data,
            model.get_series_data("_ext_data_variable").data)
        # check that the time coordinates are the same in the ds than in the
        # model
        np.testing.assert_array_equal(
            ds.data_vars["_ext_data_variable"].coords["time_#2"],
            model_re.get_series_data("_ext_data_variable").coords["time"]
            )

        # data interpolation
        np.testing.assert_array_equal(
            getattr(model_re.components, "_ext_data_data_var_3")(1.5).data,
            getattr(model.components, "_ext_data_data_var_3")(1.5).data
            )
        np.testing.assert_array_equal(
            getattr(model_re.components, "_ext_data_variable")(5.5).data,
            getattr(model.components, "_ext_data_variable")(5.5).data)

        warn_message = r"extrapolating data (above|below) the "\
            r"(maximum|minimum) value of the time"
        with pytest.warns(UserWarning, match=warn_message):
            np.testing.assert_array_equal(
                getattr(model_re.components, "_ext_data_variable")(25).data,
                getattr(model.components, "_ext_data_variable")(25).data)
            np.testing.assert_array_equal(
                getattr(model_re.components, "_ext_data_variable")(-1).data,
                getattr(model.components, "_ext_data_variable")(-1).data)

        # they should also have the same dimension names
        assert model_re["_ext_constant_const_var_1"].dims == \
            model["_ext_constant_const_var_1"].dims
        assert model_re["_ext_constant_const_var_2"].dims == \
            model["_ext_constant_const_var_2"].dims
        assert model_re["_ext_constant_const_var_3"].dims == \
            model["_ext_constant_const_var_3"].dims
        assert model_re["_ext_constant_variable"].dims == \
            model["_ext_constant_variable"].dims
        assert model_re.get_series_data("_ext_data_data_var_1").dims == \
            model.get_series_data("_ext_data_data_var_1").dims
        assert model_re.get_series_data("_ext_data_data_var_2").dims == \
            model.get_series_data("_ext_data_data_var_2").dims
        assert model_re.get_series_data("_ext_data_data_var_3").dims == \
            model.get_series_data("_ext_data_data_var_3").dims
        assert model_re.get_series_data("_ext_data_variable").dims == \
            model.get_series_data("_ext_data_variable").dims

    @pytest.mark.parametrize("model_path", [subscript_ext])
    def test_serialize_ext_subscript(self, model, tmp_path):

        # subscripts read from external file sare hardcoded in the python
        # translation, hence they should not be present in the nc file.
        result_path = tmp_path / "externals_subs.nc"

        model.serialize_externals(export_path=result_path,
                                  include_externals="all",
                                  exclude_externals=None)

        ds = xr.open_dataset(result_path)

        assert len(ds.data_vars.keys()) == 1

    @pytest.mark.parametrize(
        "model_path,include,exclude,included,excluded",
        [
            (
                multiple_excels, "all", ["input2.xls"],
                ["_ext_lookup_lookup_function"],
                ["_ext_data_data_function_table"]
            ),
            (
                multiple_excels, "all", ["input.xls"],
                ["_ext_data_data_function_table"],
                ["_ext_lookup_lookup_function"]
            ),
            (
                multiple_excels, "all", None,
                ["_ext_data_data_function_table",
                 "_ext_lookup_lookup_function"],
                []
            ),
            (
                multiple_excels, ["input.xls", ], None,
                ["_ext_lookup_lookup_function"],
                []
            ),
            (
                multiple_excels, ["input.xls", "input2.xls"], None,
                ["_ext_data_data_function_table",
                 "_ext_lookup_lookup_function"],
                []
            ),
            (
                multiple_excels, ["input.xls", "data_function_table"], None,
                ["_ext_data_data_function_table",
                 "_ext_lookup_lookup_function"],
                []
            ),
         ]
    )
    def test_serialize_combine_vars_and_excels(self, model, include, exclude,
                                               included, excluded):

        # This test combines variable names an excel sheets.
        model_path = Path(model.py_model_file)
        externals_path = model_path.with_suffix(".nc")

        model.serialize_externals(export_path=str(externals_path),
                                  include_externals=include,
                                  exclude_externals=exclude)

        ds = xr.open_dataset(externals_path)

        assert all(x in ds.data_vars.keys() for x in included)
        assert not any(x in ds.data_vars.keys() for x in excluded)

    @pytest.mark.parametrize("model_path", [incomplete_constant_def])
    @pytest.mark.filterwarnings("ignore")
    def test_incomplete_constant_definition(self, model):

        py_model = Path(model.py_model_file)
        ext_path = py_model.parent / "externals.nc"

        model.serialize_externals(ext_path, include_externals="all")

        model_re = pysd.load(py_model, initialize=False)
        model_re.initialize_external_data(ext_path)

        assert model["adimensional_const"] == model_re["adimensional_const"]

        # the reinitialized model will have nans, because the final_coords is
        # not consistent with the actual coords of the variable
        np.testing.assert_array_equal(
            model["_ext_constant_var_only_women"].data,
            model_re["_ext_constant_var_only_women"].data)

        assert getattr(
            model.components,
            "_ext_constant_var_only_women").final_coords["dim2"]\
            == ["female", "male"]

    @pytest.mark.parametrize("model_path", [subscript_ext])
    def test_exceptions(self, model):

        # not including any externals should fail
        with pytest.raises(ValueError):
            model.serialize_externals(export_path="externals.nc",
                                      include_externals=None)

        # include externals must be "all" or list
        with pytest.raises(TypeError):
            model.serialize_externals(export_path="externals.nc",
                                      include_externals="hello_world")

        # exclude externals must be a list
        with pytest.raises(TypeError):
            model.serialize_externals(export_path="externals.nc",
                                      include_externals="all",
                                      exclude_externals="hello_world")
        # externals file not found
        with pytest.raises(FileNotFoundError):
            model.initialize_external_data(externals="missing_file.nc")
