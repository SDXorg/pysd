from pathlib import Path

import pytest
import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc

from pysd.tools.benchmarking import assert_frames_close

import pysd

# TODO replace test paths by fixtures and translate and run the models
# in temporal directories

test_model = Path("test-models/samples/teacup/teacup.mdl")
test_model_subs = Path(
    "test-models/tests/subscript_2d_arrays/test_subscript_2d_arrays.mdl")
test_model_look = Path(
    "test-models/tests/get_lookups_subscripted_args/"
    + "test_get_lookups_subscripted_args.mdl")
test_model_data = Path(
    "test-models/tests/get_data_args_3d_xls/test_get_data_args_3d_xls.mdl")


more_tests = Path("more-tests")

test_model_constant_pipe = more_tests.joinpath(
    "constant_pipeline/test_constant_pipeline.mdl")


class TestPySD():

    @pytest.mark.parametrize("model_path", [test_model])
    def test_run(self, model):
        stocks = model.run()
        # return a dataframe
        assert isinstance(stocks, pd.DataFrame)
        # contains correct column
        assert "Teacup Temperature" in stocks.columns.values
        # has multiple rows
        assert len(stocks) > 3
        # there are no null values in the set
        assert stocks.notnull().all().all()

    def test_run_ignore_missing(self, _root):
        model_mdl = _root.joinpath(
            'test-models/tests/get_with_missing_values_xlsx/'
            + 'test_get_with_missing_values_xlsx.mdl')
        model_py = _root.joinpath(
            'test-models/tests/get_with_missing_values_xlsx/'
            + 'test_get_with_missing_values_xlsx.py')

        # ignore warnings for missing values
        model = pysd.read_vensim(model_mdl, missing_values="ignore")

        with pytest.warns(UserWarning, match='extrapolating data'):
            # ignore warnings for missing values
            model.run()

        with pytest.warns(UserWarning, match='missing'):
            # warnings for missing values
            model = pysd.load(model_py)

        with pytest.warns(UserWarning, match='extrapolating data'):
            # second initialization of external avoided
            model.run()

        with pytest.raises(ValueError, match='missing'):
            # errors for missing values
            pysd.load(model_py, missing_values="raise")

    @pytest.mark.parametrize("model_path", [test_model])
    def test_run_includes_last_value(self, model):
        res = model.run()
        assert res.index[-1] == model.components.final_time()

    @pytest.mark.parametrize("model_path", [test_model])
    def test_run_build_timeseries(self, model):
        res = model.run(final_time=7, time_step=2, initial_condition=(3, {}))

        actual = list(res.index)
        expected = [3.0, 5.0, 7.0]
        assert actual == expected

    @pytest.mark.parametrize("model_path", [test_model])
    def test_run_progress(self, model):
        # same as test_run but with progressbar
        stocks = model.run(progress=True)
        assert isinstance(stocks, pd.DataFrame)
        assert "Teacup Temperature" in stocks.columns.values
        assert len(stocks) > 3
        assert stocks.notnull().all().all()

    @pytest.mark.parametrize(
        "model_path",
        [Path("test-models/tests/control_vars/test_control_vars.mdl")])
    def test_run_progress_dynamic(self, model):
        # same as test_run but with progressbar
        warn_message = r"The progressbar is not compatible with dynamic "\
                       r"final time or time step\. Both variables must be "\
                       r"constants to prompt progress\."
        with pytest.warns(UserWarning, match=warn_message):
            stocks = model.run(progress=True)
        assert isinstance(stocks, pd.DataFrame)
        for var in ["FINAL TIME", "TIME STEP"]:
            # assert that control variables have change
            assert len(np.unique(stocks[var].values)) > 1

    @pytest.mark.parametrize("model_path", [test_model])
    def test_run_return_timestamps(self, model):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/17"""
        timestamps = np.random.randint(1, 5, 5).cumsum()
        stocks = model.run(return_timestamps=timestamps)
        assert (stocks.index.values == timestamps).all()

        stocks = model.run(return_timestamps=5)
        assert stocks.index[0] == 5

        timestamps = ['A', 'B']
        with pytest.raises(TypeError):
            model.run(return_timestamps=timestamps)

        # assert that return_timestamps works with float error
        stocks = model.run(time_step=0.1, return_timestamps=0.3)
        assert 0.3 in stocks.index

        # assert that return_timestamps works with float error
        stocks = model.run(
            time_step=0.1, return_timestamps=[0.3, 0.1, 10.5, 0.9])
        assert 0.1 in stocks.index
        assert 0.3 in stocks.index
        assert 0.9 in stocks.index
        assert 10.5 in stocks.index

        # assert one timestamp is not returned because is not multiple of
        # the time step
        warning_message =\
            "The returning time stamp '%s' seems to not be a multiple "\
            "of the time step. This value will not be saved in the output. "\
            "Please, modify the returning timestamps or the integration "\
            "time step to avoid this."
        # assert that return_timestamps works with float error
        with pytest.warns(UserWarning, match=warning_message % 0.55):
            stocks = model.run(
                time_step=0.1, return_timestamps=[0.3, 0.1, 0.55, 0.9])
        assert 0.1 in stocks.index
        assert 0.3 in stocks.index
        assert 0.9 in stocks.index
        assert 0.55 not in stocks.index

        with pytest.warns(UserWarning,
                          match=warning_message % "(0.15|0.55|0.95)"):
            stocks = model.run(
                time_step=0.1, return_timestamps=[0.3, 0.15, 0.55, 0.95])
        assert 0.15 not in stocks.index
        assert 0.3 in stocks.index
        assert 0.95 not in stocks.index
        assert 0.55 not in stocks.index

    @pytest.mark.parametrize("model_path", [test_model])
    def test_run_return_timestamps_past_final_time(self, model):
        """
        If the user enters a timestamp that is longer than the euler
        timeseries that is defined by the normal model file, should
        extend the euler series to the largest timestamp
        """
        return_timestamps = list(range(0, 100, 10))
        stocks = model.run(return_timestamps=return_timestamps)

        assert return_timestamps == list(stocks.index)

    @pytest.mark.parametrize("model_path", [test_model])
    def test_return_timestamps_with_range(self, model):
        """
        Tests that return timestamps may receive a 'range'.
        It will be cast to a numpy array in the end...
        """
        return_timestamps = range(0, 31, 10)
        stocks = model.run(return_timestamps=return_timestamps)
        assert list(return_timestamps) == list(stocks.index)

    @pytest.mark.parametrize("model_path", [test_model])
    def test_run_return_columns_original_names(self, model):
        """
        Addresses https://github.com/JamesPHoughton/pysd/issues/26
        - Also checks that columns are returned in the correct order
        """
        return_columns = ["Room Temperature", "Teacup Temperature"]
        result = model.run(return_columns=return_columns)
        assert set(result.columns) == set(return_columns)

    @pytest.mark.parametrize("model_path", [test_model])
    def test_run_return_columns_step(self, model):
        """
        Return only cache 'step' variables
        """
        result = model.run(return_columns='step')
        assert set(result.columns)\
            == {'Teacup Temperature', 'Heat Loss to Room'}

    @pytest.mark.parametrize("model_path", [test_model])
    def test_run_reload(self, model):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/99"""
        result0 = model.run()
        result1 = model.run(params={"Room Temperature": 1000})
        result2 = model.run()
        result3 = model.run(reload=True)

        assert (result0 == result3).all().all()
        assert not (result0 == result1).all().all()
        assert (result1 == result2).all().all()

    @pytest.mark.parametrize("model_path", [test_model])
    def test_run_return_columns_pysafe_names(self, model):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/26"""
        return_columns = ["room_temperature", "teacup_temperature"]
        result = model.run(return_columns=return_columns)
        assert set(result.columns) == set(return_columns)

    @pytest.mark.parametrize("model_path", [test_model])
    def test_initial_conditions_invalid(self, model):
        error_message = r"Invalid initial conditions\. "\
            r"Check documentation for valid entries or use "\
            r"'help\(model\.set_initial_condition\)'\."
        with pytest.raises(TypeError, match=error_message):
            model.run(initial_condition=["this is not valid"])

    @pytest.mark.parametrize("model_path", [test_model])
    def test_initial_conditions_tuple_pysafe_names(self, model):
        stocks = model.run(
            initial_condition=(3000, {"teacup_temperature": 33}),
            return_timestamps=list(range(3000, 3010))
        )

        assert stocks["Teacup Temperature"].iloc[0] == 33

    @pytest.mark.parametrize("model_path", [test_model])
    def test_initial_conditions_tuple_original_names(self, model):
        """ Responds to https://github.com/JamesPHoughton/pysd/issues/77"""
        stocks = model.run(
            initial_condition=(3000, {"Teacup Temperature": 33}),
            return_timestamps=list(range(3000, 3010)),
        )
        assert stocks.index[0] == 3000
        assert stocks["Teacup Temperature"].iloc[0] == 33

    @pytest.mark.parametrize("model_path", [test_model])
    def test_initial_conditions_current(self, model):
        stocks1 = model.run(return_timestamps=list(range(0, 31)))
        stocks2 = model.run(
            initial_condition="current", return_timestamps=list(range(30, 45))
        )
        assert stocks1["Teacup Temperature"].iloc[-1]\
            == stocks2["Teacup Temperature"].iloc[0]

    @pytest.mark.parametrize("model_path", [test_model])
    def test_initial_condition_bad_value(self, model):
        with pytest.raises(FileNotFoundError):
            model.run(initial_condition="bad value")

    @pytest.mark.parametrize("model_path", [test_model_subs])
    def test_initial_conditions_subscripted_value_with_numpy_error(self,
                                                                   model):
        input_ = np.array([[5, 3], [4, 8], [9, 3]])
        with pytest.raises(TypeError):
            model.run(initial_condition=(5, {'stock_a': input_}),
                      return_columns=['stock_a'],
                      return_timestamps=list(range(5, 10)))

    @pytest.mark.parametrize("model_path", [test_model])
    def test_set_constant_parameter(self, model):
        """ In response to:
        re: https://github.com/JamesPHoughton/pysd/issues/5"""
        model.set_components({"room_temperature": 20})
        assert model.components.room_temperature() == 20

        model.run(params={"room_temperature": 70})
        assert model.components.room_temperature() == 70

        with pytest.raises(NameError):
            model.set_components({'not_a_var': 20})

    @pytest.mark.parametrize("model_path", [test_model])
    def test_set_constant_parameter_inline(self, model):
        model.components.room_temperature = 20
        assert model.components.room_temperature() == 20

        model.run(params={"room_temperature": 70})
        assert model.components.room_temperature() == 70

        with pytest.raises(NameError):
            model.components.not_a_var = 20

    @pytest.mark.parametrize("model_path", [test_model])
    def test_set_timeseries_parameter(self, model):
        timeseries = list(range(30))
        temp_timeseries = pd.Series(
            index=timeseries,
            data=(50 + np.random.rand(len(timeseries)).cumsum())
        )
        res = model.run(
            params={"room_temperature": temp_timeseries},
            return_columns=["room_temperature"],
            return_timestamps=timeseries,
        )
        assert (res["room_temperature"] == temp_timeseries).all()

    @pytest.mark.parametrize("model_path", [test_model])
    def test_set_timeseries_parameter_inline(self, model):
        timeseries = list(range(30))
        temp_timeseries = pd.Series(
            index=timeseries,
            data=(50 + np.random.rand(len(timeseries)).cumsum())
        )
        model.components.room_temperature = temp_timeseries
        res = model.run(
            return_columns=["room_temperature"],
            return_timestamps=timeseries,
        )
        assert (res["room_temperature"] == temp_timeseries).all()

    @pytest.mark.parametrize("model_path", [test_model])
    def test_set_component_with_real_name(self, model):
        model.set_components({"Room Temperature": 20})
        assert model.components.room_temperature() == 20

        model.run(params={"Room Temperature": 70})
        assert model.components.room_temperature() == 70

    @pytest.mark.parametrize("model_path", [test_model])
    def test_set_components_warnings(self, model):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/80"""
        warn_message = r"Replacing the equation of stock "\
                       r"'Teacup Temperature' with params\.\.\."
        with pytest.warns(UserWarning, match=warn_message):
            model.set_components(
                {"Teacup Temperature": 20, "Characteristic Time": 15}
            )  # set stock value using params

    @pytest.mark.parametrize("model_path", [test_model])
    def test_set_components_with_function(self, model):
        def test_func():
            return 5

        model.set_components({"Room Temperature": test_func})
        res = model.run(return_columns=["Room Temperature"])
        assert test_func() == res["Room Temperature"].iloc[0]

    @pytest.mark.parametrize("model_path", [test_model_subs])
    def test_set_subscripted_value_with_constant(self, model):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output = xr.DataArray([[5, 5], [5, 5], [5, 5]], coords, dims)

        model.set_components({"initial_values": 5, "final_time": 10})
        res = model.run(
            return_columns=["Initial Values"], flatten_output=False)
        assert output.equals(res["Initial Values"].iloc[0])

    @pytest.mark.parametrize("model_path", [test_model_subs])
    def test_set_subscripted_value_with_partial_xarray(self, model):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output = xr.DataArray([[5, 3], [5, 3], [5, 3]], coords, dims)
        input_val = xr.DataArray(
            [5, 3],
            {"Second Dimension Subscript": ["Column 1", "Column 2"]},
            ["Second Dimension Subscript"],
        )

        model.set_components({"Initial Values": input_val, "final_time": 10})
        res = model.run(
            return_columns=["Initial Values"], flatten_output=False)
        assert output.equals(res["Initial Values"].iloc[0])

    @pytest.mark.parametrize("model_path", [test_model_subs])
    def test_set_subscripted_value_with_xarray(self, model):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output = xr.DataArray([[5, 3], [4, 8], [9, 3]], coords, dims)

        model.set_components({"initial_values": output, "final_time": 10})
        res = model.run(
            return_columns=["Initial Values"], flatten_output=False)
        assert output.equals(res["Initial Values"].iloc[0])

    @pytest.mark.parametrize("model_path", [test_model_data])
    @pytest.mark.filterwarnings("ignore")
    def test_set_parameter_data(self, model):
        timeseries = list(range(31))
        series = pd.Series(
            index=timeseries,
            data=(50+np.random.rand(len(timeseries)).cumsum())
        )

        model.set_components({"data_backward": 20, "data_forward": 70})

        out = model.run(
            return_columns=["data_backward", "data_forward"],
            flatten_output=False)

        for time in out.index:
            assert (out["data_backward"][time] == 20).all()
            assert (out["data_forward"][time] == 70).all()

        out = model.run(
            return_columns=["data_backward", "data_forward"],
            final_time=20, time_step=1, saveper=1,
            params={"data_forward": 30, "data_backward": series},
            flatten_output=False)

        for time in out.index:
            assert (out["data_forward"][time] == 30).all()
            assert (out["data_backward"][time] == series[time]).all()

    @pytest.mark.parametrize("model_path", [test_model_look])
    @pytest.mark.filterwarnings("ignore")
    def test_set_constant_parameter_lookup(self, model):
        model.set_components({"lookup_1d": 20})
        for i in range(100):
            assert model.components.lookup_1d(i) == 20

        model.run(params={"lookup_1d": 70}, final_time=1)
        for i in range(100):
            assert model.components.lookup_1d(i) == 70

        model.set_components({"lookup_2d": 20})
        for i in range(100):
            assert model.components.lookup_2d(i).equals(
                xr.DataArray(20, {"Rows": ["Row1", "Row2"]}, ["Rows"])
            )

        model.run(params={"lookup_2d": 70}, final_time=1)
        for i in range(100):
            assert model.components.lookup_2d(i).equals(
                xr.DataArray(70, {"Rows": ["Row1", "Row2"]}, ["Rows"])
            )

        xr1 = xr.DataArray([-10, 50], {"Rows": ["Row1", "Row2"]}, ["Rows"])
        model.set_components({"lookup_2d": xr1})
        for i in range(100):
            assert model.components.lookup_2d(i).equals(xr1)

        xr2 = xr.DataArray([-100, 500], {"Rows": ["Row1", "Row2"]}, ["Rows"])
        model.run(params={"lookup_2d": xr2}, final_time=1)
        for i in range(100):
            assert model.components.lookup_2d(i).equals(xr2)

    @pytest.mark.parametrize("model_path", [test_model_look])
    @pytest.mark.filterwarnings("ignore")
    def test_set_timeseries_parameter_lookup(self, model):
        timeseries = list(range(30))

        temp_timeseries = pd.Series(
            index=timeseries,
            data=(50+np.random.rand(len(timeseries)).cumsum())
        )

        res = model.run(
            params={"lookup_1d": temp_timeseries},
            return_columns=["lookup_1d_time"],
            return_timestamps=timeseries,
            flatten_output=False
        )

        assert (res["lookup_1d_time"] == temp_timeseries).all()

        res = model.run(
            params={"lookup_2d": temp_timeseries},
            return_columns=["lookup_2d_time"],
            return_timestamps=timeseries,
            flatten_output=False
        )

        assert all(
            [
                a.equals(xr.DataArray(b, {"Rows": ["Row1", "Row2"]}, ["Rows"]))
                for a, b in zip(res["lookup_2d_time"].values,
                                temp_timeseries)
            ]
        )

        temp_timeseries2 = pd.Series(
            index=timeseries,
            data=[
                xr.DataArray(
                    [50 + x, 20 - y], {"Rows": ["Row1", "Row2"]}, ["Rows"]
                )
                for x, y in zip(
                    np.random.rand(len(timeseries)).cumsum(),
                    np.random.rand(len(timeseries)).cumsum(),
                )
            ],
        )

        res = model.run(
            params={"lookup_2d": temp_timeseries2},
            return_columns=["lookup_2d_time"],
            return_timestamps=timeseries,
            flatten_output=False
        )

        assert all(
            [
                a.equals(b)
                for a, b in zip(res["lookup_2d_time"].values,
                                temp_timeseries2)
            ]
        )

    @pytest.mark.parametrize("model_path", [test_model_subs])
    def test_set_subscripted_value_with_numpy_error(self, model):
        input_ = np.array([[5, 3], [4, 8], [9, 3]])
        with pytest.raises(TypeError):
            model.set_components({"initial_values": input_, "final_time": 10})

    @pytest.mark.parametrize("model_path", [test_model_subs])
    def test_set_subscripted_timeseries_parameter_with_constant(self, model):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]

        timeseries = list(range(10))
        val_series = [50 + rd for rd in np.random.rand(len(timeseries)
                                                       ).cumsum()]
        xr_series = [xr.DataArray(val, coords, dims) for val in val_series]

        temp_timeseries = pd.Series(index=timeseries, data=val_series)
        res = model.run(
            params={"initial_values": temp_timeseries, "final_time": 10},
            return_columns=["initial_values"],
            return_timestamps=timeseries,
            flatten_output=False
        )

        assert np.all(
            [
                r.equals(t)
                for r, t in zip(res["initial_values"].values, xr_series)
            ]
        )

    @pytest.mark.parametrize("model_path", [test_model_subs])
    def test_set_subscripted_timeseries_parameter_with_partial_xarray(self,
                                                                      model):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        out_b = xr.DataArray([[0, 0], [0, 0], [0, 0]], coords, dims)
        input_val = xr.DataArray(
            [5, 3],
            {"Second Dimension Subscript": ["Column 1", "Column 2"]},
            ["Second Dimension Subscript"],
        )

        timeseries = list(range(10))
        val_series = [input_val + rd for rd in np.random.rand(len(timeseries)
                                                              ).cumsum()]
        temp_timeseries = pd.Series(index=timeseries, data=val_series)
        out_series = [out_b + val for val in val_series]
        model.set_components({"initial_values": temp_timeseries,
                              "final_time": 10})
        res = model.run(
            return_columns=["initial_values"], flatten_output=False)
        assert np.all(
            [
                r.equals(t)
                for r, t in zip(res["initial_values"].values, out_series)
            ]
        )

    @pytest.mark.parametrize("model_path", [test_model_subs])
    def test_set_subscripted_timeseries_parameter_with_xarray(self, model):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]

        init_val = xr.DataArray([[5, 3], [4, 8], [9, 3]], coords, dims)

        timeseries = list(range(10))
        temp_timeseries = pd.Series(
            index=timeseries,
            data=[init_val + rd for rd in np.random.rand(len(timeseries)
                                                         ).cumsum()],
        )
        res = model.run(
            params={"initial_values": temp_timeseries, "final_time": 10},
            return_columns=["initial_values"],
            return_timestamps=timeseries,
            flatten_output=False
        )

        assert np.all(
            [
                r.equals(t)
                for r, t in zip(
                    res["initial_values"].values, temp_timeseries.values
                )
            ]
        )

    @pytest.mark.parametrize("model_path", [test_model])
    def test_docs(self, model):
        """ Test that the model prints some documentation """
        assert isinstance(str(model), str)  # tests string conversion of
        doc = model.doc
        assert isinstance(doc, pd.DataFrame)
        assert {
            "Characteristic Time",
            "Teacup Temperature",
            "FINAL TIME",
            "Heat Loss to Room",
            "INITIAL TIME",
            "Room Temperature",
            "SAVEPER",
            "TIME STEP",
            "Time"
        } == set(doc["Real Name"].values)

        assert\
            doc[doc["Real Name"] == "Heat Loss to Room"]["Units"].values[0]\
            == "Degrees Fahrenheit/Minute"

        assert\
            doc[doc["Real Name"] == "Teacup Temperature"]["Py Name"].values[0]\
            == "teacup_temperature"

        assert\
            doc[doc["Real Name"] == "INITIAL TIME"]["Comment"].values[0]\
            == "The initial time for the simulation."

        assert\
            doc[doc["Real Name"] == "Characteristic Time"]["Type"].values[0]\
            == "Constant"

        assert\
            doc[doc["Real Name"]
                == "Characteristic Time"]["Subtype"].values[0]\
            == "Normal"

        assert\
            doc[doc["Real Name"] == "Teacup Temperature"]["Limits"].values[0]\
            == (32.0, 212.0)

    def test_stepwise_cache(self):
        from pysd.py_backend.cache import Cache

        run_history = []
        result_history = []
        cache = Cache()

        @cache
        def upstream(run_hist, res_hist):
            run_hist.append("U")
            return "up"

        def downstream(run_hist, res_hist):
            run_hist.append("D")
            result_history.append(upstream(run_hist, res_hist))
            return "down"

        # initially neither function has a chache value
        assert "upstream" not in cache.data
        assert "downstream" not in cache.data

        # when the functions are called,
        # the cache is instantiated in the upstream (cached) function
        result_history.append(downstream(run_history, result_history))
        assert "upstream" in cache.data
        assert "downstream" not in cache.data
        assert run_history == ["D", "U"]
        assert result_history == ["up", "down"]

        # at the second call, the uncached function is run,
        # but the cached upstream function returns its prior value
        result_history.append(downstream(run_history, result_history))
        assert run_history == ["D", "U", "D"]
        assert result_history == ["up", "down", "up", "down"]

        # clean step cache
        cache.clean()
        assert "upstream" not in cache.data

        result_history.append(downstream(run_history, result_history))
        assert run_history == ["D", "U", "D", "D", "U"]
        assert result_history == ["up", "down", "up", "down", "up", "down"]

    def test_runwise_cache(self):
        from pysd.py_backend.cache import constant_cache

        run_history = []
        result_history = []

        def upstream(run_hist, res_hist):
            run_hist.append("U")
            return "up"

        def downstream(run_hist, res_hist):
            run_hist.append("D")
            result_history.append(upstream(run_hist, res_hist))
            return "down"

        # when the constant cache is assigned,
        # the cache is instantiated in the upstream (cached) function
        upstream = constant_cache(upstream, run_history, result_history)
        assert run_history == ["U"]
        result_history.append(downstream(run_history, result_history))

        assert run_history == ["U", "D"]
        assert result_history == ["up", "down"]
        assert upstream.value == "up"

        # at the second call, the uncached function is run,
        # but the cached upstream function returns its prior value
        result_history.append(downstream(run_history, result_history))
        assert run_history == ["U", "D", "D"]
        assert result_history == ["up", "down", "up", "down"]

        # recover previous function
        assert hasattr(upstream, "__wrapped__")
        upstream = upstream.function
        assert not hasattr(upstream, "__wrapped__")

        result_history.append(downstream(run_history, result_history))
        assert run_history == ["U", "D", "D", "D", "U"]
        assert result_history == ["up", "down", "up", "down", "up", "down"]

    @pytest.mark.parametrize("model_path", [test_model])
    def test_initialize(self, model):
        initial_temp = model.components.teacup_temperature()
        model.run()
        final_temp = model.components.teacup_temperature()
        model.initialize()
        reset_temp = model.components.teacup_temperature()
        assert initial_temp != final_temp
        assert initial_temp == reset_temp

    def test_initialize_order(self, _root):
        model = pysd.load(_root / more_tests.joinpath(
            "initialization_order/test_initialization_order.py"))

        assert model.initialize_order == ["_integ_stock_a", "_integ_stock_b"]
        assert model.components.stock_b() == 42
        assert model.components.stock_a() == 42
        model.components.initial_par = lambda: 1
        model.initialize()
        assert model.components.stock_b() == 1
        assert model.components.stock_a() == 1

    def test_set_initial_with_deps(self, _root):
        model = pysd.load(_root / more_tests.joinpath("initialization_order/"
                          "test_initialization_order.py"))

        original_a = model.components.stock_a()

        model.set_initial_condition((0, {"_integ_stock_a": 23}))
        assert model.components.stock_a() == 23
        assert model.components.stock_b() == 23

        model.reload()
        model.set_initial_condition((0, {"_integ_stock_b": 53}))
        assert model.components.stock_a() == original_a
        assert model.components.stock_b() == 53

        model.reload()
        model.set_initial_condition((0, {"_integ_stock_a": 89,
                                         "_integ_stock_b": 73}))
        assert model.components.stock_a() == 89
        assert model.components.stock_b() == 73

    @pytest.mark.parametrize("model_path", [test_model])
    def test_set_initial_value(self, model):
        initial_temp = model.components.teacup_temperature()

        new_time = np.random.rand()

        # Test that we can set with real names
        model.set_initial_value(new_time, {'Teacup Temperature': 500})
        assert initial_temp != 500
        assert model.components.teacup_temperature() == 500
        assert model.components.time() == new_time

        # Test setting with pysafe names
        model.set_initial_value(new_time + 1, {'teacup_temperature': 202})
        assert model.components.teacup_temperature() == 202
        assert model.components.time() == new_time + 1

        # Test setting with stateful object name
        model.set_initial_value(new_time + 2,
                                {'_integ_teacup_temperature': 302})
        assert model.components.teacup_temperature() == 302
        assert model.components.time() == new_time + 2

        with pytest.raises(NameError):
            model.set_initial_value(new_time, {'not_a_var': 500})

    @pytest.mark.parametrize("model_path", [test_model_subs])
    def test_set_initial_value_subscripted_value_with_constant(self, model):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output_b = xr.DataArray([[0, 0], [0, 0], [0, 0]], coords, dims)

        new_time = np.random.rand()
        initial_stock = model.components.stock_a()

        # Test that we can set with real names
        model.set_initial_value(new_time, {'Stock A': 500})
        assert not initial_stock.equals(output_b + 500)
        assert model.components.stock_a().equals(output_b + 500)

        # Test setting with pysafe names
        model.set_initial_value(new_time + 1, {'stock_a': 202})
        assert model.components.stock_a().equals(output_b + 202)

        # Test setting with stateful object name
        model.set_initial_value(new_time + 2, {'_integ_stock_a': 302})
        assert model.components.stock_a().equals(output_b + 302)

        # Test error when coords are not a subset
        with pytest.raises(ValueError):
            model.set_initial_value(
                new_time + 2,
                {'_integ_stock_a': xr.DataArray(302, {'D': ['A', 'B']}, ['D'])}
            )

    @pytest.mark.parametrize("model_path", [test_model_subs])
    def test_set_initial_value_subscripted_value_with_partial_xarray(self,
                                                                     model):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output1 = xr.DataArray([[5, 3], [5, 3], [5, 3]], coords, dims)
        input_val1 = xr.DataArray(
            [5, 3],
            {"Second Dimension Subscript": ["Column 1", "Column 2"]},
            ["Second Dimension Subscript"],
        )

        output2 = xr.DataArray([[55, 33], [55, 33], [55, 33]], coords, dims)
        input_val2 = xr.DataArray(
            [55, 33],
            {"Second Dimension Subscript": ["Column 1", "Column 2"]},
            ["Second Dimension Subscript"],
        )

        output3 = xr.DataArray([[40, 20], [40, 20], [40, 20]], coords, dims)
        input_val3 = xr.DataArray(
            [40, 20],
            {"Second Dimension Subscript": ["Column 1", "Column 2"]},
            ["Second Dimension Subscript"],
        )

        new_time = np.random.rand()

        initial_stock = model.components.stock_a()

        # Test that we can set with real names
        model.set_initial_value(new_time, {'Stock A': input_val1})
        assert not initial_stock.equals(output1)
        assert model.components.stock_a().equals(output1)

        # Test setting with pysafe names
        model.set_initial_value(new_time + 1, {'stock_a': input_val2})
        assert model.components.stock_a().equals(output2)

        # Test setting with stateful object name
        model.set_initial_value(new_time + 2, {'_integ_stock_a': input_val3})
        assert model.components.stock_a().equals(output3)

    @pytest.mark.parametrize("model_path", [test_model_subs])
    def test_set_initial_value_subscripted_value_with_xarray(self, model):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output1 = xr.DataArray([[5, 3], [4, 8], [9, 3]], coords, dims)
        output2 = xr.DataArray([[53, 43], [84, 80], [29, 63]], coords, dims)
        output3 = xr.DataArray([[54, 32], [40, 87], [93, 93]], coords, dims)

        new_time = np.random.rand()

        initial_stock = model.components.stock_a()

        # Test that we can set with real names
        model.set_initial_value(new_time, {'Stock A': output1})
        assert not initial_stock.equals(output1)
        assert model.components.stock_a().equals(output1)

        # Test setting with pysafe names
        model.set_initial_value(new_time + 1, {'stock_a': output2})
        assert model.components.stock_a().equals(output2)

        # Test setting with stateful object name
        model.set_initial_value(new_time + 2, {'_integ_stock_a': output3})
        assert model.components.stock_a().equals(output3)

    @pytest.mark.parametrize("model_path", [test_model_subs])
    def test_set_initial_value_subscripted_value_with_numpy_error(self, model):
        input1 = np.array([[5, 3], [4, 8], [9, 3]])
        input2 = np.array([[53, 43], [84, 80], [29, 63]])
        input3 = np.array([[54, 32], [40, 87], [93, 93]])

        new_time = np.random.rand()

        # Test that we can set with real names
        with pytest.raises(TypeError):
            model.set_initial_value(new_time, {'Stock A': input1})

        # Test setting with pysafe names
        with pytest.raises(TypeError):
            model.set_initial_value(new_time + 1, {'stock_a': input2})

        # Test setting with stateful object name
        with pytest.raises(TypeError):
            model.set_initial_value(new_time + 2, {'_integ_stock_a': input3})

    @pytest.mark.parametrize("model_path", [test_model])
    def test_replace_element(self, model):
        stocks1 = model.run()
        model.components.characteristic_time = lambda: 3
        stocks2 = model.run()
        assert stocks1["Teacup Temperature"].loc[10]\
            > stocks2["Teacup Temperature"].loc[10]

    @pytest.mark.parametrize("model_path", [test_model])
    def test_set_initial_condition_origin_full(self, model):
        initial_temp = model.components.teacup_temperature()
        initial_time = model.components.time()

        new_state = {"Teacup Temperature": 500}
        new_time = 10

        model.set_initial_condition((new_time, new_state))
        set_temp = model.components.teacup_temperature()
        set_time = model.components.time()

        assert set_temp != initial_temp,\
            "Test definition is wrong, please change configuration"

        assert set_temp == 500

        assert initial_time != new_time,\
            "Test definition is wrong, please change configuration"
        assert new_time == set_time

        model.set_initial_condition("original")
        set_temp = model.components.teacup_temperature()
        set_time = model.components.time()

        assert initial_temp == set_temp
        assert initial_time == set_time

    @pytest.mark.parametrize("model_path", [test_model])
    def test_set_initial_condition_origin_short(self, model):
        initial_temp = model.components.teacup_temperature()
        initial_time = model.components.time()

        new_state = {"Teacup Temperature": 500}
        new_time = 10

        model.set_initial_condition((new_time, new_state))
        set_temp = model.components.teacup_temperature()
        set_time = model.components.time()

        assert set_temp != initial_temp,\
            "Test definition is wrong, please change configuration"

        assert set_temp == 500

        assert initial_time != new_time,\
            "Test definition is wrong, please change configuration"
        assert new_time == set_time

        model.set_initial_condition("o")
        set_temp = model.components.teacup_temperature()
        set_time = model.components.time()

        assert initial_temp == set_temp
        assert initial_time == set_time

    @pytest.mark.parametrize("model_path", [test_model])
    def test_set_initial_condition_for_stock_component(self, model):
        initial_temp = model.components.teacup_temperature()
        initial_time = model.components.time()

        new_state = {"Teacup Temperature": 500}
        new_time = 10

        model.set_initial_condition((new_time, new_state))
        set_temp = model.components.teacup_temperature()
        set_time = model.components.time()

        assert set_temp != initial_temp,\
            "Test definition is wrong, please change configuration"

        assert set_temp == 500

        assert initial_time != 10,\
            "Test definition is wrong, please change configuration"

        assert set_time == 10

    @pytest.mark.parametrize("model_path", [test_model])
    def test_set_initial_condition_for_constant_component(self, model):
        new_state = {"Room Temperature": 100}
        new_time = 10

        error_message = r"Unrecognized stateful '.*'\. If you want"\
            r" to set a value of a regular component\. Use params"
        with pytest.raises(ValueError, match=error_message):
            model.set_initial_condition((new_time, new_state))

    @pytest.mark.parametrize(
        "model_path,args",
        [
            (
                test_model,
                {
                    "Room Temperature": [],
                    "room_temperature": [],
                    "teacup_temperature": [],
                    "_integ_teacup_temperature": []
                }
            ),
            (
                test_model_look,
                {
                    "lookup 1d": ["x", "final_subs"],
                    "lookup_1d": ["x", "final_subs"],
                    "lookup 2d": ["x", "final_subs"],
                    "lookup_2d": ["x", "final_subs"],
                }
            )
        ])
    def test_get_args(self, model, args):
        for var, arg in args.items():
            assert model.get_args(var) == arg

        with pytest.raises(NameError):
            model.get_args("not_a_var")

    @pytest.mark.parametrize(
        "model_path,coords",
        [
            (
                test_model,
                {
                    "Room Temperature": None,
                    "room_temperature": None,
                    "teacup_temperature": None,
                    "_integ_teacup_temperature": None
                }
            ),
            (
                test_model_subs,
                {
                    "Initial Values": {
                        "One Dimensional Subscript": ["Entry 1", "Entry 2",
                                                      "Entry 3"],
                        "Second Dimension Subscript":["Column 1", "Column 2"]
                    },
                    "initial_values": {
                        "One Dimensional Subscript": ["Entry 1", "Entry 2",
                                                      "Entry 3"],
                        "Second Dimension Subscript":["Column 1", "Column 2"]
                    },
                    "Stock A": {
                        "One Dimensional Subscript": ["Entry 1", "Entry 2",
                                                      "Entry 3"],
                        "Second Dimension Subscript":["Column 1", "Column 2"]
                    },
                    "stock_a": {
                        "One Dimensional Subscript": ["Entry 1", "Entry 2",
                                                      "Entry 3"],
                        "Second Dimension Subscript":["Column 1", "Column 2"]
                    },
                    "_integ_stock_a": {
                        "One Dimensional Subscript": ["Entry 1", "Entry 2",
                                                      "Entry 3"],
                        "Second Dimension Subscript":["Column 1", "Column 2"]
                    }
                }
            )
        ])
    def test_get_coords(self, model, coords):
        for var, coord in coords.items():
            if coord is not None:
                coord = coord, list(coord)
            assert model.get_coords(var) == coord

        with pytest.raises(NameError):
            model.get_coords("not_a_var")

    def test_getitem(self, _root):
        model = pysd.read_vensim(_root / test_model)
        model2 = pysd.read_vensim(_root / test_model_look)
        model3 = pysd.read_vensim(_root / test_model_data)

        coords = {'Dim': ['A', 'B'], 'Rows': ['Row1', 'Row2']}
        room_temp = 70
        temp0 = 180
        temp1 = 75.37400067686977
        data0 = xr.DataArray([[0, 4], [-1, 0]], coords, list(coords))
        data1 = xr.DataArray([[5, 2], [5, 0]], coords, list(coords))
        assert model['Room Temperature'] == room_temp
        assert model['room_temperature'] == room_temp
        assert model['Teacup Temperature'] == temp0
        assert model['teacup_temperature'] == temp0
        assert model['_integ_teacup_temperature'] == temp0

        model.run()

        assert model['Room Temperature'] == room_temp
        assert model['room_temperature'] == room_temp
        assert model['Teacup Temperature'] == temp1
        assert model['teacup_temperature'] == temp1
        assert model['_integ_teacup_temperature'] == temp1

        error_message = "Trying to get the current value of a lookup"
        with pytest.raises(ValueError, match=error_message):
            model2['lookup 1d']

        assert model3['data backward'].equals(data0)
        model3.run()
        assert model3['data backward'].equals(data1)

    def test_get_series_data(self, _root):
        model = pysd.read_vensim(_root / test_model)
        model2 = pysd.read_vensim(_root / test_model_look)
        model3 = pysd.read_vensim(_root / test_model_data)

        error_message = "Trying to get the values of a constant variable."
        with pytest.raises(ValueError, match=error_message):
            model.get_series_data('Room Temperature')

        error_message = "Trying to get the values of a constant variable."
        with pytest.raises(ValueError, match=error_message):
            model.get_series_data('Teacup Temperature')

        lookup_exp = xr.DataArray(
            [0, -2, 10, 1, -5, 0, 5],
            {"lookup_dim": [0, 5, 10, 15, 20, 25, 30]},
            ["lookup_dim"])

        lookup_exp2 = xr.DataArray(
            [[0, 4], [-2, 5], [10, 5], [1, 5], [-5, 5], [0, 5], [5, 2]],
            {"lookup_dim": [0, 5, 10, 15, 20, 25, 30],
             "Rows": ["Row1", "Row2"]},
            ["lookup_dim", "Rows"])

        data_exp = xr.DataArray(
            [[[0, 4], [-1, 0]], [[-2, 5], [-3, 0]], [[10, 5], [-5, 1]],
             [[1, 5], [10, 2]], [[-5, 5], [4, 1]], [[0, 5], [5, 0]],
             [[5, 2], [5, 0]]],
            {"time": [0, 5, 10, 15, 20, 25, 30],
             "Rows": ["Row1", "Row2"], "Dim": ["A", "B"]},
            ["time", "Dim", "Rows"])

        # lookup
        lookup = model2.get_series_data('lookup 1d')
        assert lookup.equals(lookup_exp)

        lookup = model2.get_series_data('lookup_1d')
        assert lookup.equals(lookup_exp)

        lookup = model2.get_series_data('_ext_lookup_lookup_1d')
        assert lookup.equals(lookup_exp)

        lookup = model2.get_series_data('lookup 2d')
        assert lookup.equals(lookup_exp2)

        lookup = model2.get_series_data('lookup_2d')
        assert lookup.equals(lookup_exp2)

        lookup = model2.get_series_data('_ext_lookup_lookup_2d')
        assert lookup.equals(lookup_exp2)

        # data
        data = model3.get_series_data('data backward')
        assert data.equals(data_exp)

        data = model3.get_series_data('data_backward')
        assert data.equals(data_exp)

        data = model3.get_series_data('_ext_data_data_backward')
        assert data.equals(data_exp)

    @pytest.mark.parametrize("model_path", [test_model])
    def test__integrate(self, tmp_path, model):
        from pysd.py_backend.model import ModelOutput
        # TODO: think through a stronger test here...
        model.time.add_return_timestamps(list(range(0, 5, 2)))
        capture_elements = {'teacup_temperature'}

        out = ModelOutput(model, capture_elements, None)
        model._integrate(out)
        res = out.handler.ds
        assert isinstance(res, pd.DataFrame)
        assert 'teacup_temperature' in res
        assert all(res.index.values == list(range(0, 5, 2)))

        model.reload()
        model.time.add_return_timestamps(list(range(0, 5, 2)))
        out = ModelOutput(model,
                          capture_elements,
                          tmp_path.joinpath("output.nc"))
        model._integrate(out)
        res = out.handler.ds
        assert isinstance(res, nc.Dataset)
        assert 'teacup_temperature' in res.variables
        assert np.array_equal(res["time"][:].data, np.arange(0, 5, 2))
        res.close()

    def test_default_returns_with_construction_functions(self, _root):
        """
        If the run function is called with no arguments, should still be able
        to get default return functions.

        """

        model = pysd.read_vensim(
            _root.joinpath("test-models/tests/delays/test_delays.mdl"))
        ret = model.run()

        assert {
            "Initial Value",
            "Input",
            "Order Variable",
            "Output Delay1",
            "Output Delay1I",
            "Output Delay3",
        } <= set(ret.columns.values)

    @pytest.mark.parametrize(
        "model_path",
        [Path("test-models/tests/lookups/test_lookups.mdl")])
    def test_default_returns_with_lookups(self, model):
        """
        Addresses https://github.com/JamesPHoughton/pysd/issues/114
        The default settings should skip model elements with no particular
        return value
        """
        ret = model.run()
        assert {"accumulation", "rate", "lookup function call"}\
            <= set(ret.columns.values)

    @pytest.mark.parametrize("model_path", [test_model])
    def test_files(self, model, model_path, tmp_path):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/86"""

        # Path from where the model is translated
        path = tmp_path / model_path.parent.name / model_path.name

        # Check py_model_file
        assert model.py_model_file == str(path.with_suffix(".py"))
        # Check mdl_file
        assert model.mdl_file == str(path)


class TestModelInteraction():
    """ The tests in this class test pysd's interaction with itself
        and other modules. """

    def test_multiple_load(self, _root):
        """
        Test that we can load and run multiple models at the same time,
        and that the models don't interact with each other. This can
        happen if we arent careful about class attributes vs instance
        attributes

        This test responds to issue:
        https://github.com/JamesPHoughton/pysd/issues/23

        """

        model_1 = pysd.read_vensim(
            _root.joinpath("test-models/samples/teacup/teacup.mdl"))
        model_2 = pysd.read_vensim(
            _root.joinpath("test-models/samples/SIR/SIR.mdl"))

        assert "teacup_temperature" not in dir(model_2.components)
        assert "susceptible" in dir(model_2.components)

        assert "susceptible" not in dir(model_1.components)
        assert "teacup_temperature" in dir(model_1.components)

    def test_no_crosstalk(self, _root):
        """
        Need to check that if we instantiate two copies of the same model,
        changes to one copy do not influence the other copy.

        Checks for issue: https://github.com/JamesPHoughton/pysd/issues/108
        that time is not shared between the two models

        """
        # Todo: this test could be made more comprehensive

        model_1 = pysd.read_vensim(
            _root.joinpath("test-models/samples/teacup/teacup.mdl"))
        model_2 = pysd.read_vensim(
            _root.joinpath("test-models/samples/SIR/SIR.mdl"))

        model_1.components.initial_time = lambda: 10
        assert model_2.components.initial_time != 10

        # check that the model time is not shared between the two objects
        model_1.run()
        assert model_1.time() != model_2.time()

    @pytest.mark.parametrize("model_path", [test_model])
    def test_restart_cache(self, model):
        """
        Test that when we cache a model variable at the 'run' time,
        if the variable is changed and the model re-run, the cache updates
        to the new variable, instead of maintaining the old one.

        """
        model.run()
        old = model.components.room_temperature()
        model.set_components({"Room Temperature": 345})
        new = model.components.room_temperature()
        model.run()
        assert new == 345
        assert old != new

    def test_not_able_to_update_stateful_object(self):
        integ = pysd.statefuls.Integ(
            lambda: xr.DataArray([1, 2], {"Dim": ["A", "B"]}, ["Dim"]),
            lambda: xr.DataArray(0, {"Dim": ["A", "B"]}, ["Dim"]),
            "my_integ_object",
        )

        integ.initialize()

        error_message = "Could not update the value of my_integ_object"
        with pytest.raises(ValueError, match=error_message):
            integ.update(np.array([[1, 2], [3, 4]]))


class TestMultiRun():
    def test_delay_reinitializes(self, _root):
        model = pysd.read_vensim(_root.joinpath(
            "test-models/tests/delays/test_delays.mdl"))
        res1 = model.run()
        res2 = model.run()
        assert all(res1 == res2)


class TestDependencies():

    @pytest.mark.parametrize(
        "model_path,expected_dep",
        [
            (
                test_model,
                {
                    'characteristic_time': {},
                    'heat_loss_to_room': {
                        'teacup_temperature': 1,
                        'room_temperature': 1,
                        'characteristic_time': 1
                    },
                    'room_temperature': {},
                    'teacup_temperature': {'_integ_teacup_temperature': 1},
                    '_integ_teacup_temperature': {
                        'initial': {},
                        'step': {'heat_loss_to_room': 1}
                    },
                    'final_time': {},
                    'initial_time': {},
                    'saveper': {'time_step': 1},
                    'time_step': {}
                }

            ),
            (
                more_tests.joinpath(
                    "subscript_individually_defined_stocks2/"
                    "test_subscript_individually_defined_stocks2.mdl"),
                {
                    "stock_a": {"_integ_stock_a": 1, "_integ_stock_a_1": 1},
                    "inflow_a": {"rate_a": 1},
                    "inflow_b": {"rate_a": 1},
                    "initial_values": {
                        "initial_values_a": 1,
                        "initial_values_b": 1
                    },
                    "initial_values_a": {},
                    "initial_values_b": {},
                    "rate_a": {},
                    "final_time": {},
                    "initial_time": {},
                    "saveper": {"time_step": 1},
                    "time_step": {},
                    "_integ_stock_a": {
                        "initial": {"initial_values": 1},
                        "step": {"inflow_a": 1}
                    },
                    '_integ_stock_a_1': {
                        'initial': {'initial_values': 1},
                        'step': {'inflow_b': 1}
                    }
                }
            ),
            (
                test_model_constant_pipe,
                {
                    "constant1": {},
                    "constant2": {"constant1": 1},
                    "constant3": {"constant1": 3, "constant2": 1},
                    "final_time": {},
                    "initial_time": {},
                    "time_step": {},
                    "saveper": {"time_step": 1}
                }
            )
        ],
        ids=["teacup", "multiple", "constant"])
    def test_deps(self, model, expected_dep, model_path):
        assert model.dependencies == expected_dep

        if model_path == test_model_constant_pipe:
            for key, value in model.cache_type.items():
                if key != "time":
                    assert value == "run"

    @pytest.mark.parametrize("model_path", [test_model_constant_pipe])
    def test_change_constant_pipe(self, model):
        new_var = pd.Series(
            index=[0, 1, 2, 3, 4, 5],
            data=[1, 2, 3, 4, 5, 6])

        pipe = ["constant1", "constant2", "constant3"]

        out1 = model.run()

        for key in pipe:
            assert model.cache_type[key] == "run"

        # we should ensure that the constant_cache is removed
        # when passing new param
        out2 = model.run(params={"constant1": new_var})

        assert not np.all(out1 - out2 == 0)

        for key in pipe:
            assert model.cache_type[key] == "step"
            assert not (out1[key] == out2[key]).all()
            assert (np.diff(out2[key]) != 0).all()

        assert (out2["constant2"] == 4*new_var.values).all()
        assert\
            (out2["constant3"] == (5*new_var.values-1)*new_var.values).all()


class TestExportImport():

    @pytest.mark.parametrize(
        "model_path,return_ts,final_t",
        [
            (
                test_model,
                ([0, 10, 20, 30], [0, 10], [20, 30]),
                (None, 12, None)
            ),
            (
                Path('test-models/tests/delays/test_delays.mdl'),
                ([10, 20], [], [10, 20]),
                (None, 7, 34)
            ),
            (
                Path('test-models/tests/delay_fixed/test_delay_fixed.mdl'),
                ([7, 20], [7], [20]),
                (None, None, None)
            ),
            (
                Path('test-models/tests/forecast/test_forecast.mdl'),
                ([20, 30, 50], [20, 30], [50]),
                (55, 32, 52)
            ),
            (
                Path(
                    'test-models/tests/sample_if_true/test_sample_if_true.mdl'
                    ),
                ([8, 20], [8], [20]),
                (None, 15, None)
            ),
            (
                Path('test-models/tests/subscripted_smooth/'
                     'test_subscripted_smooth.mdl'),
                ([8, 20], [8], [20]),
                (None, 15, None)
            ),
            (
                Path('test-models/tests/subscripted_trend/'
                     'test_subscripted_trend.mdl'),
                ([8, 20], [8], [20]),
                (None, 15, None)

            ),
            (
                Path('test-models/tests/initial_function/test_initial.mdl'),
                ([8, 20], [8], [20]),
                (None, 15, None)
            )
        ],
        ids=["integ", "delays", "delay_fixed", "forecast", "sample_if_true",
             "smooth", "trend", "initial"]
    )
    @pytest.mark.filterwarnings("ignore")
    def test_run_export_import(self, tmp_path, model, return_ts, final_t):
        export_path = tmp_path / "export.pic"

        # Final times of each run
        finals = [final_t[i] or return_ts[i][-1] for i in range(3)]

        # Whole run
        stocks = model.run(
            return_timestamps=return_ts[0], final_time=final_t[0]
        )
        assert (stocks['INITIAL TIME'] == 0).all().all()
        assert (stocks['FINAL TIME'] == finals[0]).all().all()

        # Export run
        model.reload()
        stocks1 = model.run(
            return_timestamps=return_ts[1], final_time=final_t[1]
        )
        assert (stocks1['INITIAL TIME'] == 0).all().all()
        assert (stocks1['FINAL TIME'] == finals[1]).all().all()
        model.export(export_path)

        # Import run
        model.reload()
        stocks2 = model.run(
            initial_condition=export_path,
            return_timestamps=return_ts[2], final_time=final_t[2]
        )
        assert (stocks2['INITIAL TIME'] == finals[1]).all().all()
        assert (stocks2['FINAL TIME'] == finals[2]).all().all()

        # Compare results
        stocks.drop(columns=["INITIAL TIME", "FINAL TIME"], inplace=True)
        stocks1.drop(columns=["INITIAL TIME", "FINAL TIME"], inplace=True)
        stocks2.drop(columns=["INITIAL TIME", "FINAL TIME"], inplace=True)
        if return_ts[1]:
            assert_frames_close(stocks1, stocks.loc[return_ts[1]])
        if return_ts[2]:
            assert_frames_close(stocks2, stocks.loc[return_ts[2]])
