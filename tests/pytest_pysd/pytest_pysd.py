from pathlib import Path
from warnings import simplefilter, catch_warnings

import pytest
import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc

from pysd.tools.benchmarking import assert_frames_close
from pysd.py_backend.output import OutputHandlerInterface, DatasetHandler, \
    DataFrameHandler, ModelOutput

import pysd

# TODO replace test paths by fixtures and translate and run the models
# in temporal directories

_root = Path(__file__).parent.parent

test_model = _root.joinpath("test-models/samples/teacup/teacup.mdl")
test_model_subs = _root.joinpath(
    "test-models/tests/subscript_2d_arrays/test_subscript_2d_arrays.mdl")
test_model_look = _root.joinpath(
    "test-models/tests/get_lookups_subscripted_args/"
    + "test_get_lookups_subscripted_args.mdl")
test_model_data = _root.joinpath(
    "test-models/tests/get_data_args_3d_xls/test_get_data_args_3d_xls.mdl")
test_model_constants = _root.joinpath(
    "test-models/tests/get_constants_subranges/"
    "test_get_constants_subranges.mdl"
)
more_tests = _root.joinpath("more-tests/")

test_model_constant_pipe = more_tests.joinpath(
    "constant_pipeline/test_constant_pipeline.mdl")
test_model_numeric_coords = more_tests.joinpath(
    "numeric_subscripts/test_subscript_1d_arrays.mdl"
)

class TestPySD():

    def test_run(self):
        model = pysd.read_vensim(test_model)
        stocks = model.run()
        # return a dataframe
        assert isinstance(stocks, pd.DataFrame)
        # contains correct column
        assert "Teacup Temperature" in stocks.columns.values
        # has multiple rows
        assert len(stocks) > 3
        # there are no null values in the set
        assert stocks.notnull().all().all()

    def test_run_ignore_missing(self):
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

    def test_run_includes_last_value(self):
        model = pysd.read_vensim(test_model)
        res = model.run()
        assert res.index[-1] == model.components.final_time()

    def test_run_build_timeseries(self):
        model = pysd.read_vensim(test_model)
        res = model.run(final_time=7, time_step=2, initial_condition=(3, {}))

        actual = list(res.index)
        expected = [3.0, 5.0, 7.0]
        assert actual == expected

    def test_run_progress(self):
        # same as test_run but with progressbar
        model = pysd.read_vensim(test_model)
        stocks = model.run(progress=True)
        assert isinstance(stocks, pd.DataFrame)
        assert "Teacup Temperature" in stocks.columns.values
        assert len(stocks) > 3
        assert stocks.notnull().all().all()

    def test_run_return_timestamps(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/17"""

        model = pysd.read_vensim(test_model)
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

    def test_run_return_timestamps_past_final_time(self):
        """ If the user enters a timestamp that is longer than the euler
        timeseries that is defined by the normal model file, should
        extend the euler series to the largest timestamp"""

        model = pysd.read_vensim(test_model)
        return_timestamps = list(range(0, 100, 10))
        stocks = model.run(return_timestamps=return_timestamps)

        assert return_timestamps == list(stocks.index)

    def test_return_timestamps_with_range(self):
        """
        Tests that return timestamps may receive a 'range'.
        It will be cast to a numpy array in the end...
        """

        model = pysd.read_vensim(test_model)
        return_timestamps = range(0, 31, 10)
        stocks = model.run(return_timestamps=return_timestamps)
        assert list(return_timestamps) == list(stocks.index)

    def test_run_return_columns_original_names(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/26
        - Also checks that columns are returned in the correct order"""

        model = pysd.read_vensim(test_model)
        return_columns = ["Room Temperature", "Teacup Temperature"]
        result = model.run(return_columns=return_columns)
        assert set(result.columns) == set(return_columns)

    def test_run_return_columns_step(self):
        """
        Return only cache 'step' variables
        """
        model = pysd.read_vensim(test_model)
        result = model.run(return_columns='step')
        assert set(result.columns)\
            == {'Teacup Temperature', 'Heat Loss to Room'}

    def test_run_reload(self):
        """ Addresses https://github.com/JamesPHoughton/pysd/issues/99"""

        model = pysd.read_vensim(test_model)

        result0 = model.run()
        result1 = model.run(params={"Room Temperature": 1000})
        result2 = model.run()
        result3 = model.run(reload=True)

        assert (result0 == result3).all().all()
        assert not (result0 == result1).all().all()
        assert (result1 == result2).all().all()

    def test_run_return_columns_pysafe_names(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/26"""

        model = pysd.read_vensim(test_model)
        return_columns = ["room_temperature", "teacup_temperature"]
        result = model.run(return_columns=return_columns)
        assert set(result.columns) == set(return_columns)

    def test_run_output_file(self):

        model = pysd.read_vensim(test_model)
        model.progress = False

        error_message = "Paths must be strings or pathlib Path objects."
        with pytest.raises(TypeError, match=error_message):
            model.run(output_file=1234)

        error_message = "Unsupported output file format .txt"
        with pytest.raises(ValueError, match=error_message):
            model.run(output_file="file.txt")


    def test_initial_conditions_invalid(self):
        model = pysd.read_vensim(test_model)
        error_message = r"Invalid initial conditions\. "\
            r"Check documentation for valid entries or use "\
            r"'help\(model\.set_initial_condition\)'\."
        with pytest.raises(TypeError, match=error_message):
            model.run(initial_condition=["this is not valid"])

    def test_initial_conditions_tuple_pysafe_names(self):
        model = pysd.read_vensim(test_model)
        stocks = model.run(
            initial_condition=(3000, {"teacup_temperature": 33}),
            return_timestamps=list(range(3000, 3010))
        )

        assert stocks["Teacup Temperature"].iloc[0] == 33

    def test_initial_conditions_tuple_original_names(self):
        """ Responds to https://github.com/JamesPHoughton/pysd/issues/77"""

        model = pysd.read_vensim(test_model)
        stocks = model.run(
            initial_condition=(3000, {"Teacup Temperature": 33}),
            return_timestamps=list(range(3000, 3010)),
        )
        assert stocks.index[0] == 3000
        assert stocks["Teacup Temperature"].iloc[0] == 33

    def test_initial_conditions_current(self):
        model = pysd.read_vensim(test_model)
        stocks1 = model.run(return_timestamps=list(range(0, 31)))
        stocks2 = model.run(
            initial_condition="current", return_timestamps=list(range(30, 45))
        )
        assert stocks1["Teacup Temperature"].iloc[-1]\
            == stocks2["Teacup Temperature"].iloc[0]

    def test_initial_condition_bad_value(self):
        model = pysd.read_vensim(test_model)

        with pytest.raises(FileNotFoundError):
            model.run(initial_condition="bad value")

    def test_initial_conditions_subscripted_value_with_numpy_error(self):
        input_ = np.array([[5, 3], [4, 8], [9, 3]])

        model = pysd.read_vensim(test_model_subs)

        with pytest.raises(TypeError):
            model.run(initial_condition=(5, {'stock_a': input_}),
                      return_columns=['stock_a'],
                      return_timestamps=list(range(5, 10)))

    def test_set_constant_parameter(self):
        """ In response to:
        re: https://github.com/JamesPHoughton/pysd/issues/5"""

        model = pysd.read_vensim(test_model)
        model.set_components({"room_temperature": 20})
        assert model.components.room_temperature() == 20

        model.run(params={"room_temperature": 70})
        assert model.components.room_temperature() == 70

        with pytest.raises(NameError):
            model.set_components({'not_a_var': 20})

    def test_set_constant_parameter_inline(self):
        model = pysd.read_vensim(test_model)
        model.components.room_temperature = 20
        assert model.components.room_temperature() == 20

        model.run(params={"room_temperature": 70})
        assert model.components.room_temperature() == 70

        with pytest.raises(NameError):
            model.components.not_a_var = 20

    def test_set_timeseries_parameter(self):
        model = pysd.read_vensim(test_model)
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

    def test_set_timeseries_parameter_inline(self):
        model = pysd.read_vensim(test_model)
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

    def test_set_component_with_real_name(self):
        model = pysd.read_vensim(test_model)
        model.set_components({"Room Temperature": 20})
        assert model.components.room_temperature() == 20

        model.run(params={"Room Temperature": 70})
        assert model.components.room_temperature() == 70

    def test_set_components_warnings(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/80"""

        model = pysd.read_vensim(test_model)
        with catch_warnings(record=True) as w:
            simplefilter("always")
            model.set_components(
                {"Teacup Temperature": 20, "Characteristic Time": 15}
            )  # set stock value using params

        # check that warning references the stock
        assert "Teacup Temperature" in str(w[0].message)

    def test_set_components_with_function(self):
        def test_func():
            return 5

        model = pysd.read_vensim(test_model)
        model.set_components({"Room Temperature": test_func})
        res = model.run(return_columns=["Room Temperature"])
        assert test_func() == res["Room Temperature"].iloc[0]

    def test_set_subscripted_value_with_constant(self):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output = xr.DataArray([[5, 5], [5, 5], [5, 5]], coords, dims)

        model = pysd.read_vensim(test_model_subs)
        model.set_components({"initial_values": 5, "final_time": 10})
        res = model.run(
            return_columns=["Initial Values"], flatten_output=False)
        assert output.equals(res["Initial Values"].iloc[0])

    def test_set_subscripted_value_with_partial_xarray(self):
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

        model = pysd.read_vensim(test_model_subs)
        model.set_components({"Initial Values": input_val, "final_time": 10})
        res = model.run(
            return_columns=["Initial Values"], flatten_output=False)
        assert output.equals(res["Initial Values"].iloc[0])

    def test_set_subscripted_value_with_xarray(self):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output = xr.DataArray([[5, 3], [4, 8], [9, 3]], coords, dims)

        model = pysd.read_vensim(test_model_subs)
        model.set_components({"initial_values": output, "final_time": 10})
        res = model.run(
            return_columns=["Initial Values"], flatten_output=False)
        assert output.equals(res["Initial Values"].iloc[0])

    def test_set_parameter_data(self):
        model = pysd.read_vensim(test_model_data)
        timeseries = list(range(31))
        series = pd.Series(
            index=timeseries,
            data=(50+np.random.rand(len(timeseries)).cumsum())
        )

        with catch_warnings():
            # avoid warnings related to extrapolation
            simplefilter("ignore")
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

    def test_set_constant_parameter_lookup(self):
        model = pysd.read_vensim(test_model_look)

        with catch_warnings():
            # avoid warnings related to extrapolation
            simplefilter("ignore")
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

            xr2 = xr.DataArray([-100, 500], {"Rows": ["Row1", "Row2"]},
                               ["Rows"])
            model.run(params={"lookup_2d": xr2}, final_time=1)
            for i in range(100):
                assert model.components.lookup_2d(i).equals(xr2)

    def test_set_timeseries_parameter_lookup(self):
        model = pysd.read_vensim(test_model_look)
        timeseries = list(range(30))

        with catch_warnings():
            # avoid warnings related to extrapolation
            simplefilter("ignore")
            temp_timeseries = pd.Series(
                index=timeseries, data=(50 +
                                        np.random.rand(len(timeseries)
                                                       ).cumsum())
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
                    a.equals(xr.DataArray(b, {"Rows": ["Row1", "Row2"]},
                                          ["Rows"]))
                    for a, b in zip(res["lookup_2d_time"].values,
                                    temp_timeseries)
                ]
            )

            temp_timeseries2 = pd.Series(
                index=timeseries,
                data=[
                    xr.DataArray([50 + x, 20 - y], {"Rows": ["Row1", "Row2"]},
                                 ["Rows"])
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

    def test_set_subscripted_value_with_numpy_error(self):
        input_ = np.array([[5, 3], [4, 8], [9, 3]])

        model = pysd.read_vensim(test_model_subs)
        with pytest.raises(TypeError):
            model.set_components({"initial_values": input_, "final_time": 10})

    def test_set_subscripted_timeseries_parameter_with_constant(self):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]

        model = pysd.read_vensim(test_model_subs)
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

    def test_set_subscripted_timeseries_parameter_with_partial_xarray(self):
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

        model = pysd.read_vensim(test_model_subs)
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

    def test_set_subscripted_timeseries_parameter_with_xarray(self):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]

        init_val = xr.DataArray([[5, 3], [4, 8], [9, 3]], coords, dims)

        model = pysd.read_vensim(test_model_subs)
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

    def test_docs(self):
        """ Test that the model prints some documentation """

        model = pysd.read_vensim(test_model)
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

    def test_initialize(self):
        model = pysd.read_vensim(test_model)
        initial_temp = model.components.teacup_temperature()
        model.run()
        final_temp = model.components.teacup_temperature()
        model.initialize()
        reset_temp = model.components.teacup_temperature()
        assert initial_temp != final_temp
        assert initial_temp == reset_temp

    def test_initialize_order(self):
        model = pysd.load(more_tests.joinpath(
            "initialization_order/test_initialization_order.py"))

        assert model.initialize_order == ["_integ_stock_a", "_integ_stock_b"]
        assert model.components.stock_b() == 42
        assert model.components.stock_a() == 42
        model.components.initial_par = lambda: 1
        model.initialize()
        assert model.components.stock_b() == 1
        assert model.components.stock_a() == 1

    def test_set_initial_with_deps(self):
        model = pysd.load(more_tests.joinpath("initialization_order/"
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

    def test_set_initial_value(self):
        model = pysd.read_vensim(test_model)

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

    def test_set_initial_value_subscripted_value_with_constant(self):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output_b = xr.DataArray([[0, 0], [0, 0], [0, 0]], coords, dims)

        new_time = np.random.rand()

        model = pysd.read_vensim(test_model_subs)
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

    def test_set_initial_value_subscripted_value_with_partial_xarray(self):
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

        model = pysd.read_vensim(test_model_subs)
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

    def test_set_initial_value_subscripted_value_with_xarray(self):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]
        output1 = xr.DataArray([[5, 3], [4, 8], [9, 3]], coords, dims)
        output2 = xr.DataArray([[53, 43], [84, 80], [29, 63]], coords, dims)
        output3 = xr.DataArray([[54, 32], [40, 87], [93, 93]], coords, dims)

        new_time = np.random.rand()

        model = pysd.read_vensim(test_model_subs)
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

    def test_set_initial_value_subscripted_value_with_numpy_error(self):
        input1 = np.array([[5, 3], [4, 8], [9, 3]])
        input2 = np.array([[53, 43], [84, 80], [29, 63]])
        input3 = np.array([[54, 32], [40, 87], [93, 93]])

        new_time = np.random.rand()

        model = pysd.read_vensim(test_model_subs)

        # Test that we can set with real names
        with pytest.raises(TypeError):
            model.set_initial_value(new_time, {'Stock A': input1})

        # Test setting with pysafe names
        with pytest.raises(TypeError):
            model.set_initial_value(new_time + 1, {'stock_a': input2})

        # Test setting with stateful object name
        with pytest.raises(TypeError):
            model.set_initial_value(new_time + 2, {'_integ_stock_a': input3})

    def test_replace_element(self):
        model = pysd.read_vensim(test_model)
        stocks1 = model.run()
        model.components.characteristic_time = lambda: 3
        stocks2 = model.run()
        assert stocks1["Teacup Temperature"].loc[10]\
            > stocks2["Teacup Temperature"].loc[10]

    def test_set_initial_condition_origin_full(self):
        model = pysd.read_vensim(test_model)
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

    def test_set_initial_condition_origin_short(self):
        model = pysd.read_vensim(test_model)
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

    def test_set_initial_condition_for_stock_component(self):
        model = pysd.read_vensim(test_model)
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

    def test_set_initial_condition_for_constant_component(self):
        model = pysd.read_vensim(test_model)

        new_state = {"Room Temperature": 100}
        new_time = 10

        error_message = r"Unrecognized stateful '.*'\. If you want"\
            r" to set a value of a regular component\. Use params"
        with pytest.raises(ValueError, match=error_message):
            model.set_initial_condition((new_time, new_state))

    def test_get_args(self):
        model = pysd.read_vensim(test_model)
        model2 = pysd.read_vensim(test_model_look)

        assert model.get_args('Room Temperature') == []
        assert model.get_args('room_temperature') == []
        assert model.get_args('teacup_temperature') == []
        assert model.get_args('_integ_teacup_temperature') == []

        assert model2.get_args('lookup 1d') == ['x', 'final_subs']
        assert model2.get_args('lookup_1d') == ['x', 'final_subs']
        assert model2.get_args('lookup 2d') == ['x', 'final_subs']
        assert model2.get_args('lookup_2d') == ['x', 'final_subs']

        with pytest.raises(NameError):
            model.get_args('not_a_var')

    def test_get_coords(self):
        coords = {
            "One Dimensional Subscript": ["Entry 1", "Entry 2", "Entry 3"],
            "Second Dimension Subscript": ["Column 1", "Column 2"],
        }
        dims = ["One Dimensional Subscript", "Second Dimension Subscript"]

        coords_dims = (coords, dims)

        model = pysd.read_vensim(test_model)
        model2 = pysd.read_vensim(test_model_subs)

        assert model.get_coords("Room Temperature") is None
        assert model.get_coords("room_temperature") is None
        assert model.get_coords("teacup_temperature") is None
        assert model.get_coords("_integ_teacup_temperature") is None

        assert model2.get_coords("Initial Values") == coords_dims
        assert model2.get_coords("initial_values") == coords_dims
        assert model2.get_coords("Stock A") == coords_dims
        assert model2.get_coords("stock_a") == coords_dims
        assert model2.get_coords("_integ_stock_a") == coords_dims

        with pytest.raises(NameError):
            model.get_coords('not_a_var')

    def test_getitem(self):
        model = pysd.read_vensim(test_model)
        model2 = pysd.read_vensim(test_model_look)
        model3 = pysd.read_vensim(test_model_data)

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

    def test_get_series_data(self):
        model = pysd.read_vensim(test_model)
        model2 = pysd.read_vensim(test_model_look)
        model3 = pysd.read_vensim(test_model_data)

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

    def test__integrate(self, shared_tmpdir):
        from pysd.py_backend.model import ModelOutput
        # TODO: think through a stronger test here...
        model = pysd.read_vensim(test_model)
        model.progress = False
        model.time.add_return_timestamps(list(range(0, 5, 2)))
        capture_elements = {'teacup_temperature'}

        out = ModelOutput(model, capture_elements, None)
        model._integrate(out)
        res = out.handler.ds
        assert isinstance(res, pd.DataFrame)
        assert 'teacup_temperature' in res
        assert all(res.index.values == list(range(0, 5, 2)))

        model = pysd.read_vensim(test_model)
        model.progress = False
        model.time.add_return_timestamps(list(range(0, 5, 2)))
        out = ModelOutput(model,
                          capture_elements,
                          shared_tmpdir.joinpath("output.nc"))
        model._integrate(out)
        res = out.handler.ds
        assert isinstance(res, nc.Dataset)
        assert 'teacup_temperature' in res.variables
        assert np.array_equal(res["time"][:].data, np.arange(0, 5, 2))
        res.close()

    def test_default_returns_with_construction_functions(self):
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

    def test_default_returns_with_lookups(self):
        """
        Addresses https://github.com/JamesPHoughton/pysd/issues/114
        The default settings should skip model elements with no particular
        return value
        """

        model = pysd.read_vensim(
            _root.joinpath("test-models/tests/lookups/test_lookups.mdl"))
        ret = model.run()
        assert {"accumulation", "rate", "lookup function call"}\
            <= set(ret.columns.values)

    def test_py_model_file(self):
        """Addresses https://github.com/JamesPHoughton/pysd/issues/86"""

        model = pysd.read_vensim(test_model)
        assert model.py_model_file == str(test_model.with_suffix(".py"))

    def test_mdl_file(self):
        """Relates to https://github.com/JamesPHoughton/pysd/issues/86"""

        model = pysd.read_vensim(test_model)
        assert model.mdl_file == str(test_model)


class TestModelInteraction():
    """ The tests in this class test pysd's interaction with itself
        and other modules. """

    def test_multiple_load(self):
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

    def test_no_crosstalk(self):
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

    def test_restart_cache(self):
        """
        Test that when we cache a model variable at the 'run' time,
         if the variable is changed and the model re-run, the cache updates
         to the new variable, instead of maintaining the old one.
        """

        model = pysd.read_vensim(test_model)
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
    def test_delay_reinitializes(self):
        model = pysd.read_vensim(_root.joinpath(
            "test-models/tests/delays/test_delays.mdl"))
        res1 = model.run()
        res2 = model.run()
        assert all(res1 == res2)


class TestDependencies():
    def test_teacup_deps(self):
        from pysd import read_vensim

        model = read_vensim(test_model)

        expected_dep = {
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
        assert model.dependencies == expected_dep

    def test_multiple_deps(self):
        from pysd import read_vensim

        model = read_vensim(
            more_tests.joinpath(
                "subscript_individually_defined_stocks2/"
                + "test_subscript_individually_defined_stocks2.mdl"))

        expected_dep = {
            "stock_a": {"_integ_stock_a": 1, "_integ_stock_a_1": 1},
            "inflow_a": {"rate_a": 1},
            "inflow_b": {"rate_a": 1},
            "initial_values": {"initial_values_a": 1, "initial_values_b": 1},
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
        assert model.dependencies == expected_dep

        more_tests.joinpath(
            "subscript_individually_defined_stocks2/"
            + "test_subscript_individually_defined_stocks2.py").unlink()

    def test_constant_deps(self):
        from pysd import read_vensim

        model = read_vensim(test_model_constant_pipe)

        expected_dep = {
            "constant1": {},
            "constant2": {"constant1": 1},
            "constant3": {"constant1": 3, "constant2": 1},
            "final_time": {},
            "initial_time": {},
            "time_step": {},
            "saveper": {"time_step": 1}
        }
        assert model.dependencies == expected_dep

        for key, value in model.cache_type.items():
            if key != "time":
                assert value == "run"

        test_model_constant_pipe.with_suffix(".py").unlink()

    def test_change_constant_pipe(self):
        from pysd import read_vensim

        model = read_vensim(test_model_constant_pipe)

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

        test_model_constant_pipe.with_suffix(".py").unlink()


class TestExportImport():
    def test_run_export_import_integ(self):
        from pysd import read_vensim

        with catch_warnings():
            simplefilter("ignore")
            model = read_vensim(test_model)
            stocks = model.run(return_timestamps=[0, 10, 20, 30])
            assert (stocks['INITIAL TIME'] == 0).all().all()
            assert (stocks['FINAL TIME'] == 30).all().all()

            model.reload()
            stocks1 = model.run(return_timestamps=[0, 10], final_time=12)
            assert (stocks1['INITIAL TIME'] == 0).all().all()
            assert (stocks1['FINAL TIME'] == 12).all().all()
            model.export('teacup12.pic')
            model.reload()
            stocks2 = model.run(initial_condition='teacup12.pic',
                                return_timestamps=[20, 30])
            assert (stocks2['INITIAL TIME'] == 12).all().all()
            assert (stocks2['FINAL TIME'] == 30).all().all()
            stocks.drop('INITIAL TIME', axis=1, inplace=True)
            stocks1.drop('INITIAL TIME', axis=1, inplace=True)
            stocks2.drop('INITIAL TIME', axis=1, inplace=True)
            stocks.drop('FINAL TIME', axis=1, inplace=True)
            stocks1.drop('FINAL TIME', axis=1, inplace=True)
            stocks2.drop('FINAL TIME', axis=1, inplace=True)
            Path('teacup12.pic').unlink()

            assert_frames_close(stocks1, stocks.loc[[0, 10]])
            assert_frames_close(stocks2, stocks.loc[[20, 30]])

    def test_run_export_import_delay(self):
        from pysd import read_vensim

        with catch_warnings():
            simplefilter("ignore")
            test_delays = _root.joinpath(
                'test-models/tests/delays/test_delays.mdl')
            model = read_vensim(test_delays)
            stocks = model.run(return_timestamps=20)
            model.reload()
            model.run(return_timestamps=[], final_time=7)
            model.export('delays7.pic')
            stocks2 = model.run(initial_condition='delays7.pic',
                                return_timestamps=20)
            assert (stocks['INITIAL TIME'] == 0).all().all()
            assert (stocks2['INITIAL TIME'] == 7).all().all()
            stocks.drop('INITIAL TIME', axis=1, inplace=True)
            stocks2.drop('INITIAL TIME', axis=1, inplace=True)
            stocks.drop('FINAL TIME', axis=1, inplace=True)
            stocks2.drop('FINAL TIME', axis=1, inplace=True)
            Path('delays7.pic').unlink()

            assert_frames_close(stocks2, stocks)

    def test_run_export_import_delay_fixed(self):
        from pysd import read_vensim

        with catch_warnings():
            simplefilter("ignore")
            test_delayf = _root.joinpath(
                'test-models/tests/delay_fixed/test_delay_fixed.mdl')
            model = read_vensim(test_delayf)
            stocks = model.run(return_timestamps=20)
            model.reload()
            model.run(return_timestamps=7)
            model.export('delayf7.pic')
            stocks2 = model.run(initial_condition='delayf7.pic',
                                return_timestamps=20)
            assert (stocks['INITIAL TIME'] == 0).all().all()
            assert (stocks2['INITIAL TIME'] == 7).all().all()
            stocks.drop('INITIAL TIME', axis=1, inplace=True)
            stocks2.drop('INITIAL TIME', axis=1, inplace=True)
            stocks.drop('FINAL TIME', axis=1, inplace=True)
            stocks2.drop('FINAL TIME', axis=1, inplace=True)
            Path('delayf7.pic').unlink()

            assert_frames_close(stocks2, stocks)

    def test_run_export_import_forecast(self):
        from pysd import read_vensim

        with catch_warnings():
            simplefilter("ignore")
            test_trend = _root.joinpath(
                'test-models/tests/forecast/'
                + 'test_forecast.mdl')
            model = read_vensim(test_trend)
            stocks = model.run(return_timestamps=50, flatten_output=True)
            model.reload()
            model.run(return_timestamps=20)
            model.export('frcst20.pic')
            stocks2 = model.run(initial_condition='frcst20.pic',
                                return_timestamps=50,
                                flatten_output=True)
            assert (stocks['INITIAL TIME'] == 0).all().all()
            assert (stocks2['INITIAL TIME'] == 20).all().all()
            stocks.drop('INITIAL TIME', axis=1, inplace=True)
            stocks2.drop('INITIAL TIME', axis=1, inplace=True)
            stocks.drop('FINAL TIME', axis=1, inplace=True)
            stocks2.drop('FINAL TIME', axis=1, inplace=True)
            Path('frcst20.pic').unlink()

            assert_frames_close(stocks2, stocks)

    def test_run_export_import_sample_if_true(self):
        from pysd import read_vensim

        with catch_warnings():
            simplefilter("ignore")
            test_sample_if_true = _root.joinpath(
                'test-models/tests/sample_if_true/test_sample_if_true.mdl')
            model = read_vensim(test_sample_if_true)
            stocks = model.run(return_timestamps=20, flatten_output=True)
            model.reload()
            model.run(return_timestamps=7)
            model.export('sample_if_true7.pic')
            stocks2 = model.run(initial_condition='sample_if_true7.pic',
                                return_timestamps=20,
                                flatten_output=True)
            assert (stocks['INITIAL TIME'] == 0).all().all()
            assert (stocks2['INITIAL TIME'] == 7).all().all()
            stocks.drop('INITIAL TIME', axis=1, inplace=True)
            stocks2.drop('INITIAL TIME', axis=1, inplace=True)
            stocks.drop('FINAL TIME', axis=1, inplace=True)
            stocks2.drop('FINAL TIME', axis=1, inplace=True)
            Path('sample_if_true7.pic').unlink()

            assert_frames_close(stocks2, stocks)

    def test_run_export_import_smooth(self):
        from pysd import read_vensim

        with catch_warnings():
            simplefilter("ignore")
            test_smooth = _root.joinpath(
                'test-models/tests/subscripted_smooth/'
                + 'test_subscripted_smooth.mdl')
            model = read_vensim(test_smooth)
            stocks = model.run(return_timestamps=20, flatten_output=True)
            model.reload()
            model.run(return_timestamps=7)
            model.export('smooth7.pic')
            stocks2 = model.run(initial_condition='smooth7.pic',
                                return_timestamps=20,
                                flatten_output=True)
            assert (stocks['INITIAL TIME'] == 0).all().all()
            assert (stocks2['INITIAL TIME'] == 7).all().all()
            stocks.drop('INITIAL TIME', axis=1, inplace=True)
            stocks2.drop('INITIAL TIME', axis=1, inplace=True)
            stocks.drop('FINAL TIME', axis=1, inplace=True)
            stocks2.drop('FINAL TIME', axis=1, inplace=True)
            Path('smooth7.pic').unlink()

            assert_frames_close(stocks2, stocks)

    def test_run_export_import_trend(self):
        from pysd import read_vensim

        with catch_warnings():
            simplefilter("ignore")
            test_trend = _root.joinpath(
                'test-models/tests/subscripted_trend/'
                + 'test_subscripted_trend.mdl')
            model = read_vensim(test_trend)
            stocks = model.run(return_timestamps=20, flatten_output=True)
            model.reload()
            model.run(return_timestamps=7)
            model.export('trend7.pic')
            stocks2 = model.run(initial_condition='trend7.pic',
                                return_timestamps=20,
                                flatten_output=True)
            assert (stocks['INITIAL TIME'] == 0).all().all()
            assert (stocks2['INITIAL TIME'] == 7).all().all()
            stocks.drop('INITIAL TIME', axis=1, inplace=True)
            stocks2.drop('INITIAL TIME', axis=1, inplace=True)
            stocks.drop('FINAL TIME', axis=1, inplace=True)
            stocks2.drop('FINAL TIME', axis=1, inplace=True)
            Path('trend7.pic').unlink()

            assert_frames_close(stocks2, stocks)

    def test_run_export_import_initial(self):
        from pysd import read_vensim

        with catch_warnings():
            simplefilter("ignore")
            test_initial = _root.joinpath(
                'test-models/tests/initial_function/test_initial.mdl')
            model = read_vensim(test_initial)
            stocks = model.run(return_timestamps=20)
            model.reload()
            model.run(return_timestamps=7)
            model.export('initial7.pic')
            stocks2 = model.run(initial_condition='initial7.pic',
                                return_timestamps=20)
            assert (stocks['INITIAL TIME'] == 0).all().all()
            assert (stocks2['INITIAL TIME'] == 7).all().all()
            stocks.drop('INITIAL TIME', axis=1, inplace=True)
            stocks2.drop('INITIAL TIME', axis=1, inplace=True)
            stocks.drop('FINAL TIME', axis=1, inplace=True)
            stocks2.drop('FINAL TIME', axis=1, inplace=True)
            Path('initial7.pic').unlink()

            assert_frames_close(stocks2, stocks)


class TestOutputs():

    def test_output_handler_interface(self):
        # when the class does not inherit from OutputHandlerInterface, it must
        # implement all the interface to be a subclass of
        # OutputHandlerInterface.
        # Add any additional Handler here.
        assert issubclass(DatasetHandler, OutputHandlerInterface)
        assert issubclass(DataFrameHandler, OutputHandlerInterface)
        assert issubclass(ModelOutput, OutputHandlerInterface)

        class ThatFollowsInterface:
            """
            This class does not inherit from OutputHandlerInterface, but it
            overrides all its methods (it follows the interface).
            """

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
            empty = EmptyHandler()

        # calling methods that are not overriden returns NotImplementedError
        # this should never happen, because these methods are instance methods,
        # therefore the class needs to be instantiated first
        with pytest.raises(NotImplementedError):
            EmptyHandler.initialize(EmptyHandler, "model", "capture")

        with pytest.raises(NotImplementedError):
            EmptyHandler.update(EmptyHandler, "model", "capture")

        with pytest.raises(NotImplementedError):
            EmptyHandler.postprocess(EmptyHandler)

        with pytest.raises(NotImplementedError):
            EmptyHandler.add_run_elements(
                EmptyHandler, "model", "capture")

    def test_output_with_dimensions(self, shared_tmpdir):
        model = pysd.read_vensim(test_model_look)
        model.progress = False

        out_file = shared_tmpdir.joinpath("results.nc")

        with catch_warnings(record=True) as w:
            simplefilter("always")
            model.run(output_file=out_file)

        with nc.Dataset(out_file, "r")as ds:
            assert ds.ncattrs() == [
                'description', 'model_file', 'timestep', 'initial_time',
                'final_time']
            assert list(ds.dimensions.keys()) == ["Rows", "Dim", "time"]
            # dimensions are stored as variables
            assert ds["Rows"][:].size == 2
            assert "Rows" in ds.variables.keys()
            assert "time" in ds.variables.keys()
            # scalars do not have the time dimension
            assert ds["initial_time"][:].size == 1
            # cache step variables have the "time" dimension
            assert ds["lookup_1d_time"].dimensions == ("time",)

            assert ds["d2d"].dimensions == ("time", "Rows", "Dim")
            assert ds["d2d"].description == "Missing"
            assert ds["d2d"].units == "Missing"

        # test cache run variables with dimensions
        model2 = pysd.read_vensim(test_model_constants)
        model2.progress = False

        out_file2 = shared_tmpdir.joinpath("results2.nc")

        with catch_warnings(record=True) as w:
            simplefilter("always")
            model2.run(output_file=out_file2)

        with nc.Dataset(out_file2, "r") as ds:
            assert "constant" in list(ds.variables.keys())
            assert ds["constant"].dimensions == ("dim1",)
            assert ds["dim1"][:].data.dtype == "S1"

        # dimension with numeric coords
        model3 = pysd.read_vensim(test_model_numeric_coords)
        model3.progress = False

        out_file3 = shared_tmpdir.joinpath("results3.nc")

        with catch_warnings(record=True) as w:
            simplefilter("always")
            model3.run(output_file=out_file3)

        with nc.Dataset(out_file3, "r")as ds:
            assert np.array_equal(
                ds["time"][:].data, np.arange(0.0, 101.0, 1.0))
            # coordinates get dtype=object when their length is different
            assert ds["One Dimensional Subscript"][:].dtype == "O"
            assert np.allclose(
                ds["stock_a"][-1, :].data, np.array([1.0, 2.0, 3.0]))

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
