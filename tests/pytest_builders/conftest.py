import pytest


_CONTROL = """
********************************************************
\t.Control
********************************************************~
\t\tSimulation Control Parameters
\t|

FINAL TIME  = 10
\t~\tYear
\t~\tThe final time for the simulation.
\t|

INITIAL TIME  = 0
\t~\tYear
\t~\tThe initial time for the simulation.
\t|

SAVEPER  = 1
\t~\tYear [0,?]
\t~\tThe frequency with which output is stored.
\t|

TIME STEP  = {time_step}
\t~\tYear [0,?]
\t~\tThe time step for the simulation.
\t|
"""

#: Small models for Julia builder checks not covered by test-models.
_INLINE_MODELS = {
    # DELAY FIXED with a constant delay time: exact pipeline, 2 / 0.0625 = 32 stages
    "delay_fixed_constant": ("""{UTF-8}
Input=
\t5
\t~
\t~\tConstant input to delay.\t|

Delay Time=
\t2
\t~
\t~\tDelay time.\t|

Initial Value=
\t0
\t~
\t~\tInitial value.\t|

Output=
\tDELAY FIXED(Input, Delay Time, Initial Value)
\t~
\t~\tOutput of delay fixed.\t|
""", 0.0625),
    # Data equation (:=) without GET DATA, unsupported by the Julia builder
    "data_equation": ("""{UTF-8}
Input=
\t5
\t~
\t~\tA simple constant input.\t|

Extern Data := Input
\t~
\t~\tA DATA variable defined by an equation.\t|

Output=
\tInput + 1
\t~
\t~\tA simple output.\t|
""", 1),
}

#: Julia builder checks that use a test-models model as is.
_TEST_MODELS = {
    "trend": "trend/test_trend.mdl",
    "forecast": "forecast/test_forecast.mdl",
    "sample_if_true": "sample_if_true/test_sample_if_true.mdl",
    "allocate_available": "allocate_available/test_allocate_available.mdl",
}


@pytest.fixture
def julia_model(tmp_path, _test_models):
    """Return the path of a model for the Julia builder tests.

    Models from test-models are returned in place (callers copy them before
    translating); the small inline models are written to a temporary folder.
    """
    def get(name):
        if name in _TEST_MODELS:
            return _test_models / _TEST_MODELS[name]
        text, time_step = _INLINE_MODELS[name]
        path = tmp_path / "inline_models" / f"test_{name}.mdl"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text + _CONTROL.format(time_step=time_step), encoding="utf-8")
        return path
    return get
