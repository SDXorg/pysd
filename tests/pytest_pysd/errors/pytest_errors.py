import pytest

from pysd import read_vensim, read_xmile, load


@pytest.fixture
def model_path(_root, name, suffix):
    return _root / "more-tests" / name / f"test_{name}.{suffix}"


@pytest.mark.parametrize(
    "name,suffix,loader,raise_type,error_message",
    [
        (  # load_old_version
            "old_version",
            "py",
            load,
            ImportError,
            r"Not able to import the model\. The model was translated "
            r"with a not compatible version of PySD:\n\tPySD 1\.5\.0"
        ),
        (  # load_type
            "type_error",
            "py",
            load,
            ImportError,
            r".*Not able to import the model\. This may be because the "
            "model was compiled with an earlier version of PySD, you can "
            r"check on the top of the model file you are trying to load\..*"
        ),
        (  # not_vensim_model
            "not_vensim",
            "txt",
            read_vensim,
            ValueError,
            "The file to translate, "
            "'.*test_not_vensim.txt' is not a "
            r"Vensim model\. It must end with mdl extension\."
        ),
        (  # not_xmile_model
            "not_vensim",
            "txt",
            read_xmile,
            ValueError,
            "The file to translate, "
            "'.*test_not_vensim.txt' is not a "
            r"Xmile model\. It must end with xmile, xml or stmx extension\."
        ),
        (  # circular_reference
            "circular_reference",
            "py",
            load,
            ValueError,
            r"Circular initialization\.\.\.\nNot able to initialize the "
            "following objects:\n\t_integ_integ\n\t_delay_delay"
        ),
    ],
    ids=[
        "old_version", "load_type",
        "not_vensim_model", "not_xmile_model",
        "circular_reference"
    ]
)
def test_loading_error(loader, model_path, raise_type, error_message):
    with pytest.raises(raise_type, match=error_message):
        loader(model_path)


@pytest.mark.parametrize(
    "name,suffix",
    [
        (  # load_old_version
            "not_implemented_and_incomplete",
            "mdl",
        ),
    ]
)
def test_not_implemented_and_incomplete(model_path):
    with pytest.warns(UserWarning) as ws:
        model = read_vensim(model_path)
        assert "'incomplete var' has no equation specified"\
            in str(ws[0].message)
        assert "Trying to translate 'MY FUNC' which it is not implemented"\
            " on PySD. The translated model will crash..."\
            in str(ws[1].message)

    with pytest.warns(RuntimeWarning,
         match="Call to undefined function, calling dependencies "
         "and returning NaN"):
        model["incomplete var"]

    with pytest.raises(NotImplementedError,
         match="Not implemented function 'my_func'"):
        model["not implemented function"]
