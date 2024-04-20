import shutil
from pathlib import Path
from dataclasses import dataclass


import pytest

from pysd import read_vensim, read_xmile, load
from pysd.translators.vensim.vensim_utils import supported_extensions as\
    vensim_extensions
from pysd.translators.xmile.xmile_utils import supported_extensions as\
    xmile_extensions

from pysd.builders.python.imports import ImportsManager


@pytest.fixture(scope="session")
def _root():
    # root directory
    return Path(__file__).parent.resolve()


@pytest.fixture(scope="session")
def _test_models(_root):
    # test-models directory
    return _root.joinpath("test-models/tests")


@pytest.fixture(scope="session")
def _test_random(_root):
    # test-models directory
    return _root.joinpath("test-models/random")


@pytest.fixture(scope="class")
def shared_tmpdir(tmp_path_factory):
    # shared temporary directory for each class
    return tmp_path_factory.mktemp("shared")


@pytest.fixture
def model(_root, tmp_path, model_path):
    """
    Copy model to the tmp_path and translate it
    """
    assert (_root / model_path).exists(), "The model doesn't exist"

    target = tmp_path / model_path.parent.name
    new_path = target / model_path.name
    shutil.copytree(_root / model_path.parent, target)

    if model_path.suffix.lower() in vensim_extensions:
        return read_vensim(new_path)
    elif model_path.suffix.lower() in xmile_extensions:
        return read_xmile(new_path)
    elif model_path.suffix.lower() == ".py":
        return load(new_path)
    else:
        return ValueError("Invalid model")


@pytest.fixture(scope="session")
def ignore_warns():
    # warnings to be ignored in the integration tests
    return [
        "numpy.ndarray size changed, may indicate binary incompatibility.",
        "Creating an ndarray from ragged nested sequences.*",
        "datetime.datetime.* is deprecated and scheduled for removal in a "
        "future version. Use timezone-aware objects to represent datetimes "
        "in UTC.*",
    ]


@pytest.fixture(scope="session")
def random_size():
    # size of generated random samples
    return int(1e6)


@dataclass
class FakeComponent:
    element: str
    section: object
    subscripts_dict: dict


@dataclass
class FakeSection:
    namespace: object
    macrospace: dict
    imports: object


@dataclass
class FakeNamespace:
    cleanspace: dict


@pytest.fixture(scope="function")
def fake_component():
    # fake_component used to translate random functions to python
    return FakeComponent(
        '',
        FakeSection(FakeNamespace({}), {}, ImportsManager()),
        {}
    )
