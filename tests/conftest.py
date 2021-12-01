import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def _root():
    # root directory
    return Path(__file__).parent.resolve()


@pytest.fixture(scope="class")
def shared_tmpdir(tmpdir_factory):
    # shared temporary directory for each class
    return Path(tmpdir_factory.mktemp("shared"))
