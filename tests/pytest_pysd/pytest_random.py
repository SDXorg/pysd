
import pytest
import shutil

import pysd


class TestRandomModel:
    """Submodel selecting class"""
    # messages for selecting submodules
    @pytest.fixture(scope="class")
    def model_path(self, shared_tmpdir, _root):
        """
        Copy test folder to a temporary folder therefore we avoid creating
        PySD model files in the original folder
        """
        new_file = shared_tmpdir.joinpath("test_random.mdl")
        shutil.copy(
            _root.joinpath("more-tests/random/test_random.mdl"),
            new_file
        )
        return new_file

    def test_translate(self, model_path):
        """
        Translate the model or read a translated version.
        This way each file is only translated once.
        """
        # expected file
        model = pysd.read_vensim(model_path)
        model.run()
