import shutil
import pytest
from pathlib import Path

import pysd


@pytest.fixture(scope="class")
def input_file(shared_tmpdir, _root):
    input_path = Path("more-tests/split_model/input.xlsx")
    shutil.copy(
        _root.joinpath(input_path),
        shared_tmpdir.joinpath(input_path.name)
    )


@pytest.mark.parametrize(
    "model_path,subview_sep",
    [
        (  # teacup
            Path("test-models/samples/teacup/teacup.mdl"),
            None
        ),
        (  # split_views
            Path("more-tests/split_model/test_split_model.mdl"),
            []
        ),
        (  # split_subviews
            Path("more-tests/split_model/test_split_model_subviews.mdl"),
            ["."]
        ),
        (  # split_sub_subviews
            Path("more-tests/split_model/test_split_model_sub_subviews.mdl"),
            [".", "-"]
        )
    ],
    ids=["teacup", "split_views", "split_subviews", "split_sub_subviews"]
)
class TestModelProperties():

    @pytest.fixture
    def model(self, shared_tmpdir, model_path, subview_sep, _root, input_file):
        """
        Translate the model or read a translated version.
        This way each file is only translated once.
        """
        # expected file
        file = shared_tmpdir.joinpath(model_path.with_suffix(".py").name)
        if file.is_file():
            # load already translated file
            return pysd.load(file)
        else:
            # copy mdl file to tmp_dir and translate it
            file = shared_tmpdir.joinpath(model_path.name)
            shutil.copy(_root.joinpath(model_path), file)
            return pysd.read_vensim(
                file,
                split_views=(subview_sep is not None), subview_sep=subview_sep)

    def test_propierties(self, model):
        # test are equal to model attributes they are copying
        assert model.namespace == model._namespace
        assert model.subscripts == model._subscript_dict
        assert model.dependencies == model._dependencies
        if model._modules:
            assert model.modules == model._modules
        else:
            assert model.modules is None

        # test thatwhen modifying a propierty by the user the model
        # attribute remains the same
        ns = model.namespace
        ns["Time"] = "my_new_time"
        assert ns != model._namespace
        assert ns != model.namespace
        sd = model.subscripts
        sd["my_new_subs"] = ["s1", "s2", "s3"]
        assert sd != model._subscript_dict
        assert sd != model.subscripts
        ds = model.dependencies
        ds["my_var"] = {"time": 1}
        assert ds != model._dependencies
        assert ds != model.dependencies
        if model._modules:
            ms = model.modules
            del ms[list(ms)[0]]
            assert ms != model._modules
            assert ms != model.modules
