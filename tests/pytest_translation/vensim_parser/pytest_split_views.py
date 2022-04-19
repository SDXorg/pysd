
import pytest
from pathlib import Path
import shutil

import pysd
from pysd.tools.benchmarking import assert_frames_close


@pytest.mark.parametrize(
    "model_path,subview_sep,modules,macros,original_vars,py_vars,"
    + "stateful_objs",
    [
        (  # split_views
            Path("more-tests/split_model/test_split_model.mdl"),
            [],
            ["view_1", "view2", "view_3"],
            [],
            ["Stock"],
            ["another_var", "rate1", "varn", "variablex", "stock"],
            ["_integ_stock"]
        ),
        (  # split_subviews
            Path("more-tests/split_model/test_split_model_subviews.mdl"),
            ["."],
            ["view_1/submodule_1", "view_1/submodule_2", "view_2"],
            [],
            ["Stock"],
            ["another_var", "rate1", "varn", "variablex", "stock"],
            ["_integ_stock"]
        ),
        (  # split_sub_subviews
            Path("more-tests/split_model/test_split_model_sub_subviews.mdl"),
            [".", "-"],
            [
                "view_1/submodule_1", "view_1/submodule_2", "view_2",
                "view_3/subview_1/sview_1", "view_3/subview_1/sview_2",
                "view_3/subview_2/sview_3", "view_3/subview_2/sview_4"
            ],
            [],
            ["Stock"],
            ["another_var", "rate1", "varn", "variablex", "stock",
             "interesting_var_2", "great_var"],
            ["_integ_stock"]
        ),
        (  # split_macro
            Path("more-tests/split_model_with_macro/"
                 + "test_split_model_with_macro.mdl"),
            [".", "-"],
            ["view_1", "view_2"],
            ["expression_macro"],
            ["new var"],
            ["new_var"],
            ["_macro_macro_output"]
        ),
        (  # split_vensim_8_2_1
            Path("more-tests/split_model_vensim_8_2_1/"
                 + "test_split_model_vensim_8_2_1.mdl"),
            [],
            ["teacup", "cream"],
            [],
            ["Teacup Temperature", "Cream Temperature"],
            ["teacup_temperature", "cream_temperature"],
            ["integ_teacup_temperature", "integ_cream_temperature"]
        )
    ],
    ids=["split_views", "split_subviews", "split_sub_subviews", "split_macro",
         "split_vensim_8_2_1"]
)
class TestSplitViews:
    """
    Test for splitting Vensim views in modules and submodules
    """
    @pytest.fixture
    def model_file(self, shared_tmpdir, model_path):
        return shared_tmpdir.joinpath(model_path.name)

    @pytest.fixture
    def expected_files(self, shared_tmpdir, _root, model_path,
                       model_file, modules, macros):
        model_name = model_path.stem
        shutil.copy(
            _root.joinpath(model_path),
            model_file
        )
        modules_dir = shared_tmpdir.joinpath("modules_" + model_name)
        files = {
            shared_tmpdir.joinpath("_subscripts_" + model_name + ".json"),
            shared_tmpdir.joinpath("_dependencies_" + model_name + ".json"),
            modules_dir.joinpath("_modules.json")
        }
        [files.add(modules_dir.joinpath(module + ".py")) for module in modules]
        [files.add(shared_tmpdir.joinpath(macro + ".py")) for macro in macros]
        return files

    @pytest.mark.filterwarnings("ignore")
    def test_read_vensim_split_model(self, model_file, subview_sep,
                                     expected_files, modules,
                                     original_vars, py_vars,
                                     stateful_objs):
        # assert that the files don't exist in the temporary directory
        for file in expected_files:
            assert not file.is_file(), f"File {file} already exists..."

        # translate split model
        model_split = pysd.read_vensim(model_file, split_views=True,
                                       subview_sep=subview_sep)

        # assert that all the files have been created
        for file in expected_files:
            assert file.is_file(), f"File {file} has not been created..."

        # check the dictionaries
        assert isinstance(model_split._namespace, dict)
        assert isinstance(model_split.components._subscript_dict, dict)
        assert isinstance(model_split.components._dependencies, dict)
        assert isinstance(model_split.components._modules, dict)

        # assert taht main modules are dictionary keys
        for module in modules:
            assert module.split("/")[0]\
                in model_split.components._modules.keys()

        # assert that original variables are in the namespace
        for var in original_vars:
            assert var in model_split._namespace.keys()

        # assert that the functions are not defined in the main file
        model_py_file = model_file.with_suffix(".py")
        with open(model_py_file, 'r') as file:
            file_content = file.read()
        for var in py_vars:
            assert "def %s()" % var not in file_content
        for var in stateful_objs:
            assert "%s = " % var not in file_content

        # translation without splitting
        model_non_split = pysd.read_vensim(model_file, split_views=False)

        # assert that the functions are defined in the main file
        with open(model_py_file, 'r') as file:
            file_content = file.read()
        for var in py_vars:
            assert "def %s()" % var in file_content
        for var in stateful_objs:
            assert "%s = " % var in file_content

        # check that both models give the same result
        assert_frames_close(
            model_split.run(), model_non_split.run(), atol=0, rtol=0)


@pytest.mark.parametrize(
    "model_path,subview_sep,warning_message",
    [
        (  # warning_noviews
            Path("test-models/samples/teacup/teacup.mdl"),
            [],
            "Only a single view with no subviews was detected. The model"
            + " will be built in a single file."
        ),
        (  # not_match_separator
            Path("more-tests/split_model/test_split_model_sub_subviews.mdl"),
            ["a"],
            "The given subview separators were not matched in any view name."
        ),
    ],
    ids=["warning_noviews", "not_match_separator"]
)
class TestSplitViewsWarnings:
    """
    Test for warnings while splitting views.
    """
    @pytest.fixture
    def model(self, shared_tmpdir, model_path, _root):
        # move model file to temporary dir
        file = shared_tmpdir.joinpath(model_path.name)
        shutil.copy(_root.joinpath(model_path), file)
        return file

    def test_split_view_warnings(self, model, subview_sep, warning_message):
        with pytest.warns(UserWarning, match=warning_message):
            pysd.read_vensim(model, split_views=True, subview_sep=subview_sep)
