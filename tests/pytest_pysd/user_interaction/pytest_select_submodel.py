
import pytest
import shutil
from pathlib import Path
import numpy as np

import pysd


@pytest.mark.parametrize(
    "model_path,subview_sep,variables,modules,n_deps,dep_vars",
    [
        (  # split_views
            Path("more-tests/split_model/test_split_model.mdl"),
            [],
            ["stock"],
            [],
            (6, 1, 2, 0, 1),
            {"rate1": 4, "initial_stock": 2, "initial_stock_correction": 0}
        ),
        (  # split_subviews
            Path("more-tests/split_model/test_split_model_subviews.mdl"),
            ["."],
            [],
            ["view_1"],
            (9, 0, 0, 0, 1),
            {

            }
        ),
        (  # split_sub_subviews
            Path("more-tests/split_model/test_split_model_sub_subviews.mdl"),
            [".", "-"],
            ["variablex"],
            ["view_3/subview_1", "view_1/submodule_1"],
            (12, 0, 1, 1, 2),
            {"another_var": 5, "look_up_definition": 3}
        )
    ],
    ids=["split_views", "split_subviews", "split_sub_subviews"]
)
class TestSubmodel:
    """Submodel selecting class"""
    # messages for selecting submodules
    messages = [
            "Selected variables",
            "Dependencies for initialization only",
            "Dependencies that may change over time",
            "Lookup table dependencies",
            "Stateful objects integrated with the selected variables"
            ]
    warning = "Selecting submodel, "\
        + "to run the full model again use model.reload()"
    common_vars = {
        'initial_time', 'time_step', 'final_time', 'time', 'saveper', 'stock'
        }

    @pytest.fixture
    def model(self, shared_tmpdir, model_path, subview_sep, _root):
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
                split_views=True, subview_sep=subview_sep)

    def test__get_dependencies(self, model, variables, modules,
                               n_deps, dep_vars):

        # get selected vars and dependencies in sets and dictionary
        out = model._get_dependencies(vars=variables, modules=modules)
        assert len(out[0]) == n_deps[0]
        assert len(out[1]["initial"]) == n_deps[1]
        assert len(out[1]["step"]) == n_deps[2]
        assert len(out[1]["lookup"]) == n_deps[3]
        assert len(out[2]) == n_deps[4]

        assert self.common_vars.issubset(out[0])

    def test_get_dependencies(self, capsys, model, variables, modules,
                              n_deps, dep_vars):
        # get the dependencies information of the selected variables and
        # modules by stdout
        model.get_dependencies(vars=variables, modules=modules)

        captured = capsys.readouterr()  # capture stdout

        for n, message in zip(n_deps, self.messages):
            if n != 0:
                # check the message with the number of dependencies of
                # each type
                assert message + " (total %s):\n" % n in captured.out
            else:
                # if not dependencies not message should be printed
                assert message not in captured.out

        # assert _integ_stock is in the message as the included stateful object
        assert "_integ_stock" in captured.out

        # assert all dependencies of the submodel are in the message
        for var in dep_vars:
            assert var in captured.out

    def test_select_submodel(self, model, variables, modules,
                             n_deps, dep_vars):

        # assert original stateful elements
        assert len(model._dynamicstateful_elements) == 2
        assert "_integ_other_stock" in model._stateful_elements
        assert "_integ_other_stock" in model.components._dependencies
        assert "other_stock" in model.components._dependencies
        assert "other stock" in model._namespace
        assert "_integ_stock" in model._stateful_elements
        assert "_integ_stock" in model.components._dependencies
        assert "stock" in model.components._dependencies
        assert "Stock" in model._namespace

        # select submodel
        with pytest.warns(UserWarning) as record:
            model.select_submodel(vars=variables, modules=modules)

        # assert warning
        assert str(record[0].message) == self.warning

        # assert stateful elements change
        assert len(model._dynamicstateful_elements) == 1
        assert "_integ_other_stock" not in model._stateful_elements
        assert "_integ_other_stock" not in model.components._dependencies
        assert "other_stock" not in model.components._dependencies
        assert "other stock" not in model._namespace
        assert "_integ_stock" in model._stateful_elements
        assert "_integ_stock" in model.components._dependencies
        assert "stock" in model.components._dependencies
        assert "Stock" in model._namespace

        if not dep_vars:
            # totally independent submodels can run without producing
            # nan values
            assert not np.any(np.isnan(model.run()))
        else:
            # running the model without redefining dependencies will
            # produce nan values
            assert "Exogenous components for the following variables are"\
                + " necessary but not given:" in str(record[-1].message)
            assert "Please, set them before running the model using "\
                + "set_components method..." in str(record[-1].message)
            for var in dep_vars:
                assert var in str(record[-1].message)
            assert np.any(np.isnan(model.run()))
            # redefine dependencies
            assert not np.any(np.isnan(model.run(params=dep_vars)))

        # select submodel using contour values
        model.reload()
        with pytest.warns(UserWarning) as record:
            model.select_submodel(vars=variables, modules=modules,
                                  exogenous_components=dep_vars)

        assert not np.any(np.isnan(model.run()))


@pytest.mark.parametrize(
    "model_path,split_views,module,raise_type,error_message",
    [
        (  # module_not_found
            Path("more-tests/split_model/test_split_model.mdl"),
            True,
            "view_4",
            NameError,
            "Module or submodule 'view_4' not found..."

        ),
        (  # not_modularized_model
            Path("more-tests/split_model/test_split_model.mdl"),
            False,
            "view_1",
            ValueError,
            "Trying to get a module from a non-modularized model"

        )
    ],
    ids=["module_not_found", "not_modularized_model"]
)
class TestGetVarsInModuleErrors:
    @pytest.fixture
    def model(self, shared_tmpdir, model_path, split_views, _root):
        """
        Translate the model.
        """
        # mdl file
        file = shared_tmpdir.joinpath(model_path.name)

        if not file.is_file():
            # copy mdl file
            shutil.copy(_root.joinpath(model_path), file)

        return pysd.read_vensim(file, split_views=split_views)

    def test_get_vars_in_module_errors(self, model, module, raise_type,
                                       error_message):
        # assert raises are produced
        with pytest.raises(raise_type, match=error_message):
            model.get_vars_in_module(module)
