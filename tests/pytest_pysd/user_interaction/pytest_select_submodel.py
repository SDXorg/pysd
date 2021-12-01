
import pytest
import shutil
import numpy as np

import pysd


@pytest.mark.parametrize(
    "model_path,subview_sep,variables,modules,n_deps,dep_vars",
    [
        (
            "test_split_model",
            [],
            ["stock"],
            [],
            (6, 1, 1, 0, 1),
            {"rate1": 4, "initial_stock": 2}
        ),
        (
            "test_split_model_subviews",
            ["."],
            [],
            ["view_1"],
            (9, 0, 0, 0, 1),
            {

            }
        ),
        (
            "test_split_model_sub_subviews",
            [".", "-"],
            ["variablex"],
            ["subview_1", "submodule_1"],
            (12, 0, 1, 1, 1),
            {"another_var": 5, "look_up_definition": 3}
        )
    ],
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
    def models_dir(self, _root):
        return _root.joinpath("more-tests/split_model")

    @pytest.fixture
    def model(self, shared_tmpdir, models_dir, model_path, subview_sep):
        """
        Translate the model or read a translated version.
        This way each file is only translated once.
        """
        # expected file
        file = shared_tmpdir.joinpath(model_path + '.py')
        if file.is_file():
            # load already translated file
            return pysd.load(file)
        else:
            # copy mdl file to tmp_dir and translate it
            file = shared_tmpdir.joinpath(model_path + '.mdl')
            shutil.copy(
                models_dir.joinpath(model_path + '.mdl'),
                file)
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
        assert "other stock" in model.components._namespace
        assert "_integ_stock" in model._stateful_elements
        assert "_integ_stock" in model.components._dependencies
        assert "stock" in model.components._dependencies
        assert "Stock" in model.components._namespace

        # select submodel
        with pytest.warns(UserWarning, match=self.warning):
            model.select_submodel(vars=variables, modules=modules)

        # assert stateful elements change
        assert len(model._dynamicstateful_elements) == 1
        assert "_integ_other_stock" not in model._stateful_elements
        assert "_integ_other_stock" not in model.components._dependencies
        assert "other_stock" not in model.components._dependencies
        assert "other stock" not in model.components._namespace
        assert "_integ_stock" in model._stateful_elements
        assert "_integ_stock" in model.components._dependencies
        assert "stock" in model.components._dependencies
        assert "Stock" in model.components._namespace

        if not dep_vars:
            # totally independent submodels can run without producing
            # nan values
            assert not np.any(np.isnan(model.run()))
        else:
            # running the model without redefining dependencies will
            # produce nan values
            assert np.any(np.isnan(model.run()))
            # redefine dependencies
            assert not np.any(np.isnan(model.run(params=dep_vars)))
