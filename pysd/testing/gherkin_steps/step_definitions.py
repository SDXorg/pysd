from behave import *
import pysd
import nose.tools as test
import pandas as pd
import numpy as np
import os


@given("The model '{model_file}'")
def load_model(context, model_file):
    ext = model_file.split('.')[-1].lower()
    dir = context._config.base_dir
    model_file = os.path.join(dir, model_file)

    if ext == 'mdl':
        context.model = pysd.read_vensim(model_file)
    elif ext == 'xmile':
        context.model = pysd.read_xmile(model_file)
    elif ext == 'py':
        context.model = pysd.load(model_file)
    else:
        raise ValueError('Unknown model extension')


@when("{variable} is set to {value:g}")
def set_value(context, variable, value):
    # todo: check that this is not used for state variables, and if it is, set the value...

    context.model.set_components({variable: value})


@when('the model is run')
def run_model(context):
    context.result = context.model.run()


@then('{variable_1} is always equal to {variable_2}')
def check_equal(context, variable_1, variable_2):
    compare_variables(context, variable_1, variable_2, 'equal', 'all')


@then('{variable_1} is immediately equal to {variable_2}')
def check_equal(context, variable_1, variable_2):
    compare_variables(context, variable_1, variable_2, 'equal')


@then('{variable_1} is equal to {variable_2} at time {time:g}')
def check_equal(context, variable_1, variable_2, time):
    compare_variables(context, variable_1, variable_2, 'equal', time)


@then('{variable_1} is immediately less than {variable_2}')
def check_less(context, variable_1, variable_2):
    compare_variables(context, variable_1, variable_2, 'less')


@then('{variable_1} is always less than {variable_2}')
def check_less(context, variable_1, variable_2):
    compare_variables(context, variable_1, variable_2, 'less', 'all')


@then('{variable_1} is less than {variable_2} at time {time:g}')
def check_less(context, variable_1, variable_2, time):
    compare_variables(context, variable_1, variable_2, 'less', time)


@then('{variable_1} is immediately greater than {variable_2}')
def check_greater(context, variable_1, variable_2):
    compare_variables(context, variable_1, variable_2, 'greater')


@then('{variable_1} is always greater than {variable_2}')
def check_greater(context, variable_1, variable_2):
    compare_variables(context, variable_1, variable_2, 'greater', 'all')


@then('{variable_1} is greater than {variable_2} at time {time:g}')
def check_greater(context, variable_1, variable_2, time):
    compare_variables(context, variable_1, variable_2, 'greater', time)



def compare_variables(context, variable_1, variable_2, comparison, time=None):
    """

    Parameters
    ----------
    context
    variable_1
    variable_2
    comparison
    time
        If time is 'None', then checks that the current values in the model compare

    Returns
    -------

    """
    if time is None and variable_1 in context.model.components._namespace:
        val_1 = getattr(context.model.components,
                        context.model.components._namespace[variable_1])()

    elif hasattr(context, 'result') and variable_1 in context.result.columns:
        if time == 'all':
            val_1 = context.result[variable_1].loc[:]
        elif time in context.result.index:
            val_1 = context.result[variable_1].loc[time]
        elif context.result.index.min() < time < context.result.index.max():
            context.result.loc[time] = np.nan
            context.result = context.result.interpolate()
            val_1 = context.result[variable_1].loc[time]
        else:
            raise ValueError('Not able to find a value for %s at time %f' % (repr(variable_1),
                                                                             time))

    else:
        try:
            val_1 = float(variable_1)
        except:
            raise ValueError('Could not understand "%s" (did you run the model?)' % variable_1)

    if time is None and variable_2 in context.model.components._namespace:
        val_2 = getattr(context.model.components,
                        context.model.components._namespace[variable_2])()

    elif hasattr(context, 'result') and variable_2 in context.result.columns:
        if time == 'all':
            val_2 = context.result[variable_2].loc[:]
        elif time in context.result.index:
            val_2 = context.result[variable_2].loc[time]
        elif context.result.index.min() < time < context.result.index.max():
            context.result.loc[time] = np.nan
            context.result = context.result.interpolate()
            val_2 = context.result[variable_2].loc[time]
        else:
            raise ValueError('Not able to find a value for %s at time %f' % (repr(variable_2),
                                                                             time))
    else:
        try:
            val_2 = float(variable_2)
        except:
            raise ValueError('Could not understand "%s"' % variable_2)

    if time == 'all':
        try:
            if comparison == 'equal':
                test.assert_true(all(val_1 == val_2))
            elif comparison == 'less':
                test.assert_true(all(val_1 < val_2))
            elif comparison == 'greater':
                test.assert_true(all(val_1 > val_2))
        except AssertionError:
            raise AssertionError('%s is not %s %s' %(repr(val_1), comparison, val_2))

    else:
        if comparison == 'equal':
            test.assert_equal(val_1, val_2)
        elif comparison == 'less':
            test.assert_less(val_1, val_2)
        elif comparison == 'greater':
            test.assert_greater(val_1, val_2)


@then('we see that {scenario}')
def run_scenario(context, scenario):
    scenarios_available = context._stack[1]['feature'].scenarios
    scenario_names = [s.name for s in scenarios_available]
    if scenario in scenario_names:
        scenarios_available[scenario_names.index(scenario)].run(context._runner)
