from behave import *
import pysd
import nose.tools as test
import os


@given('The model "{model_file}"')
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


@when('{variable} is set to {value:g}')
def set_value(context, variable, value):
    context.model.set_components({variable: value})


@when('the model is run')
def run_model(context):
    context.result = context.model.run()


@then('{variable} is immediately {value:g}')
def check_immediate(context, variable, value):
    test.assert_equal(getattr(context.model.components,
                              context.model.components._namespace[variable])(),
                      value)


@then('{variable} becomes {value:g} at time {time:g}')
def check_future(context, variable, value, time):
    test.assert_equal(context.result[variable].loc[time], value)
