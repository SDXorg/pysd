from behave import *
import pysd
import nose.tools as test


@given('The model "{model_file}"')
def load_model(context, model_file):
    ext = model_file.split('.')[-1].lower()
    if ext == 'mdl':
        context.model = pysd.read_vensim(model_file)
    elif ext == 'xmile':
        context.model = pysd.read_xmile(model_file)
    elif ext == 'py':
        context.model = pysd.load(model_file)
    else:
        raise ValueError('Unknown model extension')


@when('"{variable}" is set to "{value}"')
def set_value(context, variable, value):
    context.model.set_components({variable: value})


@when('the model is run')
def run_model(context):
    context.result = context.model.run()


@then('"{variable}" are immediately "{value}"')
def check_immediate(context, variable, value):
    test.assert_equal(context.model.components[variable],
                      value)
