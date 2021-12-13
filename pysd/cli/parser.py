"""
cmdline parser
"""
import os
from ast import literal_eval
import numpy as np
import pandas as pd
from argparse import ArgumentParser, Action

from pysd import __version__

docs = "https://pysd.readthedocs.io/en/master/command_line_usage.html"

parser = ArgumentParser(
    description='Simulating System Dynamics Models in Python',
    prog='PySD')


#########################
# functions and actions #
#########################

def check_output(string):
    """
    Checks that out put file ends with .tab or .csv

    """
    if not string.endswith('.tab') and not string.endswith('.csv'):
        parser.error(
            f'when parsing {string}'
            '\nThe output file name must be .tab or .csv...')

    return string


def check_model(string):
    """
    Checks that model file ends with .py .mdl or .xmile and that exists.

    """
    if not string.lower().endswith('.mdl')\
       and not string.lower().endswith('.xmile')\
       and not string.endswith('.py'):
        parser.error(
            f'when parsing {string}'
            '\nThe model file name must be Vensim (.mdl), Xmile (.xmile)'
            ' or PySD (.py) model file...')

    if not os.path.isfile(string):
        parser.error(
            f'when parsing {string}'
            '\nThe model file does not exist...')

    return string


def check_data_file(string):
    """
    Check that data file is a tab or csv file and that exists.
    """
    if not string.endswith('.tab') and not string.endswith('.csv'):
        parser.error(
            f'when parsing {string}'
            '\nThe data file name must be .tab or .csv...')
    elif not os.path.isfile(string):
        parser.error(
            f'when parsing {string}'
            '\nThe data file does not exist...')
    else:
        return string


def split_files(string):
    """
    Splits the data files and returns and error if file doesn't exists
    --data 'file1.tab, file2.csv' -> ['file1.tab', 'file2.csv']
    --data file1.tab -> ['file1.tab']

    """
    return [check_data_file(s.strip()) for s in string.split(',')]


def split_columns(string):
    """
    Splits the return-columns argument or reads it from .txt
    --return-columns 'temperature c, "heat$"' -> ['temperature c', '"heat$"']
    --return-columns my_vars.txt -> ['temperature c', '"heat$"']

    """
    if string.endswith('.txt'):
        with open(string, 'r') as file:
            return [col.rstrip('\n').strip() for col in file]

    return [s.strip() for s in string.split(',')]


def split_timestamps(string):
    """
    Splits the return-timestamps argument
    --return-timestamps '1, 5, 15' -> array([1., 5., 15.])

    """
    try:
        return np.array([s.strip() for s in string.split(',')], dtype=float)
    except Exception:
        # error
        raise parser.error(
            f'when parsing {string}'
            '\nThe return time stamps much be separated by commas...\n'
            f'See {docs} for examples.')


def split_subview_sep(string):
    """
    Splits the subview separators
    --subview-sep ' - ,.' -> [' - ', '.']

    """
    return string.split(",")


def split_vars(string):
    """
    Splits the arguments from new_values.
    'a=5' -> {'a': ('param', 5.)}
    'b=[[1,2],[1,10]]' -> {'b': ('param', pd.Series(index=[1,2], data=[1,10]))}
    'a:5' -> {'a': ('initial', 5.)}

    """
    try:
        if '=' in string:
            # new variable value
            var, value = string.split('=')
            type = 'param'

        if ':' in string:
            # initial time value
            var, value = string.split(':')
            type = 'initial'

        if value.strip().isnumeric():
            # value is float
            return {var.strip(): (type, float(value))}

        # value is series
        assert type == 'param'
        value = literal_eval(value)
        assert len(value) == 2
        assert len(value[0]) == len(value[1])
        return {var.strip(): (type,
                              pd.Series(index=value[0], data=value[1]))}

    except Exception:
        # error
        raise parser.error(
            f'when parsing {string}'
            '\nYou must use variable=new_value to redefine values or '
            'variable:initial_value to define initial value.'
            'variable must be a model component, new_value can be a '
            'float or a list of two list, initial_value must be a float'
            '...\n'
            f'See {docs} for examples.')


class SplitVarsAction(Action):
    """
    Convert the list of split variables from new_values to a dictionary.
    [{'a': 5.}, {'b': pd.Series(index=[1, 2], data=[1, 10])}] ->
        {'a': 5., 'b': pd.Series(index=[1, 2], data=[1, 10])}
    """
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        main_dict = {'param': {}, 'initial': {}}
        for var in values:
            for var_name, (type, val) in var.items():
                main_dict[type][var_name] = val
        setattr(namespace, self.dest, main_dict)


###########
# options #
###########

parser.add_argument(
    '-v', '--version',
    action='version', version=f'PySD {__version__}')

parser.add_argument(
    '-o', '--output-file', dest='output_file',
    type=check_output, metavar='FILE',
    help='output file to save run outputs (.tab or .csv)')

parser.add_argument(
    '-p', '--progress', dest='progress',
    action='store_true', default=False,
    help='show progress bar during model integration')

parser.add_argument(
    '-r', '--return-columns', dest='return_columns',
    action='store', type=split_columns,
    metavar='\'var1, var2, .., varN\' or FILE (.txt)',
    help='provide the return columns separated by commas or a .txt file'
         ' where each row is a variable')

parser.add_argument(
    '-e', '--export', dest='export_file',
    type=str, metavar='FILE',
    help='export to a pickle stateful objects states at the end of the '
         'simulation')

parser.add_argument(
    '-i', '--import-initial', dest='import_file',
    type=str, metavar='FILE',
    help='import stateful objects states from a pickle file,'
         'if given initial conditions from var:value will be ignored')


###################
# Model arguments #
###################

model_arguments = parser.add_argument_group(
    'model arguments',
    'Modify model control variables.')

model_arguments.add_argument(
    '-I', '--initial-time', dest='initial_time',
    action='store', type=float, metavar='VALUE',
    help='modify initial time of the simulation')

model_arguments.add_argument(
    '-F', '--final-time', dest='final_time',
    action='store', type=float, metavar='VALUE',
    help='modify final time of the simulation')

model_arguments.add_argument(
    '-T', '--time-step', dest='time_step',
    action='store', type=float, metavar='VALUE',
    help='modify time step of the simulation')

model_arguments.add_argument(
    '-S', '--saveper', dest='saveper',
    action='store', type=float, metavar='VALUE',
    help='modify time step of the output')

model_arguments.add_argument(
    '-R', '--return-timestamps', dest='return_timestamps',
    action='store', type=split_timestamps,
    metavar='\'value1, value2, .., valueN\'',
    help='provide the return time stamps separated by commas, if given '
         '--saveper will be ignored')

model_arguments.add_argument(
    '-D', '--data', dest='data_files',
    action='store', type=split_files, metavar='\'FILE1, FILE2, .., FILEN\'',
    help='input data file or files to run the model')

#########################
# Translation arguments #
#########################

trans_arguments = parser.add_argument_group(
    'translation arguments',
    'Configure the translation of the original model.')

trans_arguments.add_argument(
    '--translate', dest='run',
    action='store_false', default=True,
    help='only translate the model_file, '
         'it does not run it after translation')

trans_arguments.add_argument(
    '--split-views', dest='split_views',
    action='store_true', default=False,
    help='parse the sketch to detect model elements in each model view,'
         ' and then translate each view in a separate Python file')

trans_arguments.add_argument(
    '--subview-sep', dest='subview_sep',
    action='store', type=split_subview_sep, default=[],
    metavar='\'STRING1,STRING2,..,STRINGN\'',
    help='further division of views split in subviews, by identifying the'
         'separator string in the view name, only availabe if --split-views'
         ' is used')


#######################
# Warnings and errors #
#######################

warn_err_arguments = parser.add_argument_group(
    'warning and errors arguments',
    'Modify warning and errors management.')

warn_err_arguments.add_argument(
    '--missing-values', dest='missing_values', default="warning",
    action='store', type=str, choices=['warning', 'raise', 'ignore', 'keep'],
    help='exception with missing values, \'warning\' (default) shows a '
         'warning message and interpolates the values, \'raise\' raises '
         'an error, \'ignore\' interpolates the values without showing '
         'anything, \'keep\' keeps the missing values')


########################
# Positional arguments #
########################

parser.add_argument('model_file', metavar='model_file', type=check_model,
                    help='Vensim, Xmile or PySD model file')

parser.add_argument('new_values',
                    metavar='variable=new_value', type=split_vars,
                    nargs='*', action=SplitVarsAction,
                    help='redefine the value of variable with new value.'
                    'variable must be a model component, new_value can be a '
                    'float or a a list of two list')

# The destionation new_values2 will never used as the previous argument
# is given also with nargs='*'. Nevertheless, the following variable
# is declared for documentation
parser.add_argument('new_values2',
                    metavar='variable:initial_value', type=split_vars,
                    nargs='*', action=SplitVarsAction,
                    help='redefine the initial value of variable.'
                    'variable must be a model stateful element, initial_value'
                    ' must be a float')


#########
# Usage #
#########

parser.usage = parser.format_usage().replace("usage: PySD", "python -m pysd")
