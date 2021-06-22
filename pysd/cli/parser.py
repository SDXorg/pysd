"""
cmdline parser
"""
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

def split_columns(string):
    """
    Splits the return-columns argument
    --return-columns 'temperature change, "heat$"' -> ['temperature change', '"heat$"']

    """
    if string.endswith('.txt'):
        with open(string, 'r') as file:
            return [col.rstrip('\n').strip() for col in file]

    return [s.strip() for s in string.split(',')]


def split_timestamps(string):
    """
    Splits the return-timestamps argument
    --return-timestamps '1 5 15' -> array([1., 5., 15.])

    """
    try:
        return np.array([s.strip() for s in string.split(',')], dtype=float)
    except Exception:
        # error
        raise parser.error(
            f'when parsing {string}'
            '\nThe return time stamps much be given between \'\' and split by'
            ' whitespaces...\n'
            f'See {docs} for examples.')


def split_vars(string):
    """
    Splits the arguments from new_values.
    'a=5' -> {'a': 5.}
    'b=[[1,2],[1,10]]' -> {'b': pd.Series(index=[1, 2], data=[1, 10])}

    """
    var, value = string.split('=')
    if value.strip().isnumeric():
        # value is float
        return {var.strip(): float(value)}

    try:
        # value is series
        value = literal_eval(value)
        assert len(value) == 2
        assert len(value[0]) == len(value[1])
        return {var.strip(): pd.Series(index=value[0], data=value[1])}

    except Exception:
        # error
        raise parser.error(
            f'when parsing {string}'
            '\nYou must use variable=new_value to redefine values.'
            'variable must be a model component, new_value can be a '
            'float or a a list of two list...\n'
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
        main_dict = {}
        for val in values:
            main_dict.update(val)
        setattr(namespace, self.dest, main_dict)


###########
# options #
###########

parser.add_argument(
    '-v', '--version',
    action='version', version=f'%prog {__version__}')

parser.add_argument(
    '-t', '--translate', dest='run',
    action='store_false', default=True,
    help='only translate the model_file, '
         'it does not run it after translation')

parser.add_argument(
    '-o', '--output-file', dest='output_file',
    action='store', type=str, metavar='FILE',
    help='output file to save run outputs')

parser.add_argument(
    '-p', '--progress', dest='progress',
    action='store_true', default=False,
    help='show progress bar during model integration')

parser.add_argument(
    '-r', '--return-columns', dest='return_columns',
    action='store', type=split_columns,
    metavar='\'var1, var2, .. varN\' or FILE (.txt)',
    help='provide the return columns separated by commas or a .txt file'
         ' where each row is a variable')


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
         '--final-time and --saveper will be ignored')


########################
# Positional arguments #
########################

parser.add_argument('model_file', metavar='model_file', type=str,
                    help='Vensim, Xmile or PySD model file')

parser.add_argument('new_values',
                    metavar='variable=new_value', type=split_vars,
                    nargs='*', action=SplitVarsAction,
                    help='redefine the value of variable with new value.'
                    'variable must be a model component, new_value can be a '
                    'float or a a list of two list')
