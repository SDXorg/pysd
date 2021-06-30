import sys
import os

from datetime import datetime

from .parser import parser

import pysd


def main(args):
    """
    Main function. Reads user arguments, loads the models,
    runs it and saves the output

    Parameters
    ----------
    args: list
        User arguments.

    Returns
    -------
    None

    """
    options = parser.parse_args(args)

    model = load(options.model_file, options.missing_values)

    if not options.run:
        print("\nFinished!")
        sys.exit()

    initialize(model, options)

    output = model.run(**create_configuration(model, options))

    save(output, options)
    print("\nFinished!")
    sys.exit()


def load(model_file, missing_values):
    """
    Translate and load model file.

    Paramters
    ---------
    model_file: str
        Vensim, Xmile or PySD model file.

    Returns
    -------
    pysd.model

    """
    if model_file.lower().endswith('.mdl'):
        print("\nTranslating model file...\n")
        return pysd.read_vensim(model_file, initialize=False,
                                missing_values=missing_values)
    elif model_file.lower().endswith('.xmile'):
        print("\nTranslating model file...\n")
        return pysd.read_xmile(model_file, initialize=False,
                               missing_values=missing_values)
    else:
        return pysd.load(model_file, initialize=False,
                         missing_values=missing_values)


def initialize(model, options):
    """
    Update model control variables and initialize the model.

    Parameters
    ----------
    model: pysd.model object

    options: argparse.Namespace

    Returns
    -------
    None

    """
    for func_name in ['initial_time', 'time_step', 'final_time', 'saveper']:
        if getattr(options, func_name):
            model.set_components({func_name: getattr(options, func_name)})

    model.initialize()


def create_configuration(model, options):
    """
    create configuration dict to pass to the run method.

    Parameters
    ----------
    model: pysd.model object

    options: argparse.Namespace

    Returns
    -------
    conf_dict: dict

    """
    conf_dict = {
        "progress": options.progress,
        "params": options.new_values['param'],
        "initial_condition": (model.time(), options.new_values['initial']),
        "return_columns": options.return_columns,
        "flatten_output": True,  # need to return totally flat DF
        "return_timestamps": options.return_timestamps  # given or None
    }

    return conf_dict


def save(output, options):
    """
    Saves models output.

    Paramters
    ---------
    output: pandas.DataFrame

    options: argparse.Namespace

    Returns
    -------
    None

    """
    if options.output_file:
        output_file = options.output_file
    else:
        output_file = os.path.splitext(os.path.basename(
            options.model_file
            ))[0]\
                + datetime.now().strftime("_output_%Y_%m_%d-%H_%M_%S_%f.tab")

    if output_file.endswith('.tab'):
        sep = '\t'
    else:
        sep = ','

    output.to_csv(output_file, sep, index_label='Time')

    print(f'Data saved in {output_file}')
