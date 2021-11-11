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

    model = load(options.model_file, options.data_files,
                 options.missing_values, options.split_views,
                 subview_sep=options.subview_sep)

    if not options.run:
        print("\nFinished!")
        sys.exit()

    model.initialize()

    output = model.run(**create_configuration(model, options))

    if options.export_file:
        model.export(options.export_file)

    save(output, options)
    print("\nFinished!")
    sys.exit()


def load(model_file, data_files, missing_values, split_views, **kwargs):
    """
    Translate and load model file.

    Paramters
    ---------
    model_file: str
        Vensim, Xmile or PySD model file.

    data_files: list
        If given the list of files where the necessary data to run the model
        is given.

    missing_values : str ("warning", "error", "ignore", "keep")
        What to do with missing values. If "warning" (default)
        shows a warning message and interpolates the values.
        If "raise" raises an error. If "ignore" interpolates
        the values without showing anything. If "keep" it will keep
        the missing values, this option may cause the integration to
        fail, but it may be used to check the quality of the data.

    split_views: bool (optional)
        If True, the sketch is parsed to detect model elements in each
        model view, and then translate each view in a separate python
        file. Setting this argument to True is recommended for large
        models split in many different views. Default is False.

    **kwargs: (optional)
        Additional keyword arguments.
        subview_sep:(str)
            Character used to separate views and subviews. If provided,
            and split_views=True, each submodule will be placed inside the
            folder of the parent view.

    Returns
    -------
    pysd.model

    """
    if model_file.lower().endswith(".mdl"):
        print("\nTranslating model file...\n")
        return pysd.read_vensim(model_file, initialize=False,
                                data_files=data_files,
                                missing_values=missing_values,
                                split_views=split_views, **kwargs)
    elif model_file.lower().endswith(".xmile"):
        print("\nTranslating model file...\n")
        return pysd.read_xmile(model_file, initialize=False,
                               data_files=data_files,
                               missing_values=missing_values)
    else:
        return pysd.load(model_file, initialize=False,
                         data_files=data_files,
                         missing_values=missing_values)


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
        "params": options.new_values["param"],
        "initial_condition": (options.initial_time or model.time(),
                              options.new_values["initial"]),
        "return_columns": options.return_columns,
        "final_time": options.final_time,
        "time_step": options.time_step,
        "saveper": options.saveper,
        "flatten_output": True,  # need to return totally flat DF
        "return_timestamps": options.return_timestamps  # given or None
    }

    if options.import_file:
        conf_dict["initial_condition"] = options.import_file

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

    if output_file.endswith(".tab"):
        sep = "\t"
    else:
        sep = ","

    output.to_csv(output_file, sep, index_label="Time")

    print(f"Data saved in '{output_file}'")
