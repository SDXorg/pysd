"""
pysd.py

Contains all the code that will be directly accessed by the user in normal operation.

History
--------
August 15, 2014: created
June 6 2015: Major updates - version 0.2.5
Jan 2016: Rework to handle subscripts
May 2016: Updates to handle grammar refactoring
Sept 2016: Major refactor, putting most internal code into the Model and Macro objects
"""

import sys

if sys.version_info[:2] < (3, 7):
    raise RuntimeError(
        "\n\n"
        + "Your Python version is not longer supported by PySD.\n"
        + "The current version needs to run at least Python 3.7."
        + " You are running:\n\tPython "
        + sys.version
        + "."
        + "\nPlease update your Python version or use the last "
        + " supported version:\n\t"
        + "https://github.com/JamesPHoughton/pysd/releases/tag/LastPy2"
    )


def read_xmile(xmile_file, initialize=True, missing_values="warning"):
    """
    Construct a model from `.xmile` file.

    Parameters
    ----------
    xmile_file : <string>
        The relative path filename for a raw `.xmile` file
    initialize: bool (optional)
        If False, the model will not be initialize when it is loaded.
        Default is True
    missing_values : <string> (optional)
        What to do with missing values in external objects.
        If "warning" (default) shows a warning message and
        interpolates the values. If "raise" raises an error.
        If "ignore" interpolates the values without showing anything.

    Returns
    -------
    model: a PySD class object
        Elements from the python model are loaded into the PySD class
        and ready to run

    Examples
    --------
    >>> model = read_xmile('../tests/test-models/samples/teacup/teacup.xmile')

    """
    from .py_backend.xmile.xmile2py import translate_xmile

    py_model_file = translate_xmile(xmile_file)
    model = load(py_model_file, initialize, missing_values)
    model.xmile_file = xmile_file
    return model


def read_vensim(
    mdl_file, initialize=True, missing_values="warning", split_modules=False
):
    """
    Construct a model from Vensim `.mdl` file.

    Parameters
    ----------
    mdl_file : <string>
        The relative path filename for a raw Vensim `.mdl` file
    initialize: bool (optional)
        If False, the model will not be initialize when it is loaded.
        Default is True
    missing_values : <string> (optional)
        What to do with missing values in external objects.
        If "warning" (default) shows a warning message and
        interpolates the values. If "raise" raises an error.
        If "ignore" interpolates the values without showing anything.
    split_modules: bool (optional)
        If True, the sketch is parsed to detect model elements in each
        model view, and then translate each view in a separate python 
        file. Setting this argument to True is recommended for large 
        models split in many different views.

    Returns
    -------
    model: a PySD class object
        Elements from the python model are loaded into the PySD class
        and ready to run

    Examples
    --------
    >>> model = read_vensim('../tests/test-models/samples/teacup/teacup.mdl')

    """
    from .py_backend.vensim.vensim2py import translate_vensim


    py_model_file = translate_vensim(mdl_file, split_modules)
    model = load(py_model_file, initialize, missing_values)
    model.mdl_file = mdl_file
    return model


def load(py_model_file, initialize=True, missing_values="warning"):
    """
    Load a python-converted model file.

    Parameters
    ----------
    py_model_file : <string>
        Filename of a model which has already been converted into a
        python format.
    initialize: bool (optional)
        If False, the model will not be initialize when it is loaded.
        Default is True
    missing_values : <string> (optional)
        What to do with missing values in external objects.
        If "warning" (default) shows a warning message and
        interpolates the values. If "raise" raises an error.
        If "ignore" interpolates the values without showing anything.

    Examples
    --------
    >>> model = load('../tests/test-models/samples/teacup/teacup.py')

    """
    from .py_backend import functions

    return functions.Model(py_model_file, initialize, missing_values)
