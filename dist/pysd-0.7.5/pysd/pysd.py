"""
pysd.py

Contains all the code that will be directly accessed by the user in normal operation.
Also contains some private members to facilitate integration, setup, etc.

History
--------
August 15, 2014: created
June 6 2015: Major updates - version 0.2.5
Jan 2016: Rework to handle subscripts
May 2016: Updates to handle grammar refactoring
Sept 2016: Major refactor, putting most internal code into the Model and Macro objects
"""

from . import functions


def read_vensim(mdl_file):
    """
    Construct a model from Vensim `.mdl` file.

    Parameters
    ----------
    mdl_file : <string>
        The relative path filename for a raw Vensim `.mdl` file

    Returns
    -------
    model: a PySD class object
        Elements from the python model are loaded into the PySD class and ready to run

    Examples
    --------
    >>> model = read_vensim('../tests/test-models/samples/teacup/teacup.mdl')
    """
    from .vensim2py import translate_vensim
    py_model_file = translate_vensim(mdl_file)
    model = functions.Model(py_model_file)
    model.mdl_file = mdl_file
    return model


def load(py_model_file):
    """
    Load a python-converted model file.

    Parameters
    ----------
    py_model_file : <string>
        Filename of a model which has already been converted into a
         python format.

    Examples
    --------
    >>> model = load('../tests/test-models/samples/teacup/teacup.py')
    """
    return functions.Model(py_model_file)











