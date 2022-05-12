"""
pysd.py

Contains all the code that will be directly accessed by the user in
normal operation.
"""

import sys

from pysd.py_backend.model import Model


if sys.version_info[:2] < (3, 7):  # pragma: no cover
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


def read_xmile(xmile_file, data_files=None, initialize=True,
               missing_values="warning"):
    """
    Construct a model from a Xmile file.

    Parameters
    ----------
    xmile_file:  str or pathlib.Path
        The relative path filename for a raw Xmile file.

    initialize: bool (optional)
        If False, the model will not be initialize when it is loaded.
        Default is True.

    data_files: list or str or None (optional)
        If given the list of files where the necessary data to run the model
        is given. Default is None.

    missing_values : str ("warning", "error", "ignore", "keep") (optional)
        What to do with missing values. If "warning" (default)
        shows a warning message and interpolates the values.
        If "raise" raises an error. If "ignore" interpolates
        the values without showing anything. If "keep" it will keep
        the missing values, this option may cause the integration to
        fail, but it may be used to check the quality of the data.

    Returns
    -------
    model: a PySD class object
        Elements from the python model are loaded into the PySD class
        and ready to run

    Examples
    --------
    >>> model = read_xmile('../tests/test-models/samples/teacup/teacup.xmile')

    """
    from pysd.translators.xmile.xmile_file import XmileFile
    from pysd.builders.python.python_model_builder import ModelBuilder

    # Read and parse Xmile file
    xmile_file_obj = XmileFile(xmile_file)
    xmile_file_obj.parse()

    # get AbstractModel
    abs_model = xmile_file_obj.get_abstract_model()

    # build python file
    py_model_file = ModelBuilder(abs_model).build_model()

    # load python file
    model = load(py_model_file, data_files, initialize, missing_values)
    model.xmile_file = str(xmile_file)

    return model


def read_vensim(mdl_file, data_files=None, initialize=True,
                missing_values="warning", split_views=False,
                encoding=None, **kwargs):
    """
    Construct a model from Vensim `.mdl` file.

    Parameters
    ----------
    mdl_file: str or pathlib.Path
        The relative path filename for a raw Vensim `.mdl` file.

    initialize: bool (optional)
        If False, the model will not be initialize when it is loaded.
        Default is True.

    data_files: list or str or None (optional)
        If given the list of files where the necessary data to run the model
        is given. Default is None.

    missing_values: str ("warning", "error", "ignore", "keep") (optional)
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

    encoding: str or None (optional)
        Encoding of the source model file. If None, the encoding will be
        read from the model, if the encoding is not defined in the model
        file it will be set to 'UTF-8'. Default is None.

    subview_sep: list
        Characters used to separate views and subviews (e.g. [",", "."]).
        If provided, and split_views=True, each submodule will be placed
        inside the directory of the parent view.

    **kwargs: (optional)
        Additional keyword arguments for translation.

    Returns
    -------
    model: a PySD class object
        Elements from the python model are loaded into the PySD class
        and ready to run

    Examples
    --------
    >>> model = read_vensim('../tests/test-models/samples/teacup/teacup.mdl')

    """
    from pysd.translators.vensim.vensim_file import VensimFile
    from pysd.builders.python.python_model_builder import ModelBuilder
    # Read and parse Vensim file
    ven_file = VensimFile(mdl_file, encoding=encoding)
    ven_file.parse()
    if split_views:
        # split variables per views
        subview_sep = kwargs.get("subview_sep", "")
        ven_file.parse_sketch(subview_sep)

    # get AbstractModel
    abs_model = ven_file.get_abstract_model()

    # build python file
    py_model_file = ModelBuilder(abs_model).build_model()

    # load python file
    model = load(py_model_file, data_files, initialize, missing_values)
    model.mdl_file = str(mdl_file)

    return model


def load(py_model_file, data_files=None, initialize=True,
         missing_values="warning"):
    """
    Load a python-converted model file.

    Parameters
    ----------
    py_model_file : str
        Filename of a model which has already been converted into a
        python format.

    initialize: bool (optional)
        If False, the model will not be initialize when it is loaded.
        Default is True.

    data_files: dict or list or str or None
        The dictionary with keys the name of file and variables to
        load the data from there. Or the list of names or name of the
        file to search the data in. Only works for TabData type object
        and it is neccessary to provide it. Default is None.

    missing_values : str ("warning", "error", "ignore", "keep") (optional)
        What to do with missing values. If "warning" (default)
        shows a warning message and interpolates the values.
        If "raise" raises an error. If "ignore" interpolates
        the values without showing anything. If "keep" it will keep
        the missing values, this option may cause the integration to
        fail, but it may be used to check the quality of the data.

    Examples
    --------
    >>> model = load('../tests/test-models/samples/teacup/teacup.py')

    """
    return Model(py_model_file, data_files, initialize, missing_values)
