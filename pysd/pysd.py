"""
pysd.py

Contains all the code that will be directly accessed by the user in
normal operation.
"""

import sys
from .py_backend.statefuls import Model


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


def read_xmile(xmile_file, data_files=None, initialize=True, old=False,
               missing_values="warning"):
    """
    Construct a model from `.xmile` file.

    Parameters
    ----------
    xmile_file:  str or pathlib.Path
        The relative path filename for a raw `.xmile` file.

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
    if old:
        # TODO: remove when this branch is ready to merge
        from .translation.xmile.xmile2py import translate_xmile
        py_model_file = translate_xmile(xmile_file)
    else:
        from pysd.translation.xmile.xmile_file import XmileFile
        from pysd.building.python.python_model_builder import ModelBuilder
        xmile_file_obj = XmileFile(xmile_file)
        xmile_file_obj.parse()

        abs_model = xmile_file_obj.get_abstract_model()
        #print(abs_model.dump(indent=" "))
        py_model_file = ModelBuilder(abs_model).build_model()

    model = load(py_model_file, data_files, initialize, missing_values)
    model.xmile_file = str(xmile_file)
    return model


def read_vensim(mdl_file, data_files=None, initialize=True,
                missing_values="warning", split_views=False,
                encoding=None, old=False, **kwargs):
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

    **kwargs: (optional)
        Additional keyword arguments for translation.
        subview_sep: list
            Characters used to separate views and subviews (e.g. [",", "."]).
            If provided, and split_views=True, each submodule will be placed
            inside the directory of the parent view.


    Returns
    -------
    model: a PySD class object
        Elements from the python model are loaded into the PySD class
        and ready to run

    Examples
    --------
    >>> model = read_vensim('../tests/test-models/samples/teacup/teacup.mdl')

    """
    if old:
        # TODO: remove when this branch is ready to merge
        from .translation.vensim.vensim2py import translate_vensim
        py_model_file = translate_vensim(
            mdl_file, split_views, encoding, **kwargs)
    else:
        from pysd.translation.vensim.vensim_file import VensimFile
        from pysd.building.python.python_model_builder import ModelBuilder
        ven_file = VensimFile(mdl_file)
        ven_file.parse()
        if split_views:
            subview_sep = kwargs.get("subview_sep", "")
            ven_file.parse_sketch(subview_sep)

        abs_model = ven_file.get_abstract_model()
        py_model_file = ModelBuilder(abs_model).build_model()

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

    Examples
    --------
    >>> model = load('../tests/test-models/samples/teacup/teacup.py')

    """
    return Model(py_model_file, data_files, initialize, missing_values)
