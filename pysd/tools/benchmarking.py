"""
Benchmarking tools for testing and comparing outputs between different files.
Some of these functions are also used for testing.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pysd import read_vensim, read_xmile, load
from ..py_backend.utils import load_outputs, detect_encoding
from pysd.translators.vensim.vensim_utils import supported_extensions as\
    vensim_extensions
from pysd.translators.xmile.xmile_utils import supported_extensions as\
    xmile_extensions


def runner(model_file, canonical_file=None, transpose=False, data_files=None):
    """
    Translates and runs a model and returns its output and the
    canonical output.

    Parameters
    ----------
    model_file: str
        Name of the original model file. Must be '.mdl' or '.xmile'.

    canonical_file: str or None (optional)
        Canonical output file to read. If None, will search for 'output.csv'
        and 'output.tab' in the model directory. Default is None.

    transpose: bool (optional)
        If True reads transposed canonical file, i.e. one variable per row.
        Default is False.

    data_files: list (optional)
        List of the data files needed to run the model.

    Returns
    -------
    output, canon: (pandas.DataFrame, pandas.DataFrame)
        pandas.DataFrame of the model output and the canonical output.

    """
    if isinstance(model_file, str):
        model_file = Path(model_file)

    directory = model_file.parent

    # load canonical output
    if not canonical_file:
        if directory.joinpath('output.csv').is_file():
            canonical_file = directory.joinpath('output.csv')
        elif directory.joinpath('output.tab').is_file():
            canonical_file = directory.joinpath('output.tab')
        else:
            raise FileNotFoundError("\nCanonical output file not found.")

    canon = load_outputs(canonical_file,
                         transpose=transpose,
                         encoding=detect_encoding(canonical_file))

    # load model
    if model_file.suffix.lower() in vensim_extensions:
        model = read_vensim(model_file, data_files)
    elif model_file.suffix.lower() in xmile_extensions:
        model = read_xmile(model_file, data_files)
    elif model_file.suffix.lower() == ".py":
        model = load(model_file, data_files)
    else:
        raise ValueError(
            "\nThe model file name must be a Vensim"
            f" ({', '.join(vensim_extensions)}), a Xmile "
            f"({', '.join(xmile_extensions)}) or a PySD (.py) model file...")

    # run model and return the result

    return model.run(return_columns=canon.columns), canon


def assert_frames_close(actual, expected, assertion="raise",
                        verbose=False, precision=2, **kwargs):
    """
    Compare DataFrame items by column and
    raise AssertionError if any column is not equal.

    Ordering of columns is unimportant, items are compared only by label.
    NaN and infinite values are supported.

    Parameters
    ----------
    actual: pandas.DataFrame
        Actual value from the model output.

    expected: pandas.DataFrame
        Expected model output.

    assertion: str (optional)
        "raise" if an error should be raised when not able to assert
        that two frames are close. If "warning", it will show a warning
        message. If "return" it will return information. Default is "raise".

    verbose: bool (optional)
        If True, if any column is not close the actual and expected values
        will be printed in the error/warning message with the difference.
        Default is False.

    precision: int (optional)
        Precision to print the numerical values of assertion verbosed message.
        Default is 2.

    kwargs:
        Optional rtol and atol values for assert_allclose.

    Returns
    -------
    (cols, first_false_time, first_false_cols) or None: (set, float, set) or None
        If assertion is 'return', return the sets of the all columns that are
        different. The time when the first difference was found and the
        variables that what different at that time. If assertion is not
        'return' it returns None.

    Examples
    --------
    >>> assert_frames_close(
    ...     pd.DataFrame(100, index=range(5), columns=range(3)),
    ...     pd.DataFrame(100, index=range(5), columns=range(3)))

    >>> assert_frames_close(
    ...     pd.DataFrame(100, index=range(5), columns=range(3)),
    ...     pd.DataFrame(110, index=range(5), columns=range(3)),
    ...     rtol=.2)

    >>> assert_frames_close(
    ...     pd.DataFrame(100, index=range(5), columns=range(3)),
    ...     pd.DataFrame(150, index=range(5), columns=range(3)),
    ...     rtol=.2)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    AssertionError:
    Following columns are not close:
    \t'0'

    >>> assert_frames_close(
    ...     pd.DataFrame(100, index=range(5), columns=range(3)),
    ...     pd.DataFrame(150, index=range(5), columns=range(3)),
    ...     verbose=True, rtol=.2)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    AssertionError:
    Following columns are not close:
    \t'0'
    Column '0' is not close.
    Expected values:
    \t[150, 150, 150, 150, 150]
    Actual values:
    \t[100, 100, 100, 100, 100]
    Difference:
    \t[50, 50, 50, 50, 50]

    >>> assert_frames_close(
    ...     pd.DataFrame(100, index=range(5), columns=range(3)),
    ...     pd.DataFrame(150, index=range(5), columns=range(3)),
    ...     rtol=.2, assertion="warn")
    ...
    UserWarning:
    Following columns are not close:
    \t'0'

    References
    ----------
    Derived from:
        http://nbviewer.jupyter.org/gist/jiffyclub/ac2e7506428d5e1d587b

    """
    if not isinstance(actual, pd.DataFrame)\
       or not isinstance(expected, pd.DataFrame):
        raise TypeError('\nInputs must both be pandas DataFrames.')

    expected_cols, actual_cols = set(expected.columns), set(actual.columns)

    if expected_cols != actual_cols:
        # columns are not equal
        message = ""

        if actual_cols.difference(expected_cols):
            columns = sorted([
                "'" + col + "'" for col
                in actual_cols.difference(expected_cols)])
            columns = ", ".join(columns)
            message += '\nColumns ' + columns\
                       + ' from actual values not found in expected values.'

        if expected_cols.difference(actual_cols):
            columns = sorted([
                "'" + col + "'" for col
                in expected_cols.difference(actual_cols)])
            columns = ", ".join(columns)
            message += '\nColumns ' + columns\
                       + ' from expected values not found in actual values.'

        if assertion == "raise":
            raise ValueError(
                '\nColumns from actual and expected values must be equal.'
                + message)
        else:
            warnings.warn(message)

    columns = list(actual_cols.intersection(expected_cols))

    # TODO let compare dataframes with different timestamps if "warn"
    assert np.all(np.equal(expected.index.values, actual.index.values)), \
        "test set and actual set must share a common index, "\
        "instead found %s vs %s" % (expected.index.values, actual.index.values)

    # if for Vensim outputs where constant values are only in the first row
    _remove_constant_nan(expected)
    _remove_constant_nan(actual)

    c = assert_allclose(expected[columns],
                        actual[columns],
                        **kwargs)

    if c.all().all():
        return (set(), np.nan, set()) if assertion == "return" else None

    # Get the columns that have the first different value, useful for
    # debugging
    false_index = c.apply(
        lambda x: np.where(~x)[0][0] if not x.all() else np.nan)
    index_first_false = int(np.nanmin(false_index))
    time_first_false = c.index[index_first_false]
    variable_first_false = sorted(
        false_index.index[false_index == index_first_false])

    columns = sorted(np.array(columns, dtype=str)[~c.all().values])

    assertion_details = "\nFollowing columns are not close:\n\t"\
                        + ", ".join(columns) + "\n\n"\
                        + f"First false values ({time_first_false}):\n\t"\
                        + ", ".join(variable_first_false)

    if verbose:
        for col in columns:
            assertion_details += '\n\n'\
                + f"Column '{col}' is not close."\
                + '\n\nExpected values:\n\t'\
                + np.array2string(expected[col].values,
                                  precision=precision,
                                  separator=', ')\
                + '\n\nActual values:\n\t'\
                + np.array2string(actual[col].values,
                                  precision=precision,
                                  separator=', ',
                                  suppress_small=True)\
                + '\n\nDifference:\n\t'\
                + np.array2string(expected[col].values-actual[col].values,
                                  precision=precision,
                                  separator=', ',
                                  suppress_small=True)

    if assertion == "raise":
        raise AssertionError(assertion_details)
    elif assertion == "return":
        return (set(columns), time_first_false, set(variable_first_false))
    else:
        warnings.warn(assertion_details)


def assert_allclose(x, y, rtol=1.e-5, atol=1.e-5):
    """
    Asserts if numeric values from two arrays are close.

    Parameters
    ----------
    x: ndarray
        Expected value.
    y: ndarray
        Actual value.
    rtol: float (optional)
        Relative tolerance on the error. Default is 1.e-5.
    atol: float (optional)
        Absolut tolerance on the error. Default is 1.e-5.

    Returns
    -------
    None

    """
    return ((abs(x - y) <= atol + rtol * abs(y)) + x.isna()*y.isna())


def _remove_constant_nan(df):
    """
    Removes nana values in constant value columns produced by Vensim
    """
    nan_cols = np.isnan(df.iloc[1:, :]).all()
    cols = nan_cols[nan_cols].index
    df[cols] = df[cols].apply(lambda x: x.iloc[0])
