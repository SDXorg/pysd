"""
Benchmarking tools for testing and comparing outputs between different files.
Some of these functions are also used for testing.
"""

import os.path
import warnings

import numpy as np
import pandas as pd

from pysd import read_vensim, read_xmile
from ..py_backend.utils import load_outputs, detect_encoding


def runner(model_file, canonical_file=None, transpose=False):
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

    Returns
    -------
    output, canon: (pandas.DataFrame, pandas.DataFrame)
        pandas.DataFrame of the model output and the canonical output.

    """
    directory = os.path.dirname(model_file)

    # load canonical output
    if not canonical_file:
        if os.path.isfile(os.path.join(directory, 'output.csv')):
            canonical_file = os.path.join(directory, 'output.csv')
        elif os.path.isfile(os.path.join(directory, 'output.tab')):
            canonical_file = os.path.join(directory, 'output.tab')
        else:
            raise FileNotFoundError('\nCanonical output file not found.')

    canon = load_outputs(canonical_file,
                         transpose=transpose,
                         encoding=detect_encoding(canonical_file))

    # load model
    if model_file.lower().endswith('.mdl'):
        model = read_vensim(model_file)
    elif model_file.lower().endswith(".xmile"):
        model = read_xmile(model_file)
    else:
        raise ValueError('\nModelfile should be *.mdl or *.xmile')

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
        that two frames are close. Otherwise, it will show a warning
        message. Default is "raise".

    verbose: bool (optional)
        If True, if any column is not close the actual and expected values
        will be printed in the error/warning message with the difference.
        Default is False.

    precision: int (optional)
        Precision to print the numerical values of assertion verbosed message.
        Default is 2.

    kwargs:
        Optional rtol and atol values for assert_allclose.

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
            columns = ["'" + col + "'" for col
                       in actual_cols.difference(expected_cols)]
            columns = ", ".join(columns)
            message += '\nColumns ' + columns\
                       + ' from actual values not found in expected values.'

        if expected_cols.difference(actual_cols):
            columns = ["'" + col + "'" for col
                       in expected_cols.difference(actual_cols)]
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
        'test set and actual set must share a common index' \
        'instead found' + expected.index.values + 'vs' + actual.index.values

    # if for Vensim outputs where constant values are only in the first row
    _remove_constant_nan(expected)
    _remove_constant_nan(actual)

    c = assert_allclose(expected[columns],
                        actual[columns],
                        **kwargs)

    if c.all():
        return

    columns = np.array(columns, dtype=str)[~c.values]

    assertion_details = "\nFollowing columns are not close:\n\t"\
                        + ", ".join(columns)
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
    else:
        warnings.warn(assertion_details)


def assert_allclose(x, y, rtol=1.e-5, atol=1.e-5):
    """
    Asserts if all numeric values from two arrays are close.

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
    return (abs(x - y) <= atol + rtol * abs(y)).all()


def _remove_constant_nan(df):
    """
    Removes nana values in constant value columns produced by Vensim
    """
    nan_cols = np.isnan(df.iloc[1:, :]).all()
    df.loc[:, nan_cols] = df.loc[:, nan_cols].iloc[0].values
