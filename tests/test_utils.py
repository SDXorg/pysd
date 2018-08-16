""" Utilities for aiding in testing.
Not tests of utilities... That could be confusing."""

import os.path

import numpy as np
import pandas as pd
from chardet.universaldetector import UniversalDetector

import pysd


def runner(model_file):
    directory = os.path.dirname(model_file)

    # load model
    if model_file.endswith('.mdl'):
        model = pysd.read_vensim(model_file)
    elif model_file.endswith(".xmile"):
        model = pysd.read_xmile(model_file)
    else:
        raise AttributeError('Modelfile should be *.mdl or *.xmile')

    # load canonical output
    try:
        encoding = detect_encoding(directory + '/output.csv')
        canon = pd.read_csv(directory + '/output.csv', encoding=encoding, index_col='Time')
    except IOError:
        try:
            encoding = detect_encoding(directory + '/output.tab')
            canon = pd.read_table(directory + '/output.tab', encoding=encoding, index_col='Time')
        except IOError:
            raise IOError('Canonical output file not found')

    # run model
    output = model.run(return_columns=canon.columns)

    return output, canon


def assert_frames_close(actual, expected, **kwargs):
    """
    Compare DataFrame items by column and
    raise AssertionError if any column is not equal.

    Ordering of columns is unimportant, items are compared only by label.
    NaN and infinite values are supported.

    Parameters
    ----------
    actual: pandas.DataFrame
    expected: pandas.DataFrame
    kwargs:

    Examples
    --------
    >>> assert_frames_close(pd.DataFrame(100, index=range(5), columns=range(3)),
    ...                   pd.DataFrame(100, index=range(5), columns=range(3)))

    >>> assert_frames_close(pd.DataFrame(100, index=range(5), columns=range(3)),
    ...                   pd.DataFrame(110, index=range(5), columns=range(3)),
    ...                   rtol=.2)

    >>> assert_frames_close(pd.DataFrame(100, index=range(5), columns=range(3)),
    ...                   pd.DataFrame(150, index=range(5), columns=range(3)),
    ...                   rtol=.2)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    AssertionError:
    ...

    References
    ----------
    Derived from: http://nbviewer.jupyter.org/gist/jiffyclub/ac2e7506428d5e1d587b
    """

    assert (isinstance(actual, pd.DataFrame) and
            isinstance(expected, pd.DataFrame)), \
        'Inputs must both be pandas DataFrames.'

    assert set(expected.columns) == set(actual.columns), \
        'test set columns must be equal to those in actual/observed set.'

    assert (expected.index.values == actual.index.values).all(), \
        'test set and actual set must share a common index' \
        'instead found' + expected.index.values + 'vs' + actual.index.values

    for col in expected.columns:
        try:
            assert_allclose(expected[col].values,
                            actual[col].values,
                            **kwargs)
        except AssertionError as e:
            raise AssertionError('Column: ' + str(col) + ' is not close.')


def assert_allclose(x, y, rtol=1.e-5, atol=1.e-5):
    print(np.less_equal(abs(x - y), atol + rtol * abs(y)))
    assert np.all(np.less_equal(abs(x - y), atol + rtol * abs(y)))


def detect_encoding(file):
    detector = UniversalDetector()
    for line in open(file, 'rb').readlines():
        print(line)
        detector.feed(line)
        if detector.done: break
    detector.close()
    return detector.result['encoding']
