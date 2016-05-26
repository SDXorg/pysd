import pysd
import numpy.testing as npt
import pandas as pd
import os.path


def runner(model_file):
    directory = os.path.dirname(model_file)

    # load model
    if model_file.endswith('.mdl'):
        model = pysd.read_vensim(model_file)

    elif model_file.endswith("xmile"):
        model = pysd.read_xmile(model_file)

    # load canonical output
    try:
        canon = pd.read_csv(directory + '/output.csv', index_col='Time')
    except IOError:
        try:
            canon = pd.read_table(directory + '/output.tab', index_col='Time')
        except IOError:
            raise IOError('Canonical output file not found')

    # run model
    output = model.run(return_columns=canon.columns,
                       flatten_subscripts=True)

    return output, canon



def assertFramesClose(actual, expected, **kwargs):
    """
    Compare DataFrame items by index and column and
    raise AssertionError if any item is not equal.

    Ordering is unimportant, items are compared only by label.
    NaN and infinite values are supported.

    Parameters
    ----------
    actual: pandas.DataFrame
    expected: pandas.DataFrame
    kwargs:

    Examples
    --------
    >>> assert_frames_close(pd.DataFrame(100, index=range(5), columns=range(3)),
    ...                     pd.DataFrame(100, index=range(5), columns=range(3)))

    >>> assert_frames_close(pd.DataFrame(100, index=range(5), columns=range(3)),
    ...                     pd.DataFrame(110, index=range(5), columns=range(3)),
    ...                     rtol=.2)

    >>> assert_frames_close(pd.DataFrame(100, index=range(5), columns=range(3)),
    ...                     pd.DataFrame(150, index=range(5), columns=range(3)),
    ...                     rtol=.2)  # doctest: +IGNORE_EXCEPTION_DETAIL
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

    for i, exp_row in expected.iterrows():
        assert i in actual.index, 'Expected row {!r} not found.'.format(i)

        act_row = actual.loc[i]

        for j, exp_item in exp_row.iteritems():
            assert j in act_row.index, \
                'Expected column {!r} not found.'.format(j)

            act_item = act_row[j]

            try:
                npt.assert_allclose(act_item, exp_item, **kwargs)
            except AssertionError as e:
                raise AssertionError(
                    e.message + '\n\nColumn: {!r}\nRow: {!r}'.format(j, i))