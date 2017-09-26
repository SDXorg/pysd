import numpy as _np
import pandas as _pd
import pyDOE as _pyDOE
import scipy.stats.distributions as _dist
import pysd as _pysd
import os


def create_extreme_conditions_test_matrix(model, filename=None):
    """
    Creates an empty test matrix for evaluating extreme conditions tests.

    After running this function, the user should edit the file and save with
    a separate filename to avoid overwriting.

    Todo: it would be good to make this automatically blank out elements that
    are not influenced by a variable, or *are* the variable,
    and to omit rows that have no consequences because nothing depends on them.
    Also, to omit columns that nothing influences.
    Also, omit table functions
    """
    docs = model.doc()
    docs['bounds'] = docs['Unit'].apply(_get_bounds)
    docs['Min'] = docs['bounds'].apply(lambda x: float(x[0].replace('?', '-inf')))
    docs['Max'] = docs['bounds'].apply(lambda x: float(x[1].replace('?', '+inf')))

    collector = []
    for i, row in docs.iterrows():
        collector.append({'Real Name': row['Real Name'],
                          'Comment': row['Comment'],
                          'Value': row['Min'],
                          })

        collector.append({'Real Name': row['Real Name'],
                          'Comment': row['Comment'],
                          'Value': row['Max'],
                          })

    conditions = _pd.DataFrame(collector)
    results = _pd.DataFrame(columns=list(docs['Real Name']))
    cols = ['Real Name', 'Comment', 'Value'] + sorted(list(docs['Real Name']))
    output = _pd.concat([conditions, results])[cols]

    if filename is None:
        return output
    elif filename.split('.')[-1] in ['xls', 'xlsx']:
        output.to_excel(filename, sheet_name='Extreme Conditions', index=False)
    elif filename.split('.')[-1] == 'csv':
        output.to_csv(filename, index=False)
    elif filename.split('.')[-1] == 'tab':
        output.to_csv(filename, sep='\t')
    else:
        raise ValueError('Unknown file extension %s' % filename.split('.')[-1])


def extreme_conditions_test(mdl_file, matrix=None, excel_file=None, errors='return'):
    """

    Parameters
    ----------
    mdl_file
    matrix
    excel_file
    errors

    Returns
    -------
    Error matrix

    """
    if matrix:
        pass
    elif excel_file:
        matrix = _pd.read_excel(excel_file, index_col=[0, 1, 2])
        matrix = matrix.replace('inf', _np.inf).replace('-inf', _np.inf)
    else:
        raise ValueError('Must supply a test matrix or refer to an external file')

    model = _pysd.read_vensim(mdl_file)
    py_mdl_file = model.py_model_file

    error_list = []
    for row_num, (index, row) in enumerate(matrix.iterrows()):
        try:
            model = _pysd.load(py_mdl_file)
            result = model.run(params={index[0]: index[2]},
                               return_columns=row.index.values,
                               return_timestamps=0).loc[0]

            for col_num, (key, value) in enumerate(row.items()):
                try:
                    if value not in ['-', 'x', 'nan', 'NaN', _np.nan, ''] and result[key] != value:
                        error_list.append({'Condition': '%s = %s' % (index[0], index[2]),
                                           'Variable': repr(key),
                                           'Expected': repr(value),
                                           'Observed': repr(result[key]),
                                           'Test': 'ex.%i.%i' % (row_num, col_num)
                                           })

                except Exception as e:
                    error_list.append({'Condition': '%s = %s' % (index[0], index[2]),
                                       'Variable': repr(key),
                                       'Expected': repr(value),
                                       'Observed': e,
                                       'Test': 'ex.%i.%i' % (row_num, col_num)
                                       })
        except Exception as e:
            error_list.append({'Condition': '%s = %s' % (index[0], index[2]),
                               'Variable': '',
                               'Expected': 'Run Error',
                               'Observed': e,
                               'Test': 'ex.%i.run' % row_num
                               })

    if len(error_list) == 0:
        return None

    if errors == 'return':
        df = _pd.DataFrame(error_list)
        df.set_index('Test', inplace=True)
        errors = df.sort_values(['Condition', 'Variable'])[
            ['Condition', 'Variable', 'Expected', 'Observed']]
        return errors[errors['Expected'] != 'nan']
    elif errors == 'raise':
        raise AssertionError(["When '%(Condition)s', %(Variable)s is %(Observed)s "
                              "instead of %(Expected)s" % e for e in error_list])


def _get_bounds(unit_string):
    parts = unit_string.split('[')
    return parts[-1].strip(']').split(',') if len(parts) > 1 else ['?', '?']


def create_bounds_test_matrix(model, filename=None):
    """
    Creates a test file that can be used to test that all model elements
    remain within their supported ranges.

    This supports replication of vensim's range checking functionality.

    If there are existing bounds listed in the model file, these will be incorporated.

    Parameters
    ----------
    model: PySD Model Object

    filename: string or None
        location where the test matrix may be saved

    Returns
    -------
    output: pandas DataFrame
        if filename == None, returns a DataFrame containing the test matrix
    """

    docs = model.doc()
    docs['bounds'] = docs['Unit'].apply(_get_bounds)
    docs['Min'] = docs['bounds'].apply(lambda x: float(x[0].replace('?', '-inf')))
    docs['Max'] = docs['bounds'].apply(lambda x: float(x[1].replace('?', '+inf')))

    output = docs[['Real Name', 'Comment',
                   'Unit', 'Min', 'Max']].sort_values(by='Real Name')

    if filename is None:
        return output
    elif filename.split('.')[-1] in ['xls', 'xlsx']:
        output.to_excel(filename, sheet_name='Bounds', index=False)
    elif filename.split('.')[-1] == 'csv':
        output.to_csv(filename, index=False)
    elif filename.split('.')[-1] == 'tab':
        output.to_csv(filename, sep='\t')
    else:
        raise ValueError('Unknown file extension %s' % filename.split('.')[-1])


def bounds_test(result, bounds=None, errors='return'):
    """
    Checks that the output of a simulation remains within the specified bounds.
    Also will identify if results are NaN

    Requires a test matrix probably generated by `create_bounds_test_matrix`

    Parameters
    ----------
    result : pandas dataframe
        Probably the output of a PySD run, a pandas DF whose column names are specified
         as rows in the bounds matrix, and whose values will be tested for conformance to
         the bounds.

    bounds : test file name or test matrix

    errors : 'return' or 'raise'
        if 'return' gives a list of errors
        if 'raise' will throw errors now, othewise will return them to calling function

    Raises
    ------
    AssertionError : When a parameter falls outside its support and errors=='raises'

    """

    if isinstance(bounds, _pd.DataFrame):
        bounds = bounds.set_index('Real Name')
    elif isinstance(bounds, str):
        if bounds.split('.')[-1] in ['xls', 'xlsx']:
            bounds = _pd.read_excel(bounds, sheetname='Bounds', index_col='Real Name')
        elif bounds.split('.')[-1] == 'csv':
            bounds = _pd.read_csv(bounds, index_col='Real Name')
        elif bounds.split('.')[-1] == 'tab':
            bounds = _pd.read_csv(bounds, sep='\t', index_col='Real Name')
        else:
            raise ValueError('Unknown file type: bounds')
    else:
        raise ValueError('Unknown type: bounds')

    error_list = []
    for colname in result.columns:
        if colname in bounds.index:
            lower_bound = bounds['Min'].loc[colname]
            below_bounds = result[colname] < lower_bound
            if any(below_bounds):
                error_list.append({
                    'column': colname,
                    'condition': lower_bound,
                    'type': 'below support',
                    'beginning': below_bounds[below_bounds].index[0],
                    'index': below_bounds[below_bounds].index.summary().split(':')[1],
                    'test': 'b.%i.%i' % ((bounds.index == colname).argmax(), 0)
                })

            upper_bound = bounds['Max'].loc[colname]
            above_bounds = result[colname] > upper_bound
            if any(above_bounds):
                error_list.append({
                    'column': colname,
                    'condition': upper_bound,
                    'type': 'above support',
                    'beginning': above_bounds[above_bounds].index[0],
                    'index': above_bounds[above_bounds].index.summary().split(':')[1],
                    'test': 'b.%i.%i' % ((bounds.index == colname).argmax(), 1)
                })

            nans = result[colname].isnull()
            if any(nans):
                error_list.append({
                    'column': colname,
                    'condition': '',
                    'type': 'NaN',
                    'beginning': nans[nans].index[0],
                    'index': nans[nans].index.summary().split(':')[1],
                    'test': 'b.%i.nan' % (bounds.index == colname).argmax()
                })
    if len(error_list) == 0:
        return None

    if errors == 'return':
        df = _pd.DataFrame(error_list)
        return df.sort_values(by='beginning').set_index('test')[
            ['column', 'type', 'condition', 'index']]
    elif errors == 'raise':
        raise AssertionError(["'%(column)s' is %(type) %(bound) at %(index)s" % e
                              for e in error_list])


def sample_pspace(model, param_list=None, bounds=None, samples=100):
    """
    A DataFrame where each row represents a location in the parameter
    space, locations distributed to exercise the full range of values
    that each parameter can take on.

    This is useful for quick and dirty application of tests to a bunch
    of locations in the sample space. Kind-of a fuzz-testing for
    the model.

    Uses latin hypercube sampling, with random values within
    the sample bins. The LHS sampler shuffles the bins each time,
    so a subsequent call will yield a different sample from the
    parameter space.

    When a variable has both upper and lower bounds, use a uniform
    sample between those bounds.

    When a variable has only one bound, use an exponential distribution
    with the scale set to be the difference between the bound and the
    current model value (1 if they are the same)

    When the variable has neither bound, use a normal distribution centered
    on the current model value, with scale equal to the absolute value
    of the model value (1 if that magnitude is 0)

    Parameters
    ----------
    model: pysd.Model object

    param_list: None or list of strings
        The real names of parameters to include in the explored parameter
        space.
        If None, uses all of the constants in the model except TIME STEP,
        INITIAL TIME, etc.

    bounds: DataFrame, string filename, or None
        A range test matrix as used for bounds checking.
        If None, creates one from the model
        These bounds can also place artificial limits on the
        parameter space you want to explore, even if the theoretical
        bounds on the variable are infinite.

    samples: int
        How many samples to include in the iterator?

    Returns
    -------
    lhs : pandas DataFrame
        distribution-weighted latin hypercube samples

    Note
    ----
    Executes the model by 1 time-step to get the current value of parameters.

    """
    if param_list is None:
        doc = model.doc()
        param_list = sorted(list(set(doc[doc['Type'] == 'constant']['Real Name']) -
                            {'FINAL TIME', 'INITIAL TIME', 'TIME STEP'}))

    if isinstance(bounds, _pd.DataFrame):
        bounds = bounds.set_index('Real Name')
    elif bounds is None:
        bounds = create_bounds_test_matrix(model).set_index('Real Name')
    elif isinstance(bounds, str):
        if bounds.split('.')[-1] in ['xls', 'xlsx']:
            bounds = _pd.read_excel(bounds, sheetname='Bounds', index_col='Real Name')
        elif bounds.split('.')[-1] == 'csv':
            bounds = _pd.read_csv(bounds, index_col='Real Name')
        elif bounds.split('.')[-1] == 'tab':
            bounds = _pd.read_csv(bounds, sep='\t', index_col='Real Name')
        else:
            raise ValueError('Unknown file type: bounds')
    else:
        raise ValueError('Unknown type: bounds')

    unit_lhs = _pd.DataFrame(_pyDOE.lhs(n=len(param_list), samples=samples),
                             columns=param_list)

    res = model.run(return_timestamps=[model.components.initial_time()])
    lhs = _pd.DataFrame(index=unit_lhs.index)
    for param in param_list:
        lower, upper = bounds[['Min', 'Max']].loc[param]
        value = res[param].iloc[0]

        if lower == upper:
            lhs[param] = lower

        elif _np.isfinite(lower) and _np.isfinite(upper):  # np.isfinite(0)==True
            scale = upper - lower
            lhs[param] = _dist.uniform(lower, scale).ppf(unit_lhs[param])

        elif _np.isfinite(lower) and _np.isinf(upper):
            if lower == value:
                scale = 1
            else:
                scale = value - lower
            lhs[param] = _dist.expon(lower, scale).ppf(unit_lhs[param])

        elif _np.isinf(lower) and _np.isfinite(upper):  # np.isinf(-np.inf)==True
            if upper == value:
                scale = 1
            else:
                scale = upper - value
            lhs[param] = upper - _dist.expon(0, scale).ppf(unit_lhs[param])

        elif _np.isinf(lower) and _np.isinf(upper):  # np.isinf(-np.inf)==True
            if value == 0:
                scale = 1
            else:
                scale = abs(value)
            lhs[param] = _dist.norm(value, scale).ppf(unit_lhs[param])

        else:
            raise ValueError('Problem with lower: %s or upper: %s bounds' % (lower, upper))

    return lhs


def summarize(model, cases, tests):
    """
    Runs the model at each of the test cases,
    applies each test function in the tests list
    and summarizes the results.

    Parameters
    ----------
    model: pysd Model object
    cases: Pandas Dictionary
        each row is a test condition, column names are variables
    tests: list of functions
        functions should take a model result dataframe and
        return an error dataframe,

    Returns
    -------


    """
    # Todo: This should be easily parallelizable

    synopsis = _pd.DataFrame(columns=['variable', 'type', 'condition', 'cases'])
    for case_num, case in cases.iterrows():
        result = model.run(dict(case))
        error_df = _pd.DataFrame()
        for test_func in tests:
            error_df = error_df.append(test_func(result))

        for name, error in error_df.iterrows():
            if name in synopsis.index:
                synopsis.loc[name]['cases'].append(case_num)
            else:
                synopsis.loc[name, 'variable'] = error['column']
                synopsis.loc[name, 'type'] = error['type']
                synopsis.loc[name, 'condition'] = error['condition']
                synopsis.loc[name, 'cases'] = [case_num]

    return synopsis


def timestep_test(model, threshold=.99, errors='return'):
    """
    Assess that the current timestep is appropriate for the model.
     
    This function runs the model once with its current timestep,
    and then again with the timestep at two random fractions of
    that value.
    It compares the results of these simulations across all model elements
    and ensures that the resulting output is sufficiently similar.

    The function uses two random values to get around the fact that
    some oscillatory modes might be bisectable by the timestep and
    not yield a meaningful difference in simulation. This approach
    makes that type of occurrence very unlikely.
    
    
    Parameters
    ----------
    model
    threshold
    errors

    Returns
    -------

    """
    pass


def lookup_linter(model):
    """

    Linter on lookup tables.

    Checks that tables are:
    - normalized
    - monotonic


    Parameters
    ----------
    model

    Returns
    -------

    References
    ----------
    Sterman 2000, table 14.1

    """
    pass


def behavior_test(feature_file):
    from behave.configuration import Configuration
    import behave.__main__ as bh


    config = Configuration()

    config.steps_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    'gherkin_steps')
    config.paths = [os.path.normpath(feature_file)]

    return_code = bh.run_behave(config)
    return return_code
