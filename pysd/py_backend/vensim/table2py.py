import pandas as pd
import warnings
from ...pysd import read_vensim


def read_tabular(table_file, sheetname='Sheet1'):
    """
    Reads a vensim syntax model which has been formatted as a table.

    This is useful in contexts where model building is performed
    without the aid of Vensim.

    Parameters
    ----------
    table_file: .csv, .tab or .xls(x) file

    Table should have columns titled as in the table below


    | Variable | Equation | Units | Min | Max | Comment          |
    | :------- | :------- | :---- | :-- | :-- | :--------------- |
    | Age      | 5        | Yrs   | 0   | inf | How old are you? |
    | ...      | ...      | ...   | ... | ... | ...              |

    sheetname: basestring
        if the model is specified in an excel file, what sheet?

    Returns
    -------
    PySD Model Object

    Notes
    -----
    Creates an intermediate file in vensim `.mdl` syntax, just so that
    the existing vensim parsing machinery can be used.

    """

    if isinstance(table_file, str):
        extension = table_file.split('.')[-1]
        if extension in ['xls', 'xlsx']:
            table = pd.read_excel(table_file, sheetname=sheetname)
        elif extension == 'csv':
            table = pd.read_csv(table_file)
        elif extension == 'tab':
            table = pd.read_csv(table_file, sep='\t')
        else:
            raise ValueError('Unknown file or table type')
    else:
        raise ValueError('Unknown file or table type')

    if not set(table.columns).issuperset({'Variable', 'Equation'}):
        raise ValueError('Table must contain at least columns "Variable" and "Equation"')

    if "Units" not in set(table.columns):
        warnings.warn('Column for "Units" not found', RuntimeWarning, stacklevel=2)
        table['Units'] = ''

    if "Min" not in set(table.columns):
        warnings.warn('Column for "Min" not found', RuntimeWarning, stacklevel=2)
        table['Min'] = ''

    if "Max" not in set(table.columns):
        warnings.warn('Column for "Max" not found', RuntimeWarning, stacklevel=2)
        table['Max'] = ''

    mdl_file = table_file.replace(extension, 'mdl')

    with open(mdl_file, 'w') as outfile:
        for element in table.to_dict(orient='records'):
            outfile.write(
                "%(Variable)s = \n"
                "\t %(Equation)s \n"
                "\t~\t %(Units)s [%(Min)s, %(Max)s] \n"
                "\t~\t %(Comment)s \n\t|\n\n" % element
            )

        outfile.write(r'\\\---/// Sketch information - this is where sketch stuff would go.')

    return read_vensim(mdl_file)