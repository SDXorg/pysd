from pathlib import Path

import pandas as pd


def generate(table, columns, output):
    """Generate markdown table."""
    # select only the given columns
    subtable = table[columns]
    # remove the rows where the first column is na
    subtable = subtable[~subtable[columns[0]].isna()]

    if all(subtable[columns[-1]].isna()):
        # if the comments columns (last) is all na, do not save it
        subtable = subtable[columns[:-1]]

    # Place an empty string where na values
    subtable.values[subtable.isna()] = ""

    if len(subtable.index) > 1:
        # Save values only if the table has rows
        print(f"Table generated: {output}")
        subtable.to_csv(output, index=None)


def generate_tables():
    """Generate markdown tables for documentation."""

    tables_dir = Path(__file__).parent / "tables"

    # different tables to load
    tables = {
        "binary": tables_dir / "binary.tab",
        "unary": tables_dir / "unary.tab",
        "functions": tables_dir / "functions.tab",
        "delay_functions": tables_dir / "delay_functions.tab",
        "get_functions": tables_dir / "get_functions.tab"
    }

    # different combinations to generate
    contents = {
        "vensim": [
            "Vensim", "Vensim example", "Abstract Syntax", "Vensim comments"
            ],
        "xmile": [
            "Xmile", "Xmile example", "Abstract Syntax", "Xmile comments"
            ],
        "python": [
            "Abstract Syntax", "Python Translation", "Python comments"
            ]
    }

    # load the tables
    tables = {key: pd.read_table(value) for key, value in tables.items()}

    # generate the tables
    for table, df in tables.items():
        for language, content in contents.items():
            generate(
                df,
                content,
                tables_dir / f"{table}_{language}.csv"
            )

    # transform arithmetic order table
    file = tables_dir / "arithmetic.tab"
    pd.read_table(file).to_csv(file.with_suffix(".csv"), index=None)
