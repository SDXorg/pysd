"""
The following abstract structures are used to build the Abstract Syntax
Tree (AST). In general, there is no hierarchy between them. For example,
an ArithmeticStructure can contain a CallStructure which at the same time
contains another ArithmeticStructure. However, some of them could not be
inside another structures due to the restrictions of the source languages.
For example, the GetConstantsStructure cannot be a part of another structure
because it has to appear after the '=' sign in Vensim and not be followed by
anything else.
"""
from dataclasses import dataclass
from typing import Union


@dataclass
class ArithmeticStructure:
    """
    Dataclass for an arithmetic structure.

    Parameters
    ----------
    operators: list
        List of operators applied between the arguments
    arguments: list
        The arguments of the arithmetics operations.

    """
    operators: list
    arguments: list

    def __str__(self) -> str:  # pragma: no cover
        return "ArithmeticStructure:\n\t %s %s" % (
            self.operators, self.arguments)


@dataclass
class LogicStructure:
    """
    Dataclass for a logic structure.

    Parameters
    ----------
    operators: list
        List of operators applied between the arguments
    arguments: list
        The arguments of the logic operations.

    """
    operators: list
    arguments: list

    def __str__(self) -> str:  # pragma: no cover
        return "LogicStructure:\n\t %s %s" % (
            self.operators, self.arguments)


@dataclass
class SubscriptsReferenceStructure:
    """
    Dataclass for a subscript reference structure.

    Parameters
    ----------
    subscripts: tuple
        The list of subscripts referenced.

    """
    subscripts: tuple

    def __str__(self) -> str:  # pragma: no cover
        return "SubscriptReferenceStructure:\n\t %s" % self.subscripts


@dataclass
class ReferenceStructure:
    """
    Dataclass for an element reference structure.

    Parameters
    ----------
    reference: str
        The name of the referenced element.
    subscripts: SubscriptsReferenceStructure or None
        The subscrips used in the reference.

    """
    reference: str
    subscripts: Union[SubscriptsReferenceStructure, None] = None

    def __str__(self) -> str:  # pragma: no cover
        return "ReferenceStructure:\n\t %s%s" % (
            self.reference,
            "\n\t" + str(self.subscripts or "").replace("\n", "\n\t"))


@dataclass
class CallStructure:
    """
    Dataclass for a call structure.

    Parameters
    ----------
    function: str or ReferenceStructure
        The name or the reference of the callable.
    arguments: tuple
        The list of arguments used for calling the function.

    """
    function: Union[str, object]
    arguments: tuple

    def __str__(self) -> str:  # pragma: no cover
        return "CallStructure:\n\t%s(%s)" % (
            self.function,
            "\n\t\t,".join([
                "\n\t\t" + str(arg).replace("\n", "\n\t\t")
                for arg in self.arguments
            ]))


@dataclass
class GameStructure:
    """
    Dataclass for a game structure.

    Parameters
    ----------
    expression: AST
        The expression inside the game call.

    """
    expression: object

    def __str__(self) -> str:  # pragma: no cover
        return "GameStructure:\n\t%s" % self.expression


@dataclass
class InitialStructure:
    """
    Dataclass for a initial structure.

    Parameters
    ----------
    initial: AST
        The expression inside the initial call.

    """
    initial: object

    def __str__(self) -> str:  # pragma: no cover
        return "InitialStructure:\n\t%s" % (
            self.initial)


@dataclass
class IntegStructure:
    """
    Dataclass for an integ/stock structure.

    Parameters
    ----------
    flow: AST
        The flow of the stock.
    initial: AST
        The initial value of the stock.

    """
    flow: object
    initial: object

    def __str__(self) -> str:  # pragma: no cover
        return "IntegStructure:\n\t%s,\n\t%s" % (
            self.flow,
            self.initial)


@dataclass
class DelayStructure:
    """
    Dataclass for a delay structure.

    Parameters
    ----------
    input: AST
        The input of the delay.
    delay_time: AST
        The delay time value of the delay.
    initial: AST
        The initial value of the delay.
    order: float
        The order of the delay.

    """
    input: object
    delay_time: object
    initial: object
    order: float

    def __str__(self) -> str:  # pragma: no cover
        return "DelayStructure (order %s):\n\t%s,\n\t%s,\n\t%s" % (
            self.order,
            self.input,
            self.delay_time,
            self.initial)


@dataclass
class DelayNStructure:
    """
    Dataclass for a delay n structure.

    Parameters
    ----------
    input: AST
        The input of the delay.
    delay_time: AST
        The delay time value of the delay.
    initial: AST
        The initial value of the delay.
    order: float
        The order of the delay.

    """
    input: object
    delay_time: object
    initial: object
    order: object

    # DELAY N may behave different than other delays when the delay time
    # changes during integration

    def __str__(self) -> str:  # pragma: no cover
        return "DelayNStructure (order %s):\n\t%s,\n\t%s,\n\t%s" % (
            self.order,
            self.input,
            self.delay_time,
            self.initial)


@dataclass
class DelayFixedStructure:
    """
    Dataclass for a delay fixed structure.

    Parameters
    ----------
    input: AST
        The input of the delay.
    delay_time: AST
        The delay time value of the delay.
    initial: AST
        The initial value of the delay.

    """
    input: object
    delay_time: object
    initial: object

    def __str__(self) -> str:  # pragma: no cover
        return "DelayFixedStructure:\n\t%s,\n\t%s,\n\t%s" % (
            self.input,
            self.delay_time,
            self.initial)


@dataclass
class SmoothStructure:
    """
    Dataclass for a smooth structure.

    Parameters
    ----------
    input: AST
        The input of the smooth.
    delay_time: AST
        The smooth time value of the smooth.
    initial: AST
        The initial value of the smooth.
    order: float
        The order of the smooth.

    """
    input: object
    smooth_time: object
    initial: object
    order: float

    def __str__(self) -> str:  # pragma: no cover
        return "SmoothStructure (order %s):\n\t%s,\n\t%s,\n\t%s" % (
            self.order,
            self.input,
            self.smooth_time,
            self.initial)


@dataclass
class SmoothNStructure:
    """
    Dataclass for a smooth n structure.

    Parameters
    ----------
    input: AST
        The input of the smooth.
    delay_time: AST
        The smooth time value of the smooth.
    initial: AST
        The initial value of the smooth.
    order: float
        The order of the smooth.

    """
    input: object
    smooth_time: object
    initial: object
    order: object

    # SMOOTH N may behave different than other smooths with RungeKutta
    # integration

    def __str__(self) -> str:  # pragma: no cover
        return "SmoothNStructure (order %s):\n\t%s,\n\t%s,\n\t%s" % (
            self.order,
            self.input,
            self.smooth_time,
            self.initial)


@dataclass
class TrendStructure:
    """
    Dataclass for a trend structure.

    Parameters
    ----------
    input: AST
        The input of the trend.
    average_time: AST
        The average time value of the trend.
    initial_trend: AST
        The initial trend value of the trend.

    """
    input: object
    average_time: object
    initial_trend: object

    def __str__(self) -> str:  # pragma: no cover
        return "TrendStructure:\n\t%s,\n\t%s,\n\t%s" % (
            self.input,
            self.average_time,
            self.initial)


@dataclass
class ForecastStructure:
    """
    Dataclass for a forecast structure.

    Parameters
    ----------
    input: AST
        The input of the forecast.
    averae_time: AST
        The average time value of the forecast.
    horizon: float
        The horizon value of the forecast.
    initial_trend: AST
        The initial trend value of the forecast.

    """
    input: object
    average_time: object
    horizon: object
    initial_trend: object

    def __str__(self) -> str:  # pragma: no cover
        return "ForecastStructure:\n\t%s,\n\t%s,\n\t%s,\n\t%s" % (
            self.input,
            self.average_time,
            self.horizon,
            self.initial_trend)


@dataclass
class SampleIfTrueStructure:
    """
    Dataclass for a sample if true structure.

    Parameters
    ----------
    condition: AST
        The condition of the sample if true
    input: AST
        The input of the sample if true.
    initial: AST
        The initial value of the sample if true.

    """
    condition: object
    input: object
    initial: object

    def __str__(self) -> str:  # pragma: no cover
        return "SampleIfTrueStructure:\n\t%s,\n\t%s,\n\t%s" % (
            self.condition,
            self.input,
            self.initial)


@dataclass
class LookupsStructure:
    """
    Dataclass for a lookup structure.

    Parameters
    ----------
    x: tuple
        The list of the x values of the lookup.
    y: tuple
        The list of the y values of the lookup.
    x_range: tuple
        The minimum and maximum value of x.
    y_range: tuple
        The minimum and maximum value of y.
    type: str
        The interpolation method.

    """
    x: tuple
    y: tuple
    x_range: tuple
    y_range: tuple
    type: str

    def __str__(self) -> str:  # pragma: no cover
        return "LookupStructure (%s):\n\tx %s = %s\n\ty %s = %s\n" % (
            self.type, self.x_range, self.x, self.y_range, self.y
        )


@dataclass
class InlineLookupsStructure:
    """
    Dataclass for an inline lookup structure.

    Parameters
    ----------
    argument: AST
        The argument of the inline lookup.
    lookups: LookupStructure
        The lookups definition.

    """
    argument: object
    lookups: LookupsStructure

    def __str__(self) -> str:  # pragma: no cover
        return "InlineLookupsStructure:\n\t%s\n\t%s" % (
            str(self.argument).replace("\n", "\n\t"),
            str(self.lookups).replace("\n", "\n\t")
        )


@dataclass
class DataStructure:
    """
    Dataclass for an empty data structure.

    Parameters
    ----------
    None

    """
    pass

    def __str__(self) -> str:  # pragma: no cover
        return "DataStructure"


@dataclass
class GetLookupsStructure:
    """
    Dataclass for a get lookups structure.

    Parameters
    ----------
    file: str
        The file path where the data is.
    tab: str
        The sheetname where the data is.
    x_row_or_col: str
        The pointer to the cell or cellrange name that defines the
        interpolation series data.
    cell: str
        The pointer to the cell or the cellrange name that defines the data.

    """
    file: str
    tab: str
    x_row_or_col: str
    cell: str

    def __str__(self) -> str:  # pragma: no cover
        return "GetLookupStructure:\n\t'%s', '%s', '%s', '%s'\n" % (
            self.file, self.tab, self.x_row_or_col, self.cell
        )


@dataclass
class GetDataStructure:
    """
    Dataclass for a get lookups structure.

    Parameters
    ----------
    file: str
        The file path where the data is.
    tab: str
        The sheetname where the data is.
    time_row_or_col: str
        The pointer to the cell or cellrange name that defines the
        interpolation time series data.
    cell: str
        The pointer to the cell or the cellrange name that defines the data.

    """
    file: str
    tab: str
    time_row_or_col: str
    cell: str

    def __str__(self) -> str:  # pragma: no cover
        return "GetDataStructure:\n\t'%s', '%s', '%s', '%s'\n" % (
            self.file, self.tab, self.time_row_or_col, self.cell
        )


@dataclass
class GetConstantsStructure:
    """
    Dataclass for a get lookups structure.

    Parameters
    ----------
    file: str
        The file path where the data is.
    tab: str
        The sheetname where the data is.
    cell: str
        The pointer to the cell or the cellrange name that defines the data.

    """
    file: str
    tab: str
    cell: str

    def __str__(self) -> str:  # pragma: no cover
        return "GetConstantsStructure:\n\t'%s', '%s', '%s'\n" % (
            self.file, self.tab, self.cell
        )
