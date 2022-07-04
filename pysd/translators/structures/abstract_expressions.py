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


class AbstractSyntax:
    """
    Generic class. All Abstract Synax structured are childs of that class.
    Used for typing.
    """
    pass


@dataclass
class ArithmeticStructure(AbstractSyntax):
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
class LogicStructure(AbstractSyntax):
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
class SubscriptsReferenceStructure(AbstractSyntax):
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
class ReferenceStructure(AbstractSyntax):
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
class CallStructure(AbstractSyntax):
    """
    Dataclass for a call structure.

    Parameters
    ----------
    function: str or ReferenceStructure
        The name or the reference of the callable.
    arguments: tuple
        The list of arguments used for calling the function.

    """
    function: Union[str, ReferenceStructure]
    arguments: tuple

    def __str__(self) -> str:  # pragma: no cover
        return "CallStructure:\n\t%s(%s)" % (
            self.function,
            "\n\t\t,".join([
                "\n\t\t" + str(arg).replace("\n", "\n\t\t")
                for arg in self.arguments
            ]))


@dataclass
class GameStructure(AbstractSyntax):
    """
    Dataclass for a game structure.

    Parameters
    ----------
    expression: AST
        The expression inside the game call.

    """
    expression: Union[AbstractSyntax, float]

    def __str__(self) -> str:  # pragma: no cover
        return "GameStructure:\n\t%s" % self.expression


@dataclass
class AllocateAvailableStructure(AbstractSyntax):
    """
    Dataclass for a Allocate Available structure.

    Parameters
    ----------
    request: AbstractSyntax
        The reference to the request variable.
    pp: AbstractSyntax
        The reference to the priority variable.
    avail: AbstractSyntax or float
        The total available supply.

    """
    request: AbstractSyntax
    pp: AbstractSyntax
    avail: Union[AbstractSyntax, float]

    def __str__(self) -> str:  # pragma: no cover
        return "AllocateAvailableStructure:\n\t%s,\n\t%s,\n\t%s" % (
            self.request, self.pp, self.avail
        )


@dataclass
class AllocateByPriorityStructure(AbstractSyntax):
    """
    Dataclass for a Allocate By Priority structure.

    Parameters
    ----------
    request: AbstractSyntax
        The reference to the request variable.
    priority: AbstractSyntax
        The reference to the priority variable.
    size: AbstractSyntax or int
        The size of the last dimension.
    width: AbstractSyntax or float
        The width between priorities.
    supply: AbstractSyntax or float
        The total supply.

    """
    request: AbstractSyntax
    priority: AbstractSyntax
    size: Union[AbstractSyntax, int]
    width: Union[AbstractSyntax, float]
    supply: Union[AbstractSyntax, float]

    def __str__(self) -> str:  # pragma: no cover
        return "AllocateByPriorityStructure:"\
               "\n\t%s,\n\t%s,\n\t%s,\n\t%s,\n\t%s" % (
                   self.request, self.priority, self.size,
                   self.width, self.supply
                )


@dataclass
class InitialStructure(AbstractSyntax):
    """
    Dataclass for a initial structure.

    Parameters
    ----------
    initial: AST
        The expression inside the initial call.

    """
    initial: Union[AbstractSyntax, float]

    def __str__(self) -> str:  # pragma: no cover
        return "InitialStructure:\n\t%s" % (
            self.initial)


@dataclass
class IntegStructure(AbstractSyntax):
    """
    Dataclass for an integ/stock structure.

    Parameters
    ----------
    flow: AST
        The flow of the stock.
    initial: AST
        The initial value of the stock.

    """
    flow: Union[AbstractSyntax, float]
    initial: Union[AbstractSyntax, float]

    def __str__(self) -> str:  # pragma: no cover
        return "IntegStructure:\n\t%s,\n\t%s" % (
            self.flow,
            self.initial)


@dataclass
class DelayStructure(AbstractSyntax):
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
    input: Union[AbstractSyntax, float]
    delay_time: Union[AbstractSyntax, float]
    initial: Union[AbstractSyntax, float]
    order: float

    def __str__(self) -> str:  # pragma: no cover
        return "DelayStructure (order %s):\n\t%s,\n\t%s,\n\t%s" % (
            self.order,
            self.input,
            self.delay_time,
            self.initial)


@dataclass
class DelayNStructure(AbstractSyntax):
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
    input: Union[AbstractSyntax, float]
    delay_time: Union[AbstractSyntax, float]
    initial: Union[AbstractSyntax, float]
    order: Union[AbstractSyntax, float]

    # DELAY N may behave different than other delays when the delay time
    # changes during integration

    def __str__(self) -> str:  # pragma: no cover
        return "DelayNStructure (order %s):\n\t%s,\n\t%s,\n\t%s" % (
            self.order,
            self.input,
            self.delay_time,
            self.initial)


@dataclass
class DelayFixedStructure(AbstractSyntax):
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
    input: Union[AbstractSyntax, float]
    delay_time: Union[AbstractSyntax, float]
    initial: Union[AbstractSyntax, float]

    def __str__(self) -> str:  # pragma: no cover
        return "DelayFixedStructure:\n\t%s,\n\t%s,\n\t%s" % (
            self.input,
            self.delay_time,
            self.initial)


@dataclass
class SmoothStructure(AbstractSyntax):
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
    input: Union[AbstractSyntax, float]
    smooth_time: Union[AbstractSyntax, float]
    initial: Union[AbstractSyntax, float]
    order: float

    def __str__(self) -> str:  # pragma: no cover
        return "SmoothStructure (order %s):\n\t%s,\n\t%s,\n\t%s" % (
            self.order,
            self.input,
            self.smooth_time,
            self.initial)


@dataclass
class SmoothNStructure(AbstractSyntax):
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
    input: Union[AbstractSyntax, float]
    smooth_time: Union[AbstractSyntax, float]
    initial: Union[AbstractSyntax, float]
    order: Union[AbstractSyntax, float]

    # SMOOTH N may behave different than other smooths with RungeKutta
    # integration

    def __str__(self) -> str:  # pragma: no cover
        return "SmoothNStructure (order %s):\n\t%s,\n\t%s,\n\t%s" % (
            self.order,
            self.input,
            self.smooth_time,
            self.initial)


@dataclass
class TrendStructure(AbstractSyntax):
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
    input: Union[AbstractSyntax, float]
    average_time: Union[AbstractSyntax, float]
    initial_trend: Union[AbstractSyntax, float]

    def __str__(self) -> str:  # pragma: no cover
        return "TrendStructure:\n\t%s,\n\t%s,\n\t%s" % (
            self.input,
            self.average_time,
            self.initial)


@dataclass
class ForecastStructure(AbstractSyntax):
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
    input: Union[AbstractSyntax, float]
    average_time: Union[AbstractSyntax, float]
    horizon: Union[AbstractSyntax, float]
    initial_trend: Union[AbstractSyntax, float]

    def __str__(self) -> str:  # pragma: no cover
        return "ForecastStructure:\n\t%s,\n\t%s,\n\t%s,\n\t%s" % (
            self.input,
            self.average_time,
            self.horizon,
            self.initial_trend)


@dataclass
class SampleIfTrueStructure(AbstractSyntax):
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
    condition: Union[AbstractSyntax, float]
    input: Union[AbstractSyntax, float]
    initial: Union[AbstractSyntax, float]

    def __str__(self) -> str:  # pragma: no cover
        return "SampleIfTrueStructure:\n\t%s,\n\t%s,\n\t%s" % (
            self.condition,
            self.input,
            self.initial)


@dataclass
class LookupsStructure(AbstractSyntax):
    """
    Dataclass for a lookup structure.

    Parameters
    ----------
    x: tuple
        The list of the x values of the lookup.
    y: tuple
        The list of the y values of the lookup.
    x_limits: tuple
        The minimum and maximum value of x.
    y_limits: tuple
        The minimum and maximum value of y.
    type: str
        The interpolation method.

    """
    x: tuple
    y: tuple
    x_limits: tuple
    y_limits: tuple
    type: str

    def __str__(self) -> str:  # pragma: no cover
        return "LookupStructure (%s):\n\tx %s = %s\n\ty %s = %s\n" % (
            self.type, self.x_limits, self.x, self.y_limits, self.y
        )


@dataclass
class InlineLookupsStructure(AbstractSyntax):
    """
    Dataclass for an inline lookup structure.

    Parameters
    ----------
    argument: AST
        The argument of the inline lookup.
    lookups: LookupStructure
        The lookups definition.

    """
    argument: Union[AbstractSyntax, float]
    lookups: LookupsStructure

    def __str__(self) -> str:  # pragma: no cover
        return "InlineLookupsStructure:\n\t%s\n\t%s" % (
            str(self.argument).replace("\n", "\n\t"),
            str(self.lookups).replace("\n", "\n\t")
        )


@dataclass
class DataStructure(AbstractSyntax):
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
class GetLookupsStructure(AbstractSyntax):
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
class GetDataStructure(AbstractSyntax):
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
class GetConstantsStructure(AbstractSyntax):
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
