from dataclasses import dataclass
from typing import Union


@dataclass
class ArithmeticStructure:
    operators: str
    arguments: tuple

    def __str__(self) -> str:
        return "ArithmeticStructure:\n\t %s %s" % (
            self.operators, self.arguments)


@dataclass
class LogicStructure:
    operators: str
    arguments: tuple

    def __str__(self) -> str:
        return "LogicStructure:\n\t %s %s" % (
            self.operators, self.arguments)


@dataclass
class SubscriptsReferenceStructure:
    subscripts: tuple

    def __str__(self) -> str:
        return "SubscriptReferenceStructure:\n\t %s" % self.subscripts


@dataclass
class ReferenceStructure:
    reference: str
    subscripts: Union[SubscriptsReferenceStructure, None] = None

    def __str__(self) -> str:
        return "ReferenceStructure:\n\t %s%s" % (
            self.reference,
            "\n\t" + str(self.subscripts or "").replace("\n", "\n\t"))


@dataclass
class CallStructure:
    function: Union[str, object]
    arguments: tuple

    def __str__(self) -> str:
        return "CallStructure:\n\t%s(%s)" % (
            self.function,
            "\n\t\t,".join([
                "\n\t\t" + str(arg).replace("\n", "\n\t\t")
                for arg in self.arguments
            ]))


@dataclass
class GameStructure:
    expression: object

    def __str__(self) -> str:
        return "GameStructure:\n\t%s" % self.expression


@dataclass
class InitialStructure:
    initial: object

    def __str__(self) -> str:
        return "InitialStructure:\n\t%s" % (
            self.initial)


@dataclass
class IntegStructure:
    flow: object
    initial: object

    def __str__(self) -> str:
        return "IntegStructure:\n\t%s,\n\t%s" % (
            self.flow,
            self.initial)


@dataclass
class DelayStructure:
    input: object
    delay_time: object
    initial: object
    order: float

    def __str__(self) -> str:
        return "DelayStructure (order %s):\n\t%s,\n\t%s,\n\t%s" % (
            self.order,
            self.input,
            self.delay_time,
            self.initial)


@dataclass
class DelayNStructure:
    input: object
    delay_time: object
    initial: object
    order: object

    # DELAY N may behave different than other delays when the delay time
    # changes during integration

    def __str__(self) -> str:
        return "DelayNStructure (order %s):\n\t%s,\n\t%s,\n\t%s" % (
            self.order,
            self.input,
            self.delay_time,
            self.initial)


@dataclass
class DelayFixedStructure:
    input: object
    delay_time: object
    initial: object

    def __str__(self) -> str:
        return "DelayFixedStructure:\n\t%s,\n\t%s,\n\t%s" % (
            self.input,
            self.delay_time,
            self.initial)


@dataclass
class SmoothStructure:
    input: object
    smooth_time: object
    initial: object
    order: float

    def __str__(self) -> str:
        return "SmoothStructure (order %s):\n\t%s,\n\t%s,\n\t%s" % (
            self.order,
            self.input,
            self.smooth_time,
            self.initial)


@dataclass
class SmoothNStructure:
    input: object
    smooth_time: object
    initial: object
    order: object

    # SMOOTH N may behave different than other smooths with RungeKutta
    # integration

    def __str__(self) -> str:
        return "SmoothNStructure (order %s):\n\t%s,\n\t%s,\n\t%s" % (
            self.order,
            self.input,
            self.smooth_time,
            self.initial)


@dataclass
class TrendStructure:
    input: object
    average_time: object
    initial: object

    def __str__(self) -> str:
        return "TrendStructure:\n\t%s,\n\t%s,\n\t%s" % (
            self.input,
            self.average_time,
            self.initial)


@dataclass
class ForecastStructure:
    input: object
    average_time: object
    horizon: object

    def __str__(self) -> str:
        return "ForecastStructure:\n\t%s,\n\t%s,\n\t%s" % (
            self.input,
            self.average_time,
            self.horizon)


@dataclass
class SampleIfTrueStructure:
    condition: object
    input: object
    initial: object

    def __str__(self) -> str:
        return "SampleIfTrueStructure:\n\t%s,\n\t%s,\n\t%s" % (
            self.condition,
            self.input,
            self.initial)


@dataclass
class LookupsStructure:
    x: tuple
    y: tuple
    x_range: tuple
    y_range: tuple

    def __str__(self) -> str:
        return "LookupStructure:\n\tx %s = %s\n\ty %s = %s\n" % (
            self.x_range, self.x, self.y_range, self.y
        )


@dataclass
class InlineLookupsStructure:
    argument: None
    lookups: LookupsStructure

    def __str__(self) -> str:
        return "InlineLookupsStructure:\n\t%s\n\t%s" % (
            str(self.argument).replace("\n", "\n\t"),
            str(self.lookups).replace("\n", "\n\t")
        )


@dataclass
class DataStructure:
    pass

    def __str__(self) -> str:
        return "DataStructure"


@dataclass
class GetLookupsStructure:
    file: str
    tab: str
    x_row_or_col: str
    cell: str

    def __str__(self) -> str:
        return "GetLookupStructure:\n\t'%s', '%s', '%s', '%s'\n" % (
            self.file, self.tab, self.x_row_or_col, self.cell
        )


@dataclass
class GetDataStructure:
    file: str
    tab: str
    time_row_or_col: str
    cell: str

    def __str__(self) -> str:
        return "GetDataStructure:\n\t'%s', '%s', '%s', '%s'\n" % (
            self.file, self.tab, self.time_row_or_col, self.cell
        )


@dataclass
class GetConstantsStructure:
    file: str
    tab: str
    cell: str

    def __str__(self) -> str:
        return "GetConstantsStructure:\n\t'%s', '%s', '%s'\n" % (
            self.file, self.tab, self.cell
        )
