"""
The AST structures are created with the help of the parsimonious visitors
using the structures dictionary.

"""
import re
from ..structures import abstract_expressions as ae


structures = {
    "reference":  ae.ReferenceStructure,
    "subscripts_ref": ae.SubscriptsReferenceStructure,
    "arithmetic": ae.ArithmeticStructure,
    "logic": ae.LogicStructure,
    "with_lookup": ae.InlineLookupsStructure,
    "call": ae.CallStructure,
    "game": ae.GameStructure,
    "get_xls_lookups": ae.GetLookupsStructure,
    "get_direct_lookups": ae.GetLookupsStructure,
    "get_xls_data": ae.GetDataStructure,
    "get_direct_data": ae.GetDataStructure,
    "get_xls_constants": ae.GetConstantsStructure,
    "get_direct_constants": ae.GetConstantsStructure,
    "initial": ae.InitialStructure,
    "integ": ae.IntegStructure,
    "delay1": lambda x, y: ae.DelayStructure(x, y, x, 1),
    "delay1i": lambda x, y, z: ae.DelayStructure(x, y, z, 1),
    "delay3": lambda x, y: ae.DelayStructure(x, y, x, 3),
    "delay3i": lambda x, y, z: ae.DelayStructure(x, y, z, 3),
    "delay_n": ae.DelayNStructure,
    "delay_fixed": ae.DelayFixedStructure,
    "smooth": lambda x, y: ae.SmoothStructure(x, y, x, 1),
    "smoothi": lambda x, y, z: ae.SmoothStructure(x, y, z, 1),
    "smooth3": lambda x, y: ae.SmoothStructure(x, y, x, 3),
    "smooth3i": lambda x, y, z: ae.SmoothStructure(x, y, z, 3),
    "smooth_n": ae.SmoothNStructure,
    "trend": ae.TrendStructure,
    "forecast": lambda x, y, z: ae.ForecastStructure(x, y, z, 0),
    "sample_if_true": ae.SampleIfTrueStructure,
    "lookup": ae.LookupsStructure,
    "data": ae.DataStructure
}


operators = {
    "logic_ops": [":AND:", ":OR:"],
    "not_ops": [":NOT:"],
    "comp_ops": ["=", "<>", "<=", "<", ">=", ">"],
    "add_ops": ["+", "-"],
    "prod_ops": ["*", "/"],
    "exp_ops": ["^"],
    "pre_ops": ["+", "-"]
}


parsing_ops = {
    key: "|".join(re.escape(x) for x in values)
    for key, values in operators.items()
}
