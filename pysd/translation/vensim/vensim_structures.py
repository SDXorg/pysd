import re
from ..structures import components as cs


structures = {
    "reference":  cs.ReferenceStructure,
    "subscripts_ref": cs.SubscriptsReferenceStructure,
    "arithmetic": cs.ArithmeticStructure,
    "logic": cs.LogicStructure,
    "with_lookup": cs.InlineLookupsStructure,
    "call": cs.CallStructure,
    "game": cs.GameStructure,
    "get_xls_lookups": cs.GetLookupsStructure,
    "get_direct_lookups": cs.GetLookupsStructure,
    "get_xls_data": cs.GetDataStructure,
    "get_direct_data": cs.GetDataStructure,
    "get_xls_constants": cs.GetConstantsStructure,
    "get_direct_constants": cs.GetConstantsStructure,
    "initial": cs.InitialStructure,
    "integ": cs.IntegStructure,
    "delay1": lambda x, y: cs.DelayStructure(x, y, x, 1),
    "delay1i": lambda x, y, z: cs.DelayStructure(x, y, z, 1),
    "delay3": lambda x, y: cs.DelayStructure(x, y, x, 3),
    "delay3i": lambda x, y, z: cs.DelayStructure(x, y, z, 3),
    "delay_n": cs.DelayNStructure,
    "delay_fixed": cs.DelayFixedStructure,
    "smooth": lambda x, y: cs.SmoothStructure(x, y, x, 1),
    "smoothi": lambda x, y, z: cs.SmoothStructure(x, y, z, 1),
    "smooth3": lambda x, y: cs.SmoothStructure(x, y, x, 3),
    "smooth3i": lambda x, y, z: cs.SmoothStructure(x, y, z, 3),
    "smooth_n": cs.SmoothNStructure,
    "trend": cs.TrendStructure,
    "forecast": cs.ForecastStructure,
    "sample_if_true": cs.SampleIfTrueStructure,
    "lookup": cs.LookupsStructure,
    "data": cs.DataStructure
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
