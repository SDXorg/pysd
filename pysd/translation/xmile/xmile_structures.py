import re
from ..structures import abstract_expressions as ae


structures = {
    "reference":  ae.ReferenceStructure,
    "subscripts_ref": ae.SubscriptsReferenceStructure,
    "arithmetic": ae.ArithmeticStructure,
    "logic": ae.LogicStructure,
    "inline_lookup": ae.InlineLookupsStructure,
    "lookup": ae.LookupsStructure,
    "call": ae.CallStructure,
    "init": ae.InitialStructure,
    "stock": ae.IntegStructure,
    "delay1": {
        2: lambda x, y: ae.DelayStructure(x, y, x, 1),
        3: lambda x, y, z: ae.DelayStructure(x, y, z, 1)
    },
    "delay3": {
        2: lambda x, y: ae.DelayStructure(x, y, x, 3),
        3: lambda x, y, z: ae.DelayStructure(x, y, z, 3),
    },
    "delayn": {
        3: lambda x, y, n: ae.DelayNStructure(x, y, x, n),
        4: lambda x, y, n, z: ae.DelayNStructure(x, y, z, n),
    },
    "smth1": {
        2: lambda x, y: ae.SmoothStructure(x, y, x, 1),
        3: lambda x, y, z: ae.SmoothStructure(x, y, z, 1)
    },
    "smth3": {
        2: lambda x, y: ae.SmoothStructure(x, y, x, 3),
        3: lambda x, y, z: ae.SmoothStructure(x, y, z, 3)
    },
    "smthn": {
        3: lambda x, y, n: ae.SmoothNStructure(x, y, x, n),
        4: lambda x, y, n, z: ae.SmoothNStructure(x, y, z, n)
    },
    "trend": {
        2: lambda x, y: ae.TrendStructure(x, y, 0),
        3: ae.TrendStructure,
    },
    "forcst": {
        3: lambda x, y, z: ae.ForecastStructure(x, y, z, 0),
        4: ae.ForecastStructure
    },
    "safediv": {
        2: lambda x, y: ae.CallStructure(
            ae.ReferenceStructure("zidz"), (x, y)),
        3: lambda x, y, z: ae.CallStructure(
            ae.ReferenceStructure("xidz"), (x, y, z))
    },
    "if_then_else": lambda x, y, z: ae.CallStructure(
            ae.ReferenceStructure("if_then_else"), (x, y, z)),
    "negative": lambda x: ae.ArithmeticStructure(["negative"], (x,)),
    "int": lambda x: ae.CallStructure(
            ae.ReferenceStructure("integer"), (x,))
}


operators = {
    "logic_ops": ["and", "or"],
    "not_ops": ["not"],
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
