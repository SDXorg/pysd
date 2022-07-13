from typing import Union, List, Iterable
from itertools import chain

from pysd.translators.structures.abstract_model import\
    AbstractComponent, AbstractElement, AbstractModel, AbstractSection

from pysd.translators.structures.abstract_expressions import *


def get_aux_names(entry_ast_node):
    match entry_ast_node:
        case int():
            return []
        case ArithmeticStructure(operators, arguments):
            return list(chain.from_iterable([get_aux_names(argument) for argument in arguments]))
        case ReferenceStructure(reference, subscripts):
            return [entry_ast_node.reference]
        case CallStructure(function, arguments):
            return list(chain.from_iterable([get_aux_names(argument) for argument in arguments]))
        case IntegStructure(flow, initial):
            return get_aux_names(flow) + get_aux_names(initial)
        case InlineLookupsStructure(argument, lookups):
            return get_aux_names(lookups)


def ast_codegen(node) -> str:
    match node:
        case int(x):
            return f"{x}"

        case str(x):
            return x

        case ArithmeticStructure(operators, arguments):
            output_string = ""
            last_argument_index = len(arguments) - 1
            for index, argument in enumerate(arguments):
                output_string += ast_codegen(argument)
                if index < last_argument_index:
                    output_string += " "
                    output_string += operators[index]
                    output_string += " "
            return output_string

        case ReferenceStructure(reference, subscripts):
            return reference

        case CallStructure(function, arguments):
            output_string = ""
            function_name = ast_codegen(function)
            match function_name:
                case "min":
                    function_name = "fmin"
                case "max":
                    function_name = "fmax"
                case "xidz":
                    assert len(arguments) == 3, "number of arguments for xidz must be 3"
                    arg1 = ast_codegen(arguments[0])
                    arg2 = ast_codegen(arguments[1])
                    arg3 = ast_codegen(arguments[2])
                    output_string += f" (fabs({arg2}) <= 1e-6) ? {arg3} : ({arg1}) / ({arg2})"
                    return output_string
                case "zidz":
                    assert len(arguments) == 2, "number of arguments for zidz must be 2"
                    arg1 = ast_codegen(arguments[0])
                    arg2 = ast_codegen(arguments[1])
                    output_string += f" (fabs({arg2}) <= 1e-6) ? 0 : ({arg1}) / ({arg2})"
                    return output_string
                case "ln":
                    function_name = "log"

            output_string += function_name
            output_string += "("
            output_string += ",".join([ast_codegen(argument) for argument in arguments])
            output_string += ")"

            return output_string

        case IntegStructure(flow, initial):
            return ast_codegen(flow)


class AbstractComponentWrapper:
    def __init__(self, component: AbstractComponent):
        self.component = component
        self.ast = component.ast

    def get_required_variable_name(self):
        match self.ast:
            case int():
                return []
            case ArithmeticStructure(ops, args):
                pass


