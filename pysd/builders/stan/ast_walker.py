from typing import Union, List, Iterable, Dict, Tuple
from itertools import chain
from dataclasses import dataclass, field
from .utilities import IndentedString
from pysd.translators.structures.abstract_model import\
    AbstractComponent, AbstractElement, AbstractModel, AbstractSection

from pysd.translators.structures.abstract_expressions import *


class BaseNodeWaler:
    def walk(self, ast_node):
        raise NotImplementedError


class AuxNameWalker(BaseNodeWaler):
    def walk(self, ast_node) -> List[str]:
        match ast_node:
            case int():
                return []
            case ArithmeticStructure(operators, arguments):
                return list(chain.from_iterable([self.walk(argument) for argument in arguments]))
            case ReferenceStructure(reference, subscripts):
                return [ast_node.reference]
            case CallStructure(function, arguments):
                return list(chain.from_iterable([self.walk(argument) for argument in arguments]))
            case IntegStructure(flow, initial):
                return self.walk(flow) + self.walk(initial)
            case InlineLookupsStructure(argument, lookups):
                return self.walk(lookups)

@dataclass
class LookupCodegenWalker(BaseNodeWaler):
    generated_lookup_function_names: Dict[Tuple, str] = field(default_factory=dict)
    # This dict holds the generated function names of each individual lookup function.
    # Key is x + y + x_limits + y_limits, value is function name
    n_lookups = 0
    code = IndentedString(indent_level=1)

    @staticmethod
    def get_lookup_keyname(lookup_node: LookupsStructure):
        return lookup_node.x + lookup_node.y + lookup_node.x_limits + lookup_node.y_limits

    def walk(self, ast_node) -> None:
        match ast_node:
            case InlineLookupsStructure(argument, lookups):
                self.walk(lookups)
            case LookupsStructure(x, y, x_limits, y_limits, type):
                assert type == "interpolate", "Type of Lookup must be 'interpolate'"
                identifier_key = LookupCodegenWalker.get_lookup_keyname(ast_node)
                function_name = f"lookupFunc_{self.n_lookups}"
                self.generated_lookup_function_names[identifier_key] = function_name
                self.n_lookups += 1
                self.code += f"real {function_name}(real x){{\n"
                self.code.indent_level += 1
                # Enter function body
                self.code += f"# x {x_limits} = {x}\n"
                self.code += f"# y {y_limits} = {y}\n"
                self.code += "real slope;\n"
                self.code += "real intercept;\n\n"
                n_intervals = len(x)
                for lookup_index in range(n_intervals):
                    if lookup_index == 0:
                        continue
                    if lookup_index == 1:
                        self.code += f"if(x <= {x[lookup_index]})\n"
                    else:
                        self.code += f"else if(x <= {x[lookup_index]})\n"

                    self.code.indent_level += 1
                    # enter conditional body
                    self.code += f"intercept = {y[lookup_index - 1]}\n"
                    self.code += f"slope = ({y[lookup_index]} - {y[lookup_index - 1]}) / ({x[lookup_index]} - {x[lookup_index - 1]});\n"
                    self.code += f"return intercept + slope * (x - {x[lookup_index - 1]});\n"
                    self.code.indent_level -= 1
                    # exit conditional body

                self.code.indent_level -= 1
                # exit function body
                self.code += "}\n\n"

            case _:
                return None


@dataclass
class BlockCodegenWalker(BaseNodeWaler):
    lookup_function_names: Dict[Tuple, str]

    def walk(self, ast_node) -> str:
        match ast_node:
            case int(x):
                return f"{x}"

            case str(x):
                return x

            case ArithmeticStructure(operators, arguments):
                # ArithmeticStructure consists of chained arithmetic expressions.
                # We parse them one by one into a single expression
                output_string = ""
                last_argument_index = len(arguments) - 1
                for index, argument in enumerate(arguments):
                    output_string += self.walk(argument)
                    if index < last_argument_index:
                        output_string += " "
                        output_string += operators[index]
                        output_string += " "
                return output_string

            case ReferenceStructure(reference, subscripts):
                # ReferenceSTructure denotes invoking the value of another variable
                # Subscripts are ignored for now
                return reference

            case CallStructure(function, arguments):
                output_string = ""
                function_name = self.walk(function)
                match function_name:
                    case "min":
                        function_name = "fmin"
                    case "max":
                        function_name = "fmax"
                    case "xidz":
                        assert len(arguments) == 3, "number of arguments for xidz must be 3"
                        arg1 = self.walk(arguments[0])
                        arg2 = self.walk(arguments[1])
                        arg3 = self.walk(arguments[2])
                        output_string += f" (fabs({arg2}) <= 1e-6) ? {arg3} : ({arg1}) / ({arg2})"
                        return output_string
                    case "zidz":
                        assert len(arguments) == 2, "number of arguments for zidz must be 2"
                        arg1 = self.walk(arguments[0])
                        arg2 = self.walk(arguments[1])
                        output_string += f" (fabs({arg2}) <= 1e-6) ? 0 : ({arg1}) / ({arg2})"
                        return output_string
                    case "ln":
                        # natural log in stan is just log
                        function_name = "log"

                output_string += function_name
                output_string += "("
                output_string += ",".join([self.walk(argument) for argument in arguments])
                output_string += ")"

                return output_string

            case IntegStructure(flow, initial):
                return self.walk(flow)

            case InlineLookupsStructure(argument, lookups):
                lookup_func_name = self.lookup_function_names[LookupCodegenWalker.get_lookup_keyname(lookups)]
                return f"{lookup_func_name}({self.walk(argument)})"

@dataclass
class InitialValueCodeGenWalker(BlockCodegenWalker):
    lookup_function_names: Dict[Tuple, str]

    def walk(self, ast_node):
        match ast_node:
            case IntegStructure(flow, initial):
                return self.walk(initial)
            case SmoothStructure(input, smooth_time, initial, order):
                return self.walk(initial)
            case _:
                return super().walk(ast_node)

