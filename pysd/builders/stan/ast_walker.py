from typing import Union, List, Iterable, Dict, Tuple, Any
from itertools import chain
from dataclasses import dataclass, field
from .utilities import IndentedString
from pysd.translators.structures.abstract_model import (
    AbstractComponent,
    AbstractElement,
    AbstractModel,
    AbstractSection,
)

from pysd.translators.structures.abstract_expressions import *


class BaseNodeWaler:
    def walk(self, ast_node):
        raise NotImplementedError


class AuxNameWalker(BaseNodeWaler):
    def walk(self, ast_node) -> List[str]:
        if isinstance(ast_node, int):
            return []
        elif isinstance(ast_node, float):
            return []
        elif isinstance(ast_node, ArithmeticStructure):
            return list(
                chain.from_iterable(
                    [self.walk(argument) for argument in ast_node.arguments]
                )
            )
        elif isinstance(ast_node, ReferenceStructure):
            return [ast_node.reference]
        elif isinstance(ast_node, CallStructure):
            return list(
                chain.from_iterable(
                    [self.walk(argument) for argument in ast_node.arguments]
                )
            )
        elif isinstance(ast_node, IntegStructure):
            return self.walk(ast_node.flow) + self.walk(ast_node.initial)
        elif isinstance(ast_node, InlineLookupsStructure):
            return self.walk(ast_node.lookups)
        else:
            raise Exception(
                f"AST node of type {ast_node.__class__.__name__} is not supported."
            )


@dataclass
class LookupCodegenWalker(BaseNodeWaler):
    generated_lookup_function_names: Dict[Tuple, str] = field(
        default_factory=dict
    )
    # This dict holds the generated function names of each individual lookup function.
    # Key is x + y + x_limits + y_limits, value is function name
    n_lookups = 0
    code = IndentedString(indent_level=1)

    @staticmethod
    def get_lookup_keyname(lookup_node: LookupsStructure):
        return (
            lookup_node.x
            + lookup_node.y
            + lookup_node.x_limits
            + lookup_node.y_limits
        )

    def walk(self, ast_node) -> None:
        if isinstance(ast_node, InlineLookupsStructure):
            self.walk(ast_node.lookups)
        elif isinstance(ast_node, LookupsStructure):
            assert (
                ast_node.type == "interpolate"
            ), "Type of Lookup must be 'interpolate'"
            identifier_key = LookupCodegenWalker.get_lookup_keyname(ast_node)
            function_name = f"lookupFunc_{self.n_lookups}"
            self.generated_lookup_function_names[
                identifier_key
            ] = function_name
            self.n_lookups += 1
            self.code += f"real {function_name}(real x){{\n"
            self.code.indent_level += 1
            # Enter function body
            self.code += f"# x {ast_node.x_limits} = {ast_node.x}\n"
            self.code += f"# y {ast_node.y_limits} = {ast_node.y}\n"
            self.code += "real slope;\n"
            self.code += "real intercept;\n\n"
            n_intervals = len(ast_node.x)
            for lookup_index in range(n_intervals):
                if lookup_index == 0:
                    continue
                if lookup_index == 1:
                    self.code += f"if(x <= {ast_node.x[lookup_index]})\n"
                else:
                    self.code += f"else if(x <= {ast_node.x[lookup_index]})\n"

                self.code.indent_level += 1
                # enter conditional body
                self.code += f"intercept = {ast_node.y[lookup_index - 1]};\n"
                self.code += f"slope = ({ast_node.y[lookup_index]} - {ast_node.y[lookup_index - 1]}) / ({ast_node.x[lookup_index]} - {ast_node.x[lookup_index - 1]});\n"
                self.code += f"return intercept + slope * (x - {ast_node.x[lookup_index - 1]});\n"
                self.code.indent_level -= 1
                # exit conditional body

            self.code.indent_level -= 1
            # exit function body
            self.code += "}\n\n"
        else:
            return None


@dataclass
class BlockCodegenWalker(BaseNodeWaler):
    lookup_function_names: Dict[Tuple, str]

    def walk(self, ast_node) -> str:

        if isinstance(ast_node, int):
            return f"{ast_node}"
        elif isinstance(ast_node, float):
            return f"{ast_node}"
        elif isinstance(ast_node, str):
            return ast_node
        elif isinstance(ast_node, ArithmeticStructure):
            # ArithmeticStructure consists of chained arithmetic expressions.
            # We parse them one by one into a single expression
            output_string = ""
            last_argument_index = len(ast_node.arguments) - 1
            for index, argument in enumerate(ast_node.arguments):
                output_string += self.walk(argument)
                if index < last_argument_index:
                    output_string += " "
                    output_string += ast_node.operators[index]
                    output_string += " "
            return output_string

        elif isinstance(ast_node, ReferenceStructure):
            # ReferenceSTructure denotes invoking the value of another variable
            # Subscripts are ignored for now
            return ast_node.reference

        elif isinstance(ast_node, CallStructure):
            output_string = ""
            function_name = self.walk(ast_node.function)
            if function_name == "min":
                function_name = "fmin"
            elif function_name == "max":
                function_name = "fmax"
            elif function_name == "xidz":
                assert (
                    len(ast_node.arguments) == 3
                ), "number of arguments for xidz must be 3"
                arg1 = self.walk(ast_node.arguments[0])
                arg2 = self.walk(ast_node.arguments[1])
                arg3 = self.walk(ast_node.arguments[2])
                output_string += (
                    f" (fabs({arg2}) <= 1e-6) ? {arg3} : ({arg1}) / ({arg2})"
                )
                return output_string
            elif function_name == "zidz":
                assert (
                    len(ast_node.arguments) == 2
                ), "number of arguments for zidz must be 2"
                arg1 = self.walk(ast_node.arguments[0])
                arg2 = self.walk(ast_node.arguments[1])
                output_string += (
                    f" (fabs({arg2}) <= 1e-6) ? 0 : ({arg1}) / ({arg2})"
                )
                return output_string
            elif function_name == "ln":
                # natural log in stan is just log
                function_name = "log"

            output_string += function_name
            output_string += "("
            output_string += ",".join(
                [self.walk(argument) for argument in ast_node.arguments]
            )
            output_string += ")"

            return output_string

        elif isinstance(ast_node, IntegStructure):
            return self.walk(ast_node.flow)

        elif isinstance(ast_node, InlineLookupsStructure):
            lookup_func_name = self.lookup_function_names[
                LookupCodegenWalker.get_lookup_keyname(ast_node.lookups)
            ]
            return f"{lookup_func_name}({self.walk(ast_node.argument)})"

        else:
            raise Exception("Got unknown node", ast_node)


@dataclass
class InitialValueCodegenWalker(BlockCodegenWalker):
    variable_ast_dict: Dict[str, AbstractSyntax]
    lookup_function_names: Dict[Tuple, str]

    def walk(self, ast_node):
        if isinstance(ast_node, IntegStructure):
            return self.walk(ast_node.initial)

        elif isinstance(ast_node, SmoothStructure):
            return self.walk(ast_node.initial)

        elif isinstance(ast_node, ReferenceStructure):
            if ast_node.reference in self.variable_ast_dict:
                return self.walk(self.variable_ast_dict[ast_node.reference])
            else:
                return super().walk(ast_node)

        elif isinstance(ast_node, ArithmeticStructure):
            # ArithmeticStructure consists of chained arithmetic expressions.
            # We parse them one by one into a single expression
            output_string = ""
            last_argument_index = len(ast_node.arguments) - 1
            for index, argument in enumerate(ast_node.arguments):
                output_string += self.walk(argument)
                if index < last_argument_index:
                    output_string += " "
                    output_string += ast_node.operators[index]
                    output_string += " "
            return output_string
        else:
            return super().walk(ast_node)


@dataclass
class RNGCodegenWalker(InitialValueCodegenWalker):
    variable_ast_dict: Dict[str, AbstractSyntax]
    lookup_function_names: Dict[Tuple, str]
    total_timestep: int

    def walk(self, ast_node) -> str:
        if isinstance(ast_node, CallStructure):
            function_name = self.walk(ast_node.function)
            if function_name in (
                "random_beta",
                "random_binomial",
                "random_binomial",
                "random_exponential",
                "random_gamma",
                "random_normal",
                "random_poisson",
            ):
                argument_codegen = [
                    self.walk(argument) for argument in ast_node.arguments
                ]
                return self.rng_codegen(function_name, argument_codegen)
            else:
                return super().walk(ast_node)

        elif isinstance(ast_node, IntegStructure):
            raise Exception(
                "RNG function arguments cannot contain stock variables which change with time and thus must be constant!"
            )

        elif isinstance(ast_node, SmoothStructure):
            raise Exception(
                "RNG function arguments cannot contain stock variables which change with time and thus must be constant!"
            )

        elif isinstance(ast_node, ReferenceStructure):
            if ast_node.reference in self.variable_ast_dict:
                return self.walk(ast_node.reference)
            else:
                return super().walk(ast_node)

        elif isinstance(ast_node, ArithmeticStructure):
            # ArithmeticStructure consists of chained arithmetic expressions.
            # We parse them one by one into a single expression
            output_string = ""
            last_argument_index = len(ast_node.arguments) - 1
            for index, argument in enumerate(ast_node.arguments):
                output_string += self.walk(argument)
                if index < last_argument_index:
                    output_string += " "
                    output_string += ast_node.operators[index]
                    output_string += " "
            return output_string

        else:
            return super().walk(ast_node)

    def rng_codegen(self, rng_type: str, arguments: List[Any]):
        if rng_type == "random_normal":
            lower, upper, mean, std, _ = arguments
            return f"fmin(fmax(normal_rng({mean}, {std}), {lower}), {upper})"
        elif rng_type == "random_uniform":
            lower, upper, _ = arguments
            return f"uniform_rng({lower}, {upper})"
        elif rng_type == "random_poisson":
            lower, upper, _lambda, offset, multiply, _ = arguments
            return f"fmin(fmax(fma(poisson_rng({_lambda}), {multiply}, {offset}), {lower}), {upper})"
        else:
            raise Exception(f"RNG function {rng_type} not implemented")
