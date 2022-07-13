import os
from pathlib import Path
from typing import Union, List, Dict, Set, Iterable

from .ast_walker import *

from pysd.translators.structures.abstract_model import\
    AbstractComponent, AbstractElement, AbstractModel, AbstractSection


class IndentedString:
    def __init__(self, indent_level=0):
        self.indent_level = indent_level
        self.string = " " * 4 * self.indent_level

    def __iadd__(self, other: str):
        prefix = " " * 4 * self.indent_level
        if other != "\n":
            self.string += prefix
        self.string += other
        return self

    def __str__(self):
        return self.string


def name_to_identifier(name: str):
    return name.lower().replace(" ", "_")


class StanModelBuilder:
    def __init__(self, abstract_model: AbstractModel):
        self.abstract_model = abstract_model

    def create_stan_program(self, input_variable_names, output_variable_names, function_name="vensim_func"):
        self.code = IndentedString()


        self.code += StanFunctionBuilder(self.abstract_model).build_function_block(input_variable_names,
                                                                                   output_variable_names, function_name)

        self.code += "data{\n}\n"
        self.code += "transformed data{\n}\n"
        self.code += "parameters{\n}\n"
        self.code += StanTransformedParametersBuilder(self.abstract_model).build_block(input_variable_names, output_variable_names, function_name)
        self.code += "model{\n}\n"

        self.code += "generated quantities{\n}"

        return self.code


class StanTransformedParametersBuilder:
    def __init__(self, abstract_model: AbstractModel):
        self.abstract_model = abstract_model

    def build_block(self, input_variable_names, output_variable_names, function_name):
        self.code = IndentedString()
        self.code += "transformed parameters {\n"
        self.code.indent_level += 1

        argument_variables = []
        for var in input_variable_names:
            match var:
                case str(x):
                    argument_variables.append(x)
                case (str(type), str(var_name)):
                    argument_variables.append(var_name)

        self.code += f"vector[{len(output_variable_names)}] initial_state;\n"
        self.code += f"initial_state = {{{', '.join(output_variable_names)}}};\n"

        self.code += f"array[] vector integrated_result = integrate_ode_rk45({function_name}, initial_state, initial_time, times, {','.join(argument_variables)});\n"
        self.code.indent_level -= 1
        self.code += "}\n"

        return str(self.code)


class StanFunctionBuilder:
    def __init__(self, abstract_model: AbstractModel, function_name: str = "vensim_ode"):

        self.abstract_model = abstract_model
        self.elements = self.abstract_model.sections[0].elements
        self.function_name = function_name



    def create_dependency_graph(self):
        dependency_graph: Dict[str, Set] = {}
        for element in self.elements:
            for component in element.components:
                if element.name not in dependency_graph:
                    dependency_graph[element.name.lower().replace(" ", "_")] = set()

                dependent_aux_names = get_aux_names(component.ast)
                dependency_graph[element.name.lower().replace(" ", "_")].update(dependent_aux_names)

        return dependency_graph

    def print_variable_names(self):
        var_names = []
        max_length = len("original name") + 1
        for element in self.elements:
            var_names.append((element.name, element.name.lower().replace(" ", "_")))
            max_length = max(max_length, len(element.name) + 1)

        print(f"{'original name'.ljust(max_length)}stan variable name")
        print("-" * 10)
        for x in var_names:
            print(f"{x[0].ljust(max_length)}{x[1]}")

    def build_function_block(self, input_variable_names, output_variable_names, function_name="vensim_func"):
        self.code = IndentedString()
        self.code += "functions {\n"
        self.code.indent_level += 1
        dgraph = self.create_dependency_graph()
        eval_order = []

        def recursive_order_search(current, visited):
            if current in visited:
                return
            visited.add(current)
            if current in eval_order:
                return
            for child in dgraph[current]:
                if child == current: continue
                recursive_order_search(child, visited)
            eval_order.append(current)

        for var_name in dgraph.keys():
            recursive_order_search(var_name, set())

        self.elements = sorted(self.elements, key=lambda x: eval_order.index(x.name.lower().replace(" ", "_")))
        self.code += f"vector {function_name}(real time, vector state, "
        argument_strings = []
        argument_variables = []
        for var in input_variable_names:
            match var:
                case str(x):
                    argument_variables.append(x)
                    argument_strings.append("real " + x)
                case (str(type), str(var_name)):
                    argument_variables.append(var_name)
                    argument_strings.append(f"{type} {var_name}")

        self.code += ", ".join(argument_strings)
        self.code += "){"
        self.code += "\n"
        self.code.indent_level += 1

        for index, output_variable_name in enumerate(output_variable_names, 1):
            self.code += f"real {output_variable_name} = state[{index}];\n"

        self.code += "\n"

        for element in self.elements:
            stan_varname = name_to_identifier(element.name)
            if stan_varname in argument_variables:
                continue
            elif stan_varname in output_variable_names:
                stan_varname += "_dydt"
            for component in element.components:
                self.code += f"real {stan_varname} = {ast_codegen(component.ast)};\n"

        self.code += "\n"
        output_variable_names = [name + "_dydt" for name in output_variable_names]
        self.code += f"return {{{', '.join(output_variable_names)}}};\n"
        self.code.indent_level -= 1
        self.code += "}\n"

        self.code.indent_level -= 1
        self.code += "}\n"
        return str(self.code)

    def build_lookups(self):
        pass
