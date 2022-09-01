import os
from pathlib import Path
from typing import Union, List, Dict, Set, Sequence, Iterable
from numbers import Number
from .ast_walker import *
from .utilities import *
from pysd.translators.structures.abstract_model import (
    AbstractComponent,
    AbstractElement,
    AbstractModel,
    AbstractSection,
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .stan_model import SamplingStatement


class StanTransformedParametersBuilder:
    def __init__(self, abstract_model: AbstractModel):
        self.abstract_model = abstract_model

    def build_block(
        self,
        predictor_variable_names,
        outcome_variable_names,
        lookup_function_dict,
        function_name,
    ):
        self.code = IndentedString()
        self.code += "transformed parameters {\n"
        self.code.indent_level += 1

        argument_variables = []
        for var in predictor_variable_names:
            if isinstance(var, str):
                argument_variables.append(var)
            elif isinstance(var, tuple):
                var_name = var[1]
                argument_variables.append(var_name)

        variable_ast_dict: Dict[str, AbstractSyntax] = {}
        for element in self.abstract_model.sections[0].elements:
            stan_varname = vensim_name_to_identifier(element.name)
            variable_ast_dict[stan_varname] = element.components[0].ast

        self.code += "# Initial ODE values\n"
        for outcome_variable_name in outcome_variable_names:
            for element in self.abstract_model.sections[0].elements:
                if (
                    vensim_name_to_identifier(element.name)
                    == outcome_variable_name
                ):
                    component = element.components[0]
                    assert isinstance(
                        component.ast, IntegStructure
                    ), "Output variable component must be an INTEG."
                    self.code += f"real {outcome_variable_name}_initial = {InitialValueCodegenWalker(lookup_function_dict, variable_ast_dict).walk(component.ast)};\n"
                    break

        self.code += "\n"
        self.code += f"vector[{len(outcome_variable_names)}] initial_outcome;  # Initial ODE state vector\n"
        for index, name in enumerate(outcome_variable_names, 1):
            self.code += f"initial_outcome[{index}] = {name}_initial;\n"

        self.code += "\n"

        self.code += f"vector[{len(outcome_variable_names)}] integrated_result[n_t] = ode_rk45({function_name}, initial_outcome, initial_time, times, {', '.join(argument_variables)});\n"
        self.code.indent_level -= 1
        self.code += "}\n"

        return str(self.code)


class StanParametersBuilder:
    def __init__(self, sampling_statements: Iterable["SamplingStatement"]):
        self.sampling_statements = sampling_statements

    def build_block(self):
        code = IndentedString()
        code += "parameters{\n"
        code.indent_level += 1  # Enter parameters block

        for statement in self.sampling_statements:
            code += f"real {statement.lhs_name};\n"

        code.indent_level -= 1  # Exit parameters block
        code += "}\n"
        return code.string


class StanDataBuilder:
    def __init__(self):
        pass

    def build_block(self):
        code = IndentedString()
        code += "data{\n"
        code.indent_level += 1
        code += "int <lower = 1> n_obs_state;\n"
        code.indent_level -= 1
        code += "}\n"
        return code.string


class StanTransformedDataBuilder:
    def __init__(self, initial_time, integration_times: Iterable[Number]):
        self.initial_time = initial_time
        self.integration_times = integration_times

    def build_block(self) -> str:
        n_t = len(self.integration_times)
        code = IndentedString()
        code += "transformed data{\n"
        code.indent_level += 1
        code += f"real initial_time = {self.initial_time};\n"
        code += f"int n_t = {n_t};\n"
        code += f"array[n_t] real times = {{{', '.join([str(x) for x in self.integration_times])}}};\n"
        code.indent_level -= 1
        code += "}\n"
        return code.string


class StanModelBuilder:
    def __init__(self, sampling_statements: Iterable["SamplingStatement"]):
        self.sampling_statements = sampling_statements

    def build_block(self):
        code = IndentedString()
        code += "model{\n"
        code.indent_level += 1
        for statement in self.sampling_statements:
            code += f"{statement.lhs_name} ~ {statement.distribution_type}({', '.join([str(arg) for arg in statement.distribution_args])});\n"

        code.indent_level -= 1
        code += "}\n"
        return str(code)

    
class StanGeneratedQuantitiesBuilder:
    def __init__(self, sampling_statements: Iterable["SamplingStatement"]):
        self.sampling_statements = sampling_statements

    def build_block(self):
        code = IndentedString()
        code += "generated quantities{\n"
        code.indent_level += 1
        for statement in self.sampling_statements:
            code += f"{statement.lhs_name}_tilde ~ {statement.distribution_type}({', '.join([str(arg) for arg in statement.distribution_args])});\n"

        code.indent_level -= 1
        code += "}\n"
        return str(code)

    
class StanFunctionBuilder:
    def __init__(
        self, abstract_model: AbstractModel, function_name: str = "vensim_func"
    ):

        self.abstract_model = abstract_model
        self.elements = self.abstract_model.sections[0].elements
        self.ode_function_name = function_name
        self.lookup_builder_walker = LookupCodegenWalker()
        self.variable_dependency_graph: Dict[
            str, Set
        ] = (
            {}
        )  # in order to evaluate 'key' variable, we need 'element' variables
        self.code = IndentedString()

    def get_generated_lookups_dict(self):
        return self.lookup_builder_walker.generated_lookup_function_names

    def _create_dependency_graph(self):
        self.variable_dependency_graph = {}
        walker = AuxNameWalker()
        for element in self.elements:
            for component in element.components:
                if element.name not in self.variable_dependency_graph:
                    self.variable_dependency_graph[
                        vensim_name_to_identifier(element.name)
                    ] = set()

                dependent_aux_names = walker.walk(component.ast)
                if dependent_aux_names:
                    self.variable_dependency_graph[
                        vensim_name_to_identifier(element.name)
                    ].update(dependent_aux_names)

        return self.variable_dependency_graph

    def build_functions(
        self,
        predictor_variable_names: Iterable[Tuple[str, str]],
        outcome_variable_names: Iterable[str],
        function_name: str = "vensim_func",
    ):
        self.code = IndentedString()

        # Build the lookup functions
        self.build_lookups()
        lookup_functions_code = str(self.lookup_builder_walker.code).rstrip()
        if lookup_functions_code:
            self.code += lookup_functions_code
            self.code += "\n\n"

        self.code += "# Begin ODE declaration\n"
        self._create_dependency_graph()

        # Identify the minimum number of variables needed for calculating outcomes
        required_variables = set()
        bfs_stack = []
        bfs_stack.extend(outcome_variable_names)
        while len(bfs_stack) > 0:
            variable = bfs_stack.pop(0)
            required_variables.add(variable)
            for next_var in self.variable_dependency_graph[variable]:
                if next_var in required_variables:
                    continue
                bfs_stack.append(next_var)
            required_variables |= self.variable_dependency_graph[variable]

        eval_order = []

        def recursive_order_search(current, visited):
            # if current in visited:
            #     return
            visited.add(current)
            # if current in eval_order:
            #     return
            for child in self.variable_dependency_graph[current]:
                if child == current:
                    continue
                if child in outcome_variable_names:
                    continue
                if child not in visited:
                    recursive_order_search(child, visited)
            eval_order.append(current)

        for var_name in required_variables:
            recursive_order_search(var_name, set())

        self.elements = [
            element
            for element in self.elements
            if vensim_name_to_identifier(element.name) in required_variables
        ]
        self.elements = sorted(
            self.elements,
            key=lambda x: eval_order.index(vensim_name_to_identifier(x.name)),
        )

        #################
        # Create function declaration
        self.code += f"vector {function_name}(real time, vector outcome, "
        argument_strings = []
        argument_variables = (
            []
        )  # this list holds the names of the argument variables
        for var in predictor_variable_names:
            if isinstance(var, str):
                argument_variables.append(var)
                argument_strings.append("real " + var)
            elif isinstance(var, tuple):
                var_type, var_name = var
                argument_variables.append(var_name)
                argument_strings.append(f"{var_type} {var_name}")

        self.code += ", ".join(argument_strings)
        self.code += "){"
        self.code += "\n"
        #############
        self.code.indent_level += 1
        # Enter function body

        self.code += f"vector[{len(outcome_variable_names)}] dydt;  # Return vector of the ODE function\n"
        self.code += "\n"

        self.code += "# State variables\n"
        for index, outcome_variable_name in enumerate(
            outcome_variable_names, 1
        ):
            self.code += f"real {outcome_variable_name} = outcome[{index}];\n"

        self.code += "\n"

        codegen_walker = BlockCodegenWalker(
            self.lookup_builder_walker.generated_lookup_function_names
        )
        for element in self.elements:
            stan_varname = vensim_name_to_identifier(element.name)
            if stan_varname in argument_variables:
                continue
            elif stan_varname in outcome_variable_names:
                stan_varname += "_dydt"
            elif stan_varname not in required_variables:
                continue
            for component in element.components:
                self.code += f"real {stan_varname} = {codegen_walker.walk(component.ast)};\n"

        self.code += "\n"

        # Generate code for returning outcomes of interest
        outcome_variable_names = [
            name + "_dydt" for name in outcome_variable_names
        ]
        for index, outcome_variable_name in enumerate(
            outcome_variable_names, 1
        ):
            self.code += f"dydt[{index}] = {outcome_variable_name};\n"

        self.code += "\n"
        self.code += "return dydt;\n"

        self.code.indent_level -= 1
        # Exit function body
        self.code += "}\n"

        return str(self.code)

    def build_lookups(self):
        for element in self.elements:
            for component in element.components:
                self.lookup_builder_walker.walk(component.ast)


