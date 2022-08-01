import os
from pathlib import Path
from typing import Union, List, Dict, Set, Sequence

from .ast_walker import *
from .utilities import *
from pysd.translators.structures.abstract_model import\
    AbstractComponent, AbstractElement, AbstractModel, AbstractSection


class StanModelBuilder:
    def __init__(self, abstract_model: AbstractModel):
        self.abstract_model = abstract_model

        self.variable_ast_dict: Dict[str, AbstractSyntax] = {}
        assert len(self.abstract_model.sections) == 1, "Number of sections in AbstractModel must be 1."
        for element in self.abstract_model.sections[0].elements:
            stan_varname = vensim_name_to_identifier(element.name)
            assert len(element.components) == 1, f"Number of components in AbstractElement must be 1, but {element.name} has {len(element.components)}"
            self.variable_ast_dict[stan_varname] = element.components[0].ast


    def create_stan_program(self, predictor_variable_names: List[Union[str, Tuple[str, str]]], outcome_variable_names: Sequence[str] = (), function_name="vensim_func"):
        """

        Parameters
        ----------
        predictor_variable_names: List of name of variables within the SD model that are handled by stan. The code for
        these variables will not be generated, but instead taken from the argument of the ODE system function.
        outcome_variable_names: Sequence of name of the variables which are the measured as system state.
        Normally this will be the observed outcomes among the stock variables. If it is not specified, it will automatically
        identify stock variable names and use them.
        function_name: Name of the stan function to be generated. default is "vensim_func"

        Returns
        -------

        """
        # Santize vensim names to stan-compliant identifiers
        sanitized_predictor_variable_names = []
        for var in predictor_variable_names:
            match var:
                case str(x):
                    sanitized_predictor_variable_names.append(vensim_name_to_identifier(x))
                case (str(type), str(var_name)):
                    sanitized_predictor_variable_names.append((type, vensim_name_to_identifier(var_name)))
                case _:
                    raise Exception("predictor_variable_names must be a list of strings and/or a tuple of the form(T, Name), where T is a string denoting the variable's stan type and Name a string denoting the variable name")

        predictor_variable_names = sanitized_predictor_variable_names
        outcome_variable_names = self.get_stock_variable_stan_names() if not outcome_variable_names else [vensim_name_to_identifier(name) for name in outcome_variable_names]
        if not outcome_variable_names:
            raise Exception("There are no stock variables defined in the model, hence nothing to integrate.")

        self.code = IndentedString()

        function_block_builder = StanFunctionBuilder(self.abstract_model)

        self.code += function_block_builder.build_function_block(predictor_variable_names, outcome_variable_names, function_name)

        self.code += "data{\n}\n"
        # self.code += StanDataBuilder(self.abstract_model).build_block(predictor_variable_names, outcome_variable_names)
        self.code += "transformed data{\n}\n"
        self.code += "parameters{\n}\n"
        self.code += StanTransformedParametersBuilder(self.abstract_model).build_block(predictor_variable_names, outcome_variable_names, function_block_builder.lookup_builder_walker.generated_lookup_function_names, function_name)
        self.code += "model{\n}\n"

        self.code += "generated quantities{\n}"

        return self.code

    def print_variable_info(self):
        var_names = []
        max_length = len("original name") + 1
        for element in self.abstract_model.sections[0].elements:
            is_stock = False
            for component in element.components:
                if isinstance(component.ast, IntegStructure):
                    is_stock = True
                    break

            var_names.append((element.name, vensim_name_to_identifier(element.name), is_stock))
            max_length = max(max_length, len(element.name) + 1)

        header = 'original name'.ljust(max_length) + "stan variable name".ljust(max_length) + "is stock"
        print(header)
        print("-" * len(header))
        for x in var_names:
            print(x[0].ljust(max_length) + x[1].ljust(max_length) + ("V" if x[2] else ""))

    def get_stock_variable_stan_names(self) -> List[str]:
        """
        Iterate through the AST and find stock variables
        Returns
        -------

        """
        stock_varible_names = []
        for element in self.abstract_model.sections[0].elements:
            for component in element.components:
                if isinstance(component.ast, IntegStructure):
                    stock_varible_names.append(vensim_name_to_identifier(element.name))
                    break

        return stock_varible_names

""" class StanDataBuilder:
    def __init__(self, abstract_model: AbstractModel):
        self.abstract_model = abstract_model

    def build_block(self, predictor_variable_names, outcome_variable_names):
        self.code = IndentedString()
        self.code += "data {\n"
        self.code.indent_level += 1        

        self.code += f"predictor= {{{', '.join(predictor_variable_names)}}};\n"
        self.code += f"initial_outcome = {{{', '.join(outcome_variable_names)}}};\n"
        self.code += f"observed_outcome = {{{', '.join(outcome_variable_names)}}};\n"
        self.code += f"times = {{{', '.join(outcome_variable_names)}}};\n"
        self.code.indent_level -= 1
        self.code += "}\n" """


class StanTransformedDataBuilder:
    def __init__(self, abstract_model: AbstractModel):
        self.abstract_model = abstract_model


class StanTransformedParametersBuilder:
    def __init__(self, abstract_model: AbstractModel):
        self.abstract_model = abstract_model

    def build_block(self, predictor_variable_names, outcome_variable_names, lookup_function_dict, function_name):
        self.code = IndentedString()
        self.code += "transformed parameters {\n"
        self.code.indent_level += 1

        argument_variables = []
        for var in predictor_variable_names:
            match var:
                case str(x):
                    argument_variables.append(x)
                case (str(type), str(var_name)):
                    argument_variables.append(var_name)

        variable_ast_dict: Dict[str, AbstractSyntax] = {}
        for element in self.abstract_model.sections[0].elements:
            stan_varname = vensim_name_to_identifier(element.name)
            variable_ast_dict[stan_varname] = element.components[0].ast

        for outcome_variable_name in outcome_variable_names:
            for element in self.abstract_model.sections[0].elements:
                if vensim_name_to_identifier(element.name) == outcome_variable_name:
                    component = element.components[0]
                    assert isinstance(component.ast, IntegStructure), "Output variable component must be an INTEG."
                    self.code += f"real {outcome_variable_name}_initial = {InitialValueCodegenWalker(lookup_function_dict, variable_ast_dict).walk(component.ast)};\n"
                    break

        self.code += f"vector[{len(outcome_variable_names)}] initial_outcome = {{{', '.join([x + '_initial' for x in outcome_variable_names])}}};\n"

        self.code += f"array[] vector integrated_result = integrate_ode_rk45({function_name}, initial_outcome, initial_time, times, {','.join(argument_variables)});\n"
        self.code.indent_level -= 1
        self.code += "}\n"

        return str(self.code)


class StanFunctionBuilder:
    def __init__(self, abstract_model: AbstractModel, function_name: str = "vensim_ode"):

        self.abstract_model = abstract_model
        self.elements = self.abstract_model.sections[0].elements
        self.ode_function_name = function_name
        self.lookup_builder_walker = LookupCodegenWalker()
        self.variable_dependency_graph: Dict[str, Set] = {}  # in order to evaluate 'key' variable, we need 'element' variables
        self.code = IndentedString()

    def _create_dependency_graph(self):
        self.variable_dependency_graph = {}
        walker = AuxNameWalker()
        for element in self.elements:
            for component in element.components:
                if element.name not in self.variable_dependency_graph:
                    self.variable_dependency_graph[vensim_name_to_identifier(element.name)] = set()

                dependent_aux_names = walker.walk(component.ast)
                if dependent_aux_names:
                    self.variable_dependency_graph[vensim_name_to_identifier(element.name)].update(dependent_aux_names)

        return self.variable_dependency_graph

    def build_function_block(self, predictor_variable_names: List[Tuple[str, str]], outcome_variable_names: List[str], function_name: str ="vensim_func"):
        self.code = IndentedString()
        self.code += "functions {\n"

        # Build the lookup functions
        self.build_lookups()
        lookup_functions_code = str(self.lookup_builder_walker.code).rstrip()
        if lookup_functions_code:
            self.code += lookup_functions_code
            self.code += "\n\n"

        self.code.indent_level += 1
        self.code += "# Begin ODE declaration\n"
        # Enter function block
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

        #print(self.variable_dependency_graph)
        #print("rv:", required_variables)

        eval_order = []
        def recursive_order_search(current, visited):
            # if current in visited:
            #     return
            visited.add(current)
            # if current in eval_order:
            #     return
            for child in self.variable_dependency_graph[current]:
                if child == current: continue
                if child in outcome_variable_names: continue
                if child not in visited:
                    recursive_order_search(child, visited)
            eval_order.append(current)

        #for var_name in self.variable_dependency_graph.keys():
        for var_name in required_variables:
            recursive_order_search(var_name, set())

        self.elements = [element for element in self.elements if vensim_name_to_identifier(element.name) in required_variables]
        self.elements = sorted(self.elements, key=lambda x: eval_order.index(vensim_name_to_identifier(x.name)))


        #################
        # Create function declaration
        self.code += f"vector {function_name}(real time, vector outcome, "
        argument_strings = []
        argument_variables = []  # this list holds the names of the argument variables
        for var in predictor_variable_names:
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
        #############
        self.code.indent_level += 1
        # Enter function body

        for index, outcome_variable_name in enumerate(outcome_variable_names, 1):
            self.code += f"real {outcome_variable_name} = outcome[{index}];\n"

        self.code += "\n"

        codegen_walker = BlockCodegenWalker(self.lookup_builder_walker.generated_lookup_function_names)
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
        outcome_variable_names = [name + "_dydt" for name in outcome_variable_names]
        self.code += f"return {{{', '.join(outcome_variable_names)}}};\n"
        self.code.indent_level -= 1
        # Exit function body
        self.code += "}\n"

        self.code.indent_level -= 1
        # Exit function block
        self.code += "}\n"
        return str(self.code)

    def build_lookups(self):
        for element in self.elements:
            for component in element.components:
                self.lookup_builder_walker.walk(component.ast)


class StanTransformedDataBuilder:
    def __init__(self, abstract_model: AbstractModel, function_name: str = "vensim_ode"):

        self.abstract_model = abstract_model
        self.elements = self.abstract_model.sections[0].elements
        self.function_name = function_name
