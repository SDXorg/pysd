from typing import List, Set, Type, Tuple
from numbers import Number
from dataclasses import dataclass, field
import ast, os, pathlib, warnings, glob, re
from .stan_block_builder import *
from .utilities import vensim_name_to_identifier
from pysd.translators.structures.abstract_expressions import *


class SamplingStatement:
    lhs_name: str
    distribution_type: str
    distribution_return_type: Type
    distribution_args: Tuple[str]
    init_state: bool

    def __init__(self, lhs_name, distribution_type, *distribution_args, init_state=False):
        self.lhs_name = lhs_name
        self.distribution_type = distribution_type
        self.distribution_args = distribution_args
        self.init_state = init_state

    def __post_init__(self):
        if self.distribution_type in ("bernoulli", "binomial", "beta_binomial", "neg_binomial", "poisson"):
            self.distribution_return_type = int
        else:
            # TODO: Check if it's a valid stan distribution
            self.distribution_return_type = float


@dataclass
class StanModelContext:
    sample_statements: List[SamplingStatement] = field(default_factory=list)
    exposed_parameters: Set[str] = field(default_factory=set)


class VensimModelContext:
    def __init__(self, abstract_model):
        self.variable_names = set()
        self.stock_variable_names = set()

        # Some basic checks to make sure the AM is compatible
        assert len(abstract_model.sections) == 1, "Number of sections in AbstractModel must be 1."

        for element in abstract_model.sections[0].elements:
            assert len(element.components) == 1, f"Number of components in AbstractElement must be 1, but {element.name} has {len(element.components)}"
            self.variable_names.add(vensim_name_to_identifier(element.name))

        for element in abstract_model.sections[0].elements:
            for component in element.components:
                if isinstance(component.ast, IntegStructure):
                    self.stock_variable_names.add(vensim_name_to_identifier(element.name))
                    break

    def print_variable_info(self, abstract_model):
        var_names = []
        max_length = len("original name") + 1
        for element in abstract_model.sections[0].elements:
            is_stock = False
            for component in element.components:
                if isinstance(component.ast, IntegStructure):
                    is_stock = True
                    break

            var_names.append((element.name, vensim_name_to_identifier(element.name), is_stock,))
            max_length = max(max_length, len(element.name) + 1)

        header = ("original name".ljust(max_length) + "stan variable name".ljust(max_length) + "is stock")
        print(header)
        print("-" * len(header))
        for x in var_names:
            print(x[0].ljust(max_length) + x[1].ljust(max_length) + ("V" if x[2] else ""))


class StanVensimModel:
    def __init__(self, model_name: str, abstract_model, initial_time: float, integration_times: Iterable[Number]):
        self.abstract_model = abstract_model
        self.model_name = model_name
        self.initial_time = float(initial_time)
        self.integration_times = integration_times
        self.stan_model_context = StanModelContext()
        self.vensim_model_context = VensimModelContext(self.abstract_model)
        if initial_time in integration_times:
            raise Exception("initial_time shouldn't be present in integration_times")

        self.init_variable_regex = re.compile(".+?(?=_init$)")
        # This regex is to match all preceding characters that come before '_init' at the end of the string.
        # So something like stock_var_init_init would match into stock_var_init.
        # This is used to parse out the corresponding stock names for init parameters.

    def print_info(self):
        print("- Vensim model information:")
        self.vensim_model_context.print_variable_info(self.abstract_model)
        print("*" * 10)
        print("- Stan model information:")

    def set_prior(self, variable_name: str, distribution_type: str, *args, init_state=False):
        if init_state:
            # This means the initial value of the ODE state variable.
            if variable_name not in self.vensim_model_context.stock_variable_names:
                raise Exception("init_state may be set to True only for stock variables.")
            self.stan_model_context.sample_statements.append(SamplingStatement(f"{variable_name}_init", distribution_type, *args, init_state=init_state))
        else:
            for arg in args:
                if isinstance(arg, str):
                    # If the distribution argument is an expression, parse the dependant variables
                    # We're using the python parser here, which might be problematic
                    used_variable_names = [node.id for node in ast.walk(ast.parse(arg)) if isinstance(node, ast.Name)]
                    for name in used_variable_names:
                        if name in self.vensim_model_context.variable_names:
                            self.stan_model_context.exposed_parameters.update(used_variable_names)

            if variable_name in self.vensim_model_context.variable_names:
                self.stan_model_context.exposed_parameters.add(variable_name)
            self.stan_model_context.sample_statements.append(SamplingStatement(variable_name, distribution_type, *args, init_state=init_state))


    def build_stan_functions(self):
        """
        We build the stan file that holds the ODE function. From the sample statements that the user have provided,
        we identify which variables within the ODE model should be treated as stan parameters instead if variables within
        the function block. This means that instead of the variable being defined locally within the function body,
        it instead gets defined within the transformed parameters/model block.
        Returns
        -------

        """
        if glob.glob(os.path.join(os.getcwd(), f"{self.model_name}_functions.stan")):
            if input(f"{self.model_name}_functions.stan already exists in the current working directory. Overwrite? (Y/N):").lower() != "y":
                raise Exception("Code generation aborted by user")



        with open(os.path.join(os.getcwd(), f"{self.model_name}_functions.stan"), "w") as f:
            self.function_builder = StanFunctionBuilder(self.abstract_model)
            f.write(self.function_builder.build_functions(self.stan_model_context.exposed_parameters, self.vensim_model_context.stock_variable_names))

    def data2draws(self, data_file_path: str):
        with open(os.path.join(os.getcwd(), f"{self.model_name}_data2draws.stan"), "w") as f:
            # Include the function
            f.write("functions{\n")
            f.write(f"    #include {self.model_name}_functions.stan\n")
            f.write("}\n\n")

            f.write(StanDataBuilder().build_block())
            f.write("\n")

            f.write(StanTransformedDataBuilder(self.initial_time, self.integration_times).build_block())
            f.write("\n")

            f.write(StanParametersBuilder(self.stan_model_context.sample_statements).build_block())
            f.write("\n")

            transformed_params_builder = StanTransformedParametersBuilder(self.abstract_model)
            # Find sampling statements for init
            stock_initials = {}
            for statement in self.stan_model_context.sample_statements:
                if statement.init_state:
                    stock_variable_name = self.init_variable_regex.findall(statement.lhs_name)[0]
                    stock_initials[stock_variable_name] = statement.lhs_name

            f.write(transformed_params_builder.build_block(self.stan_model_context.exposed_parameters,
                                                           self.vensim_model_context.stock_variable_names,
                                                           self.function_builder.get_generated_lookups_dict(),
                                                           self.function_builder.ode_function_name,
                                                           stock_initials))
            f.write("\n")

            f.write(StanModelBuilder(self.stan_model_context.sample_statements).build_block())
            f.write("\n")


