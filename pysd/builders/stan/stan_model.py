from typing import List, Set, Type
from dataclasses import dataclass, field
import ast, os, pathlib, warnings, glob
from .stan_model_builder import StanFunctionBuilder
from .utilities import vensim_name_to_identifier

@dataclass
class SamplingStatement:
    lhs_name: str
    distribution_type: str
    distribution_return_type: Type = field(init=False)
    distribution_args: List[str]

    def __post_init__(self):
        if self.distribution_type in ("bernoulli", "binomial", "beta_binomial", "neg_binomial", "poisson"):
            self.distribution_return_type = int
        else:
            # TODO: Check if it's a valid stan distribution
            self.distribution_return_type = float


@dataclass
class StanModelContext:
    sample_statements: List[SamplingStatement] = field(default_factory=list)
    exposed_parameters: Set[str] = field(default_factory=list)


class VensimModelContext:
    def __init__(self, abstract_model):

        self.variable_names = set()
        for element in abstract_model.sections[0].elements:
            self.variable_names.add(vensim_name_to_identifier(element.name))


class StanVensimModel:
    def __init__(self, model_name: str, abstract_model):
        self.abstract_model = abstract_model
        self.model_name = model_name
        self.stan_model_context = StanModelContext()
        self.vensim_model_context = VensimModelContext(self.abstract_model)

    def set_prior(self, variable_name: str, distribution_type: str, *args):
        for arg in args:
            if isinstance(arg, str):
                # If the distribution argument is an expression, parse the dependant variables
                # We're using the python parser here, which might be problematic
                used_variable_names = [node.id for node in ast.walk(ast.parse(arg)) if isinstance(node, ast.Name)]
                self.stan_model_context.exposed_parameters.update(used_variable_names)

        self.stan_model_context.sample_statements.append(SamplingStatement(variable_name, distribution_type, *args))


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
            function_builder = StanFunctionBuilder(self.abstract_model)
            f.write(function_builder.build_functions())

    def data2draws(self):
        pass

