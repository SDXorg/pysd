
## 1. What is this PR doing and how?

The main contribution of this PR is inference; its benefits are estimating parameter values and calibration. Currently pysd is doing a good job in data generation conditional on parameter values but lacks modules that retrieves parameter from the generated data. This pr aims to fill the gap; Stan is a computational Bayesian statistics package, with a large eco-system, providing state of the art inference algorithms including MCMC (HMC-NUTS) and variational inference. Given the following input from users, our output is posterior samples of estimated parameters.

1. dynamic model in .mdl or .xmile or .stmx format

2. user's classification of 

    a. assumed parameter: 
        a-1. time series of parameters (e.g. temperature or demand)
        a-2. fixed value of parameter (e.g. size of one franchise branch)
    
    b. estimated parameter (e.g. delay time of shipment) 
        b-1. prior distribution 
        b-2. prior parameter
        
    c. observed state (stock variables with observed data)

(SQ2. might we need addition input signature for `estimated parameter`? currently we `predictor_variable_names`: List[Union[str, Tuple[str, str]]] is a, `outcome_variable_name`: Sequence[str] = () is c. also could you explain the difference of two current signature types?)

Under the hood, files under `pysd/builders/stan` folder, we parse `xmile`, `mdl`, `stmx` files into one Stan code file `.stan` and pass it to Stan's state of the art MCMC infefinference engine, which serves inverse process of pysd's data generation. Stan has a large ecosystem that supports both calibration and model expansion (e.g. hierarchical regression, generalized linear). 

These are tested in `test_scripts` folder, and main parsing logics are in [here](https://github.com/Dashadower/pysd/blob/master/pysd/builders/stan/ast_walker.py) and [here](https://github.com/Dashadower/pysd/blob/master/pysd/builders/stan/stan_model_builder.py). (SQ3. could you add some detailes on each file in the last sentence?)

## 2. Future plans
Our work is casting dynamic model into statistical model structure (JQ1. may I classify it as stochastic process; especially isn't representing demand with three parameter approach seems like gaussian process whose `alpha`, `rho` are `scale` and `1/corr` param) as part of [Bayesian workflow](https://arxiv.org/abs/2011.01808) project which provides principled guidelines for model building.

Two works are planned for the near future and it would be tremendously helpful some support could be given in the second.

- Translating vensim and/or stella function using python [here](https://github.com/Dashadower/pysd/blob/master/pysd/builders/stan/ast_walker.py)
Currently `integration`, `random generation`, `lookup` are completed (but need testing) and `smooth` is in progress. Some parts we are missing: `if then else`, `step` (using `if then else`), `pulse`(point mass + divide by dt), `delay` (hard..), `material memory` (not possible in stan as it does not allow).

## 3. Our question
May we ask what test or other functions would be need for pysd for our PR to be most helpful? Thanks for the awesome package!
