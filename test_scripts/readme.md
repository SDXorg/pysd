
## 1. What is this PR doing and how?

The main contribution of this PR is inference; its benefits are estimating parameter values and calibration. Currently pysd is doing a good job in data generation conditional on parameter values but lacks modules that retrieves parameter from the generated data. This pr aims to fill the gap; Stan is a computational Bayesian statistics package, with a large eco-system, providing state of the art inference algorithms including MCMC (HMC-NUTS) and variational inference. Given the following input from users, our output is posterior samples of estimated parameters.

1. dynamic model in .mdl or .xmile or .stmx format (SQ1. can we support .stmx?)
2. user's classification of 
    a. assumed parameter: 
        a-1. time series of parameters (e.g. temperature or demand)
        a-2. fixed value of parameter (e.g. size of one franchise branch)
    b. estimated parameters (e.g. delay time of shipment) 
        b-1. prior distribution 
        b-2. prior parameter
    c. observed state (stock variables with observed data)

(SQ2. don't we need input signature of driving data? e.g. demand in static approach) 

Under the hood, files under `pysd/builders/stan` folder, we parse `xmile`, `mdl`, `stmx` files into one Stan code file `.stan` and pass it to Stan's state of the art MCMC infefinference engine, which serves inverse process of pysd's data generation. Stan has a large ecosystem that supports both calibration and model expansion (e.g. hierarchical regression, generalized linear). 

These are tested in `test_scripts` folder, and main parsing logics are in [here](https://github.com/Dashadower/pysd/blob/master/pysd/builders/stan/ast_walker.py) and [here](https://github.com/Dashadower/pysd/blob/master/pysd/builders/stan/stan_model_builder.py). (SQ3. could you add some detailes on each file in the last sentence?)

## 2. Future plans
Our work is casting dynamic model into statistical model structure (JQ1. may I classify it as stochastic process; especially isn't representing demand with three parameter approach seems like gaussian process whose `alpha`, `rho` are `scale` and `1/corr` param) as part of [Bayesian workflow](https://arxiv.org/abs/2011.01808) project which provides principled guidelines for model building.

Two works are planned for the near future and it would be tremendously helpful some support could be given in the second.

- Developing case studies on workflow which includes the following:
```
1. prior predictive
2. calibration 
- posterior credible interval and SBC
- sensitivity check on prior distribution and prior parameter 
- compare posterior with prior
3. posterior predictive
```
- Translating vensim and/or stella function using python [here](https://github.com/Dashadower/pysd/blob/master/pysd/builders/stan/ast_walker.py)
Currently `integration`, `random generation`, `lookup` are completed (but need testing) and `smooth` is in progress. (SQ4. am I missing something? JQ2. are we missing any main structure?) 

## 3. Our question
May we ask what test or other functions would be need for pysd for our PR to be most helpful? Thanks for the awesome package!

---
Q. I am making the notebook but is this necessary for pr? Could the pr and making notebook be proceeded parallelly?
SQ5. Could you give us a code review as we can receive Jair's feedback? I am curious on 
- the interface btw `ast_walker` and `stan_model_builder` 
- the role of `RNGCodegenWalker`
