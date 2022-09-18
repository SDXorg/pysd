from pysd.builders.stan.stan_model import StanVensimModel
from pysd.translators.vensim.vensim_file import VensimFile
import pandas as pd
import numpy as np
import cmdstanpy #; cmdstanpy.install_cmdstan(overwrite=True)
def draws2data(am, data_draws2data):
    ## 1. D
    n_t = data_draws2data.get('n_t')
    ## 2. P
    ### a. set_prior_struc
    model = StanVensimModel("prey-predator", am, 0.0, list(range(1, n_t + 1)), data_dict = data_draws2data)

    ### b. set_prior_var
    ##### 1) ode parameter prior
    model.set_prior("alpha", "normal", 0.55, 0.1)
    model.set_prior("gamma", "normal", 0.8, 0.1)
    model.set_prior("beta", "normal", 0.028, 0.01)
    model.set_prior("delta", "normal", 0.024, 0.01)

    ##### 2) sampling distribution parameter (measruement error) prior
    model.set_prior("sigma", "lognormal", np.log(0.01), 0.1)

    ##### 3)  measurement \tilde{y}_{1..t} ~ f(\theta, t)_{1..t}
    model.set_prior("predator_obs", "lognormal", "predator", "sigma")
    model.set_prior("prey_obs", "lognormal", "prey", "sigma")

    ##### 1) + 2) + 3)
    model.build_stan_functions()

    ### c. set_prior_demand #TODO

    ## 1+2. P(D)
    model.stanify_draws2data()
    draws2data_model = cmdstanpy.CmdStanModel(stan_file="stan_files/prey-predator_draws2data.stan")

    ## 3. A(P(D))
    model.print_info()
    print(draws2data_model.sample(data=data_draws2data, fixed_param=True).summary())

def data2draws(am, obs_stock_df):
    # ORDER is important
    ## 1. D
    n_t = obs_stock_df.shape[0] - 1
    data_data2draws = {
        "n_obs_state" : 2,
        "n_t": n_t,
        "predator_obs":  obs_stock_df.loc[1:, 'Predator'].values.tolist(),
        "prey_obs": obs_stock_df.loc[1:, 'Prey'].values.tolist(),
    }

    ## 2. P
    ### a. set_prior_struc
    model = StanVensimModel("prey-predator", am, 0.0, list(range(1, n_t + 1)), data_dict=data_data2draws)

    ### b. set_prior_var
    ##### 1) ode parameter prior
    model.set_prior("alpha", "normal", 0.8, 0.1)
    model.set_prior("gamma", "normal", 0.8, 0.1)
    model.set_prior("beta", "normal", 0.05, 0.001)
    model.set_prior("delta", "normal", 0.05, 0.001)

    ##### 2) sampling distribution parameter (measruement error) prior
    model.set_prior("sigma", "lognormal",  np.log(0.01), 0.1)

    ##### 3)  measurement \tilde{y}_{1..t} ~ f(\theta, t)_{1..t}
    model.set_prior("predator_obs", "lognormal", "predator", "sigma")
    model.set_prior("prey_obs", "lognormal", "prey", "sigma")

    ##### 1) + 2) + 3)
    model.build_stan_functions()  # TODO check cache and build if not exist

    ### c. set_prior_demand #TODO

    ## 1+2. P(D)
    model.stanify_data2draws()
    data2draws_model = cmdstanpy.CmdStanModel(stan_file="stan_files/prey-predator_data2draws.stan")

    ## 3. A(P(D))
    print("DATA2DRAWS=========================================================")
    print(data2draws_model.sample(data=data_data2draws).summary())

def draws2data2draws():

    return
## compare with vensim output
# vensim_df = pd.read_csv("vensim_models/prey-predator/output.csv")
# predator_obs = vensim_df[vensim_df['Time']=="Predator"]
# prey_obs = vensim_df[vensim_df['Time']=="Prey"]
# print(predator_obs)
# print(prey_obs)
