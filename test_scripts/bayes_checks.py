from pysd.builders.stan.stan_model import StanVensimModel
from pysd.translators.vensim.vensim_file import VensimFile
from pysd.translators.xmile.xmile_file import XmileFile
import pandas as pd
import cmdstanpy
import numpy as np
from cmdstanpy import install_cxx_toolchain
config = install_cxx_toolchain.get_config('C:\\RTools', True)
print(install_cxx_toolchain.get_toolchain_name())
import cmdstanpy; cmdstanpy.install_cmdstan(overwrite=True)
# def data2draws():
## 1. D
obs_stock_df = pd.read_csv('data/hudson-bay-lynx-hare.csv')
n_t = obs_stock_df.shape[0] - 1
data_draws2data = {
    "n_t": n_t
}
## 2. P
### a. set_prior_struc
vf = VensimFile("vensim_models/prey-predator.mdl")
vf.parse()
am = vf.get_abstract_model()
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

##### 1+2+3)
model.build_stan_functions()

### c. set_prior_demand ??

#model.print_info()

## 1+2. P(D)
model.draws2data() # write stanfile
draws2data_model = cmdstanpy.CmdStanModel(stan_file="stan_files/prey-predator_draws2data.stan")

## 3. A(P(D))
print("DRAWS2DATA========================================================")
print(draws2data_model.sample(data=data_draws2data, fixed_param=True).summary())

# def data2draws():
## 1. D
data_data2draws = {
    "n_obs_state" : 2,
    "initial_time" : 0,
    "times": [i+1 for i in np.arange(n_t)],
    "n_t": n_t,
    "predator_obs":  obs_stock_df.loc[1:, 'Predator'].values.tolist(),
    "prey_obs": obs_stock_df.loc[1:, 'Prey'].values.tolist(),
}

## 2. P
### a. set_prior_struc
model.build_stan_functions() # TODO check cache and build if not exist

### b. set_prior_var
##### 1) ode parameter prior
model.set_prior("alpha", "normal", 0.8, 0.1)
model.set_prior("gamma", "normal", 0.8, 0.1)
model.set_prior("beta", "normal", 0.05, 0.001)
model.set_prior("delta", "normal", 0.05, 0.001)

##### 2) sampling distribution parameter (measruement error) prior
model.set_prior("sigma", "lognormal",  np.log(0.01), 0.1)

model.draws2data("")
model.data2draws("")

##### 3)  measurement \tilde{y}_{1..t} ~ f(\theta, t)_{1..t}
model.set_prior("predator_obs", "lognormal", "predator", "sigma")
model.set_prior("prey_obs", "lognormal", "prey", "sigma")

#model.print_info()

## 1+2. P(D)
model.data2draws() # write stanfile
data2draws_model = cmdstanpy.CmdStanModel(stan_file="stan_files/prey-predator_data2draws.stan")

## 3. A(P(D))
print("DATA2DRAWS=========================================================")
print(data2draws_model.sample(data=data_data2draws).summary())

# def draws2data2draws():

## compare with vensim output
# vensim_df = pd.read_csv("vensim_models/prey-predator/output.csv")
# predator_obs = vensim_df[vensim_df['Time']=="Predator"]
# prey_obs = vensim_df[vensim_df['Time']=="Prey"]
# print(predator_obs)
# print(prey_obs)
