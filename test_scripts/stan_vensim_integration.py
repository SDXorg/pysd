from pysd.builders.stan.stan_model import StanVensimModel
from pysd.translators.vensim.vensim_file import VensimFile
from pysd.translators.xmile.xmile_file import XmileFile
import pandas as pd
obs_stock_df = pd.read_csv('data/hudson-bay-lynx-hare.csv')
vf = VensimFile("vensim_models/prey-predator.mdl")
vf.parse()
am = vf.get_abstract_model()

# set time
n_t = obs_stock_df.shape[0] - 1
premodel = StanVensimModel("prey-predator", am, 0.0, list(range(1, n_t + 1)))
premodel.print_info()

n_t = obs_stock_df.shape[0] - 1
data_data2draws = {
    "n_t": n_t,
    "predator_obs": obs_stock_df.loc[1:, 'Predator'].values.tolist(),
    "prey_obs": obs_stock_df.loc[1:, 'Prey'].values.tolist(),
}


premodel = StanVensimModel("prey-predator", am, 0.0, list(range(1, n_t + 1)), data_dict = data_data2draws)
# ode parameter
premodel.set_prior("alpha", "normal", 0.55, 0.1)
premodel.set_prior("gamma", "normal", 0.8, 0.1)
premodel.set_prior("beta", "normal", 0.028, 0.01)
premodel.set_prior("delta", "normal", 0.024, 0.01)

# sampling distribution parameter
premodel.set_prior("sigma", "normal", 0, 1)

premodel.set_prior("predator_obs", "lognormal", "predator", "sigma")
premodel.set_prior("prey_obs", "lognormal", "prey", "sigma")
premodel.build_stan_functions()
premodel.draws2data()
#print(premodel.vensim_model_context.variable_names)


#cmdstan_model = premodel.data2draws(data_data2draws)
#result = cmdstan_model.sample(data=data_data2draws)
#result.summary()