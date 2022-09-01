from pysd.builders.stan.stan_model import StanVensimModel
from pysd.translators.vensim.vensim_file import VensimFile
from pysd.translators.xmile.xmile_file import XmileFile

vf = VensimFile("vensim_models/ds_white_sterman.mdl")
vf.parse()
am = vf.get_abstract_model()

model = StanVensimModel("ds_white_sterman", am, 0.0, list(range(1, 10)))
model.print_info()

model.set_prior("inventory_adjustment_time", "normal", 0, 1)
model.set_prior("minimum_order_processing_time", "normal", 0, 1)
model.set_prior("alpha", "normal", 0, 1, lower=0.0)
model.set_prior("inventory", "normal", 0, 1, init_state=True)

print(model.vensim_model_context.variable_names)

model.build_stan_functions()

cmdstan_model = model.data2draws({})
result = cmdstan_model.sample()
result.summary()