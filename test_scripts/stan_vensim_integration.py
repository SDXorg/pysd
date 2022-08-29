from pysd.builders.stan.stan_model import StanVensimModel
from pysd.translators.vensim.vensim_file import VensimFile
from pysd.translators.xmile.xmile_file import XmileFile

vf = VensimFile("vensim_models/ds_white_sterman.mdl")
vf.parse()
am = vf.get_abstract_model()

model = StanVensimModel("ds_white_sterman", am, 0.0, list(range(0, 10)))
model.set_prior("inventory_adjustment_time", "normal", 0, 1)
model.set_prior("minimum_order_processing_time", "normal", 0, 1)

print(model.vensim_model_context.variable_names)

model.build_stan_functions()

model.data2draws("")