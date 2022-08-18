from pysd.translators.vensim.vensim_file import VensimFile
from pysd.translators.xmile.xmile_file import XmileFile
from pysd.builders.stan.stan_model_builder import *

vf = VensimFile("vensim_models/ds_white_sterman.mdl")
#vf = VensimFile("vensim_models/arithmetic.mdl")
#vf = XmileFile("vensim_models/repair.xmile")
vf.parse()

am = vf.get_abstract_model()

stan_builder = StanModelBuilder(am)
stan_builder.print_variable_info()

ass_param_lst = ["customer_order_rate", "inventory_coverage", "manufacturing_cycle_time", "time_to_average_order_rate", "wip_adjustment_time"]
obs_stock_lst = ["work_in_process_inventory", "inventory"]

#print(stan_builder.create_stan_program(ass_param_lst, obs_stock_lst))

f_builder = StanFunctionBuilder(am)
print(f_builder.build_function_block(ass_param_lst, obs_stock_lst))
# for section in am.sections:
#     for element in section.elements:
#         print("*" * 10)
#         print(f"name: {element.name}")
#         print(f"length: {len(element.components)}")
#         for component in element.components:
#             print(f"type: {component.type}")
#             print(f"subtype: {component.subtype}")
#             print(f"subscript: {component.subscripts}")
#             print(component.ast)