import pysd
import os

os.path.abspath(__file__)

model=pysd.read_vensim('tests\stamps\pdatvensim8k_RW_v2.mdl')

stocks=model.run(flatten_subscripts=True)

stocks
