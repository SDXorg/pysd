import pysd
import os
import pandas as pd

# display information about the pysd version we're using
print pysd.__version__
print pysd.__file__
os.path.abspath(pysd.__file__)

mdl_file = 'tests/test-models/tests/subscript_2d_arrays/test_subscript_2d_arrays.mdl'
model = pysd.read_vensim(mdl_file)

stocks = model.run(flatten_subscripts=True)
print stocks.head()