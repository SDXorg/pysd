import timeit
import time

test_model = 'test-models/samples/teacup/teacup.mdl'

times = { }

###################
from _version import __version__

times['version'] = __version__
times['timestamp'] = time.time()


####################

times['import'] = timeit.timeit('reload(pysd)', number=100,
                                 setup='import pysd')/100

print 'Import:', times['import']

####################

times['load teacup'] = timeit.timeit("pysd.read_vensim('%s')" % test_model,
                                     setup='import pysd', number=100)/100

print 'Load Teacup:', times['load teacup']

##################

times['run teacup baseline'] = timeit.timeit("model.run()",
                                    setup="import pysd; model = pysd.read_vensim('%s')" % test_model,
                                    number=1000)/1000

print 'Run Teacup Baseline:', times['run teacup baseline']


##############

times['run teacup modify params'] = timeit.timeit("model.run(params={'room_temperature':20})",
                                    setup="import pysd; model = pysd.read_vensim(%s)" % test_model,
                                    number=1000)/1000

print 'Run Teacup Mofifying Params:', times['run teacup modify params']


###########

times['run teacup return extra columns'] = timeit.timeit("model.run(return_columns=['teacup_temperature','heat_loss_to_room'])",
                                    setup="import pysd; model = pysd.read_vensim(%s)" % test_model,
                                    number=100)/100

print 'Run Teacup Returning Extra Columns:', times['run teacup return extra columns']

###############

times['monte carlo'] = \
    timeit.timeit("[model.run(initial_condition=(0, {'teacup_temperature':50*np.random.rand()})) for i in range(1000)]",
                  setup="import numpy as np; import pysd; model = pysd.read_vensim(%s)" % test_model,
                  number=10)/10

print 'Monte Carlo:', times['monte carlo']


#---------------------------------
with open('speedtest_results.json', 'a') as outfile:
    outfile.write(str(times)+'\n')
