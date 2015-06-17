import timeit
import time

times = { }

###################
from pysd._version import __version__

times['version'] = __version__
times['timestamp'] = time.time()


####################

times['import'] = timeit.timeit('reload(pysd)', number=100,
                                 setup='import pysd')/100

print 'Import:', times['import']

####################

times['load teacup'] = timeit.timeit("pysd.read_vensim('tests/vensim/Teacup.mdl')",
                                     setup='import pysd', number=100)/100

print 'Load Teacup:', times['load teacup']

##################

times['run teacup baseline'] = timeit.timeit("model.run()",
                                    setup="import pysd; model = pysd.read_vensim('tests/vensim/Teacup.mdl')",
                                    number=1000)/1000

print 'Run Teacup Baseline:', times['run teacup baseline']


##############

times['run teacup modify params'] = timeit.timeit("model.run(params={'room_temperature':20})",
                                    setup="import pysd; model = pysd.read_vensim('tests/vensim/Teacup.mdl')",
                                    number=1000)/1000

print 'Run Teacup Mofifying Params:', times['run teacup modify params']


with open('tests/speedtest_results.json', 'a') as outfile:
    outfile.write(str(times)+'\n')
