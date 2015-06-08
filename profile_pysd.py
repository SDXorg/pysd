import pstats, cProfile
import pysd

model = pysd.read_vensim('tests/vensim/Teacup.mdl')

def run():
    for i in range(0,100):
        model.run()

cProfile.runctx('run()', globals(), locals(), 'tests/profile')
