import glob
import os.path
import traceback
from parsimonious.exceptions import ParseError, IncompleteParseError
import pandas as pd
import pysd


test_dir = 'tests/test-models-master/'

if not os.path.isdir(test_dir):
    import get_tests


vensim_testfiles = glob.glob(test_dir+'*/*/*.mdl')
xmile_testfiles = glob.glob(test_dir+'*/*/*.xmile')
testfiles = vensim_testfiles + xmile_testfiles

err_str = "'Testing module at location: %s\n"%pysd.__file__
threshold = 1

for modelfile in testfiles:
    #print modelfile
    directory = os.path.dirname(modelfile)
    try:
        if modelfile[-3:] == "mdl":
            model = pysd.read_vensim(modelfile)
        elif modelfile[-5:] == "xmile":
            model = pysd.read_xmile(modelfile)
        
        canon = pd.read_csv(directory+'/output.csv', index_col='Time')
        canon.columns = [pysd.builder.make_python_identifier(x) for x in canon.columns.values]
        
        output = model.run(return_columns=list(canon.columns.values))
        
        assert (canon-output).max().max() < 1
        
        print '.',
        
    except ParseError as e:
        print 'F',
        
        err_str += '-'*60 + '\n'
        err_str += 'Parse Error in: %s \n'%modelfile
        err_str += 'At line: %i, column%i \n'%(e.line(), e.column())
        err_str += 'On rule: %s \n'%e.expr.name
        err_str += e.expr.__repr__() + '\n\n'
        err_str += e.text.splitlines()[e.line()-1] + '\n' #line numbers are 1 based, most likely
        err_str += '^'.rjust(e.column()) + '\n\n'
        
    except IOError as e:
        print 'E',
        
        err_str += '-'*60 + '\n'
        err_str += 'Failure loading canonical output for: %s \n'%modelfile 
        
    except AssertionError as e:
        print 'F',
        
        err_str += '-'*60 + '\n'
        err_str += 'Model output does not match canon for: %s \n\n'%modelfile
        err_str += str((canon-output).max())
        err_str += '\n' 

        

print err_str