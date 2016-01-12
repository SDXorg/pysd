import glob
import os.path
import traceback
from parsimonious.exceptions import ParseError, IncompleteParseError, VisitationError
import pandas as pd
import pysd
import sys
import timeit
starttime = timeit.time.time()


test_dir = 'tests/test-models-master/'


# get tests from github, using another script
if not os.path.isdir(test_dir):
    import get_tests


vensim_testfiles = glob.glob(test_dir+'*/*/*.mdl')
#xmile_testfiles = glob.glob(test_dir+'*/*/*.xmile')
xmile_testfiles = []
testfiles = vensim_testfiles + xmile_testfiles

print "Testing module at location: %s\n"%pysd.__file__
err_str = '\n\n'
threshold = 1

success_count = 0
err_count = 0
fail_count = 0

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
        success_count += 1
        
    except ParseError as e:
        print 'F',
        
        err_str += '='*60 + '\n'
        err_str += 'Test Failure of: %s \n'%modelfile
        err_str += '-'*60 + '\n'
        err_str += 'Parsing Error at line: %i, column%i.\n'%(e.line(), e.column())
        err_str += 'On rule: %s \n\n'%e.expr.__repr__()
        err_str += str(e)
        #err_str += e.text.splitlines()[e.line()-1] + '\n' #line numbers are 1 based, most likely
        #err_str += '^'.rjust(e.column())
        err_str += '\n\n'

        fail_count += 1

    except VisitationError as e:
        print 'F',

        err_str += '='*60 + '\n'
        err_str += 'Test Failure of: %s \n'%modelfile
        err_str += '-'*60 + '\n'
        err_str += str(e.args[0])
        err_str += '\n\n'
        
    except IOError as e:
        print 'E',
        
        err_str += '='*60 + '\n'
        err_str += 'Test Error attempting: %s \n'%modelfile
        err_str += '-'*60 + '\n'
        err_str += 'Could not load canonical output\n'
        err_str += '\n\n'

        err_count += 1

    except AssertionError as e:
        print 'F',

        err_str += '='*60 + '\n'
        err_str += 'Test Failure of: %s \n'%modelfile
        err_str += '-'*60 + '\n'
        err_str += 'Model output does not match canon.\n'
        err_str += 'Variable       Maximum Discrepancy\n'
        err_str += str((canon-output).max())
        err_str += '\n\n'

        fail_count += 1

    except Exception as e:
        print 'E',
        err_str += '='*60 + '\n'
        err_str += 'Unknown issue with: %s \n'%modelfile
        err_str += '-'*60 + '\n'
        err_str += str(e.args[0])
        err_str += '\n\n'

        err_count += 1

endtime = timeit.time.time()

err_str += '='*60 + '\n'
err_str += 'Attempted %i tests in %.02f seconds \n'%(len(testfiles), (endtime-starttime))
err_str += '%i Successes, %i Failures, %i Errors'%(success_count, fail_count, err_count)

print err_str