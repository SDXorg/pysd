import os.path
import textwrap
import glob
from pysd import builder

test_dir = 'test-models/'
vensim_test_files = glob.glob(test_dir+'tests/*/*.mdl')

tests = []
for file_path in vensim_test_files:
    (path, file_name) = os.path.split(file_path)
    (name, ext) = os.path.splitext(file_name)

    test_name = builder.make_python_identifier(path.split('/')[-1])[0]

    test_func_string = """
        def test_%(test_name)s(self):
            from test_utils import runner, assert_frames_close
            output, canon = runner('%(file_path)s')
            assert_frames_close(output, canon, rtol=rtol)
        """ % {
        'file_path': file_path,
        'test_name': test_name,
    }
    tests.append(test_func_string)

file_string = textwrap.dedent("""
    from unittest import TestCase

    rtol = .05


    class TestIntegrationExamples(TestCase):
    %(tests)s

    """ % {'tests': ''.join(tests)})

with open('integration_test_pysd.py', 'w') as ofile:
    ofile.write(file_string)
