import os
import unittest
import tempfile
from io import StringIO

from pysd.py_backend.xmile.xmile2py import translate_xmile



class TestEquationStringParsing(unittest.TestCase):

    def test_multiline_equation():
        with open('tests/test-models/tests/game/test_game.stmx', 'r') as stmx:
            contents = stmx.read()

        # Insert line break in equation definition
        contents = contents.replace('<eqn>(Stock+Constant)</eqn>', '<eqn>(Stock+\nConstant)</eqn>')

        # Write out contents to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(contents)

        # Convert file (should not raise error)
        generated_file = translate_xmile(temp_file.name)

        with open(generated_file, 'r') as fp:
            contents = fp.read()

        idx = contents.find('stock() + constant()')

        assert idx > 0, 'Correct, generated, equation not found'

        os.remove(temp_file.name)
        os.remove(generated_file+'.py')
