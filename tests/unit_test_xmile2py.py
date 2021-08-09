import os
import unittest
import tempfile
from io import StringIO

from pysd.py_backend.xmile.xmile2py import translate_xmile

_root = os.path.dirname(__file__)
TARGET_STMX_FILE = os.path.join(_root, "test-models/tests/game/test_game.stmx")


class TestXmileConversion(unittest.TestCase):

    def test_python_file_creation(self):
        with open(TARGET_STMX_FILE, 'r') as stmx:
            contents = stmx.read()

        # Write out contents to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(contents)

        # Convert file (should not raise error)
        generated_file = translate_xmile(temp_file.name)

        # Check if both source file and python file exists
        try:
            assert generated_file != temp_file.name, "Accidental replacement of original model file!"
            assert generated_file.endswith('.py'), 'File created without python extension'
            assert os.path.exists(temp_file.name) and os.path.exists(generated_file), 'Expected files are missing'
        finally:
            os.remove(temp_file.name)

            try:
                os.remove(generated_file)
            except FileNotFoundError:
                # Okay if python file is missing
                pass

    def test_multiline_equation(self):
        with open(TARGET_STMX_FILE, 'r') as stmx:
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

        try:
            assert idx > 0, 'Correct, generated, equation not found'
        finally:
            os.remove(temp_file.name)
            os.remove(generated_file)
