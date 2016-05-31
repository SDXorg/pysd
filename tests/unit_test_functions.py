from unittest import TestCase


class TestPySD(TestCase):
    def test_ramp(self):
        from pysd import functions

        def time():
            return 4

        functions.ramp(.3, 0, 7)