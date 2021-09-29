from pysd import cache

__pysd_version__ = "2.99.3"

__data = {'scope': None, 'time': lambda: 0}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


def initial_time():
    return 0
