__pysd_version__ = "2.99.3"

_namespace = {}
_dependencies = {}

__data = {'scope': None, 'time': lambda: 0}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


def initial_time():
    return 0
