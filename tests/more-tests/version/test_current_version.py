__pysd_version__ = "2.99.3"

_namespace = {}
_dependencies = {}

__data = {'scope': None, 'time': lambda: 0}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


def _time_step():
    return 0.5


def _initial_time():
    return 0


def _final_time():
    return 0.5


def _saveper():
    return 0.5
