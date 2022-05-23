__pysd_version__ = "1.5.0"

_dependencies = {}

__data = {'scope': None, 'time': lambda: 0}

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 20,
    "time_step": lambda: 1,
    "saveper": lambda: time_step()
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


def time():
    return __data["time"]()


def initial_time():
    return __data["time"].initial_time()


def final_time():
    return __data["time"].final_time()


def time_step():
    return __data["time"].time_step()


def saveper():
    return __data["time"].saveper()
