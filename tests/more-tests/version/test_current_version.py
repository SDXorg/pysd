from pysd import Component

__pysd_version__ = "3.0.0"

_dependencies = {}

__data = {'scope': None, 'time': lambda: 0}

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 20,
    "time_step": lambda: 1,
    "saveper": lambda: time_step()
}

component = Component()


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


@component.add(name="Time")
def time():
    return __data["time"]()


@component.add(name="Initial time")
def initial_time():
    return __data["time"].initial_time()


@component.add(name="Final time")
def final_time():
    return __data["time"].final_time()


@component.add(name="Time step")
def time_step():
    return __data["time"].time_step()


@component.add(name="Saveper")
def saveper():
    return __data["time"].saveper()
