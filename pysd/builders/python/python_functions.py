
# functions that can be diretcly applied over an array
functionspace = {
    # directly build functions without dependencies
    "elmcount": ("len(%(0)s)", None),

    # directly build numpy based functions
    "abs": ("np.abs(%(0)s)", ("numpy",)),
    "min": ("np.minimum(%(0)s, %(1)s)", ("numpy",)),
    "max": ("np.maximum(%(0)s, %(1)s)", ("numpy",)),
    "exp": ("np.exp(%(0)s)", ("numpy",)),
    "sin": ("np.sin(%(0)s)", ("numpy",)),
    "cos": ("np.cos(%(0)s)", ("numpy",)),
    "tan": ("np.tan(%(0)s)", ("numpy",)),
    "arcsin": ("np.arcsin(%(0)s)", ("numpy",)),
    "arccos": ("np.arccos(%(0)s)", ("numpy",)),
    "arctan": ("np.arctan(%(0)s)", ("numpy",)),
    "sinh": ("np.sinh(%(0)s)", ("numpy",)),
    "cosh": ("np.cosh(%(0)s)", ("numpy",)),
    "tanh": ("np.tanh(%(0)s)", ("numpy",)),
    "sqrt": ("np.sqrt(%(0)s)", ("numpy",)),
    "ln": ("np.log(%(0)s)", ("numpy",)),
    "log": ("(np.log(%(0)s)/np.log(%(1)s))", ("numpy",)),
    # NUMPY: "invert_matrix": ("np.linalg.inv(%(0)s)", ("numpy",)),

    # vector functions with axis to apply over
    # NUMPY:
    # "prod": "np.prod(%(0)s, axis=%(axis)s)", ("numpy",)),
    # "sum": "np.sum(%(0)s, axis=%(axis)s)", ("numpy",)),
    # "vmax": "np.max(%(0)s, axis=%(axis)s)", ("numpy", )),
    # "vmin": "np.min(%(0)s, axis=%(axis)s)", ("numpy",))
    "prod": ("prod(%(0)s, dim=%(axis)s)", ("functions", "prod")),
    "sum": ("sum(%(0)s, dim=%(axis)s)", ("functions", "sum")),
    "vmax": ("vmax(%(0)s, dim=%(axis)s)", ("functions", "vmax")),
    "vmin": ("vmin(%(0)s, dim=%(axis)s)", ("functions", "vmin")),

    # functions defined in pysd.py_bakcend.functions
    "active_initial": (
        "active_initial(__data[\"time\"].stage, lambda: %(0)s, %(1)s)",
        ("functions", "active_initial")),
    "if_then_else": (
        "if_then_else(%(0)s, lambda: %(1)s, lambda: %(2)s)",
        ("functions", "if_then_else")),
    "integer": (
        "integer(%(0)s)",
        ("functions", "integer")),
    "invert_matrix": (  # NUMPY: remove
        "invert_matrix(%(0)s)",
        ("functions", "invert_matrix")),  # NUMPY: remove
    "modulo": (
        "modulo(%(0)s, %(1)s)",
        ("functions", "modulo")),
    "pulse": (
        "pulse(__data['time'], %(0)s, width=%(1)s)",
        ("functions", "pulse")),
    "Xpulse": (
        "pulse(__data['time'], %(0)s, magnitude=%(1)s)",
        ("functions", "pulse")),
    "pulse_train": (
        "pulse(__data['time'], %(0)s, repeat_time=%(1)s, width=%(2)s, "\
        "end=%(3)s)",
        ("functions", "pulse")),
    "Xpulse_train": (
        "pulse(__data['time'], %(0)s, repeat_time=%(1)s, magnitude=%(2)s)",
        ("functions", "pulse")),
    "quantum": (
        "quantum(%(0)s, %(1)s)",
        ("functions", "quantum")),
    "Xramp": (
        "ramp(__data['time'], %(0)s, %(1)s)",
        ("functions", "ramp")),
    "ramp": (
        "ramp(__data['time'], %(0)s, %(1)s, %(2)s)",
        ("functions", "ramp")),
    "step": (
        "step(__data['time'], %(0)s, %(1)s)",
        ("functions", "step")),
    "xidz": (
        "xidz(%(0)s, %(1)s, %(2)s)",
        ("functions", "xidz")),
    "zidz": (
        "zidz(%(0)s, %(1)s)",
        ("functions", "zidz")),

    # random functions must have the shape of the component subscripts
    # most of them are shifted, scaled and truncated
    # TODO: it is difficult to find same parametrization in Python,
    # maybe build a new model
    "random_0_1": (
        "np.random.uniform(0, 1, size=%(size)s)",
        ("numpy",)),
    "random_uniform": (
        "np.random.uniform(%(0)s, %(1)s, size=%(size)s)",
        ("numpy",)),
    "random_normal": (
        "stats.truncnorm.rvs(%(0)s, %(1)s, loc=%(2)s, scale=%(3)s,"
        " size=%(size)s)",
        ("scipy", "stats")),
}
