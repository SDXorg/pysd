Julia Builder
=============

PySD can translate Vensim (``.mdl``) and Stella (``.xmile`` / ``.stmx``) models
into standalone Julia files that run without Python or PySD at runtime.
Two backends are available:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Backend
     - How it works
     - When to use
   * - ``"ode"`` *(default)*
     - Emits a plain ``rhs!(du, u, p, t)`` function solved by
       ``OrdinaryDiffEq.jl``
     - General use; fast startup, full subscript support
   * - ``"mtk"``
     - Emits a ``ModelingToolkit.ODESystem``; ModelingToolkit performs
       symbolic simplification before solving
     - Small-to-medium models where you want symbolic analysis, LaTeX
       export, or automatic sparsity detection

.. note::
   The MTK backend runs ``structural_simplify`` before solving, which can
   take minutes to hours for large subscripted models.  For production runs
   the ODE backend is recommended.


Prerequisites
-------------

Julia 1.10 or later. Install via `juliaup <https://github.com/JuliaLang/juliaup>`_::

   curl -fsSL https://install.julialang.org | sh

Install the required Julia packages once::

   julia -e 'using Pkg; Pkg.add([
       "OrdinaryDiffEq",
       "OrdinaryDiffEqLowOrderRK",
       "DataInterpolations",
       "NCDatasets",
       "JSON3",
       "ModelingToolkit",   # only needed for the mtk backend
       "Symbolics",         # only needed for the mtk backend
   ])'

Then install the ``PySD.jl`` companion library. It lives in a submodule of
the PySD repo (``pysd/builders/julia/PySD.jl/``) and can also be found at
https://github.com/rogersamso/PySD.jl.  From the root of your PySD checkout::

   git submodule update --init pysd/builders/julia/PySD.jl
   julia -e 'using Pkg; Pkg.develop(path="pysd/builders/julia/PySD.jl")'


Translating a model
--------------------

From Python
^^^^^^^^^^^

.. code-block:: python

   import pysd

   # ODE backend (default)
   path = pysd.translate_to_julia("model.mdl")

   # MTK backend
   path = pysd.translate_to_julia("model.mdl", backend="mtk")

   # Split views — one module file per Vensim view
   path = pysd.translate_to_julia("model.mdl", split_views=True)

   # JSON data format — companion _data.json instead of inline Excel reads
   path = pysd.translate_to_julia("model.mdl", data_format="json")

**Parameters**

``backend``
   ``"ode"`` (default) or ``"mtk"``.

``split_views``
   When ``True`` and the model has multiple Vensim views, the output is
   split into a main ``.jl`` file and one module file per view under a
   ``modules_<name>/`` directory.

``data_format``
   ``"hardcoded"`` (default) reads Excel files at Julia startup via
   ``PySD.jl`` helpers.  ``"json"`` writes a companion
   ``<model>_data.json`` file and reads it via ``JSON3.jl``.

``encoding``
   Source file encoding (Vensim only). If ``None`` the encoding is
   detected from the model file header.

From the command line
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -c "import pysd; pysd.translate_to_julia('model.mdl')"


Running the model
-----------------

The generated ``.jl`` file is self-contained and can be run directly::

   julia --project=/path/to/PySD.jl model.jl

It prints progress to stdout, runs the simulation, and writes a NetCDF
results file (``<model>_results.nc``) next to the ``.jl`` file.

You can also ``include`` the file interactively:

.. code-block:: julia

   include("model.jl")       # defines run_model, u0, tspan, …

   sol = run_model()         # run with defaults

The ``run_model`` function accepts keyword arguments:

.. code-block:: julia

   # Higher-order solver
   sol = run_model(solver=Tsit5())

   # Finer time step
   sol = run_model(dt=0.01)

   # Custom time span
   sol = run_model(tspan=(2000.0, 2100.0))

   # Custom initial conditions
   sol = run_model(u0=my_u0)


Choosing a solver
^^^^^^^^^^^^^^^^^

The default is ``Euler()``, which matches Vensim's integration method.
All solvers from `OrdinaryDiffEq.jl
<https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/>`_ work.

.. list-table::
   :header-rows: 1

   * - Solver
     - Notes
   * - ``Euler()``
     - Default; matches Vensim output exactly
   * - ``Tsit5()``
     - Fast, accurate explicit solver; good general replacement
   * - ``Rodas5P()``
     - Stiff systems (widely different time scales)
   * - ``RK4()``
     - Classic 4th-order Runge-Kutta


Accessing results
^^^^^^^^^^^^^^^^^

``sol`` is a standard
`DiffEq solution object <https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/>`_.

**ODE backend** — use ``observe(u, t)`` to read any variable at any saved
time step:

.. code-block:: julia

   # All saved time points
   sol.t

   # Read a scalar variable at every time step
   obs = [mod.observe(sol.u[i], sol.t[i]) for i in eachindex(sol.t)]
   population = [o["population"] for o in obs]

   # Read a subscripted variable (returns a Vector)
   stock_a = [o["stock_a"] for o in obs]

   # Individual subscript element
   stock_a_entry1 = [o["stock_a"][1] for o in obs]

**MTK backend** — access variables symbolically via ``sys``:

.. code-block:: julia

   sol[sys.population]        # time series for a scalar variable
   sol(2025.0)[sys.gdp]       # interpolate at a specific time


Saving results
^^^^^^^^^^^^^^

The generated file calls ``save_results`` automatically, writing a
`NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ file.  You can
also call it manually:

.. code-block:: julia

   # ODE backend
   save_results(sol, _state_map, _dim_labels, "output.nc")

   # MTK backend
   save_results(sol, sys, _dim_labels, "output.nc")

Read the results with any NetCDF library, e.g. in Python:

.. code-block:: python

   import xarray as xr
   ds = xr.open_dataset("model_results.nc")
   print(ds["population"])


External data (Excel files)
----------------------------

Models that use ``GET DIRECT CONSTANTS``, ``GET DIRECT LOOKUPS``, or
``GET DIRECT DATA`` are fully supported.  The translated Julia model reads
from the **same Excel files at runtime** — no intermediate conversion is
needed.

Excel file paths in the generated code are relative to the ``.jl`` file
(via ``@__DIR__``), so Excel files must remain at their original locations
relative to the model.

All Vensim cell reference modes are supported:

- **Named ranges** — ``GET DIRECT CONSTANTS('data.xlsx', 'Sheet1', 'param_name')``
- **Cell references** — ``GET DIRECT CONSTANTS('data.xlsx', 'Sheet1', 'B2')``
- **Row/column mode** — ``GET DIRECT LOOKUPS('data.xlsx', 'Sheet1', '4', 'C5')``

Excel files are cached in memory after the first read.


PySD.jl companion library
--------------------------

``PySD.jl`` (located at ``pysd/builders/julia/PySD.jl/``) provides the
runtime helper functions used by generated models, imported via
``using PySD``.

**Vensim built-in functions**

- ``pysd_xidz(x, y, z)`` — safe division; returns ``z`` when ``y == 0``
- ``pysd_zidz(x, y)`` — safe division; returns ``0`` when ``y == 0``
- ``pysd_pulse(t, start, width)``
- ``pysd_pulse_train(t, start, interval, width, end_time)``
- ``pysd_ramp(t, slope, start, end)``
- ``pysd_step(t, height, step_time)``
- ``pysd_log_base(x, base)``
- ``pysd_logical_and(a, b)``, ``pysd_logical_or(a, b)``, ``pysd_logical_not(a)``
- ``pysd_safe(x)`` — replaces ``NaN``/``Inf`` with ``0.0`` (guards array
  allocations against uninitialised reads in subscripted equations)
- ``pysd_allocate_by_priority(request, priority, width, supply)`` — Vensim priority allocation
- ``pysd_allocate_available(request, pp, avail)`` — Vensim profile-based demand allocation

**Excel data readers**

- ``pysd_xlsx_read_constant(path, sheet, name)``
- ``pysd_xlsx_read_series(path, sheet, x_ref, y_ref)``
- ``pysd_xlsx_build_lookup_dispatch(path, sheet, x_ref, y_ref)``

**Result writer**

- ``save_results(sol, state_map_or_sys, dim_labels, path)`` — writes a
  NetCDF file; dispatches on ODE ``state_map`` (``AbstractVector``) or MTK
  ``sys`` (``AbstractSystem``)

**LaTeX export** *(MTK backend only)*

- ``pysd_export_latex(sys; filename=nothing)`` — returns a LaTeX string of
  the simplified ODE system; writes a standalone ``.tex`` file when
  ``filename`` is given


Exporting equations to LaTeX (MTK only)
----------------------------------------

.. code-block:: julia

   include("model.jl")           # MTK backend

   # LaTeX string
   tex = export_latex()

   # Write a standalone compilable .tex file
   export_latex(filename="equations.tex")

The exported equations correspond to the **structurally simplified** system
after ModelingToolkit's index reduction — redundant auxiliaries are
substituted away.


Supported Vensim features
--------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - ODE backend
     - MTK backend
   * - Stocks (``INTEG``)
     - Supported
     - Supported
   * - Auxiliaries
     - Supported
     - Supported
   * - Constants / parameters
     - Supported
     - Supported
   * - Subscripts / arrays (1D, 2D)
     - Supported
     - Supported
   * - Lookup tables (inline)
     - Supported
     - Supported
   * - ``GET DIRECT CONSTANTS``
     - Supported
     - Supported
   * - ``GET DIRECT LOOKUPS``
     - Supported
     - Supported
   * - ``GET DIRECT DATA``
     - Supported
     - Supported
   * - ``GET DIRECT SUBSCRIPT`` (subscript ranges from Excel)
     - Supported
     - Supported (read at translation time; correct array shapes)
   * - ``SMOOTH`` / ``SMOOTH3`` / ``SMOOTHN``
     - Supported
     - Supported
   * - ``DELAY1`` / ``DELAY3`` / ``DELAYN``
     - Supported
     - Supported
   * - ``DELAY FIXED``
     - Supported (exact N-stage Euler pipeline; falls back to first-order ODE if delay time is dynamic)
     - Supported
   * - ``TREND``, ``FORECAST``
     - Supported
     - Supported
   * - ``SAMPLE IF TRUE``
     - Supported (instantaneous ifelse output)
     - Supported
   * - ``INITIAL``
     - Supported
     - Supported
   * - ``IF THEN ELSE``
     - Supported (``ifelse``)
     - Supported (``ifelse``)
   * - ``PULSE``, ``STEP``, ``RAMP``, ``PULSE TRAIN``
     - Supported
     - Supported
   * - ``XIDZ``, ``ZIDZ``
     - Supported
     - Supported
   * - Multiple views (``split_views=True``)
     - Supported
     - Supported
   * - ``GAME``
     - Supported (passes through; interactive play ignored)
     - Supported
   * - ``ELMCOUNT``
     - Supported (resolved to integer literal at translation time)
     - Supported
   * - ``DATA`` variables (tab-delimited ``.tab`` files)
     - Supported (runtime ``_tab_val`` interpolation via ``tab_data_files=`` parameter)
     - Not supported
   * - ``DATA`` variables fed from another model's NetCDF output
     - Supported (pass ``nc_data_files=["results.nc"]`` to ``run_model()``; scalars and
       subscripted variables supported)
     - Not supported
   * - Subscripted ``GET DIRECT LOOKUPS`` > 2D
     - Partial (flattened to first column with warning)
     - Partial
   * - ``SMOOTH``/``DELAY`` with non-integer order
     - Partial (order rounded to nearest integer with warning)
     - Partial
   * - ``EXCEPT`` exclusion on 3-D subscripts
     - Supported (per-index comprehension equations)
     - Supported
   * - Macros (stateless)
     - Supported (companion ``.jl`` function file per macro)
     - Supported
   * - Macros (stateful — ``INTEG`` inside macro)
     - Not supported (placeholder ``return 0.0`` with warning)
     - Not supported
   * - XMILE ``MIN``/``MAX`` aggregation (``vmin_xmile``, ``vmax_xmile``)
     - Supported
     - Supported
   * - XMILE ``DELAY`` embedded in expression
     - Supported (lifted to pipeline auxiliary stocks)
     - Supported
   * - ``ALLOCATE AVAILABLE`` / ``ALLOCATE BY PRIORITY``
     - Supported (exact Vensim algorithm via PySD.jl helpers)
     - Supported

.. note::
   When the builder encounters an unsupported or partially-supported construct
   it emits a Python ``UserWarning`` during translation and writes a
   placeholder (``0.0``) in the generated file.  Always review warnings after
   translation to identify gaps.

Comparison with the Python builder
-----------------------------------

The Python builder supports every Vensim/Stella construct that PySD can
parse.  The Julia builder does not yet cover:

.. list-table::
   :header-rows: 1

   * - Feature
     - Python builder
     - Julia builder
   * - ``ALLOCATE AVAILABLE`` / ``ALLOCATE BY PRIORITY``
     - Full
     - Supported via ``pysd_allocate_available`` / ``pysd_allocate_by_priority`` in PySD.jl
   * - ``DATA`` variables (tab-delimited ``.tab`` file source)
     - Full
     - Supported — pass ``tab_data_files=["data.tab"]`` to ``run_model()``
   * - ``DATA`` variables fed from another model's NetCDF output
     - Full
     - Supported — pass ``nc_data_files=["other_model_results.nc"]`` to ``run_model()``
   * - Step-by-step execution (``model.step()``)
     - Full — essential for ABM coupling (e.g. Mesa)
     - Not supported — ``run_model()`` always runs the full simulation in one call
   * - Mid-run parameter injection (``model.set_components()``)
     - Full — swap variable equations between steps
     - Not supported — parameters can only be changed before calling ``run_model()``
   * - State export/import (``model.export()`` / ``model.import_()``)
     - Full — snapshot and restore model state for warm restarts or ensemble branching
     - Not supported
   * - Submodel selection (``model.select_submodel()``)
     - Full — prune to a variable subset for faster targeted simulation
     - Not supported — the full model is always simulated
   * - Subscripted lookups with > 2 subscript dimensions
     - Full
     - Flattened to first column (with warning)
   * - ``SMOOTH``/``DELAY`` with non-integer order
     - Full (arbitrary real order)
     - Order rounded to nearest integer (with warning)
   * - ``EXCEPT`` exclusion on 3-D subscripts
     - Full
     - Supported (per-index comprehension equations, same as 1-D and 2-D)
   * - Stateless macros (``MACRO`` … ``END OF MACRO`` without ``INTEG``)
     - Full (inlined)
     - Companion ``.jl`` function file generated; included via ``include``
   * - Stateful macros (``INTEG`` inside ``MACRO`` … ``END OF MACRO``)
     - Full
     - Not supported; placeholder ``return 0.0`` emitted with warning
   * - ``GAME`` interactive input
     - Full
     - Passes through; interactive value ignored in batch simulation
   * - ``DELAY FIXED`` exact semantics
     - Full (discrete transport delay)
     - Supported (exact N-stage Euler pipeline matching Vensim ring-buffer semantics); dynamic delay times fall back to a first-order ODE approximation with a warning
   * - ``SAMPLE IF TRUE`` exact semantics
     - Full (holds last-true value)
     - Supported (instantaneous ifelse output; hold stock updated each step)


Limitations
-----------

- **MTK structural analysis** scales poorly with model size.  For models
  with hundreds of subscripted equations (which expand into thousands of
  scalar equations) ``structural_simplify`` can take hours.  Use the ODE
  backend for large models.

- **EXCEPT subscript exclusion** on 4-D or higher subscripts emits a
  plain broadcast equation (the exclusion is ignored) with a warning.
  1-D, 2-D, and 3-D EXCEPT are fully supported.

- **SAMPLE IF TRUE** uses an ODE hold-stock to track the last sampled value
  and instantaneously outputs ``ifelse(condition, input, hold)`` at each step.
  Behaviour matches Vensim exactly when using the Euler solver.

- **DELAY FIXED** is implemented as an exact N-stage Euler pipeline
  (``N = round(delay_time / time_step)``), which matches Vensim's ring-buffer
  transport delay exactly when the delay time is a static constant.  If the
  delay time cannot be evaluated at translation time (e.g. it is a dynamic
  expression), the builder falls back to a first-order ODE approximation and
  emits a warning.

- **Non-integer SMOOTH/DELAY order** is rounded to the nearest integer
  (defaulting to 3) with a warning; the Python builder supports arbitrary
  real-valued orders.

- **Tab-delimited DATA variables** (``.tab`` file sources) are supported for
  the ODE backend.  Pass the file paths as ``run_model(tab_data_files=["data.tab"])``;
  the model reads and interpolates the time series at runtime.  The MTK backend
  does not yet support tab-delimited DATA variables.

- **Step-by-step execution** is not available.  The Python builder exposes
  ``model.set_stepper()`` / ``model.step()`` for advancing the simulation one
  time step at a time, which is the standard pattern for coupling with
  agent-based frameworks (e.g. Mesa, Agents.jl).  The Julia builder has no
  equivalent — ``run_model()`` always executes the full simulation in a single
  call.

- **Mid-run parameter injection** is not available.  The Python builder's
  ``model.set_components()`` can replace any variable's equation with a new
  function or constant value at any point during a run.  In the Julia builder,
  parameters can only be changed before calling ``run_model()`` (e.g. by
  modifying ``u0`` or editing the generated constants).

- **State export/import** is not available.  The Python builder's
  ``model.export()`` / ``model.import_()`` snapshot and restore the full model
  state — stock values, stateful caches, and current time — enabling warm
  restarts and ensemble branching from a common saved point.  The Julia builder
  writes results to NetCDF via ``save_results()`` but cannot restore mid-run
  state.

- **Submodel selection** is not available.  The Python builder's
  ``model.select_submodel()`` prunes the model to a requested subset of
  variables, which can substantially reduce simulation time when only part of
  the model is needed.  The Julia builder always simulates the full model.

- The Euler solver (default) produces output that matches Vensim's built-in
  integration.  Higher-order solvers (e.g. ``Tsit5()``) are generally more
  accurate but may produce slightly different results.
