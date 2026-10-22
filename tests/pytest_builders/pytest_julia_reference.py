"""
Julia builder: reference validation against the test-models outputs.

Every model of the Python integration pathways (``vensim_test`` and
``xmile_test``) is translated with each Julia backend, run, and compared with
its reference output (Vensim/Stella) using the same tools as the Python
pathway tests (``load_outputs`` and ``assert_frames_close``).

Models that are known not to work yet are listed in ``KNOWN_FAILURES`` and
marked as strict xfails: a fix shows up as XPASS, so the entry can be removed.

All models of one backend run in a single Julia process (packages are loaded
and compiled once).  A model that does not finish within ``MODEL_TIMEOUT``
seconds is reported as a timeout and the remaining models continue in a new
process.

Environment variables:
    PYSD_JULIA_BACKENDS  comma-separated backends to test (default "ode,mtk")
    PYSD_JULIA_MTK       "all" (default) or "subset" (MTK_SUBSET only)
    PYSD_JULIA_TIMEOUT   per-model timeout in seconds (default 600)
    PYSD_JULIA_STARTUP   extra seconds for the first model of each Julia
                         process, which loads the packages (default 600)
    PYSD_JULIA_REQUIRED  if set, fail (instead of skip) when Julia or the
                         packages a backend needs are not found

Julia's output is written to ``_julia.log`` in each backend's work folder.
"""
import functools
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _pytest.mark.structures import ParameterSet

from pysd.builders.julia.julia_model_builder import expand_subscript_ranges
from pysd.builders.julia.namespace import JuliaNamespaceManager
from pysd.py_backend.utils import detect_encoding, load_outputs
from pysd.tools.benchmarking import assert_frames_close

from tests.pytest_integration import pytest_integration_vensim_pathway as _vensim
from tests.pytest_integration import pytest_integration_xmile_pathway as _xmile


TEST_MODELS = Path(__file__).parents[1] / "test-models" / "tests"
BACKENDS = [b for b in os.environ.get("PYSD_JULIA_BACKENDS", "ode,mtk").split(",") if b]
MODEL_TIMEOUT = int(os.environ.get("PYSD_JULIA_TIMEOUT", "600"))
#: Extra seconds for the first model of each Julia process (package loading).
STARTUP_ALLOWANCE = int(os.environ.get("PYSD_JULIA_STARTUP", "600"))

#: Simulation-control columns of the reference outputs -> Julia names.
CONTROL_COLUMNS = {
    "INITIAL TIME": "initial_time", "FINAL TIME": "final_time",
    "TIME STEP": "time_step", "SAVEPER": "saveper",
}

#: MTK is slow (symbolic processing and compilation per model).  On pull
#: requests it runs on this representative subset; the full set runs weekly,
#: on demand, and on pull requests labelled ``test-mtk`` (PYSD_JULIA_MTK=all).
MTK_SUBSET = {
    # stocks, initialisation, control
    "vensim:abs", "vensim:chained_initialization", "vensim:time", "vensim:number_handling",
    # built-in functions
    "vensim:builtin_max", "vensim:exp", "vensim:if_stmt", "vensim:logicals", "vensim:xidz_zidz",
    "vensim:input_functions", "vensim:nested_functions", "vensim:invert_matrix",
    # lookups
    "vensim:lookups", "vensim:lookups_inline", "vensim:lookups_with_expr", "vensim:with_lookup",
    # stateful functions
    "vensim:smooth", "vensim:sample_if_true", "vensim:trend", "vensim:delay_parentheses",
    # subscripts
    "vensim:subscript_1d_arrays", "vensim:subscript_2d_arrays", "vensim:subscript_3d_arrays",
    "vensim:subscript_individually_defined_stocks", "vensim:subscript_transposition",
    "vensim:subscript_subranges", "vensim:subscript_mixed_assembly", "vensim:subscripted_flows",
    "vensim:subscripted_if_then_else",
    # XMILE
    "xmile:abs", "xmile:lookups", "xmile:non_negative_flows", "xmile:min_max_1arg",
}
MTK_SCOPE = os.environ.get("PYSD_JULIA_MTK", "all")

#: (case, backend) -> reason.  Strict xfail: remove the entry once it passes.
KNOWN_FAILURES = {
    # values differ from the reference (#499)
    ("vensim:active_initial", "mtk"): "values differ from the reference (#499)",
    ("vensim:active_initial", "ode"): "values differ from the reference (#499)",
    ("vensim:active_initial_circular", "ode"): "values differ from the reference (#499)",
    ("vensim:arithmetics", "mtk"): "values differ from the reference (#499)",
    ("vensim:arithmetics", "ode"): "values differ from the reference (#499)",
    ("vensim:arithmetics_exp", "mtk"): "values differ from the reference (#499)",
    ("vensim:arithmetics_exp", "ode"): "values differ from the reference (#499)",
    ("vensim:chained_initialization", "ode"): "values differ from the reference (#499)",
    ("vensim:data_from_other_model", "ode"): "values differ from the reference (#499)",
    ("vensim:delay_fixed", "mtk"): "values differ from the reference (#499)",
    ("vensim:delay_fixed", "ode"): "values differ from the reference (#499)",
    ("vensim:delay_pipeline", "mtk"): "values differ from the reference (#499)",
    ("vensim:delay_pipeline", "ode"): "values differ from the reference (#499)",
    ("vensim:delays", "mtk"): "values differ from the reference (#499)",
    ("vensim:delays", "ode"): "values differ from the reference (#499)",
    ("vensim:get_values_order", "mtk"): "values differ from the reference (#499)",
    ("vensim:get_values_order", "ode"): "values differ from the reference (#499)",
    ("vensim:get_xls_cellrange", "ode"): "values differ from the reference (#499)",
    ("vensim:lookups_inline_spaces", "mtk"): "values differ from the reference (#499)",
    ("vensim:macro_stock", "ode"): "values differ from the reference (#499)",
    ("vensim:rounding", "mtk"): "values differ from the reference (#499)",
    ("vensim:rounding", "ode"): "values differ from the reference (#499)",
    ("vensim:smooth_and_stock", "ode"): "values differ from the reference (#499)",
    ("vensim:subscripted_round", "mtk"): "values differ from the reference (#499)",
    ("vensim:subscripted_round", "ode"): "values differ from the reference (#499)",
    ("vensim:with_lookup", "ode"): "values differ from the reference (#499)",
    ("xmile:arithmetics_exp", "mtk"): "values differ from the reference (#499)",
    ("xmile:arithmetics_exp", "ode"): "values differ from the reference (#499)",
    ("xmile:chained_initialization", "ode"): "values differ from the reference (#499)",
    ("xmile:delay_xmile", "mtk"): "values differ from the reference (#499)",
    ("xmile:delay_xmile", "ode"): "values differ from the reference (#499)",
    ("xmile:non_negative_all1", "mtk"): "values differ from the reference (#499)",
    ("xmile:non_negative_all1", "ode"): "values differ from the reference (#499)",
    ("xmile:non_negative_all2", "mtk"): "values differ from the reference (#499)",
    ("xmile:non_negative_all2", "ode"): "values differ from the reference (#499)",
    ("xmile:non_negative_stocks", "mtk"): "values differ from the reference (#499)",
    ("xmile:non_negative_stocks", "ode"): "values differ from the reference (#499)",
    ("xmile:non_negative_stocks_behavior", "mtk"): "values differ from the reference (#499)",
    ("xmile:non_negative_stocks_behavior", "ode"): "values differ from the reference (#499)",
    ("xmile:rounding", "mtk"): "values differ from the reference (#499)",
    ("xmile:rounding", "ode"): "values differ from the reference (#499)",
    # generated code refers to an undefined name (#500)
    ("vensim:arguments", "ode"): "generated code refers to an undefined name (#500)",
    ("vensim:control_vars", "ode"): "generated code refers to an undefined name (#500)",
    ("vensim:data_from_other_model", "mtk"): "generated code refers to an undefined name (#500)",
    ("vensim:dynamic_final_time", "ode"): "generated code refers to an undefined name (#500)",
    ("vensim:macro_cross_reference", "mtk"): "generated code refers to an undefined name (#500)",
    ("vensim:macro_expression", "mtk"): "generated code refers to an undefined name (#500)",
    ("vensim:macro_multi_expression", "mtk"): "generated code refers to an undefined name (#500)",
    ("vensim:macro_multi_macros", "mtk"): "generated code refers to an undefined name (#500)",
    ("vensim:macro_stock", "mtk"): "generated code refers to an undefined name (#500)",
    ("vensim:macro_trailing_definition", "mtk"): "generated code refers to an undefined name (#500)",
    # .xls input files not supported (XLSX.jl reads .xlsx only) (#501)
    ("vensim:get_data_args_3d_xls", "mtk"): ".xls input files not supported (XLSX.jl reads .xlsx only) (#501)",
    ("vensim:get_data_args_3d_xls", "ode"): ".xls input files not supported (XLSX.jl reads .xlsx only) (#501)",
    ("vensim:get_lookups_data_3d_xls", "mtk"): ".xls input files not supported (XLSX.jl reads .xlsx only) (#501)",
    ("vensim:get_lookups_data_3d_xls", "ode"): ".xls input files not supported (XLSX.jl reads .xlsx only) (#501)",
    ("vensim:get_lookups_subscripted_args", "mtk"): ".xls input files not supported (XLSX.jl reads .xlsx only) (#501)",
    ("vensim:get_lookups_subscripted_args", "ode"): ".xls input files not supported (XLSX.jl reads .xlsx only) (#501)",
    ("vensim:get_lookups_subset", "mtk"): ".xls input files not supported (XLSX.jl reads .xlsx only) (#501)",
    ("vensim:get_lookups_subset", "ode"): ".xls input files not supported (XLSX.jl reads .xlsx only) (#501)",
    # variables missing from the Julia output (#499)
    ("vensim:get_subscript_3d_arrays_xls", "mtk"): "variables missing from the Julia output (#499)",
    ("vensim:get_subscript_3d_arrays_xls", "ode"): "variables missing from the Julia output (#499)",
    ("vensim:initial_function", "mtk"): "variables missing from the Julia output (#499)",
    ("vensim:subscripted_xidz", "mtk"): "variables missing from the Julia output (#499)",
    ("vensim:unchangeable_constant", "mtk"): "variables missing from the Julia output (#499)",
    ("xmile:initial_function", "mtk"): "variables missing from the Julia output (#499)",
    # MTK: indexing a symbolic array out of its declared shape (#502)
    ("vensim:allocate_available", "mtk"): "MTK: indexing a symbolic array out of its declared shape (#502)",
    ("vensim:subscripted_smooth", "mtk"): "MTK: indexing a symbolic array out of its declared shape (#502)",
    # Excel data referenced by row/column number not supported (#501)
    ("vensim:get_mixed_definitions", "mtk"): "Excel data referenced by row/column number not supported (#501)",
    ("vensim:get_mixed_definitions", "ode"): "Excel data referenced by row/column number not supported (#501)",
    ("vensim:partial_range_definitions", "mtk"): "Excel data referenced by row/column number not supported (#501)",
    ("vensim:partial_range_definitions", "ode"): "Excel data referenced by row/column number not supported (#501)",
    # lookup emitted as a constant but called as a function (#500)
    ("vensim:get_time_value", "mtk"): "lookup emitted as a constant but called as a function (#500)",
    ("vensim:get_time_value", "ode"): "lookup emitted as a constant but called as a function (#500)",
    ("vensim:lookups_funcnames", "ode"): "lookup emitted as a constant but called as a function (#500)",
    # ALLOCATE with subscripted arguments not supported (#500)
    ("vensim:allocate_available", "ode"): "ALLOCATE with subscripted arguments not supported (#500)",
    ("vensim:allocate_by_priority", "ode"): "ALLOCATE with subscripted arguments not supported (#500)",
    # MTK: array minus scalar in symbolic equations (#502)
    ("vensim:forecast", "mtk"): "MTK: array minus scalar in symbolic equations (#502)",
    ("vensim:subscripted_trend", "mtk"): "MTK: array minus scalar in symbolic equations (#502)",
    # MTK: symbolic value in an if/else condition (#502)
    ("vensim:control_vars", "mtk"): "MTK: symbolic value in an if/else condition (#502)",
    ("vensim:dynamic_final_time", "mtk"): "MTK: symbolic value in an if/else condition (#502)",
    # VECTOR SELECT called with wrong arguments (#500)
    ("vensim:vector_select", "mtk"): "VECTOR SELECT called with wrong arguments (#500)",
    ("vensim:vector_select", "ode"): "VECTOR SELECT called with wrong arguments (#500)",
    # lookup called with extra arguments (#500)
    ("vensim:vector_order", "mtk"): "lookup called with extra arguments (#500)",
    ("vensim:vector_order", "ode"): "lookup called with extra arguments (#500)",
    # lookup x values not sorted (#500)
    ("vensim:get_with_missing_values_xlsx", "mtk"): "lookup x values not sorted (#500)",
    ("vensim:get_with_missing_values_xlsx", "ode"): "lookup x values not sorted (#500)",
    # subscripted TREND/FORECAST: array minus scalar (#500)
    ("vensim:forecast", "ode"): "subscripted TREND/FORECAST: array minus scalar (#500)",
    ("vensim:subscripted_trend", "ode"): "subscripted TREND/FORECAST: array minus scalar (#500)",
    # MTK: MODULO not callable on symbolic values (#502)
    ("vensim:lookups_funcnames", "mtk"): "MTK: MODULO not callable on symbolic values (#502)",
    # MTK: initial value depends on other variables (#502)
    ("vensim:active_initial_circular", "mtk"): "MTK: initial value depends on other variables (#502)",
    # MTK: numeric conversion of a symbolic value (#502)
    ("vensim:allocate_by_priority", "mtk"): "MTK: numeric conversion of a symbolic value (#502)",
    # MTK: unbalanced system (#502)
    ("vensim:tabbed_arrays", "mtk"): "MTK: unbalanced system (#502)",
    # macro: variable from another macro expression (#500)
    ("vensim:macro_multi_expression", "ode"): "macro: variable from another macro expression (#500)",
    # subscripted SMOOTH: wrong state shape (#500)
    ("vensim:subscripted_smooth", "ode"): "subscripted SMOOTH: wrong state shape (#500)",
    # tabbed array constant generates invalid Julia (#500)
    ("vensim:tabbed_arrays", "ode"): "tabbed array constant generates invalid Julia (#500)",
}


def _cases():
    """Pathway test cases as {case id: spec}, plus the cases the Python
    pathway itself marks as xfail {case id: reason}."""
    cases, python_xfail = {}, {}
    for kind, table in (("vensim", _vensim.vensim_test), ("xmile", _xmile.xmile_test)):
        for name, entry in table.items():
            case = f"{kind}:{name}"
            if isinstance(entry, ParameterSet):
                cases[case] = entry.values[0]
                for mark in entry.marks:
                    if mark.name == "xfail":
                        python_xfail[case] = mark.kwargs.get("reason", "")
            else:
                cases[case] = entry
    return cases, python_xfail


CASES, PYTHON_XFAIL = _cases()


#: Julia packages the generated models load, per backend.
JULIA_PACKAGES = {
    "ode": ["PySD", "OrdinaryDiffEq", "NCDatasets", "DataInterpolations"],
    "mtk": ["PySD", "OrdinaryDiffEq", "NCDatasets", "DataInterpolations",
            "ModelingToolkit", "Symbolics"],
}


@functools.lru_cache(maxsize=None)
def _missing_julia_packages():
    """Packages of JULIA_PACKAGES that Julia cannot find (None: no julia binary).

    GitHub's runner images ship Julia without these packages, so checking the
    binary alone is not enough.
    """
    if shutil.which("julia") is None:
        return None
    wanted = sorted(set().union(*JULIA_PACKAGES.values()))
    script = "for p in ARGS; Base.find_package(p) === nothing && println(p); end"
    try:
        out = subprocess.run(
            ["julia", "--startup-file=no", "-e", script, *wanted],
            capture_output=True, text=True, timeout=300, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return frozenset(wanted)
    return frozenset(out.split())


def _julia_unavailable(backend):
    """Why *backend* cannot run here, or None if it can."""
    missing = _missing_julia_packages()
    if missing is None:
        return "julia binary not found"
    missing = [p for p in JULIA_PACKAGES[backend] if p in missing]
    if missing:
        return f"Julia packages not found: {', '.join(missing)}"
    return None


def _cases_for(backend):
    if backend == "mtk" and MTK_SCOPE == "subset":
        return [case for case in CASES if case in MTK_SUBSET]
    return list(CASES)


def _params():
    for backend in BACKENDS:
        for case in _cases_for(backend):
            marks = [pytest.mark.julia]
            if (case, backend) in KNOWN_FAILURES:
                marks.append(pytest.mark.xfail(
                    reason=KNOWN_FAILURES[case, backend], strict=True))
            elif case in PYTHON_XFAIL:
                # Julia may well pass where Python fails: not strict
                marks.append(pytest.mark.xfail(
                    reason=f"xfail in the Python pathway: {PYTHON_XFAIL[case]}"))
            yield pytest.param(case, backend, id=f"{backend}-{case}", marks=marks)


# ---------------------------------------------------------------------------
# Translation and Julia batch runner
# ---------------------------------------------------------------------------

def _own_ranges(elem, ranges):
    """The range a variable is laid out over, at each subscript position.

    It is the range whose elements are exactly the ones the variable's
    definitions cover (e.g. `lower` for a variable defined over `lower`,
    the parent for one defined over `lower` and `Layer1`, `dim2: c, b` for
    one defined for `c` and `b`), or None (numbered by the first range
    listing each element).
    """
    subs = [c.subscripts[0] for c in elem.components]
    out = []
    for i in range(min(map(len, subs), default=0)):
        names = [s[i].lower() for s in subs]
        covered = set()
        for name in names:
            covered |= set(ranges.get(name, [name.strip()]))
        own = next((ranges[n] for n in names if n in ranges and set(ranges[n]) == covered), None)
        if own is None:
            own = next((r for r in ranges.values() if set(r) == covered), None)
        out.append(own)
    return out


def _id_map(model_file, columns):
    """Reference column -> Julia identifier ('Stock A[B1,C2]' -> 'stock_a[1, 2]')."""
    if model_file.suffix.lower() in (".xmile", ".stmx", ".xml"):
        from pysd.translators.xmile.xmile_file import XmileFile as File
    else:
        from pysd.translators.vensim.vensim_file import VensimFile as File
    vf = File(model_file)
    vf.parse()
    am = expand_subscript_ranges(vf.get_abstract_model())
    ns = JuliaNamespaceManager()
    labels = {}
    ranges = {}
    own_ranges = {}
    for section in am.sections:
        for sr in section.subscripts:
            if isinstance(sr.subscripts, list):
                ranges[sr.name.lower()] = [s.lower().strip() for s in sr.subscripts]
                for i, label in enumerate(sr.subscripts):
                    labels.setdefault(label.lower().strip(), i + 1)
    for section in am.sections:
        for sr in section.subscripts:
            # an alias (subdim1 copy <-> subdim1) has the elements of its range
            if isinstance(sr.subscripts, str) and sr.subscripts.lower() in ranges:
                ranges[sr.name.lower()] = ranges[sr.subscripts.lower()]
    for section in am.sections:
        for elem in section.elements:
            ns.add_to_namespace(elem.name)
            own_ranges[elem.name.lower().strip()] = _own_ranges(elem, ranges)
    ids = {}
    for col in columns:
        if col.upper() in CONTROL_COLUMNS:
            ids[col] = CONTROL_COLUMNS[col.upper()]
            continue
        base, _, rest = col.partition("[")
        jl = ns.get(base.strip())
        if jl is None:
            continue
        if rest:
            own = own_ranges.get(base.strip().lower(), [])
            idx = [
                own[i].index(lb) + 1 if i < len(own) and own[i] and lb in own[i]
                else labels.get(lb)
                for i, lb in enumerate(s.lower().strip() for s in rest.rstrip("]").split(","))
            ]
            if None in idx:
                continue
            jl += "[" + ", ".join(map(str, idx)) + "]"
        ids[col] = jl
    return ids


def _data_files(spec, folder):
    files = spec.get("data_files")
    if not files:
        return []
    if isinstance(files, str):
        files = [files]
    return [str((folder / f).resolve()) for f in files]


_JULIA_PRELUDE = r'''
using Printf

function _pysd_values(m, sol, ids::Vector{String}, t_ref::Vector{Float64})
    rows = [argmin(abs.(sol.t .- t)) for t in t_ref]
    out = fill(NaN, length(rows), length(ids))
    parsed = map(ids) do id
        mt = match(r"^(\w+)(?:\[([\d, ]+)\])?$", id)
        (Symbol(mt[1]), mt[2] === nothing ? Int[] : parse.(Int, split(mt[2], ",")))
    end
    pick(v, idx) = isempty(idx) ? Float64(v) : Float64(v[idx...])
    if isdefined(m, :observe)                       # ODE backend
        for (r, i) in enumerate(rows)
            obs = Base.invokelatest(m.observe, sol.u[i], sol.t[i])
            for (c, (s, idx)) in enumerate(parsed)
                v = get(obs, String(s), nothing)
                v === nothing && isdefined(m, s) && (v = getfield(m, s))
                v === nothing && continue
                try out[r, c] = pick(v, idx) catch end
            end
        end
    else                                            # MTK backend
        for (c, (s, idx)) in enumerate(parsed)
            isdefined(m, s) || continue
            v = getfield(m, s)
            # plain numbers are constants; symbolic variables (Num <: Number) are not
            const_types = Union{AbstractFloat, Integer}
            if v isa const_types || v isa AbstractArray{<:const_types}
                try out[:, c] .= pick(v, idx) catch end
                continue
            end
            target = isempty(idx) ? v : v[idx...]
            series = try
                sol[target]
            catch
                try fill(sol.ps[target], length(sol.t)) catch; nothing end
            end
            series === nothing && continue
            for (r, i) in enumerate(rows)
                try out[r, c] = Float64(series[i]) catch end
            end
        end
    end
    return out
end
'''


def _model_block(name, jl_path, ids, t_ref, kwargs):
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
    ids_jl = "String[" + ", ".join(f'"{i}"' for i in ids) + "]"
    t_jl = "Float64[" + ", ".join(repr(float(t)) for t in t_ref) + "]"
    kw = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    return f'''
println("===START={name}===")
module _M_{safe}
    load_error = nothing
    try
        include("{jl_path.as_posix()}")
    catch e
        global load_error = e
    end
end
try
    _M_{safe}.load_error === nothing || throw(_M_{safe}.load_error)
    local t0 = time()
    local sol = Base.invokelatest(_M_{safe}.run_model; {kw})
    local vals = _pysd_values(_M_{safe}, sol, {ids_jl}, {t_jl})
    println("===MODEL={name}===")
    for r in axes(vals, 1)
        println(join([@sprintf("%.17g", x) for x in vals[r, :]], ","))
    end
    println("===END=", round(time() - t0; digits=2), "===")
catch e
    msg = replace(sprint(showerror, e), "\\n" => " ")
    println("===ERROR={name}===", first(msg, 600))
end
flush(stdout)
'''


def _parse(stdout):
    """Batch stdout -> {case: ndarray | error string}; plus the last started case."""
    results, current, rows, started = {}, None, [], None
    for line in stdout.splitlines():
        if line.startswith("===START="):
            started = line[9:-3]
        elif line.startswith("===MODEL="):
            current, rows = line[9:-3], []
        elif line.startswith("===END=") and current is not None:
            results[current] = np.array(rows, dtype=float)
            current = None
        elif line.startswith("===ERROR="):
            name, _, msg = line[9:].partition("===")
            results[name] = msg
            current = None
        elif current is not None:
            rows.append([float(x) for x in line.split(",")] if line else [])
    return results, started


def _run_julia_streaming(script, log):
    """Run a batch script, stopping it when one model exceeds its time limit.

    Returns (stdout lines, name of the model that timed out or None).  The
    first model of a process also loads and compiles the Julia packages, so
    it gets STARTUP_ALLOWANCE extra seconds.
    """
    lines = queue.Queue()
    out, current, started, n_started = [], None, time.time(), 0
    # the with block closes the output pipe (pytest.ini turns ResourceWarnings into errors)
    with subprocess.Popen(["julia", "--startup-file=no", str(script)],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as proc:

        def reader():
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        while True:
            try:
                line = lines.get(timeout=5)
            except queue.Empty:
                limit = MODEL_TIMEOUT + (STARTUP_ALLOWANCE if n_started <= 1 else 0)
                if current is not None and time.time() - started > limit:
                    proc.kill()
                    thread.join()
                    return out, current
                continue
            if line is None:
                break
            log.write(line)
            log.flush()
            line = line.rstrip("\n")
            out.append(line)
            if line.startswith("===START="):
                current, started, n_started = line[9:-3], time.time(), n_started + 1
            elif line.startswith(("===END=", "===ERROR=")):
                current = None
    return out, None


def _run_batch(blocks, workdir):
    """Run model blocks in as few Julia processes as possible, with per-model timeouts.

    Julia's output is also written to ``_julia.log`` in *workdir*.
    """
    results = {}
    pending = list(blocks)
    with open(workdir / "_julia.log", "a") as log:
        while pending:
            script = workdir / f"_batch_{len(results)}.jl"
            script.write_text(_JULIA_PRELUDE + "".join(code for _, code in pending))
            out, timed_out = _run_julia_streaming(script, log)
            done, _ = _parse("\n".join(out))
            results.update(done)
            remaining = [(n, c) for n, c in pending if n not in results]
            if not remaining:
                break
            # The first unfinished model hung or crashed Julia: record it, go on.
            culprit = timed_out or remaining[0][0]
            results[culprit] = (f"timeout after {MODEL_TIMEOUT} s" if timed_out
                                else "Julia process ended: " + " ".join(out[-5:])[-600:])
            pending = [(n, c) for n, c in remaining if n != culprit]
    return results


class _Batch:
    """Translate and run all cases of a backend, once per session."""

    def __init__(self, tmp_factory):
        self.tmp_factory = tmp_factory
        self.results = {}
        self.refs = {}
        self.columns = {}

    def get(self, backend):
        if backend not in self.results:
            # Translate every backend here (the builder is not thread-safe),
            # then run one Julia process per backend in parallel.
            from concurrent.futures import ThreadPoolExecutor
            todo = [b for b in BACKENDS if b not in self.results]
            prepared = {b: self._prepare(b) for b in todo}

            def execute(b):
                blocks, results, workdir = prepared[b]
                start = time.time()
                results.update(_run_batch(blocks, workdir))
                results["__seconds__"] = time.time() - start
                return results

            with ThreadPoolExecutor(max_workers=len(todo)) as pool:
                for b, res in zip(todo, pool.map(execute, todo)):
                    self.results[b] = res
        return self.results[backend]

    def _prepare(self, backend):
        """Translate all cases; return (Julia blocks, early results, workdir)."""
        from pysd import translate_to_julia
        workdir = self.tmp_factory.mktemp(f"julia_{backend}")
        blocks, results = [], {}
        for case in _cases_for(backend):
            spec = CASES[case]
            folder = workdir / re.sub(r"[^A-Za-z0-9_]", "_", case)
            shutil.copytree(TEST_MODELS / spec["folder"], folder)
            model = folder / spec["file"]
            ref_file = folder / spec.get("canonical_file", "")
            if not ref_file.is_file():
                ref_file = next((folder / n for n in ("output.csv", "output.tab")
                                 if (folder / n).is_file()), None)
            if ref_file is None:
                results[case] = "no reference output"
                continue
            canon = load_outputs(ref_file, encoding=detect_encoding(ref_file))
            self.refs[case] = canon
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    jl_path = translate_to_julia(model, backend=backend)
                    ids = _id_map(model, list(canon.columns))
            except Exception as exc:  # translation error is a test result
                results[case] = f"translation failed: {type(exc).__name__}: {exc}"
                continue
            self.columns[case] = list(ids)
            kwargs = {}  # run_model defaults to Euler(), as Vensim
            data = _data_files(spec, folder)
            if data:
                key = "nc_data_files" if data[0].endswith(".nc") else "tab_data_files"
                kwargs[key] = "[" + ", ".join(f'"{d}"' for d in data) + "]"
            blocks.append((case, _model_block(case, jl_path, list(ids.values()),
                                              canon.index.values, kwargs)))
        return blocks, results, workdir


@pytest.fixture(scope="session")
def julia_reference(tmp_path_factory):
    return _Batch(tmp_path_factory)


@pytest.mark.parametrize("case,backend", list(_params()))
def test_reference_output(case, backend, julia_reference):
    unavailable = _julia_unavailable(backend)
    if unavailable:
        # CI's Julia job sets PYSD_JULIA_REQUIRED, so it cannot pass untested.
        if os.environ.get("PYSD_JULIA_REQUIRED"):
            pytest.fail(f"{backend}: {unavailable}", pytrace=False)
        pytest.skip(unavailable)
    results = julia_reference.get(backend)
    result = results[case]
    if isinstance(result, str):
        pytest.fail(f"{case} ({backend}): {result}", pytrace=False)
    canon = julia_reference.refs[case]
    actual = pd.DataFrame(result, index=canon.index, columns=julia_reference.columns[case])
    # Variables Julia could not provide count as missing columns.
    actual = actual.loc[:, ~(actual.isna().all() & canon[actual.columns].notna().any())]
    spec = CASES[case]
    kwargs = {k: spec[k] for k in ("rtol", "atol") if k in spec}
    assert_frames_close(actual, canon, verbose=True, **kwargs)
