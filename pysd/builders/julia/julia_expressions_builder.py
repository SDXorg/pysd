"""
Converts AMR Abstract Syntax Tree nodes to Julia expression strings.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from warnings import warn

from pysd.translators.structures.abstract_expressions import (
    AbstractSyntax,
    AllocateAvailableStructure,
    AllocateByPriorityStructure,
    ArithmeticStructure,
    CallStructure,
    DataStructure,
    ForecastStructure,
    GameStructure,
    GetConstantsStructure,
    GetDataStructure,
    GetLookupsStructure,
    InitialStructure,
    InlineLookupsStructure,
    IntegStructure,
    LogicStructure,
    LookupsStructure,
    ReferenceStructure,
    SampleIfTrueStructure,
    SubscriptsReferenceStructure,
    TrendStructure,
)

# ---------------------------------------------------------------------------
# Operator tables
# ---------------------------------------------------------------------------

# Vensim arithmetic operator  →  Julia operator
ARITHMETIC_OPS: dict = {
    "+": "+",
    "-": "-",
    "negative": "-",   # Vensim unary negation AST operator
    "*": "*",
    "/": "/",
    "^": "^",
    "**": "^",
    "mod": "mod",
}

# Vensim logic operator  →  Julia operator
LOGIC_OPS: dict = {
    "=": "==",
    "<>": "!=",
    "<": "<",
    ">": ">",
    "<=": "<=",
    ">=": ">=",
    ":AND:": "&&",
    ":OR:": "||",
    ":NOT:": "!",
    "AND": "&&",
    "OR": "||",
    "NOT": "!",
}

# Vensim built-in function name  →  Julia function name
# Keys are matched after .upper(), so include both "SPACE FORM" and "UNDERSCORE_FORM"
# because the Vensim parser may store names either way.
BUILTIN_FUNCTIONS: dict = {
    # Basic math
    "ABS": "abs",
    "EXP": "exp",
    "LN": "log",
    "SQRT": "sqrt",
    "SIN": "sin",
    "COS": "cos",
    "TAN": "tan",
    "ARCSIN": "asin",
    "ARCCOS": "acos",
    "ARCTAN": "atan",
    "SINH": "sinh",
    "COSH": "cosh",
    "TANH": "tanh",
    "INTEGER": "pysd_trunc",
    "INT": "pysd_trunc",
    "POWER": "pysd_power",
    "MIN": "min",
    "MAX": "max",
    "MODULO": "mod",
    "QUANTUM": "pysd_quantum",
    "PI": "pysd_pi",
    # Control flow — parser stores as "if_then_else" (underscores).
    # Use pysd_ifelse: Symbolics' `ifelse` has type issues with SymReal
    # conditions, so PySD.jl provides a dispatching wrapper.
    "IF THEN ELSE": "pysd_ifelse",
    "IF_THEN_ELSE": "pysd_ifelse",
    # Array operations
    "SUM": "sum",
    "PROD": "prod",
    "VMAX": "maximum",
    "VMIN": "minimum",
    # XMILE emits vmax_xmile/vmin_xmile for whole-array MIN/MAX reductions.
    "VMAX_XMILE": "maximum",
    "VMIN_XMILE": "minimum",
    "ELMCOUNT": "pysd_elmcount",   # resolved to literal size by caller
    "INVERT MATRIX": "inv",
    "INVERT_MATRIX": "inv",
    "TRANSPOSE": "transpose",
    # ACTIVE INITIAL(expr, initial) — for ODE simulation just return expr
    "ACTIVE INITIAL": "pysd_active_initial",
    "ACTIVE_INITIAL": "pysd_active_initial",
    # SD helpers provided by PySD.jl
    "LOG": "pysd_log_base",
    "XIDZ": "pysd_xidz",
    "ZIDZ": "pysd_zidz",
    "PULSE": "pysd_pulse",
    "PULSE TRAIN": "pysd_pulse_train",
    "PULSE_TRAIN": "pysd_pulse_train",
    "RAMP": "pysd_ramp",
    "STEP": "pysd_step",
    # XMILE pulse/ramp variants
    "XPULSE": "pysd_xpulse",
    "XPULSE_TRAIN": "pysd_xpulse_train",
    "XRAMP": "pysd_xramp",
    # Random functions
    "RANDOM 0 1": "pysd_random_0_1",
    "RANDOM_0_1": "pysd_random_0_1",
    "RANDOM UNIFORM": "pysd_random_uniform",
    "RANDOM_UNIFORM": "pysd_random_uniform",
    "RANDOM NORMAL": "pysd_random_normal",
    "RANDOM_NORMAL": "pysd_random_normal",
    "RANDOM EXPONENTIAL": "pysd_random_exponential",
    "RANDOM_EXPONENTIAL": "pysd_random_exponential",
    # Vector operations
    "VECTOR SELECT": "pysd_vector_select",
    "VECTOR_SELECT": "pysd_vector_select",
    "VECTOR SORT ORDER": "pysd_vector_sort_order",
    "VECTOR_SORT_ORDER": "pysd_vector_sort_order",
    "VECTOR REORDER": "pysd_vector_reorder",
    "VECTOR_REORDER": "pysd_vector_reorder",
    "VECTOR RANK": "pysd_vector_rank",
    "VECTOR_RANK": "pysd_vector_rank",
    # Time value
    "GET TIME VALUE": "pysd_get_time_value",
    "GET_TIME_VALUE": "pysd_get_time_value",
}

# Names of helper functions provided by the PySD.jl companion package.
# These are no longer inlined into generated files — the generated model does
# `using PySD` which re-exports every ``pysd_*`` helper.  The set is retained so
# the AST visitor / model builder can track which helpers an equation requires
# (e.g. to pull in ``pysd_trunc`` when ``pysd_quantum`` is used).
HELPER_IMPLEMENTATIONS: frozenset = frozenset({
    "pysd_trunc", "pysd_log_base", "pysd_xidz", "pysd_zidz",
    "pysd_pulse", "pysd_pulse_train", "pysd_ramp", "pysd_step",
    "pysd_active_initial", "pysd_ifelse",
    "pysd_inv_mat2d_elem", "pysd_inv_mat3d_elem",
    "pysd_power", "pysd_quantum", "pysd_pi",
    "pysd_xpulse", "pysd_xpulse_train", "pysd_xramp",
    "pysd_random_0_1", "pysd_random_uniform",
    "pysd_random_normal", "pysd_random_exponential",
    "pysd_vector_select", "pysd_vector_sort_order",
    "pysd_vector_reorder", "pysd_vector_rank",
    "pysd_get_time_value",
    "pysd_logical_and", "pysd_logical_or", "pysd_logical_not",
})

# Helper functions that receive the current time *t* as their first argument
_TIME_HELPERS: frozenset = frozenset({
    "pysd_pulse", "pysd_pulse_train", "pysd_ramp", "pysd_step",
    "pysd_xpulse", "pysd_xpulse_train", "pysd_xramp", "pysd_get_time_value",
})


# ---------------------------------------------------------------------------
# Lookup-table utilities
# ---------------------------------------------------------------------------

class InlineLookupRegistry:
    """Collects inline lookup tables encountered during AST traversal.

    Each inline lookup is given a unique name so the generated file can
    declare a named interpolant constant and a one-argument wrapper.
    """

    def __init__(self) -> None:
        self._entries: List[Tuple[str, tuple, tuple, str]] = []
        self._counter: int = 0

    def register(self, xs: tuple, ys: tuple, itp_type: str) -> str:
        """Register an inline lookup table and return its function name."""
        self._counter += 1
        name = f"_inline_lookup_{self._counter}"
        self._entries.append((name, xs, ys, itp_type))
        return name

    @property
    def entries(self) -> List[Tuple[str, tuple, tuple, str]]:
        return list(self._entries)


def format_number(value: Any) -> str:
    """Format a Python numeric value as a Julia floating-point literal."""
    if isinstance(value, float):
        if value == float("inf"):
            return "Inf"
        if value == float("-inf"):
            return "-Inf"
        if value != value:  # NaN
            return "NaN"
    return repr(float(value))


def format_vector(values: tuple) -> str:
    """Format a tuple of numbers as a Julia Float64 vector literal."""
    return "[" + ", ".join(format_number(v) for v in values) + "]"


def lookup_interpolation_code(
    name: str, xs: tuple, ys: tuple, itp_type: str
) -> Tuple[str, str, str]:
    """Return ``(const_decl, func_decl, register_decl)`` for a named lookup table.

    ``itp_type`` controls the DataInterpolations constructor:

    * ``"interpolate"`` / ``"extrapolate"`` → ``LinearInterpolation`` (default)
    * ``"hold_forward"``  → ``ConstantInterpolation`` (previous-value hold)
    * ``"hold_backward"`` → ``ConstantInterpolation(...; dir=:right)`` (next-value hold)

    ``@register_symbolic`` tells ModelingToolkit that this is an opaque
    external function so it is called at every timestep rather than being
    constant-folded during structural_simplify.
    """
    xs_vec = format_vector(xs)
    ys_vec = format_vector(ys)
    itp_name = f"{name}_itp"

    _extrap = "ExtrapolationType.Constant"
    if itp_type == "hold_forward":
        const_decl = (
            f"const {itp_name} = ConstantInterpolation({ys_vec}, {xs_vec};"
            f" extrapolation_left = {_extrap}, extrapolation_right = {_extrap})"
        )
    elif itp_type == "hold_backward":
        const_decl = (
            f"const {itp_name} = ConstantInterpolation({ys_vec}, {xs_vec};"
            f" dir=:right, extrapolation_left = {_extrap}, extrapolation_right = {_extrap})"
        )
    else:
        # "interpolate", "extrapolate", or any unrecognised type → linear
        const_decl = (
            f"const {itp_name} = LinearInterpolation({ys_vec}, {xs_vec};"
            f" extrapolation_left = {_extrap}, extrapolation_right = {_extrap})"
        )

    func_decl = f"{name}(x) = {itp_name}(x)"
    register_decl = f"@register_symbolic {name}(x::Real)"
    return const_decl, func_decl, register_decl


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------

class JuliaASTVisitor:
    """Recursively converts an AMR AST node to a Julia expression string.

    Parameters
    ----------
    namespace:
        A :class:`~pysd.builders.julia.namespace.JuliaNamespaceManager`.
    inline_registry:
        Accumulator for inline lookup tables found during traversal.
    needed_helpers:
        Mutable set; the visitor adds the names of any helper functions
        (``_pulse``, ``_xidz``, …) it emits, so the builder can include
        their implementations in the generated file.
    """

    def __init__(
        self,
        namespace,
        inline_registry: InlineLookupRegistry,
        needed_helpers: Set[str],
        active_subs: Optional[Dict[str, str]] = None,
        var_dims: Optional[Dict[str, List[str]]] = None,
        subs_sizes: Optional[Dict[str, int]] = None,
        subs_elems: Optional[Dict[str, List[str]]] = None,
        lookup_names: Optional[Set[str]] = None,
        root=None,
        macro_names: Optional[Set[str]] = None,
    ) -> None:
        self.namespace = namespace
        self.registry = inline_registry
        self.needed_helpers = needed_helpers
        # active_subs: dim_name -> julia index variable (e.g. {"sector": "_i"})
        self.active_subs = active_subs or {}
        # _clean_active_subs: normalised-dim-name -> julia index variable, for
        # case-insensitive lookup when subscript names appear as bare references.
        self._clean_active_subs = {
            re.sub(r"[^a-z0-9_]", "_", k.lower()): v
            for k, v in self.active_subs.items()
        }
        # var_dims: julia identifier -> list of dim names it is subscripted over
        self.var_dims = var_dims or {}
        # subs_sizes: subscript range name -> integer size (for ELMCOUNT)
        self.subs_sizes = subs_sizes or {}
        # _clean_subs_sizes: normalised name -> size, for case-insensitive ELMCOUNT lookup
        self._clean_subs_sizes = {
            re.sub(r"[^a-z0-9_]", "_", k.lower()): v
            for k, v in self.subs_sizes.items()
        }
        # lookup_names: identifiers that are GET DATA / GET LOOKUPS functions
        # — bare references to these should be auto-called as f(t) or f(i, t)
        self.lookup_names = lookup_names or set()
        # subs_elems: range_name -> ordered list of element labels
        self.subs_elems = subs_elems or {}
        # Pre-compute element_label -> {range_name: 1-based-index} for fast lookups
        self._elem_index: Dict[str, Dict[str, int]] = {}
        # _clean_elem_index: normalised-label -> {range_name: 1-based-index}
        # for case-insensitive resolution of bare element references in equations.
        self._clean_elem_index: Dict[str, Dict[str, int]] = {}
        for rng, elems in self.subs_elems.items():
            for i, lbl in enumerate(elems):
                if lbl not in self._elem_index:
                    self._elem_index[lbl] = {}
                self._elem_index[lbl][rng] = i + 1
                clean_lbl = re.sub(r"[^a-z0-9_]", "_", lbl.lower())
                if clean_lbl not in self._clean_elem_index:
                    self._clean_elem_index[clean_lbl] = {}
                self._clean_elem_index[clean_lbl][rng] = i + 1
        # root: Path to the model directory (for reading external files)
        self._root = root
        # macro_names: Julia identifiers of known Vensim macros (no warning on call)
        self._macro_names: Set[str] = set(macro_names) if macro_names else set()
        # Embedded DelayFixedStructure nodes encountered during expression traversal.
        # The model builder drains this list after each element to lift them out
        # into dedicated auxiliary pipeline stocks.
        self._pending_delay_fixed: List[tuple] = []

    def _jl_n(self, dim_name: str) -> str:
        """Julia constant name for the size of *dim_name* (``N_DIMNAME``)."""
        return "N_" + re.sub(r"[^a-z0-9]", "_", dim_name.lower()).upper()

    def _collect_bang_subs(self, node) -> List[str]:
        """Return unique '!'-subscript strings found anywhere in *node*'s subtree."""
        result: List[str] = []
        seen: set = set()

        def _scan(n: Any) -> None:
            if isinstance(n, ReferenceStructure):
                subs = (
                    n.subscripts.subscripts
                    if n.subscripts is not None and hasattr(n.subscripts, "subscripts")
                    else []
                )
                for s in subs:
                    if s.endswith("!") and s not in seen:
                        seen.add(s)
                        result.append(s)
            elif isinstance(n, CallStructure):
                fsubs = (
                    n.function.subscripts.subscripts
                    if n.function.subscripts is not None
                    and hasattr(n.function.subscripts, "subscripts")
                    else []
                )
                for s in fsubs:
                    if s.endswith("!") and s not in seen:
                        seen.add(s)
                        result.append(s)
                for arg in n.arguments:
                    _scan(arg)
            elif isinstance(n, (ArithmeticStructure, LogicStructure)):
                for arg in n.arguments:
                    _scan(arg)

        _scan(node)
        return result

    def _with_extra_subs(self, extra: Dict[str, str]) -> "JuliaASTVisitor":
        """Return a child visitor with *extra* entries added to active_subs."""
        return JuliaASTVisitor(
            self.namespace,
            self.registry,
            self.needed_helpers,
            active_subs={**self.active_subs, **extra},
            var_dims=self.var_dims,
            subs_sizes=self.subs_sizes,
            subs_elems=self.subs_elems,
            lookup_names=self.lookup_names,
            root=self._root,
            macro_names=self._macro_names,
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def visit(self, node: Any) -> str:
        """Return the Julia expression string for *node*."""
        if node is None:
            return "0.0"

        if isinstance(node, bool):
            return "true" if node else "false"

        if isinstance(node, (int, float)):
            return format_number(node)

        if isinstance(node, str):
            # Bare strings occasionally appear as numeric literals in the AMR
            try:
                return format_number(float(node))
            except ValueError:
                return repr(node)

        # numpy arrays (e.g. from GetConstantsStructure values embedded inline)
        try:
            import numpy as np
            if isinstance(node, np.ndarray):
                if node.ndim == 0:
                    return format_number(float(node))
                if node.ndim == 1:
                    vals = ", ".join(format_number(float(v)) for v in node)
                    return f"[{vals}]"
                # Higher dims: flatten
                vals = ", ".join(format_number(float(v)) for v in node.flat)
                return f"[{vals}]"
        except ImportError:  # pragma: no cover
            pass

        if isinstance(node, ArithmeticStructure):
            return self._arithmetic(node)

        if isinstance(node, LogicStructure):
            return self._logic(node)

        if isinstance(node, ReferenceStructure):
            return self._reference(node)

        if isinstance(node, CallStructure):
            return self._call(node)

        if isinstance(node, InlineLookupsStructure):
            return self._inline_lookup(node)

        if isinstance(node, InitialStructure):
            # INITIAL(x) — in an ODE context we just use the expression value
            return self.visit(node.initial)

        if isinstance(node, GameStructure):
            # GAME passes through in simulation (non-interactive) mode
            return self.visit(node.expression)

        if isinstance(node, GetConstantsStructure):
            # GetConstantsStructure nested inside an expression — read the
            # external value at translation time and emit it as a Julia literal.
            try:
                from pysd.py_backend.external import ExtConstant
                from pysd.builders.julia.julia_model_builder import _format_julia_value
                import pathlib as _pathlib
                root = self._root or _pathlib.Path(".")
                ext = ExtConstant(
                    file_name=node.file,
                    tab=node.tab,
                    cell=node.cell,
                    coords={},
                    root=root,
                    final_coords={},
                    py_name="_inline_const",
                )
                ext.initialize()
                return _format_julia_value(ext.data)
            except Exception as exc:  # pragma: no cover
                warn(
                    f"GetConstantsStructure inside expression could not be read "
                    f"({exc}); emitting placeholder 0.0."
                )
                return "0.0"

        if isinstance(node, SubscriptsReferenceStructure):
            # A subscript reference used as a value — emit the first subscript name.
            # This handles cases like ELMCOUNT(SECTORS) where the parser produces
            # a bare SubscriptsReferenceStructure for the subscript range name.
            if node.subscripts:
                ref = node.subscripts[0]
                return self.namespace.get(ref) or repr(ref)
            return "0.0"

        # DelayFixedStructure embedded inside another expression (XMILE pattern where
        # DELAY(x, n) appears inline rather than as a top-level element equation).
        # Queue it to be lifted into a dedicated auxiliary by the model builder.
        from pysd.translators.structures.abstract_expressions import DelayFixedStructure as _DFS
        if isinstance(node, _DFS):
            edf_name = f"_edf{len(self._pending_delay_fixed)}"
            self.namespace.namespace[edf_name] = edf_name
            self._pending_delay_fixed.append((edf_name, node))
            return edf_name

        # Structures that are handled at the element level should not appear
        # inside other expressions; warn and emit a placeholder.
        warn(
            f"Unsupported AST node type '{type(node).__name__}' inside expression "
            "— emitting placeholder 0.0."
        )
        return "0.0"

    # ------------------------------------------------------------------
    # Node handlers
    # ------------------------------------------------------------------

    def _arithmetic(self, node: ArithmeticStructure) -> str:
        args = [self.visit(a) for a in node.arguments]
        ops = node.operators

        if len(args) == 1:
            # Unary operator (negation)
            op = ARITHMETIC_OPS.get(ops[0], ops[0])
            return f"({op}{args[0]})"

        # Build expression, using pysd_power for ^ to handle negative bases
        result = args[0]
        for op, arg in zip(ops, args[1:]):
            julia_op = ARITHMETIC_OPS.get(op, op)
            if julia_op == "^":
                result = f"pysd_power({result}, {arg})"
            else:
                result = f"({result} {julia_op} {arg})"
        return result

    def _logic(self, node: LogicStructure) -> str:
        args = [self.visit(a) for a in node.arguments]
        ops = node.operators

        # AND / OR / NOT: use helper functions so the expression remains valid
        # when called with symbolic (Num) arguments inside MTK equations.
        # Julia's &&/|| require a concrete Bool; the helpers use ifelse instead.
        if len(args) == 1:
            op_key = ops[0].upper().strip(":")
            if op_key == "NOT":
                self.needed_helpers.add("pysd_logical_not")
                return f"pysd_logical_not({args[0]})"
            op = LOGIC_OPS.get(ops[0], ops[0])
            return f"({op}{args[0]})"

        result = args[0]
        for op, arg in zip(ops, args[1:]):
            op_key = op.upper().strip(":")
            if op_key == "AND":
                self.needed_helpers.add("pysd_logical_and")
                result = f"pysd_logical_and({result}, {arg})"
            elif op_key == "OR":
                self.needed_helpers.add("pysd_logical_or")
                result = f"pysd_logical_or({result}, {arg})"
            else:
                julia_op = LOGIC_OPS.get(op, op)
                result = f"({result} {julia_op} {arg})"
        return result

    def _reference(self, node: ReferenceStructure) -> str:
        # Subscript dimension names appear as bare references in equations like
        # I_Matrix[s, s1] = IF_THEN_ELSE(s = s1, 1, 0).  When inside an active
        # subscript loop, emit the corresponding loop-index variable directly.
        if self._clean_active_subs:
            clean_ref = re.sub(r"[^a-z0-9_]", "_", node.reference.lower())
            idx_var = self._clean_active_subs.get(clean_ref)
            if idx_var is not None:
                return idx_var

        julia_name = self.namespace.get(node.reference)
        if julia_name is None:
            clean_ref = re.sub(r"[^a-z0-9_]", "_", node.reference.lower())
            # Check if the reference is a subscript element label (e.g. "B" in
            # dimA: A, B, C).  When it is, emit the 1-based integer index so
            # comparisons like "dimA = B" become "_i0 == 2" in generated Julia.
            if clean_ref in self._clean_elem_index:
                ranges_map = self._clean_elem_index[clean_ref]
                # Prefer a range that is currently being iterated (active dim)
                idx = None
                for rng, pos in ranges_map.items():
                    clean_rng = re.sub(r"[^a-z0-9_]", "_", rng.lower())
                    if clean_rng in self._clean_active_subs:
                        idx = pos
                        break
                if idx is None:
                    idx = next(iter(ranges_map.values()))
                return str(idx)
            warn(
                f"Variable '{node.reference}' not found in namespace; "
                "using a sanitised fallback identifier."
            )
            julia_name = clean_ref
        # Apply subscript indices.  Two sources:
        #
        # (A) Explicit subscripts in the AST node  (e.g. share_FEH[solids])
        #     Each entry is either a range name (→ use active loop variable) or
        #     a specific element label (→ resolve to 1-based numeric index).
        # (B) Active loop variables from the enclosing comprehension context
        #     (only when the AST carries no explicit subscripts).
        node_subs = (
            node.subscripts.subscripts
            if node.subscripts is not None and hasattr(node.subscripts, "subscripts")
            else []
        )

        # If this identifier is a GET DATA/LOOKUPS function referenced bare (no
        # call syntax), auto-call it with the active subscript indices + t.
        if julia_name in self.lookup_names and not node_subs:
            dims = self.var_dims.get(julia_name, [])
            if self.active_subs:
                indices = [self.active_subs[d] for d in dims if d in self.active_subs]
                return f"{julia_name}({', '.join(indices + ['t'])})"
            elif dims:
                # Scalar context, subscripted lookup: broadcast over all dims
                idx_vars = [f"_ii{k}" for k in range(len(dims))]
                ranges = ", ".join(
                    f"{iv} in 1:{self._jl_n(d)}" for iv, d in zip(idx_vars, dims)
                )
                return f"[{julia_name}({', '.join(idx_vars + ['t'])}) for {ranges}]"
            else:
                return f"{julia_name}(t)"

        if node_subs:
            # (A) Explicit: resolve each subscript to a Julia index expression.
            var_dims_list = self.var_dims.get(julia_name, [])

            # Aggregation subscripts (ending with '!') are handled here.
            # Normally the outer _call for sum/prod/vmax/vmin pre-populates
            # active_subs for ! dims so that all references sharing the same !
            # subscript are inside ONE comprehension (not separate comprehensions
            # multiplied together).  If an ! dim is already in active_subs we
            # reuse that loop variable; otherwise we generate a new comprehension.
            if any(sub.endswith("!") for sub in node_subs):
                bang_ranges: List[str] = []
                ii_count = 0
                dim_to_idx: Dict[str, str] = {}

                for sub in node_subs:
                    clean_sub = re.sub(r"[^a-z0-9_]", "_", sub.lower())
                    if sub.endswith("!"):
                        bare = sub[:-1]
                        clean_bare = re.sub(r"[^a-z0-9_]", "_", bare.lower())
                        if clean_bare in self._clean_active_subs:
                            # Already being iterated by an outer loop (added by sum
                            # handler) — reuse the existing loop variable.
                            iv = self._clean_active_subs[clean_bare]
                            dim_to_idx[clean_bare] = iv
                        else:
                            # Find the matching dim in var_dims_list (by normalised name)
                            dim_name = next(
                                (d for d in var_dims_list
                                 if re.sub(r"[^a-z0-9_]", "_", d.lower()) == clean_bare),
                                bare,
                            )
                            iv = f"_ii{ii_count}"
                            ii_count += 1
                            dim_to_idx[re.sub(r"[^a-z0-9_]", "_", dim_name.lower())] = iv
                            bang_ranges.append(f"{iv} in 1:{self._jl_n(dim_name)}")
                    elif sub in self.active_subs:
                        dim_to_idx[clean_sub] = self.active_subs[sub]
                    elif sub in self.subs_elems:
                        idx_var = self.active_subs.get(sub)
                        if idx_var:  # pragma: no cover  # unreachable: elif subs_elems only runs when sub not in active_subs
                            dim_to_idx[clean_sub] = idx_var
                    else:
                        if sub in self._elem_index:
                            # Key dim_to_idx by the variable's DIMENSION NAME (not
                            # the element label) so the index assembly over
                            # var_dims_list can find it.  Also prefer the variable's
                            # own declared dim to avoid picking a larger parent range.
                            target_dim = None
                            for d in var_dims_list:
                                if d in self._elem_index[sub]:
                                    target_dim = d
                                    break
                            if target_dim is None:
                                target_dim = next(iter(self._elem_index[sub]))
                            clean_dim = re.sub(r"[^a-z0-9_]", "_", target_dim.lower())
                            dim_to_idx[clean_dim] = str(self._elem_index[sub][target_dim])

                # Assemble indices in var_dims_list (declaration) order.
                # When a dim name doesn't match any key in dim_to_idx (common when
                # Vensim aliases differ, e.g. 'sectors' decl vs 'sectors1' in ref),
                # fall back to size-matching then positional assignment.
                bang_iv_pool = [
                    dim_to_idx[re.sub(r"[^a-z0-9_]", "_", sub[:-1].lower())]
                    for sub in node_subs
                    if sub.endswith("!")
                    and re.sub(r"[^a-z0-9_]", "_", sub[:-1].lower()) in dim_to_idx
                ]
                used_ivars: set = set()
                if var_dims_list:
                    indices = []
                    for d in var_dims_list:
                        clean_d = re.sub(r"[^a-z0-9_]", "_", d.lower())
                        if clean_d in dim_to_idx:
                            iv = dim_to_idx[clean_d]
                            indices.append(iv)
                            used_ivars.add(iv)
                        else:
                            # Size-based fallback first
                            d_size = (self.subs_sizes.get(d, 0) or
                                      self._clean_subs_sizes.get(clean_d, 0))
                            matched = None
                            for sub in node_subs:
                                if not sub.endswith("!"):
                                    continue
                                bare = sub[:-1]
                                cb = re.sub(r"[^a-z0-9_]", "_", bare.lower())
                                iv = dim_to_idx.get(cb)
                                if iv is None or iv in used_ivars:
                                    continue
                                bare_size = (self.subs_sizes.get(bare, 0) or
                                             self._clean_subs_sizes.get(cb, 0))
                                if d_size > 0 and bare_size == d_size:
                                    matched = iv
                                    used_ivars.add(iv)
                                    break
                            if matched is None:
                                # Positional fallback
                                for iv in bang_iv_pool:
                                    if iv not in used_ivars:
                                        matched = iv
                                        used_ivars.add(iv)
                                        break
                            if matched is not None:
                                indices.append(matched)
                else:
                    # No var_dims info: use node_subs order as fallback
                    indices = []
                    for sub in node_subs:
                        key = re.sub(
                            r"[^a-z0-9_]", "_",
                            (sub[:-1] if sub.endswith("!") else sub).lower(),
                        )
                        if key in dim_to_idx:
                            indices.append(dim_to_idx[key])

                inner = f"{julia_name}[{', '.join(indices)}]"
                if bang_ranges:
                    for_clause = ", ".join(bang_ranges)
                    return f"[{inner} for {for_clause}]"
                else:
                    # All ! dims were already active — no new comprehension
                    return inner

            indices = []
            # Track which active loop vars have been consumed by alignment so
            # two different range names (e.g. sectors_a_matrix and
            # sectors_a_matrix1) that both map to the same element-set don't
            # both resolve to the same variable.
            used_align_vars: set = set()
            for pos, sub in enumerate(node_subs):
                if sub in self.active_subs:
                    # Range name matching an active loop variable
                    lv = self.active_subs[sub]
                    indices.append(lv)
                    used_align_vars.add(lv)
                elif sub in self.subs_elems:
                    # Range name with all elements — use active loop var if available.
                    # First try direct name match, then fall back to element-set
                    # alignment (handles aliases like sectors_a_matrix ↔ sectors).
                    idx_var = self.active_subs.get(sub)
                    if idx_var and idx_var not in used_align_vars:  # pragma: no cover  # unreachable: elif subs_elems only runs when sub not in active_subs
                        indices.append(idx_var)
                        used_align_vars.add(idx_var)
                    elif not idx_var:
                        # Aligned range: find an active range with the same elements
                        sub_elems = self.subs_elems.get(sub, [])
                        aligned = None
                        # Element-set match (exact) — prefer first unused
                        for ar, lv in self.active_subs.items():
                            if lv in used_align_vars:
                                continue
                            if sub_elems and self.subs_elems.get(ar, []) == sub_elems:
                                aligned = lv
                                break
                        # Size match fallback
                        if aligned is None and sub_elems:
                            sub_size = len(sub_elems)
                            for ar, lv in self.active_subs.items():
                                if lv in used_align_vars:
                                    continue
                                if len(self.subs_elems.get(ar, [])) == sub_size:
                                    aligned = lv
                                    break
                        if aligned:
                            indices.append(aligned)
                            used_align_vars.add(aligned)
                        # else: genuinely unresolvable — skip (rare)
                else:
                    # Specific element label → numeric index in the variable's own dim.
                    # Prefer the variable's declared dim at this position so that
                    # sub-ranges (e.g. matter_final_sources) yield a local index,
                    # not the index from a larger parent range (e.g. final_sources).
                    parent_range = None
                    if pos < len(var_dims_list):
                        candidate = var_dims_list[pos]
                        if sub in self._elem_index and candidate in self._elem_index[sub]:
                            parent_range = candidate
                    if parent_range is None:
                        # Fallback: any of the variable's declared dims that contain sub
                        for rng in var_dims_list:
                            if sub in self._elem_index and rng in self._elem_index[sub]:
                                parent_range = rng
                                break
                    if parent_range is None and sub in self._elem_index:
                        # Last resort: first known range (may be wrong for sub-ranges)
                        parent_range = next(iter(self._elem_index[sub]))
                    if parent_range is not None and sub in self._elem_index:
                        indices.append(str(self._elem_index[sub][parent_range]))
                    elif sub in self._elem_index:  # pragma: no cover  # unreachable: parent_range is always set when sub in _elem_index
                        idx_val = next(iter(self._elem_index[sub].values()))
                        indices.append(str(idx_val))
            if indices:
                # GET DATA / LOOKUPS functions must use call syntax f(i, t),
                # not array-index syntax f[i].
                if julia_name in self.lookup_names:
                    return f"{julia_name}({', '.join(indices + ['t'])})"
                julia_name = julia_name + "[" + ", ".join(indices) + "]"
        elif self.active_subs and self.var_dims:
            # (B) No explicit subscripts: apply active loop variables.
            dims = self.var_dims.get(julia_name, [])
            indices = [self.active_subs[d] for d in dims if d in self.active_subs]
            if indices:
                julia_name = julia_name + "[" + ", ".join(indices) + "]"

        return julia_name

    def _call(self, node: CallStructure) -> str:
        func_upper = node.function.reference.upper()
        julia_func = BUILTIN_FUNCTIONS.get(func_upper)

        if julia_func is None:
            # Check whether the function name is a model variable (lookup table).
            # Vensim allows calling a lookup variable as a function:
            #   result = my_lookup_table(input_value)
            # We check the namespace and emit the variable name directly
            # (which will be a Julia interpolation function if loaded correctly).
            julia_id = self.namespace.get(node.function.reference)
            if julia_id is not None:
                args = [self.visit(a) for a in node.arguments]

                # Handle aggregation subscripts on the function reference itself,
                # e.g. f[dim1!, dim2](t) → [f(_ii0, _i0, t) for _ii0 in 1:N_DIM1]
                func_node_subs = (
                    node.function.subscripts.subscripts
                    if node.function.subscripts is not None
                    and hasattr(node.function.subscripts, "subscripts")
                    else []
                )
                if func_node_subs and any(s.endswith("!") for s in func_node_subs):
                    var_dims_list = self.var_dims.get(julia_id, [])
                    bang_ranges_c: List[str] = []
                    ii_count_c = 0
                    dim_to_idx_c: Dict[str, str] = {}
                    for sub in func_node_subs:
                        clean_sub = re.sub(r"[^a-z0-9_]", "_", sub.lower())
                        if sub.endswith("!"):
                            bare = sub[:-1]
                            clean_bare = re.sub(r"[^a-z0-9_]", "_", bare.lower())
                            if clean_bare in self._clean_active_subs:
                                # Already iterated by an outer SUM comprehension —
                                # reuse the existing loop variable, don't add a new range.
                                iv = self._clean_active_subs[clean_bare]
                                dim_to_idx_c[clean_bare] = iv
                            else:
                                dim_name = next(
                                    (d for d in var_dims_list
                                     if re.sub(r"[^a-z0-9_]", "_", d.lower()) == clean_bare),
                                    bare,
                                )
                                iv = f"_ii{ii_count_c}"
                                ii_count_c += 1
                                dim_to_idx_c[re.sub(r"[^a-z0-9_]", "_", dim_name.lower())] = iv
                                bang_ranges_c.append(f"{iv} in 1:{self._jl_n(dim_name)}")
                        elif sub in self.active_subs:
                            dim_to_idx_c[clean_sub] = self.active_subs[sub]
                        elif sub in self.subs_elems:
                            idx_var = self.active_subs.get(sub)
                            if idx_var:  # pragma: no cover  # unreachable: elif subs_elems only when sub not in active_subs
                                dim_to_idx_c[clean_sub] = idx_var
                        else:
                            if sub in self._elem_index:
                                idx_val = next(iter(self._elem_index[sub].values()))
                                dim_to_idx_c[clean_sub] = str(idx_val)
                    # Assemble call indices in var_dims_list order with fallback
                    bang_iv_pool_c = [
                        dim_to_idx_c[re.sub(r"[^a-z0-9_]", "_", sub[:-1].lower())]
                        for sub in func_node_subs
                        if sub.endswith("!")
                        and re.sub(r"[^a-z0-9_]", "_", sub[:-1].lower()) in dim_to_idx_c
                    ]
                    used_ivars_c: set = set()
                    if var_dims_list:
                        call_indices = []
                        for d in var_dims_list:
                            clean_d = re.sub(r"[^a-z0-9_]", "_", d.lower())
                            if clean_d in dim_to_idx_c:
                                iv = dim_to_idx_c[clean_d]
                                call_indices.append(iv)
                                used_ivars_c.add(iv)
                            else:
                                d_size = (self.subs_sizes.get(d, 0) or
                                          self._clean_subs_sizes.get(clean_d, 0))
                                matched = None
                                for sub in func_node_subs:
                                    if not sub.endswith("!"):
                                        continue
                                    bare = sub[:-1]
                                    cb = re.sub(r"[^a-z0-9_]", "_", bare.lower())
                                    iv = dim_to_idx_c.get(cb)
                                    if iv is None or iv in used_ivars_c:
                                        continue
                                    bare_size = (self.subs_sizes.get(bare, 0) or
                                                 self._clean_subs_sizes.get(cb, 0))
                                    if d_size > 0 and bare_size == d_size:
                                        matched = iv
                                        used_ivars_c.add(iv)
                                        break
                                if matched is None:
                                    for iv in bang_iv_pool_c:
                                        if iv not in used_ivars_c:
                                            matched = iv
                                            used_ivars_c.add(iv)
                                            break
                                if matched is not None:
                                    call_indices.append(matched)
                    else:
                        call_indices = []
                        for sub in func_node_subs:
                            key = re.sub(
                                r"[^a-z0-9_]", "_",
                                (sub[:-1] if sub.endswith("!") else sub).lower(),
                            )
                            if key in dim_to_idx_c:
                                call_indices.append(dim_to_idx_c[key])
                    inner_call = f"{julia_id}({', '.join(call_indices + args)})"
                    if bang_ranges_c:
                        for_clause_c = ", ".join(bang_ranges_c)
                        return f"[{inner_call} for {for_clause_c}]"
                    else:
                        # All ! dims already active via outer comprehension.
                        return inner_call

                # Explicit function subscripts without '!': resolve each subscript
                # positionally (range name → active loop var; element label →
                # literal index), matching the logic in _reference for explicit
                # node_subs.  This handles e.g.
                #   Historic_water_use[sectors, water](Time)
                # where var_dims uses the parent dim 'sectors_and_households'
                # which doesn't appear in active_subs, but 'sectors' does.
                if func_node_subs:
                    var_dims_list = self.var_dims.get(julia_id, [])
                    call_indices: List[str] = []
                    used_align_vars_c2: set = set()
                    for pos, sub in enumerate(func_node_subs):
                        if sub in self.active_subs:
                            lv = self.active_subs[sub]
                            call_indices.append(lv)
                            used_align_vars_c2.add(lv)
                        elif sub in self.subs_elems:
                            idx_var = self.active_subs.get(sub)
                            if idx_var and idx_var not in used_align_vars_c2:  # pragma: no cover  # unreachable: elif subs_elems only when sub not in active_subs
                                call_indices.append(idx_var)
                                used_align_vars_c2.add(idx_var)
                            elif not idx_var:
                                sub_elems = self.subs_elems.get(sub, [])
                                aligned: Optional[str] = None
                                for ar, lv in self.active_subs.items():
                                    if lv in used_align_vars_c2:
                                        continue
                                    if sub_elems and self.subs_elems.get(ar, []) == sub_elems:
                                        aligned = lv
                                        break
                                if aligned is None and sub_elems:
                                    sub_size = len(sub_elems)
                                    for ar, lv in self.active_subs.items():
                                        if lv in used_align_vars_c2:
                                            continue
                                        if len(self.subs_elems.get(ar, [])) == sub_size:
                                            aligned = lv
                                            break
                                if aligned:
                                    call_indices.append(aligned)
                                    used_align_vars_c2.add(aligned)
                        else:
                            parent_range: Optional[str] = None
                            if pos < len(var_dims_list):
                                candidate = var_dims_list[pos]
                                if sub in self._elem_index and candidate in self._elem_index[sub]:
                                    parent_range = candidate
                            if parent_range is None:
                                for rng in var_dims_list:
                                    if sub in self._elem_index and rng in self._elem_index[sub]:
                                        parent_range = rng
                                        break
                            if parent_range is None and sub in self._elem_index:
                                parent_range = next(iter(self._elem_index[sub]))
                            if parent_range is not None and sub in self._elem_index:
                                call_indices.append(str(self._elem_index[sub][parent_range]))
                            elif sub in self._elem_index:  # pragma: no cover  # unreachable: parent_range is always set when sub in _elem_index
                                call_indices.append(str(next(iter(self._elem_index[sub].values()))))
                    if call_indices:
                        return f"{julia_id}({', '.join(call_indices + args)})"

                if self.var_dims:
                    dims = self.var_dims.get(julia_id, [])
                    if dims:
                        if self.active_subs:
                            # Subscript comprehension context: prepend active indices.
                            # historic_gfcf(t) → historic_gfcf(_i0, t)
                            indices = [
                                self.active_subs[d] for d in dims if d in self.active_subs
                            ]
                            if indices:
                                args = indices + args
                        else:
                            # Scalar context: broadcast over all dim indices.
                            # sum(historic_labour_compensation(t))
                            # → sum([historic_labour_compensation(_ii0, t) for _ii0 in 1:N_SECTORS])
                            idx_vars = [f"_ii{k}" for k in range(len(dims))]
                            full_args = idx_vars + args
                            ranges = ", ".join(
                                f"{iv} in 1:{self._jl_n(d)}"
                                for iv, d in zip(idx_vars, dims)
                            )
                            return f"[{julia_id}({', '.join(full_args)}) for {ranges}]"
                return f"{julia_id}({', '.join(args)})"
            # Check if the function is a known Vensim macro — no warning needed.
            clean_ref = re.sub(r"[^a-z0-9_]", "_", node.function.reference.lower())
            if clean_ref in self._macro_names:
                args = [self.visit(a) for a in node.arguments]
                return f"{clean_ref}({', '.join(args)})"
            warn(f"Unknown Vensim function '{node.function.reference}'; using lowercase name.")
            julia_func = re.sub(r"[^a-z0-9_]", "_", node.function.reference.lower())

        # ELMCOUNT(SubscriptRange) → emit the integer literal size
        if julia_func == "pysd_elmcount":
            if node.arguments:
                arg = node.arguments[0]
                if isinstance(arg, ReferenceStructure):
                    size = self.subs_sizes.get(arg.reference)
                    if size is None:
                        # Case-insensitive fallback (abstract model may use different
                        # casing from the expression parser)
                        clean = re.sub(r"[^a-z0-9_]", "_", arg.reference.lower())
                        size = self._clean_subs_sizes.get(clean)
                    if size is not None:
                        return str(size)
                # Fall back: try to visit the argument and return it
                return self.visit(arg)
            return "0"

        if julia_func in HELPER_IMPLEMENTATIONS:
            self.needed_helpers.add(julia_func)
            if julia_func == "pysd_quantum":
                self.needed_helpers.add("pysd_trunc")

        # sum/prod/vmax/vmin with ! subscripts: generate ONE comprehension that
        # covers ALL references sharing the same ! dim, rather than separate
        # per-reference comprehensions that would be multiplied/added as arrays.
        if julia_func in ("sum", "prod", "maximum", "minimum") and len(node.arguments) == 1:
            bang_subs = self._collect_bang_subs(node.arguments[0])
            new_bang_subs = [
                s for s in bang_subs
                if re.sub(r"[^a-z0-9_]", "_", s[:-1].lower())
                not in self._clean_active_subs
            ]
            if new_bang_subs:
                extra_subs: Dict[str, str] = {}
                agg_ranges: List[str] = []
                ii_cnt = 0
                for sub in new_bang_subs:
                    bare = sub[:-1]
                    clean_bare = re.sub(r"[^a-z0-9_]", "_", bare.lower())
                    iv = f"_ii{ii_cnt}"
                    ii_cnt += 1
                    extra_subs[bare] = iv
                    extra_subs[clean_bare] = iv
                    agg_ranges.append(f"{iv} in 1:{self._jl_n(bare)}")
                child = self._with_extra_subs(extra_subs)
                arg_expr = child.visit(node.arguments[0])
                for_clause_agg = ", ".join(agg_ranges)
                return f"{julia_func}([{arg_expr} for {for_clause_agg}])"

        args = [self.visit(a) for a in node.arguments]

        # pysd_ifelse dispatches on the condition type.  Vensim IF THEN ELSE
        # accepts any numeric condition (nonzero = true), so a bare variable or
        # arithmetic expression must be wrapped with `!= 0` to produce a Bool.
        # Only LogicStructure arguments (comparisons like `<`, `>`, `==`, and
        # logical operators) are already Bool — leave them untouched.
        if julia_func == "pysd_ifelse" and args:
            if not isinstance(node.arguments[0], LogicStructure):
                args[0] = f"({args[0]} != 0)"

        # Time-dependent helpers receive the symbolic *t* as their first arg
        if julia_func in _TIME_HELPERS:
            return f"{julia_func}(t, {', '.join(args)})"

        return f"{julia_func}({', '.join(args)})"

    def _inline_lookup(self, node: InlineLookupsStructure) -> str:
        arg_expr = self.visit(node.argument)
        name = self.registry.register(node.lookups.x, node.lookups.y, node.lookups.type)
        return f"{name}({arg_expr})"
