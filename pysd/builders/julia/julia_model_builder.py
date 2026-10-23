"""
Translates a PySD AbstractModel into a standalone Julia file that uses
ModelingToolkit.jl.  The generated file requires no PySD or Python at runtime.

Entry point::

    from pysd.builders.julia.julia_model_builder import JuliaModelBuilder
    path = JuliaModelBuilder(abstract_model).build_model()
"""
from __future__ import annotations

import dataclasses
import itertools
import math
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from warnings import warn

from pysd._version import __version__
from pysd.translators.structures.abstract_model import (
    AbstractComponent,
    AbstractControlElement,
    AbstractData,
    AbstractElement,
    AbstractLookup,
    AbstractModel,
    AbstractSection,
    AbstractUnchangeableConstant,
)
from pysd.translators.structures.abstract_expressions import (
    AllocateAvailableStructure,
    AllocateByPriorityStructure,
    ArithmeticStructure,
    CallStructure,
    DataStructure,
    DelayFixedStructure,
    DelayNStructure,
    DelayStructure,
    ForecastStructure,
    GetConstantsStructure,
    GetDataStructure,
    GetLookupsStructure,
    InitialStructure,
    IntegStructure,
    LookupsStructure,
    ReferenceStructure,
    SampleIfTrueStructure,
    SmoothNStructure,
    SmoothStructure,
    TrendStructure,
)

from .julia_expressions_builder import (
    HELPER_IMPLEMENTATIONS,
    InlineLookupRegistry,
    JuliaASTVisitor,
    format_number,
    lookup_interpolation_code,
)
from .macro_inliner import inline_macros
from .namespace import JuliaNamespaceManager


_CELL_REF_RE = re.compile(r"^\$?([A-Za-z]{1,3})\$?([0-9]+)$")


def _cell_block(cell: str, coords: dict, transpose: bool) -> str:
    """Turn a top-left cell into the block of cells a variable reads.

    As Python PySD (``ExtConstant``): the shape is the variable's
    dimensions without the ones of length 1, completed to 2D with a 1 on
    the left (so a 1D variable reads one row), and reversed when the
    reference is transposed (``*``).  Named ranges and single-element
    variables are returned unchanged.
    """
    m = _CELL_REF_RE.match(cell)
    sizes = [len(v) for v in coords.values() if len(v) > 1]
    if not m or not sizes or len(sizes) > 2:
        return cell
    shape = [1] * (2 - len(sizes)) + sizes
    if transpose:
        shape.reverse()
    col = 0
    for ch in m.group(1).upper():
        col = col * 26 + ord(ch) - ord("A") + 1
    row = int(m.group(2))
    last_col, last_row = col + shape[1] - 1, row + shape[0] - 1
    letters = ""
    while last_col:
        last_col, rem = divmod(last_col - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return f"{m.group(1).upper()}{row}:{letters}{last_row}"



def _cell_blocks(cells) -> List[Tuple[List[int], ...]]:
    """Split a set of index tuples into rectangular blocks (one list of
    indices per dimension) that cover exactly those cells.

    Values of the first index with the same remaining cells share a block;
    the remaining cells are split the same way, dimension by dimension.
    """
    cells = set(cells)
    if not cells:
        return []
    if len(next(iter(cells))) == 1:
        return [(sorted(c[0] for c in cells),)]
    rest: Dict[int, frozenset] = {}
    for c in cells:
        rest.setdefault(c[0], set()).add(c[1:])
    groups: Dict[frozenset, List[int]] = {}
    for i, sub in sorted(rest.items()):
        groups.setdefault(frozenset(sub), []).append(i)
    return [
        (firsts, *block)
        for sub, firsts in groups.items()
        for block in _cell_blocks(sub)
    ]


def _outer_for_clause(eq: str) -> str:
    """Return the for-clause of the comprehension equation ``[lhs ~ rhs for ...]...``.

    Only the outer clause (at bracket depth 0) counts: the right-hand side may
    contain aggregations with their own loops, e.g. ``sum(w[_ii0] for _ii0 in 1:N_B)``.
    Returns "" when there is none.
    """
    inner = eq.strip().rstrip(",")
    inner = inner[:-3] if inner.endswith("...") else inner
    inner = inner[1:-1] if inner.startswith("[") and inner.endswith("]") else inner
    depth = 0
    for i in range(len(inner) - 1, 3, -1):
        c = inner[i]
        if c in ")]":
            depth += 1
        elif c in "([":
            depth -= 1
        elif depth == 0 and inner[i:i + 4] == "for " and inner[i - 1] == " ":
            return inner[i + 4:]
    return ""


#: PySD.jl version the generated models are written for (checked on load).
PYSD_JL_VERSION = "0.2.6"

# Control variable identifiers produced by Vensim
_CONTROL_IDENTIFIERS = frozenset(
    {"initial_time", "final_time", "time_step", "saveper"}
)

# Structures that expand to auxiliary state variables (handled at element level)
_STATEFUL_STRUCTURES = (
    IntegStructure,
    SmoothStructure,
    SmoothNStructure,
    DelayStructure,
    DelayNStructure,
    DelayFixedStructure,
)

# Structures not yet supported — emit a warning and a placeholder equation
# Note: DataStructure is handled explicitly in _process_element (when the
# component is AbstractData with a keyword, it reads from a .tab file at
# runtime).  Non-AbstractData DataStructure ASTs still fall through here.
_UNSUPPORTED_STRUCTURES = ()


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


#: Functions _eval_ast_at_t0 can evaluate on numbers (Vensim names, lower case)
_T0_MATH = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp,
    "ln": math.log, "log": math.log10, "sqrt": math.sqrt, "abs": abs,
    "integer": math.trunc, "max": max, "min": min,
}


def _non_negative_flow(ast: IntegStructure, flow_expr: str, stock_ref: str) -> str:
    """Flow of a stock, limited so a non-negative stock (XMILE) stays >= 0.

    PySD's NonNegativeInteg sets the stock to max(stock + dt * flow, 0) after
    each Euler step; with a fixed step that is the same as limiting the flow
    to -stock / dt.
    """
    if getattr(ast, "non_negative", False):
        return f"max({flow_expr}, -{stock_ref} / time_step)"
    return flow_expr


def expand_subscript_ranges(model: AbstractModel) -> AbstractModel:
    """Expand subscript ranges defined with other ranges.

    Vensim allows ``dim: A, subdim1, B`` where ``subdim1`` is itself a range;
    the parser keeps the range name.  As Python PySD, the range is the list
    of elements with each nested range replaced by its elements (recursively),
    so ``dim`` has 2 + len(subdim1) elements.  Copy definitions such as
    ``subdim2 copy: subdim2`` become the elements of the copied range.
    """
    import dataclasses

    lists = {
        sr.name: sr.subscripts
        for section in model.sections for sr in section.subscripts
        if isinstance(sr.subscripts, list)
    }

    def expand(name: str, seen: tuple) -> list:
        out = []
        for entry in lists[name]:
            if entry in lists and entry not in seen:
                out.extend(expand(entry, seen + (entry,)))
            else:
                out.append(entry)
        return out

    sections = []
    for section in model.sections:
        ranges = tuple(
            dataclasses.replace(sr, subscripts=expand(sr.name, (sr.name,)))
            if isinstance(sr.subscripts, list) else sr
            for sr in section.subscripts
        )
        sections.append(dataclasses.replace(section, subscripts=ranges))
    return dataclasses.replace(model, sections=tuple(sections))


class JuliaModelBuilder:
    """Build a standalone Julia/ModelingToolkit model from an AbstractModel.

    Parameters
    ----------
    abstract_model:
        The abstract model produced by a PySD translator.
    data_format : str, optional
        How to store external numeric data.  ``"hardcoded"`` (default) inlines
        all values as Julia literals in the generated ``.jl`` file.
        ``"json"`` writes a companion ``<model>_data.json`` file and generates
        Julia code that reads it at startup via ``JSON3.jl``.
    """

    def __init__(
        self,
        abstract_model: AbstractModel,
        data_format: str = "hardcoded",
        backend: str = "ode",
    ) -> None:
        if data_format not in ("hardcoded", "json"):
            raise ValueError(
                f"data_format must be 'hardcoded' or 'json', got {data_format!r}"
            )
        if backend not in ("ode", "mtk"):
            raise ValueError(
                f"backend must be 'ode' or 'mtk', got {backend!r}"
            )
        abstract_model = expand_subscript_ranges(abstract_model)
        self.original_path = abstract_model.original_path
        # Macros are expanded in place (macro_inliner), leaving the main section
        abstract_model = inline_macros(abstract_model)
        self.sections = [
            JuliaSectionBuilder(section, data_format=data_format, backend=backend)
            for section in abstract_model.sections
        ]

    def build_model(self) -> Path:
        """Translate the model and return the path to the main ``.jl`` file."""
        main = self.sections[0]
        main.build_section()
        if main.backend == "ode" and not (main.split and main.views_dict):
            tidy_julia_file(main.path)
        return main.path


def tidy_julia_file(path: Path) -> bool:
    """Write the expressions of a generated file with only the parentheses
    they need, with Julia's own printer (``tidy.jl``): the expressions are
    the same.  Needs ``julia`` on the PATH; without it the file is left as it
    is, with a warning.  ``PYSD_JULIA_TIDY=0`` turns it off."""
    import os
    import shutil
    import subprocess
    if os.environ.get("PYSD_JULIA_TIDY", "1") == "0":
        return False
    julia = shutil.which("julia")
    if julia is None:
        warn("julia not found: the generated model keeps its extra parentheses "
             "(install Julia to have them removed)")
        return False
    script = Path(__file__).with_name("tidy.jl")
    try:
        subprocess.run([julia, "--startup-file=no", str(script), str(path)],
                       check=True, capture_output=True, text=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as exc:
        warn(f"the generated model keeps its extra parentheses: {exc}")
        return False
    return True


# ---------------------------------------------------------------------------
# Section builder
# ---------------------------------------------------------------------------

class JuliaSectionBuilder:
    """Build one section (main model or macro) of the Julia output.

    Parameters
    ----------
    abstract_section:
        The abstract section to translate.
    data_format : str, optional
        ``"hardcoded"`` (default) or ``"json"``.  When ``"json"``, numeric data
        is written to a companion ``<model>_data.json`` file and the generated
        Julia code reads it at startup via ``JSON3.jl``.
    """

    def __init__(
        self,
        abstract_section: AbstractSection,
        data_format: str = "hardcoded",
        backend: str = "ode",
    ) -> None:
        self.backend: str = backend
        self._ode_active_initial: bool = False
        self._ode_p_names: Set[str] = set()
        self._ode_stock_ndims: Dict[str, int] = {}
        # constants whose value the translation used: name -> what it fixed
        self._fixed_inputs: Dict[str, str] = {}
        self._fix_reason: Optional[str] = None
        self.name: str = abstract_section.name
        self.path: Path = abstract_section.path.with_suffix(".jl")
        self.root: Path = self.path.parent
        self.model_name: str = self.path.stem
        self.split: bool = abstract_section.split
        self.views_dict: Optional[dict] = abstract_section.views_dict
        self.abstract_elements: List[AbstractElement] = list(abstract_section.elements)
        self._abstract_subscripts = abstract_section.subscripts
        # Julia ids of macros, for the visitor (empty: macros are expanded in place)
        self._known_macro_names: Set[str] = set()
        self.data_format: str = data_format
        # JSON data accumulator — populated when data_format == "json"
        self._json_data: Dict[str, dict] = {
            "constants": {},
            "lookups": {},
            "data": {},
        }

        self.namespace = JuliaNamespaceManager()
        self.inline_registry = InlineLookupRegistry()
        self.needed_helpers: Set[str] = set()

        # Map subscript range name → number of elements
        self._subs_sizes: Dict[str, int] = {}
        # Map subscript range name → ordered list of element labels
        self._subs_elems: Dict[str, List[str]] = {}

        for sr in self._abstract_subscripts:
            if isinstance(sr.subscripts, list):
                self._subs_sizes[sr.name] = len(sr.subscripts)
                self._subs_elems[sr.name] = list(sr.subscripts)
            elif isinstance(sr.subscripts, str):
                # copy alias — resolve later if needed, default to 0
                self._subs_sizes[sr.name] = 0
            elif isinstance(sr.subscripts, dict):
                # External subscript (GET DIRECT SUBSCRIPT from Excel/XLS).
                # Read element labels at translation time so that constants
                # defined over these ranges can be shaped correctly.
                try:
                    from pysd.py_backend.external import ExtSubscript
                    ext = ExtSubscript(
                        file_name=sr.subscripts["file"],
                        tab=sr.subscripts["tab"],
                        firstcell=sr.subscripts["firstcell"],
                        lastcell=sr.subscripts["lastcell"],
                        prefix=sr.subscripts["prefix"],
                        root=self.root,
                    )
                    elems = ext.subscript
                    self._subs_sizes[sr.name] = len(elems)
                    self._subs_elems[sr.name] = elems
                except Exception:  # pragma: no cover
                    self._subs_sizes[sr.name] = 0

        # Resolve string-alias subscript ranges (e.g. "SEC ALL MAP = SEC ALL")
        for sr in self._abstract_subscripts:
            if isinstance(sr.subscripts, str):
                aliased = sr.subscripts
                if self._subs_sizes.get(aliased, 0) > 0:
                    self._subs_sizes[sr.name] = self._subs_sizes[aliased]
                if aliased in self._subs_elems:
                    self._subs_elems[sr.name] = self._subs_elems[aliased]

        # Map Julia identifier → comment string (units / documentation)
        self._var_comments: Dict[str, str] = {}

        # Accumulated declarations
        self.stock_decls: List[str] = []
        self.aux_decls: List[str] = []
        self.param_decls: List[str] = []
        self.ext_const_decls: List[str] = []
        self.lookup_const_decls: List[str] = []
        self.lookup_func_decls: List[str] = []
        self.lookup_register_decls: List[str] = []
        self.lookup_identifiers: Set[str] = set()
        self.subs_const_decls: List[str] = []
        self.u0_entries: List[str] = []
        # Map julia identifier -> list of dim names (for subscripted vars)
        self._var_dims: Dict[str, List[str]] = {}
        # Names of identifiers that are lookup/data functions (need `(t)` when referenced bare)
        self._lookup_func_names: Set[str] = set()
        # GET DATA variables (functions of time in params), reported by observe
        self._data_names: Set[str] = set()
        # Tab-data entries: list of (julia_id, real_name, method_sym, dim_elem_lists)
        # where dim_elem_lists is a list of element-label lists (one per subscript dim)
        self._tab_data_entries: List[Tuple[str, str, str, List[List[str]]]] = []

        # Reverse map: element label → parent range name (for per-element component coords)
        self._elem_to_range: Dict[str, str] = {}
        for sr in self._abstract_subscripts:
            if isinstance(sr.subscripts, list):
                for elem_label in sr.subscripts:
                    if elem_label not in self._elem_to_range:
                        self._elem_to_range[elem_label] = sr.name

        self.control_vals: Dict[str, Optional[str]] = {
            "initial_time": None,
            "final_time": None,
            "time_step": None,
            "saveper": None,
        }

        # Maps Julia identifier -> (equations, is_control_var)
        self.built_elements: Dict[str, Tuple[List[str], bool]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_section(self) -> None:
        """Build the section, writing its ``.jl`` file."""
        # First pass: populate the namespace with all element names
        for elem in self.abstract_elements:
            self.namespace.add_to_namespace(elem.name)

        # Pre-populate _var_dims for every subscripted element so that the
        # subscript-indexed visitor can correctly index forward-referenced
        # variables even when they haven't been processed yet.
        for elem in self.abstract_elements:
            identifier = self.namespace.namespace.get(elem.name)
            if identifier:
                dims = self._element_dims(elem)
                if dims:
                    self._var_dims[identifier] = [d for d, _ in dims]

        # Pre-populate _lookup_func_names for GET DATA / GET LOOKUPS elements
        # so that bare references to them in equations auto-call f(t).
        for elem in self.abstract_elements:
            identifier = self.namespace.namespace.get(elem.name)
            if identifier and elem.components:
                if all(isinstance(c.ast, GetLookupsStructure) for c in elem.components) or \
                   any(isinstance(c.ast, GetDataStructure) for c in elem.components):
                    self._lookup_func_names.add(identifier)
                if any(isinstance(c.ast, GetDataStructure) for c in elem.components):
                    self._data_names.add(identifier)

        # Emit subscript size constants (const N_DIMNAME = n)
        for name, size in sorted(self._subs_sizes.items()):
            if size > 0:
                jl_name = "N_" + re.sub(r"[^a-z0-9]", "_", name.lower()).upper()
                self.subs_const_decls.append(f"const {jl_name} = {size}")

        # Pre-scan: build a map of identifier → float value for scalar numeric
        # constants.  This allows constructs like DELAY FIXED to resolve a named
        # constant as their delay time even when that constant is defined later in
        # the model file (i.e. before its element has been processed).
        self._prescanned_const_vals: Dict[str, float] = {}
        for _elem in self.abstract_elements:
            _id = self.namespace.namespace.get(_elem.name)
            if not _id or not _elem.components:
                continue
            _comp = _elem.components[0]
            _ast = _comp.ast
            if isinstance(_ast, (int, float)):
                try:
                    self._prescanned_const_vals[_id] = float(_ast)
                except (ValueError, TypeError):  # pragma: no cover
                    pass

        # Second pass: process control elements first so that control_vals
        # (especially time_step) are available for constructs like SAMPLE IF TRUE.
        non_control_elems = []
        initial_elems = []
        for elem in self.abstract_elements:
            identifier = self.namespace.namespace[elem.name]
            is_control = isinstance(elem, AbstractControlElement)
            comp = elem.components[0] if elem.components else None
            if is_control:
                eqs = self._process_element(elem, identifier, is_control)
                self.built_elements[identifier] = (eqs, is_control)
            elif comp is not None and isinstance(comp.ast, InitialStructure):
                initial_elems.append((elem, identifier, is_control))
            else:
                non_control_elems.append((elem, identifier, is_control))

        # Third pass: process non-control, non-INITIAL elements
        for elem, identifier, is_control in non_control_elems:
            eqs = self._process_element(elem, identifier, is_control)
            self.built_elements[identifier] = (eqs, is_control)

        # Fourth pass: INITIAL elements (u0_entries now complete)
        for elem, identifier, is_control in initial_elems:
            eqs = self._process_element(elem, identifier, is_control)
            self.built_elements[identifier] = (eqs, is_control)

        self._sort_u0_entries()

        # Register any inline lookups collected while visiting ASTs
        for lut_name, xs, ys, itp_type in self.inline_registry.entries:
            const_decl, func_decl, reg_decl = lookup_interpolation_code(lut_name, xs, ys, itp_type)
            self.lookup_const_decls.append(const_decl)
            self.lookup_func_decls.append(func_decl)
            self.lookup_register_decls.append(reg_decl)
            if self.data_format == "json":
                self._json_data["lookups"][lut_name] = {
                    "x": list(xs), "y": list(ys),
                    "interp_type": itp_type, "subscripts": [],
                }

        if self.split and self.views_dict and self.backend == "ode":
            raise ValueError(
                "split_views=True is not supported with backend='ode'. "
                "Use backend='mtk' for modular (multi-file) builds."
            )
        if self.split and self.views_dict:
            self._build_modular()
        else:
            self._build()

    # ------------------------------------------------------------------
    # Subscript helpers
    # ------------------------------------------------------------------

    def _dispatch_n(self, comp: "AbstractComponent") -> str:
        """Keyword arguments for pysd_xlsx_build_lookup_dispatch: ``n``, the
        number of series (rows or columns from the start cell) that *comp*
        covers, which PySD.jl needs in row and column mode; and ``method``,
        the DATA keyword (HOLD BACKWARD, LOOK FORWARD), when given."""
        kwargs = []
        n = 1
        for labels in self._comp_coords(comp).values():
            n *= max(len(labels), 1)
        if n > 1:
            kwargs.append(f"n={n}")
        keyword = getattr(comp, "keyword", None)
        if keyword in ("hold_backward", "look_forward"):
            kwargs.append(f"method=:{keyword}")
        return "; " + ", ".join(kwargs) if kwargs else ""

    def _comp_coords(self, comp: "AbstractComponent") -> Dict[str, list]:
        """Build an ``{range_name: [element_labels]}`` coords dict for *comp*.

        Each item in ``comp.subscripts[0]`` may be either a subscript-range
        name (→ use all its elements) or a specific element name (→ resolve to
        its parent range with a single-element list).  This mirrors what the
        Python builder passes to ``ExtLookup``/``ExtData``/``ExtConstant``.
        """
        def_subs = comp.subscripts[0] if comp.subscripts else []
        if not def_subs:
            return {}
        result: Dict[str, list] = {}
        for s in def_subs:
            if s in self._subs_elems:
                # Range name → full element list
                result[s] = self._subs_elems[s]
            elif s in self._elem_to_range:
                # Specific element → map to parent range with single-element list
                parent = self._elem_to_range[s]
                result[parent] = [s]
            else:
                result[s] = []
        return result

    def _infer_parent_range(self, elements: List[str]) -> Optional[str]:
        """Return the smallest subscript range that contains *all* given elements.

        Used to resolve ambiguity when the same element name appears in
        multiple ranges (e.g. ``Agriculture`` is in both ``sectors`` and
        ``SECTORS_and_HOUSEHOLDS``).
        """
        candidates = []
        for sr in self._abstract_subscripts:
            if isinstance(sr.subscripts, list) and all(
                e in sr.subscripts for e in elements
            ):
                candidates.append((len(sr.subscripts), sr.name))
        return min(candidates, key=lambda x: x[0])[1] if candidates else None

    def _infer_parent_range_not_in(
        self, elements: List[str], exclude: Optional[set] = None
    ) -> Optional[str]:
        """Like :meth:`_infer_parent_range` but skips ranges in *exclude*.

        Used when a candidate range is already occupied by a non-split
        subscript position (e.g. ``final_sources`` used for pos 1 should
        not also be the parent for the split pos 2 — use ``final_sources1``
        instead).
        """
        exclude = exclude or set()
        candidates = []
        for sr in self._abstract_subscripts:
            if isinstance(sr.subscripts, list) and sr.name not in exclude:
                if all(e in sr.subscripts for e in elements):
                    candidates.append((len(sr.subscripts), sr.name))
        return min(candidates, key=lambda x: x[0])[1] if candidates else None

    def _detect_split_ranges(
        self, components: List["AbstractComponent"]
    ) -> Dict[int, str]:
        """Return ``{position: parent_range}`` for subscript positions that
        carry different element names across *components*.

        Non-split positions' ranges are recorded first so that the split
        position is assigned a *different* range when the naive best-fit would
        collide (e.g. ``efficiency_rate_of_substitution`` has ``final_sources``
        at pos 1 and the split at pos 2 also maps to ``final_sources`` — we
        instead assign ``final_sources1`` to avoid the collision).
        """
        all_subs = [
            c.subscripts[0]
            for c in components
            if c.subscripts and c.subscripts[0]
        ]
        if not all_subs:
            return {}
        n_pos = len(all_subs[0])

        # Collect ranges committed by non-split (constant) positions
        committed: set = set()
        for pos in range(n_pos):
            vals = list({s[pos] for s in all_subs if len(s) > pos})
            if len(vals) == 1:
                s = vals[0]
                if s in self._subs_elems:
                    committed.add(s)
                elif s in self._elem_to_range:
                    committed.add(self._elem_to_range[s])

        result: Dict[int, str] = {}
        for pos in range(n_pos):
            vals = list({s[pos] for s in all_subs if len(s) > pos})
            if len(vals) > 1:
                parent = self._infer_parent_range_not_in(vals, exclude=committed)
                if parent is None:
                    parent = self._infer_parent_range(vals)
                if parent:
                    result[pos] = parent
        return result

    def _comp_coords_split(
        self,
        comp: "AbstractComponent",
        split_ranges: Dict[int, str],
    ) -> Dict[str, list]:
        """Like :meth:`_comp_coords` but uses *split_ranges* to override the
        parent-range lookup for positions that vary across components.
        """
        subs = comp.subscripts[0] if comp.subscripts else []
        result: Dict[str, list] = {}
        for pos, s in enumerate(subs):
            if pos in split_ranges:
                result[split_ranges[pos]] = [s]
            elif s in self._subs_elems:
                result[s] = self._subs_elems[s]
            elif s in self._elem_to_range:
                result[self._elem_to_range[s]] = [s]
            else:
                result[s] = []
        return result

    def _element_dims(self, elem: "AbstractElement") -> List[Tuple[str, int]]:
        """Return ``[(dim_name, dim_size), ...]`` for *elem*'s defining subscripts.

        Uses the first component's first subscript list.  Dims with size == 0
        (unresolved aliases) are filtered out.

        When a multi-component element has per-element subscripts (e.g.
        ``['electricity']``, ``['heat']``, …) rather than a range name, the
        parent range is inferred from all components' element labels so the
        variable is correctly declared as an array.
        """
        if not elem.components:
            return []
        comp = elem.components[0]
        if not comp.subscripts or not comp.subscripts[0]:
            return []
        dims = []
        for pos, dim_name in enumerate(comp.subscripts[0]):
            size = self._subs_sizes.get(dim_name, 0)
            if size > 0:
                # Check whether other components reference elements that fall
                # outside this range (the "range + sibling-element" pattern,
                # e.g. C_in_Deep_Ocean[upper] + C_in_Deep_Ocean[Layer4]).
                if len(elem.components) > 1:
                    range_elems = set(self._subs_elems.get(dim_name, []))
                    all_labels: set = set()
                    for c in elem.components:
                        if c.subscripts and len(c.subscripts[0]) > pos:
                            s = c.subscripts[0][pos]
                            if s in self._subs_elems:
                                all_labels.update(self._subs_elems[s])
                            elif s in self._elem_to_range:
                                all_labels.add(s)
                    if all_labels and not all_labels <= range_elems:
                        parent = self._infer_parent_range(list(all_labels))
                        if parent and self._subs_sizes.get(parent, 0) > size:
                            dim_name = parent
                            size = self._subs_sizes[parent]
                dims.append((dim_name, size))
            elif dim_name in self._elem_to_range:
                # dim_name is a specific element — infer the parent range from
                # all components' element at this position, or fall back to the
                # first known parent.
                if len(elem.components) > 1:
                    all_elems_at_pos = list({
                        c.subscripts[0][pos]
                        for c in elem.components
                        if c.subscripts and len(c.subscripts[0]) > pos
                    })
                    parent = self._infer_parent_range(all_elems_at_pos) or self._elem_to_range[dim_name]
                else:
                    parent = self._elem_to_range[dim_name]
                parent_size = self._subs_sizes.get(parent, 0)
                if parent_size > 0:
                    dims.append((parent, parent_size))
        return dims

    def _jl_n(self, dim_name: str) -> str:
        """Julia constant name for the size of a subscript dimension."""
        return "N_" + re.sub(r"[^a-z0-9]", "_", dim_name.lower()).upper()

    def _range_str(self, dims: List[Tuple[str, int]]) -> str:
        """Build ``'1:N_D0, 1:N_D1, ...'`` for array declarations."""
        return ", ".join(f"1:{self._jl_n(d)}" for d, _ in dims)

    def _per_index_subs(
        self,
        dim_name: str,
        dim_elems: List[str],
        abs_idx: int,
        def_range_name: Optional[str],
    ) -> Dict[str, str]:
        """Build ``active_subs`` for a per-index visitor in EXCEPT expansion.

        When iterating over a sub-range (*def_range_name*), Vensim aligns
        same-size ranges positionally: if we are at position *p* within the
        defining range, a reference ``[other_range]`` of the same size refers
        to element *other_range[p]*.  We pre-compute the absolute Julia array
        index for each such range so the expression visitor resolves them
        correctly without needing to understand range aliasing.
        """
        subs: Dict[str, str] = {dim_name: str(abs_idx)}
        if def_range_name is None or def_range_name == dim_name:
            return subs

        def_elems = self._subs_elems.get(def_range_name, [])
        if not def_elems:
            return subs

        element_label = dim_elems[abs_idx - 1]
        if element_label not in def_elems:
            return subs

        pos = def_elems.index(element_label)   # 0-based position within def_range
        def_size = len(def_elems)

        # Map the defining sub-range to its 1-based index WITHIN the sub-range,
        # not the absolute parent-dimension index.  Variables declared over the
        # sub-range (e.g. `sectors`) are 1-indexed from 1, so using the parent
        # dimension's absolute index (abs_idx) would produce out-of-bounds access
        # when the sub-range is offset within the parent (e.g. Households at 1,
        # sectors at 2-15).
        subs[def_range_name] = str(pos + 1)

        # For every range of the same size, Vensim aligns them positionally:
        # position *pos* in def_range corresponds to position *pos* in the other
        # range.  Each such range is also 1-indexed from 1 in Julia, so the
        # correct index is always pos+1.
        for sr in self._abstract_subscripts:
            if (
                isinstance(sr.subscripts, list)
                and len(sr.subscripts) == def_size
                and sr.name != def_range_name
                and sr.name != dim_name
            ):
                subs[sr.name] = str(pos + 1)

        return subs

    def _idx_vars(self, ndim: int) -> List[str]:
        """Generate index variable names ``_i0, _i1, ...`` for comprehensions."""
        return [f"_i{k}" for k in range(ndim)]

    def _for_clause(self, dims: List[Tuple[str, int]], idx_vars: List[str]) -> str:
        """Build ``'_i0 in 1:N_D0, _i1 in 1:N_D1, ...'`` for comprehensions."""
        return ", ".join(
            f"{iv} in 1:{self._jl_n(d)}" for (d, _), iv in zip(dims, idx_vars)
        )

    @staticmethod
    def _nd_cells(dims: List[Tuple[str, int]], idx_vars: List[str], expr: str):
        """Yield ``(index, expr)`` for every cell of *dims*, with the loop
        variables in *expr* replaced by that cell's indices."""
        for combo in itertools.product(*[range(1, n + 1) for _, n in dims]):
            cell_expr = expr
            for iv, val in zip(idx_vars, combo):
                cell_expr = re.sub(rf"\b{iv}\b", str(val), cell_expr)
            yield ", ".join(map(str, combo)), cell_expr

    def _nd_visitor(self, dims: List[Tuple[str, int]], idx_vars: List[str]) -> "JuliaASTVisitor":
        """Return a visitor with active subscript index context for N dims."""
        active_subs = {d: iv for (d, _), iv in zip(dims, idx_vars)}
        return JuliaASTVisitor(
            self.namespace, self.inline_registry, self.needed_helpers,
            active_subs=active_subs, var_dims=self._var_dims,
            subs_sizes=self._subs_sizes, subs_elems=self._subs_elems,
            lookup_names=self._lookup_func_names, root=self.root,
            macro_names=self._known_macro_names,
        )


    # ------------------------------------------------------------------
    # Limits helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _limits_comment(elem: "AbstractElement") -> str:
        """Return a block comment ``#= limits: [min, max] =#`` if *elem* has
        non-trivial limits, otherwise return an empty string.

        Block comments are used (rather than line comments ``#``) so that the
        trailing ``,`` separator added by the equation-array join is placed
        AFTER the closing ``=#`` and is therefore visible to the Julia parser.
        A line comment would swallow the comma, removing the array separator
        and causing a ParseError.
        """
        lims = getattr(elem, "limits", (None, None))
        if not lims or (lims[0] is None and lims[1] is None):
            return ""
        lo = "-Inf" if lims[0] is None else format_number(float(lims[0]))
        hi = "Inf" if lims[1] is None else format_number(float(lims[1]))
        return f"  #= limits: [{lo}, {hi}] =#"

    def _json_add_limits(self, elem: "AbstractElement", identifier: str) -> None:
        """Store limits metadata into ``_json_data["constants"]`` when in json mode."""
        lims = getattr(elem, "limits", (None, None))
        if not lims or (lims[0] is None and lims[1] is None):
            return
        entry = self._json_data["constants"].get(identifier)
        if entry is not None:
            entry["limits"] = [
                None if lims[0] is None else float(lims[0]),
                None if lims[1] is None else float(lims[1]),
            ]

    # ------------------------------------------------------------------
    # Element processing
    # ------------------------------------------------------------------

    def _process_element(
        self,
        elem: AbstractElement,
        identifier: str,
        is_control: bool,
    ) -> List[str]:
        """Return the equation string(s) for *elem*.

        Variable/parameter declarations and initial conditions are registered
        as side-effects on ``self``.
        """
        if not elem.components:
            return []

        # ---- Documentation comment ----------------------------------------
        if not is_control:
            parts = []
            if elem.units and elem.units.strip():
                parts.append(f"units: {elem.units.strip()}")
            if elem.documentation and elem.documentation.strip():
                doc = elem.documentation.strip().replace("\n", " ")
                parts.append(doc)
            if parts:
                self._var_comments[identifier] = " | ".join(parts)

        # ---- EXCEPT subscript exclusion / per-element multi-component ----
        # Delegate when:
        # (a) at least one component has an :EXCEPT: clause, OR
        # (b) multiple components each cover a specific element (not a full range)
        #     of the same subscript dimension — this is the Vensim pattern for
        #     piecewise-defined auxiliaries (e.g. hist_share[elec]=0, [heat]=0,
        #     [liquids]=f(...)).
        if len(elem.components) > 1:
            elem = self._lift_mixed_constants(elem, identifier)
            _has_except = any(comp.subscripts[1] for comp in elem.components)
            # Only apply per-element detection to plain auxiliary/constant
            # components — skip when the element uses external structures
            # (GET LOOKUPS, GET DATA, GET CONSTANTS) which have their own
            # dedicated handlers.
            _is_external = any(
                isinstance(c.ast, (GetLookupsStructure, GetDataStructure, GetConstantsStructure))
                for c in elem.components
            )
            _has_per_elem = not _is_external and any(
                sub not in self._subs_elems and sub in self._elem_to_range
                for c in elem.components if c.subscripts
                for sub in c.subscripts[0]
            )
            # Multi-component inline lookup tables (e.g. lookup1dim[A](...) ~~|
            # lookup1dim[B](...)) are all AbstractLookup + LookupsStructure.
            # Route them to a dedicated handler rather than the EXCEPT path.
            _all_inline_lookups = all(
                isinstance(c, AbstractLookup) and isinstance(c.ast, LookupsStructure)
                for c in elem.components
            )
            if _all_inline_lookups:
                return self._process_subscripted_inline_lookup(elem, identifier)

            # Components over different subranges (x[Disp, Mat] = 0;
            # x[Var, Mat] = ...): each covers its own elements
            _subs_lists = [tuple(c.subscripts[0]) for c in elem.components if c.subscripts]
            _has_subrange_split = not _is_external and len(set(_subs_lists)) > 1

            if _has_except or _has_per_elem or _has_subrange_split:
                return self._process_except_element(elem, identifier, is_control)

        comp = elem.components[0]
        ast = comp.ast

        # Determine subscript dimensionality for this element
        dims = self._element_dims(elem)
        ndim = len(dims)

        # Register var dims for use by visitors in 2D contexts
        if ndim > 0 and not is_control:
            self._var_dims[identifier] = [d for d, _ in dims]

        # Scalar visitor (no active subscript context)
        visitor = JuliaASTVisitor(
            self.namespace, self.inline_registry, self.needed_helpers,
            var_dims=self._var_dims, subs_sizes=self._subs_sizes,
            subs_elems=self._subs_elems, lookup_names=self._lookup_func_names,
            root=self.root, macro_names=self._known_macro_names,
        )

        # ---- Named lookup table ----------------------------------------
        if isinstance(comp, AbstractLookup) and isinstance(ast, LookupsStructure):
            const_decl, func_decl, reg_decl = lookup_interpolation_code(
                identifier, ast.x, ast.y, ast.type
            )
            self.lookup_const_decls.append(const_decl)
            self.lookup_func_decls.append(func_decl)
            self.lookup_register_decls.append(reg_decl)
            return []

        # ---- INITIAL() — freeze inner expression at t=0 ----------------
        # Vensim's INITIAL(x) returns the value of x at t=0.  Two strategies:
        #
        # (a) If the inner value can be resolved at translation time → @parameters.
        # (b) Otherwise → implement as a zero-derivative stock so MTK evaluates
        #     the initial condition at t0 and holds it constant:
        #       D(initial_var) ~ 0.0 ; initial_var(t0) = inner_expr
        if isinstance(ast, InitialStructure):
            if self.backend == "ode" and not isinstance(ast.initial, (int, float)):
                # evaluated by initial_state from the parameters, so that it
                # follows the constants given to params()
                val = None
            else:
                val = self._resolve_initial_value(ast.initial)
            if val is not None:
                if not is_control:
                    self.param_decls.append(f"@parameters {identifier} = {val}")
                return []
            # Fall back: frozen stock — D = 0, initial value = inner expression.
            return self._expand_initial_frozen_stock(
                identifier, ast.initial, dims, ndim
            )

        # ---- Stock (INTEG) ---------------------------------------------
        if isinstance(ast, IntegStructure):
            flow_expr = visitor.visit(ast.flow)
            initial_expr = visitor.visit(ast.initial)
            if ndim == 0:
                self.stock_decls.append(f"@variables {identifier}(t)")
                self.u0_entries.append(f"{identifier} => {initial_expr}")
                flow_expr = _non_negative_flow(ast, flow_expr, identifier)
                return [f"D({identifier}) ~ {flow_expr}"]
            elif ndim == 1:
                (d0, n0) = dims[0]
                vnd1 = self._nd_visitor(dims, ["_i0"])
                flow_nd1 = vnd1.visit(ast.flow)
                self.stock_decls.append(
                    f"@variables {identifier}(t)[{self._range_str(dims)}]"
                )
                # Numpy 1D literal: use scalar per element to avoid assigning
                # a full vector to each scalar u0 entry.
                try:
                    import numpy as _np
                    if isinstance(ast.initial, _np.ndarray) and ast.initial.ndim == 1 and len(ast.initial) == n0:
                        for i, v in enumerate(ast.initial, 1):
                            self.u0_entries.append(
                                f"{identifier}[{i}] => {format_number(float(v))}"
                            )
                    else:
                        raise TypeError
                except (ImportError, TypeError, ValueError):
                    init_nd1 = vnd1.visit(ast.initial)
                    for i in range(1, n0 + 1):
                        self.u0_entries.append(
                            f"{identifier}[{i}] => {init_nd1.replace('_i0', str(i))}"
                        )
                flow_nd1 = _non_negative_flow(ast, flow_nd1, f"{identifier}[_i0]")
                return [
                    f"[D({identifier}[_i0]) ~ {flow_nd1} "
                    f"for _i0 in 1:{self._jl_n(d0)}]..."
                ]
            else:
                # N≥2 dims: comprehension with N index variables
                idx_vars = self._idx_vars(ndim)
                vnd = self._nd_visitor(dims, idx_vars)
                flow_nd = vnd.visit(ast.flow)
                idx_str = ", ".join(idx_vars)
                self.stock_decls.append(
                    f"@variables {identifier}(t)[{self._range_str(dims)}]"
                )
                # Numpy ndarray initial: generate per-element scalar u0 entries.
                # Variable-reference initial: use nd_visitor (which inserts index
                # variables into subscripted references) then substitute each
                # index variable with its concrete value — same strategy as
                # ndim==1.  This prevents assigning a full N-D array to each
                # scalar u0 entry (causes MTK "Cannot equate arrays of different
                # sizes" error).
                try:
                    import numpy as _np
                    expected_shape = tuple(n for _, n in dims)
                    if isinstance(ast.initial, _np.ndarray) and ast.initial.shape == expected_shape:
                        for idx in _np.ndindex(*expected_shape):
                            idx_s = ", ".join(str(i + 1) for i in idx)
                            val = format_number(float(ast.initial[idx]))
                            self.u0_entries.append(f"{identifier}[{idx_s}] => {val}")
                    else:
                        raise TypeError
                except (ImportError, TypeError, ValueError):
                    init_nd = vnd.visit(ast.initial)
                    ranges_nd = [range(1, size + 1) for _, size in dims]
                    for idx_combo in itertools.product(*ranges_nd):
                        idx_s = ", ".join(str(i) for i in idx_combo)
                        init_val = init_nd
                        for var, val in zip(idx_vars, idx_combo):
                            init_val = init_val.replace(var, str(val))
                        self.u0_entries.append(f"{identifier}[{idx_s}] => {init_val}")
                flow_nd = _non_negative_flow(ast, flow_nd, f"{identifier}[{idx_str}]")
                return [
                    f"[D({identifier}[{idx_str}]) ~ {flow_nd} "
                    f"for {self._for_clause(dims, idx_vars)}]..."
                ]

        # ---- First-order Smooth ----------------------------------------
        if isinstance(ast, SmoothStructure) and ast.order == 1:
            return self._expand_smooth(identifier, ast, visitor, order=1, dims=dims)

        # ---- Higher-order Smooth / SmoothN -----------------------------
        if isinstance(ast, (SmoothStructure, SmoothNStructure)):
            try:
                order = int(ast.order)
            except (TypeError, ValueError):
                order = None
                if isinstance(ast, SmoothNStructure):
                    t0_val = self._eval_fixing(ast.order, f"the order of SMOOTH N in {identifier}")
                    if t0_val is not None:
                        order = max(1, round(t0_val))
                if order is None:
                    warn(
                        f"SMOOTH with non-integer order for '{elem.name}'; defaulting to 3."
                    )
                    order = 3
            return self._expand_smooth(identifier, ast, visitor, order=order, dims=dims)

        # ---- Delay (integer order) -------------------------------------
        if isinstance(ast, (DelayStructure, DelayNStructure)):
            try:
                order = int(ast.order)
            except (TypeError, ValueError):
                order = None
                if isinstance(ast, DelayNStructure):
                    # the order is read once, at the initial time, and truncated
                    t0_val = self._eval_fixing(ast.order, f"the order of DELAY N in {identifier}")
                    if t0_val is not None:
                        order = max(1, int(t0_val))
                if order is None:
                    warn(
                        f"DELAY with non-integer order for '{elem.name}'; defaulting to 3."
                    )
                    order = 3
            # As Python PySD: lower the order while order * time_step exceeds
            # the delay time (at the initial time)
            delay0 = self._eval_ast_at_t0(ast.delay_time)
            ts0 = self._eval_ast_at_t0(ReferenceStructure("time_step"))
            if delay0 is not None and ts0 and order > 1 and order * ts0 > delay0:
                # the order depends on the delay time
                self._eval_fixing(ast.delay_time, f"the order of the delay in {identifier}")
            if delay0 is not None and ts0:
                while order > 1 and order * ts0 > delay0:
                    order -= 1
            return self._expand_delay(identifier, ast, visitor, order=order, dims=dims)

        # ---- DELAY FIXED ------------------------------------------------
        # N-stage pipeline, N from the delay time at the initial time (first-order
        # approximation only if that cannot be evaluated); see _expand_delay_fixed.
        if isinstance(ast, DelayFixedStructure):
            return self._expand_delay_fixed(identifier, ast, visitor, dims=dims)

        # ---- External constant (GET XLS/DIRECT CONSTANTS) ----------------
        # Also catches piecewise-constant elements where some components are
        # GCS and others are plain numeric literals (e.g. var[fuel1]=GCS,
        # var[electricity]=0, var[heat]=0).
        # Require at least one GCS so pure-literal or stock elements are not
        # accidentally routed here.
        _const_like = any(
            isinstance(c.ast, GetConstantsStructure) for c in elem.components
        ) and all(
            isinstance(c.ast, GetConstantsStructure) or isinstance(c.ast, (int, float))
            for c in elem.components
        )
        if _const_like:
            julia_val = self._read_get_constants(elem, identifier)
            if julia_val is not None:
                if is_control:
                    if identifier in self.control_vals:
                        self.control_vals[identifier] = julia_val
                    return []
                if self.data_format == "json":
                    self._json_accumulate_constant(elem, identifier, julia_val)
                has_subs = any(
                    bool(self._comp_coords(c)) for c in elem.components
                )
                if (has_subs
                    or julia_val.startswith("[")
                    or julia_val.startswith("reshape(")
                    or julia_val.startswith("vcat(")
                    or "pysd_xlsx_read_constant" in julia_val and "[" in julia_val):
                    self.ext_const_decls.append(f"const {identifier} = {julia_val}")
                else:
                    self.param_decls.append(f"@parameters {identifier} = {julia_val}")
                return []
            # fall through to unsupported handler if reading failed

        # ---- GET XLS/DIRECT LOOKUPS -------------------------------------
        if all(isinstance(c.ast, GetLookupsStructure) for c in elem.components):
            return self._process_get_lookups(elem, identifier)

        # ---- GET XLS/DIRECT DATA ----------------------------------------
        # Guard: only route to GET DATA when at least one component actually
        # carries a GetDataStructure (avoids misrouting variables that are typed
        # as AbstractData but whose equation is a plain CallStructure).
        _has_get_data_ast = any(
            isinstance(c.ast, GetDataStructure) for c in elem.components
        )
        if (isinstance(comp, AbstractData) and not _has_get_data_ast
                and not isinstance(ast, DataStructure)):
            warn(
                f"'{elem.name}' is a DATA variable but its equation is not "
                "GET DATA — data-override mechanism not supported in the Julia "
                "builder; emitting as a regular auxiliary."
            )
        if (isinstance(ast, GetDataStructure) or isinstance(comp, AbstractData)) and _has_get_data_ast:
            return self._process_get_data(elem, identifier, comp)

        # ---- TREND ------------------------------------------------------
        if isinstance(ast, TrendStructure):
            return self._expand_trend(identifier, ast, visitor, dims=dims)

        # ---- FORECAST ---------------------------------------------------
        if isinstance(ast, ForecastStructure):
            return self._expand_forecast(identifier, ast, visitor, dims=dims)

        # ---- SAMPLE IF TRUE ---------------------------------------------
        if isinstance(ast, SampleIfTrueStructure):
            return self._expand_sample_if_true(identifier, ast, visitor, dims=dims, ndim=ndim)

        # ---- ALLOCATE AVAILABLE / ALLOCATE BY PRIORITY ------------------
        if isinstance(ast, (AllocateAvailableStructure, AllocateByPriorityStructure)):
            return self._expand_allocate(identifier, ast, visitor, dims=dims)

        # ---- DataStructure (tab-file DATA variable) ----------------------
        # AbstractData + DataStructure = DATA variable reading from a .tab file
        if isinstance(ast, DataStructure):
            if isinstance(comp, AbstractData):
                return self._process_tab_data_structure(elem, identifier, comp)
            # Non-AbstractData with DataStructure AST: fall through to unsupported
            warn(
                f"'DataStructure' for '{elem.name}' is not supported in the "
                "Julia builder — emitting placeholder equation."
            )
            self.aux_decls.append(f"@variables {identifier}(t)")
            return [f"# UNSUPPORTED(DataStructure): {identifier} ~ 0.0"]

        # ---- Remaining unsupported structures ---------------------------
        if isinstance(ast, _UNSUPPORTED_STRUCTURES):  # pragma: no cover  # empty tuple, always False
            warn(
                f"'{type(ast).__name__}' for '{elem.name}' is not supported in the "
                "Julia builder — emitting placeholder equation."
            )
            self.aux_decls.append(f"@variables {identifier}(t)")
            return [f"# UNSUPPORTED({type(ast).__name__}): {identifier} ~ 0.0"]

        # ---- Constant / unchangeable constant --------------------------
        if isinstance(comp, AbstractUnchangeableConstant) or comp.type == "Constant":
            value_expr = visitor.visit(ast)
            if is_control:
                if identifier in self.control_vals:
                    self.control_vals[identifier] = value_expr
                return []
            lim_comment = self._limits_comment(elem)
            if identifier in self._var_comments:
                self.param_decls.append(f"# {self._var_comments[identifier]}")
            if ndim == 0:
                self.param_decls.append(
                    f"@parameters {identifier} = {value_expr}{lim_comment}"
                )
                if self.data_format == "json":
                    try:
                        self._json_data["constants"][identifier] = {
                            "dims": [], "coords": {},
                            "values": float(value_expr),
                            "units": elem.units or "",
                        }
                        self._json_add_limits(elem, identifier)
                    except (ValueError, TypeError):
                        pass
            else:
                import numpy as _np
                if isinstance(ast, _np.ndarray) and ndim >= 2:
                    # The values as a Julia array of the variable's shape, from
                    # Vensim's row-major order (Julia arrays are column-major)
                    flat = ", ".join(format_number(float(v)) for v in ast.flatten())
                    sizes = [self._jl_n(d) for d, _ in dims]
                    perm = ", ".join(str(k) for k in range(ndim, 0, -1))
                    value_expr = (f"permutedims(reshape(Float64[{flat}], "
                                  f"{', '.join(reversed(sizes))}), ({perm}))")
                self.param_decls.append(
                    f"@parameters {identifier}[{self._range_str(dims)}] = {value_expr}{lim_comment}"
                )
            return []

        # ---- Auxiliary variable (algebraic) ----------------------------
        if ndim == 0:
            rhs_expr = visitor.visit(ast)
            self._drain_embedded_delays(visitor)
            if is_control:
                if identifier in self.control_vals:
                    self.control_vals[identifier] = rhs_expr
                return []
            lim_comment = self._limits_comment(elem)
            self.aux_decls.append(f"@variables {identifier}(t)")
            return [f"{identifier} ~ {rhs_expr}{lim_comment}"]
        elif ndim == 1:
            (d0, n0) = dims[0]
            # Special case: literal numpy array RHS.  A comprehension would put
            # the full N-element vector on each scalar LHS, causing an MTK shape
            # mismatch error ("Cannot add arguments of different sizes").
            # Generate individual per-element equations instead.
            try:
                import numpy as _np
                if isinstance(ast, _np.ndarray) and ast.ndim == 1 and len(ast) == n0:
                    if is_control:
                        return []
                    self.aux_decls.append(
                        f"@variables {identifier}(t)[{self._range_str(dims)}]"
                    )
                    return [
                        f"{identifier}[{i + 1}] ~ {format_number(float(ast[i]))}"
                        for i in range(n0)
                    ]
            except (ImportError, TypeError, ValueError):  # pragma: no cover  # requires numpy import failure
                pass
            vnd1 = self._nd_visitor(dims, ["_i0"])
            rhs_nd1 = vnd1.visit(ast)
            self._drain_embedded_delays(vnd1)
            if is_control:
                if identifier in self.control_vals:
                    self.control_vals[identifier] = rhs_nd1
                return []
            self.aux_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            return [
                f"[{identifier}[_i0] ~ {rhs_nd1} "
                f"for _i0 in 1:{self._jl_n(d0)}]..."
            ]
        else:
            # Special case: INVERT_MATRIX → matrix-level Symbolics.scalarize equations
            if self._is_invert_matrix(ast):
                return self._build_invert_matrix_equations(
                    identifier, ast, dims, is_control
                )
            # Special case: literal numpy array RHS for N≥2 dim auxiliary.
            # Like the 1D case, a comprehension would put the full array on each
            # scalar LHS.  Generate per-element equations instead.
            try:
                import numpy as _np
                expected_shape = tuple(n for _, n in dims)
                if (isinstance(ast, _np.ndarray) and ast.shape != expected_shape
                        and ast.size == int(_np.prod(expected_shape))):
                    # e.g. TABBED ARRAY gives the values as one row-major list
                    ast = ast.reshape(expected_shape)
                if isinstance(ast, _np.ndarray) and ast.shape == expected_shape:
                    if is_control:
                        return []
                    self.aux_decls.append(
                        f"@variables {identifier}(t)[{self._range_str(dims)}]"
                    )
                    # Iterate over all multi-index combinations (Fortran column-major
                    # order is NOT assumed — we iterate in C order but Julia indices
                    # are 1-based).
                    eqs = []
                    for idx in _np.ndindex(*expected_shape):
                        julia_idx = ", ".join(str(i + 1) for i in idx)
                        val = format_number(float(ast[idx]))
                        eqs.append(f"{identifier}[{julia_idx}] ~ {val}")
                    return eqs
            except (ImportError, TypeError, ValueError):  # pragma: no cover  # requires numpy import failure
                pass
            # N≥2 dims: comprehension with N index variables
            idx_vars = self._idx_vars(ndim)
            vnd = self._nd_visitor(dims, idx_vars)
            rhs_nd = vnd.visit(ast)
            self._drain_embedded_delays(vnd)
            if is_control:
                if identifier in self.control_vals:
                    self.control_vals[identifier] = rhs_nd
                return []
            idx_str = ", ".join(idx_vars)
            self.aux_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            return [
                f"[{identifier}[{idx_str}] ~ {rhs_nd} "
                f"for {self._for_clause(dims, idx_vars)}]..."
            ]

    @staticmethod
    def _is_invert_matrix(ast) -> bool:
        """Return True if *ast* is a Vensim INVERT_MATRIX call."""
        return (
            isinstance(ast, CallStructure)
            and isinstance(ast.function, ReferenceStructure)
            and ast.function.reference.lower().replace(" ", "_") == "invert_matrix"
        )

    def _build_invert_matrix_equations(
        self,
        identifier: str,
        ast: "CallStructure",
        dims: List[Tuple[str, int]],
        is_control: bool,
    ) -> List[str]:
        """Emit element-wise INVERT_MATRIX equations via registered helper functions.

        Symbolics symbolic arrays do not support colon (:) slice indexing and
        calling inv(Matrix{Num}) triggers a full symbolic LU decomposition which
        hangs for matrices larger than ~4x4.  Instead we emit equations that use
        @register_symbolic black-box helpers (_inv_mat2d_elem / _inv_mat3d_elem)
        that are evaluated numerically at solve time.

        For 2-D LHS (no batch dims):
            [result[i,j] ~ _inv_mat2d_elem(mat, i, j) for i in 1:N0, j in 1:N1]...
        For 3-D+ LHS (first N-2 dims are batch):
            [result[b,i,j] ~ _inv_mat3d_elem(mat, b, i, j)
             for b in 1:N0, i in 1:N1, j in 1:N2]...
        """
        if is_control:
            return []

        mat_ref = ast.arguments[0]
        mat_name = self.namespace.get(mat_ref.reference) or re.sub(
            r"[^a-z0-9_]", "_", mat_ref.reference.lower()
        )

        self.aux_decls.append(
            f"@variables {identifier}(t)[{self._range_str(dims)}]"
        )

        mat_dims = dims[-2:]
        (d1, _), (d2, _) = mat_dims
        n1 = self._jl_n(d1)
        n2 = self._jl_n(d2)

        ndim = len(dims)
        if ndim == 2:
            self.needed_helpers.add("pysd_inv_mat2d_elem")
            return [
                f"[{identifier}[_i1, _i2] ~ pysd_inv_mat2d_elem({mat_name}, _i1, _i2) "
                f"for _i1 in 1:{n1}, _i2 in 1:{n2}]..."
            ]

        # ndim >= 3: first N-2 dims are batch dims.
        self.needed_helpers.add("pysd_inv_mat3d_elem")
        batch_dims = dims[:-2]
        batch_idx_vars = [f"_ib{k}" for k in range(len(batch_dims))]
        batch_idx = ", ".join(batch_idx_vars)
        all_idx = ", ".join(batch_idx_vars + ["_i1", "_i2"])
        batch_for = self._for_clause(batch_dims, batch_idx_vars)
        return [
            f"[{identifier}[{all_idx}] ~ pysd_inv_mat3d_elem({mat_name}, {batch_idx}, _i1, _i2) "
            f"for {batch_for}, _i1 in 1:{n1}, _i2 in 1:{n2}]..."
        ]

    # ------------------------------------------------------------------
    # EXCEPT subscript exclusion
    # ------------------------------------------------------------------

    def _lift_mixed_constants(
        self, elem: AbstractElement, identifier: str
    ) -> AbstractElement:
        """A variable defined in parts where some parts are GET CONSTANTS and
        others are equations (``x[a1] = GET DIRECT CONSTANTS(...)``,
        ``x[a2] = y * 2``): each constant
        part is read into its own module constant and referenced from that
        part, so the variable is built like any variable defined in parts."""
        comps = elem.components
        if not any(isinstance(c.ast, GetConstantsStructure) for c in comps) or all(
            isinstance(c.ast, (GetConstantsStructure, int, float)) for c in comps
        ):
            return elem
        lifted = []
        for k, comp in enumerate(comps):
            if isinstance(comp.ast, GetConstantsStructure):
                name = f"_{identifier}_constant_{k}"
                part = AbstractElement(name=name, components=[comp])
                value = self._read_get_constants(part, name)
                self.namespace.namespace[name] = name
                self.ext_const_decls.append(f"const {name} = {value}")
                comp = dataclasses.replace(comp, ast=ReferenceStructure(name))
            lifted.append(comp)
        return dataclasses.replace(elem, components=lifted)

    def _process_except_element(
        self,
        elem: AbstractElement,
        identifier: str,
        is_control: bool,
    ) -> List[str]:
        """Handle multi-component elements that use ``:EXCEPT:`` subscript exclusion.

        For each component we determine which integer indices it covers (the
        component's defined range minus the EXCEPT-excluded elements) and
        emit one equation per covered index.  The variable declaration
        (``@variables`` or ``@parameters``) is still emitted once for the
        full range.

        Limitations:
        - Only 1-D subscripted auxiliaries and constants are handled.
        - Components covering more than one dimension are not yet supported
          and fall back to a ``UserWarning`` + plain broadcast equation.
        """
        dims = self._element_dims(elem)
        ndim = len(dims)

        if ndim != 1:
            if ndim == 2:
                return self._process_except_element_2d(
                    elem, identifier, dims, is_control
                )
            if ndim == 3:
                return self._process_except_element_3d(
                    elem, identifier, dims, is_control
                )
            warn(
                f"EXCEPT subscript exclusion for '{elem.name}' with {ndim}D "
                "subscripts is not yet supported — emitting plain broadcast equation."
            )
            # Fallback: use first component, ignore EXCEPT
            comp = elem.components[0]
            visitor = JuliaASTVisitor(
                self.namespace, self.inline_registry, self.needed_helpers,
                subs_sizes=self._subs_sizes, root=self.root,
                macro_names=self._known_macro_names,
            )
            rhs = visitor.visit(comp.ast)
            if not is_control:
                self.aux_decls.append(
                    f"@variables {identifier}(t)[{self._range_str(dims)}]"
                )
            return [f"Symbolics.scalarize({identifier} .~ {rhs})..."]

        dim_name, dim_size = dims[0]
        dim_elems = self._subs_elems.get(dim_name, [])

        # Build a map: element_label → 1-based index
        label_to_idx: Dict[str, int] = {
            label: i + 1 for i, label in enumerate(dim_elems)
        }

        # Pre-scan: detect stock and delay-fixed components so we can choose
        # the right declaration type and pre-allocate internal state arrays.
        has_integ = any(isinstance(c.ast, IntegStructure) for c in elem.components)
        has_delay_fixed = any(
            isinstance(c.ast, DelayFixedStructure) for c in elem.components
        )
        df_name: Optional[str] = None
        if has_delay_fixed:
            df_name = f"_df_{identifier}"
            self.namespace.namespace[f"__internal_df_{identifier}"] = df_name
            self.stock_decls.append(
                f"@variables {df_name}(t)[{self._range_str(dims)}]"
            )

        equations: List[str] = []

        for comp in elem.components:
            # Collect the excluded element labels for this component
            excluded_labels: set = set()
            for except_list in comp.subscripts[1]:
                for label in except_list:
                    excluded_labels.add(label)

            # Determine which indices this component covers.
            # The defining subscript (comp.subscripts[0]) may be:
            #   (a) the full dimension range name  → all elements
            #   (b) a sub-range name               → elements of that sub-range
            #   (c) specific element label(s)      → those elements only
            def_subs = comp.subscripts[0] if comp.subscripts else []
            def_range_name: Optional[str] = None  # non-None only for sub-ranges (b)

            if def_subs and def_subs[0] == dim_name:
                # (a) Full range
                candidate_labels = dim_elems
            elif def_subs and def_subs[0] in self._subs_sizes:
                # (b) A named sub-range — expand to its elements within dim_elems
                def_range_name = def_subs[0]
                range_elems = self._subs_elems.get(def_range_name, [])
                candidate_labels = [e for e in range_elems if e in label_to_idx]
            else:
                # (c) Specific element label(s)
                candidate_labels = [s for s in def_subs if s in label_to_idx]

            covered_indices = [
                label_to_idx[label]
                for label in candidate_labels
                if label not in excluded_labels
            ]

            if comp.type in ("Constant", ) or isinstance(comp, AbstractUnchangeableConstant):
                # Constant component — emit as parameter entries
                visitor = JuliaASTVisitor(
                    self.namespace, self.inline_registry, self.needed_helpers,
                    var_dims=self._var_dims, subs_sizes=self._subs_sizes,
                    subs_elems=self._subs_elems, lookup_names=self._lookup_func_names,
                    root=self.root, macro_names=self._known_macro_names,
                )
                value_expr = visitor.visit(comp.ast)
                for idx in covered_indices:
                    if not is_control:
                        equations.append(f"{identifier}[{idx}] ~ {value_expr}")

            elif isinstance(comp.ast, IntegStructure):
                # Stock component — emit per-index ODE + initial condition.
                for idx in covered_indices:
                    vis_idx = JuliaASTVisitor(
                        self.namespace, self.inline_registry, self.needed_helpers,
                        active_subs=self._per_index_subs(
                            dim_name, dim_elems, idx, def_range_name
                        ),
                        var_dims=self._var_dims, subs_sizes=self._subs_sizes,
                        subs_elems=self._subs_elems, lookup_names=self._lookup_func_names,
                        root=self.root, macro_names=self._known_macro_names,
                    )
                    flow_expr = vis_idx.visit(comp.ast.flow)
                    init_expr = vis_idx.visit(comp.ast.initial)
                    self.u0_entries.append(f"{identifier}[{idx}] => {init_expr}")
                    flow_expr = _non_negative_flow(
                        comp.ast, flow_expr, f"{identifier}[{idx}]")
                    equations.append(f"D({identifier}[{idx}]) ~ {flow_expr}")

            elif isinstance(comp.ast, DelayFixedStructure):
                # DELAY FIXED — approximate as first-order ODE (same as
                # _expand_delay_fixed) but per-index with separate initials.
                for idx in covered_indices:
                    vis_idx = JuliaASTVisitor(
                        self.namespace, self.inline_registry, self.needed_helpers,
                        active_subs=self._per_index_subs(
                            dim_name, dim_elems, idx, def_range_name
                        ),
                        var_dims=self._var_dims, subs_sizes=self._subs_sizes,
                        subs_elems=self._subs_elems, lookup_names=self._lookup_func_names,
                        root=self.root, macro_names=self._known_macro_names,
                    )
                    input_expr = vis_idx.visit(comp.ast.input)
                    delay_expr = vis_idx.visit(comp.ast.delay_time)
                    init_expr = vis_idx.visit(comp.ast.initial)
                    self.u0_entries.append(f"{df_name}[{idx}] => {init_expr}")
                    equations.append(
                        f"D({df_name}[{idx}]) ~ "
                        f"({input_expr} - {df_name}[{idx}]) / {delay_expr}"
                    )
                    equations.append(f"{identifier}[{idx}] ~ {df_name}[{idx}]")

            elif (def_subs and def_subs[0] == dim_name and covered_indices
                  and not isinstance(comp.ast, (int, float))):
                # Auxiliary component over the full range (but the excluded
                # elements): one loop over its elements, as without EXCEPT
                rhs_expr = self._nd_visitor(dims, ["_i0"]).visit(comp.ast)
                index_list = (f"1:{self._jl_n(dim_name)}"
                              if covered_indices == list(range(1, len(dim_elems) + 1))
                              else "[" + ", ".join(str(i) for i in covered_indices) + "]")
                equations.append(
                    f"[{identifier}[_i0] ~ {rhs_expr} for _i0 in {index_list}]...")

            else:
                # Auxiliary component — use a per-index visitor with aligned
                # subscripts so cross-range references resolve correctly.
                for idx in covered_indices:
                    vis_idx = JuliaASTVisitor(
                        self.namespace, self.inline_registry, self.needed_helpers,
                        active_subs=self._per_index_subs(
                            dim_name, dim_elems, idx, def_range_name
                        ),
                        var_dims=self._var_dims, subs_sizes=self._subs_sizes,
                        subs_elems=self._subs_elems, lookup_names=self._lookup_func_names,
                        root=self.root, macro_names=self._known_macro_names,
                    )
                    rhs_expr = vis_idx.visit(comp.ast)
                    equations.append(f"{identifier}[{idx}] ~ {rhs_expr}")

        if not is_control:
            if has_integ:
                self.stock_decls.append(
                    f"@variables {identifier}(t)[{self._range_str(dims)}]"
                )
            else:
                self.aux_decls.append(
                    f"@variables {identifier}(t)[{self._range_str(dims)}]"
                )
        return equations

    def _process_except_element_2d(
        self,
        elem: "AbstractElement",
        identifier: str,
        dims: List[Tuple[str, int]],
        is_control: bool,
    ) -> List[str]:
        """Handle 2-D EXCEPT subscript exclusion.

        For each component, resolve its subscript specification (which may name a
        full subscript range or a specific element) plus any EXCEPT exclusions to a
        concrete set of 1-based (row, col) index pairs, then emit one comprehension
        equation per component covering exactly those pairs.
        """
        dim0_name, _ = dims[0]
        dim1_name, _ = dims[1]
        dim0_elems = self._subs_elems.get(dim0_name, [])
        dim1_elems = self._subs_elems.get(dim1_name, [])

        def _resolve_spec(spec: str, dim_elems: List[str]) -> List[int]:
            """Return 1-based indices in *dim_elems* for *spec*.

            *spec* is either a subscript-range name (all its elements that appear
            in dim_elems are included) or a bare element name (only that element).
            """
            if spec in self._subs_sizes:
                range_elems = set(self._subs_elems.get(spec, []))
                return [i + 1 for i, e in enumerate(dim_elems) if e in range_elems]
            # Bare element name
            return [i + 1 for i, e in enumerate(dim_elems) if e == spec]

        # Pre-scan: detect stock components so we choose the right declaration.
        has_integ_2d = any(isinstance(c.ast, IntegStructure) for c in elem.components)

        equations: List[str] = []

        for comp in elem.components:
            sub0_spec = comp.subscripts[0][0] if comp.subscripts[0] else dim0_name
            sub1_spec = comp.subscripts[0][1] if len(comp.subscripts[0]) > 1 else dim1_name

            covered0 = _resolve_spec(sub0_spec, dim0_elems)
            covered1 = _resolve_spec(sub1_spec, dim1_elems)

            # Build set of excluded (i0, i1) pairs from EXCEPT clauses
            excluded: set = set()
            for exc_clause in comp.subscripts[1]:
                exc0_spec = exc_clause[0] if len(exc_clause) > 0 else None
                exc1_spec = exc_clause[1] if len(exc_clause) > 1 else None
                exc0_idx = _resolve_spec(exc0_spec, dim0_elems) if exc0_spec else list(range(1, len(dim0_elems) + 1))
                exc1_idx = _resolve_spec(exc1_spec, dim1_elems) if exc1_spec else list(range(1, len(dim1_elems) + 1))
                for i in exc0_idx:
                    for j in exc1_idx:
                        excluded.add((i, j))

            # The cells this component defines, grouped into blocks of rows
            # that cover the same columns (EXCEPT may remove single cells)
            row_cols: Dict[int, Tuple[int, ...]] = {}
            for i in covered0:
                cols = tuple(j for j in covered1 if (i, j) not in excluded)
                if cols:
                    row_cols[i] = cols
            blocks: Dict[Tuple[int, ...], List[int]] = {}
            for i, cols in row_cols.items():
                blocks.setdefault(cols, []).append(i)
            if not blocks:
                continue
            # Ranges of this component that stand for the loops over the
            # variable's dimensions (x[dim1up, d3] inside x[dim1, dim2])
            aliases = {
                spec: dim for spec, dim in ((sub0_spec, dim0_name), (sub1_spec, dim1_name))
                if spec != dim and spec in self._subs_sizes
            }

            if isinstance(comp.ast, IntegStructure):
                # Stock component: one u0 entry per cell, and the flows as one
                # comprehension per block of rows that cover the same columns
                vnd = self._nd_visitor(dims, ["_i0", "_i1"])
                vnd.sub_aliases = aliases
                flow_expr = vnd.visit(comp.ast.flow)
                init_expr = vnd.visit(comp.ast.initial)
                flow_expr = _non_negative_flow(
                    comp.ast, flow_expr, f"{identifier}[_i0, _i1]")
                for i0, cols in row_cols.items():
                    for i1 in cols:
                        init_val = re.sub(r"\b_i0\b", str(i0), init_expr)
                        init_val = re.sub(r"\b_i1\b", str(i1), init_val)
                        self.u0_entries.append(f"{identifier}[{i0}, {i1}] => {init_val}")
                for cols, rows in blocks.items():
                    row_str = "[" + ", ".join(str(i) for i in rows) + "]"
                    col_str = "[" + ", ".join(str(j) for j in cols) + "]"
                    equations.append(
                        f"[D({identifier}[_i0, _i1]) ~ {flow_expr} "
                        f"for _i0 in {row_str}, _i1 in {col_str}]..."
                    )
            else:
                # Auxiliary component — one comprehension per block of rows
                # that cover the same columns.
                full_rows = list(range(1, len(dim0_elems) + 1))
                full_cols = list(range(1, len(dim1_elems) + 1))
                if len(blocks) == 1:
                    (cols, rows), = blocks.items()
                    arr_eqs = self._ndarray_component_eqs(
                        identifier, comp.ast, [rows, list(cols)]
                    )
                    if arr_eqs is not None:
                        equations.extend(arr_eqs)
                        continue

                vnd = self._nd_visitor(dims, ["_i0", "_i1"])
                vnd.sub_aliases = aliases
                rhs_expr = vnd.visit(comp.ast)

                for cols, rows in blocks.items():
                    row_str = (
                        f"1:{self._jl_n(dim0_name)}" if rows == full_rows
                        else "[" + ", ".join(str(i) for i in rows) + "]"
                    )
                    col_str = (
                        f"1:{self._jl_n(dim1_name)}" if list(cols) == full_cols
                        else "[" + ", ".join(str(j) for j in cols) + "]"
                    )
                    equations.append(
                        f"[{identifier}[_i0, _i1] ~ {rhs_expr} "
                        f"for _i0 in {row_str}, _i1 in {col_str}]..."
                    )

        if not is_control:
            if has_integ_2d:
                self.stock_decls.append(
                    f"@variables {identifier}(t)[{self._range_str(dims)}]"
                )
            else:
                self.aux_decls.append(
                    f"@variables {identifier}(t)[{self._range_str(dims)}]"
                )
        return equations

    @staticmethod
    def _ndarray_component_eqs(
        identifier: str, value, covered: List[List[int]]
    ) -> Optional[List[str]]:
        """Per-element equations for a component whose value is a numpy array.

        *covered* holds, per dimension, the 1-based indices the component
        defines (e.g. ``[[1, 2, 3], [1, 2], [2]]`` for ``x[A, B, b2]``).  The
        array holds one value per covered element (dimensions of length 1 may
        be dropped), so it is reshaped to the covered shape and walked in the
        same order as the index combinations.  Returns ``None`` when *value*
        is not an array of matching size.
        """
        import numpy as _np
        if not isinstance(value, _np.ndarray):
            return None
        shape = [len(c) for c in covered]
        if value.size != int(_np.prod(shape)):
            return None
        arr = value.astype(float).reshape(shape)
        return [
            f"{identifier}[{', '.join(map(str, idx))}] ~ {format_number(float(arr[pos]))}"
            for pos, idx in zip(_np.ndindex(*shape), itertools.product(*covered))
        ]

    def _process_except_element_3d(
        self,
        elem: "AbstractElement",
        identifier: str,
        dims: List[Tuple[str, int]],
        is_control: bool,
    ) -> List[str]:
        """Handle 3-D EXCEPT subscript exclusion.

        Generalises :meth:`_process_except_element_2d` to three dimensions.
        For each component, resolve the subscript spec + EXCEPT exclusions to
        a concrete set of 1-based (i0, i1, i2) index triples, then emit one
        comprehension equation per component covering exactly those triples.
        """
        dim0_name, _ = dims[0]
        dim1_name, _ = dims[1]
        dim2_name, _ = dims[2]
        dim0_elems = self._subs_elems.get(dim0_name, [])
        dim1_elems = self._subs_elems.get(dim1_name, [])
        dim2_elems = self._subs_elems.get(dim2_name, [])

        def _resolve_spec(spec: str, dim_elems: List[str]) -> List[int]:
            if spec in self._subs_sizes:
                range_elems = set(self._subs_elems.get(spec, []))
                return [i + 1 for i, e in enumerate(dim_elems) if e in range_elems]
            return [i + 1 for i, e in enumerate(dim_elems) if e == spec]

        equations: List[str] = []
        has_integ = False

        for comp in elem.components:
            s0 = comp.subscripts[0][0] if len(comp.subscripts[0]) > 0 else dim0_name
            s1 = comp.subscripts[0][1] if len(comp.subscripts[0]) > 1 else dim1_name
            s2 = comp.subscripts[0][2] if len(comp.subscripts[0]) > 2 else dim2_name

            covered0 = _resolve_spec(s0, dim0_elems)
            covered1 = _resolve_spec(s1, dim1_elems)
            covered2 = _resolve_spec(s2, dim2_elems)

            # Build set of excluded triples from EXCEPT clauses
            excluded: set = set()
            for exc_clause in comp.subscripts[1]:
                ec0 = exc_clause[0] if len(exc_clause) > 0 else None
                ec1 = exc_clause[1] if len(exc_clause) > 1 else None
                ec2 = exc_clause[2] if len(exc_clause) > 2 else None
                exc0 = _resolve_spec(ec0, dim0_elems) if ec0 else list(range(1, len(dim0_elems) + 1))
                exc1 = _resolve_spec(ec1, dim1_elems) if ec1 else list(range(1, len(dim1_elems) + 1))
                exc2 = _resolve_spec(ec2, dim2_elems) if ec2 else list(range(1, len(dim2_elems) + 1))
                for i in exc0:
                    for j in exc1:
                        for k in exc2:
                            excluded.add((i, j, k))

            # The cells this component defines (EXCEPT may remove single
            # cells), as rectangular blocks
            cells = set(itertools.product(covered0, covered1, covered2)) - excluded
            blocks = _cell_blocks(cells)
            if not blocks:
                continue

            if len(blocks) == 1:
                arr_eqs = self._ndarray_component_eqs(
                    identifier, comp.ast, list(blocks[0])
                )
                if arr_eqs is not None:
                    equations.extend(arr_eqs)
                    continue

            vnd = self._nd_visitor(dims, ["_i0", "_i1", "_i2"])
            # Ranges of this component that stand for the loops over the
            # variable's dimensions (x[REGION, SEC A, …] inside x[REGION, SEC ALL, …])
            vnd.sub_aliases = {
                spec: dim for spec, dim in ((s0, dim0_name), (s1, dim1_name), (s2, dim2_name))
                if spec != dim and spec in self._subs_sizes
            }

            def _idx_str(indices: List[int], dim_elems: List[str], dim_name: str) -> str:
                full = list(range(1, len(dim_elems) + 1))
                if indices == full:
                    return f"1:{self._jl_n(dim_name)}"
                return "[" + ", ".join(str(i) for i in indices) + "]"

            def _for_clause(block) -> str:
                return "for " + ", ".join(
                    f"{iv} in {_idx_str(idx, elems, name)}"
                    for iv, idx, elems, name in zip(
                        ("_i0", "_i1", "_i2"), block,
                        (dim0_elems, dim1_elems, dim2_elems),
                        (dim0_name, dim1_name, dim2_name)))

            if isinstance(comp.ast, IntegStructure):
                # Stock component (e.g. Stock A[A, B, Depth 1] = INTEG(...)):
                # one u0 entry per covered element, flows as a comprehension.
                has_integ = True
                flow_expr = vnd.visit(comp.ast.flow)
                init_expr = vnd.visit(comp.ast.initial)
                for combo in sorted(cells):
                    init_val = init_expr
                    for var, val in zip(("_i0", "_i1", "_i2"), combo):
                        init_val = re.sub(rf"\b{var}\b", str(val), init_val)
                    idx_s = ", ".join(str(i) for i in combo)
                    self.u0_entries.append(f"{identifier}[{idx_s}] => {init_val}")
                flow_expr = _non_negative_flow(
                    comp.ast, flow_expr, f"{identifier}[_i0, _i1, _i2]")
                for block in blocks:
                    equations.append(
                        f"[D({identifier}[_i0, _i1, _i2]) ~ {flow_expr} {_for_clause(block)}]..."
                    )
                continue

            rhs_expr = vnd.visit(comp.ast)
            for block in blocks:
                equations.append(
                    f"[{identifier}[_i0, _i1, _i2] ~ {rhs_expr} {_for_clause(block)}]..."
                )

        if not is_control:
            decls = self.stock_decls if has_integ else self.aux_decls
            decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
        return equations

    # ------------------------------------------------------------------
    # Nested structure materialisation
    # ------------------------------------------------------------------

    def _materialize_input(
        self,
        node,
        base_id: str,
        visitor: "JuliaASTVisitor",
        eqs_accumulator: List[str],
        dims: Optional[List[Tuple[str, int]]] = None,
    ) -> str:
        """Return a Julia expression string for *node*, creating an intermediate
        auxiliary variable if *node* is itself a complex structure (DelayStructure,
        SmoothStructure, SmoothNStructure, IntegStructure, ForecastStructure,
        TrendStructure).

        When a complex structure is detected the intermediate variable is expanded
        immediately (its equations are appended to *eqs_accumulator*) and its
        identifier is returned so the caller can use it in an outer expression.
        """
        from pysd.translators.structures.abstract_expressions import (
            DelayStructure as _Delay,
        )

        if not isinstance(node, (_Delay,)):
            return visitor.visit(node)

        # Pick a collision-free name for the intermediate variable
        interm_id = f"_inter_{base_id}"
        counter = 0
        while f"__internal_{interm_id}" in self.namespace.namespace.values():
            counter += 1
            interm_id = f"_inter_{base_id}_{counter}"
        self.namespace.namespace[f"__internal_{interm_id}"] = interm_id

        if isinstance(node, _Delay):
            order = int(node.order) if node.order else 3
            inner_eqs = self._expand_delay(interm_id, node, visitor, order=order, dims=dims)
            eqs_accumulator.extend(inner_eqs)

        return interm_id

    # ------------------------------------------------------------------
    # Smooth expansion
    # ------------------------------------------------------------------

    def _expand_smooth(
        self,
        identifier: str,
        ast,
        visitor: JuliaASTVisitor,
        order: int,
        dims: Optional[List[Tuple[str, int]]] = None,
    ) -> List[str]:
        """Expand a SMOOTH(N) into *order* chained first-order ODE levels.

        The output variable ``identifier`` is declared as an auxiliary equal
        to the final level.  When *dims* is provided the internal levels are
        subscripted arrays and the equations are emitted as comprehensions.
        """
        dims = dims or []
        eqs: List[str] = []

        if dims:
            # Subscripted SMOOTH — each internal level is an array over the
            # variable's dimensions (any number of them).
            idx = self._idx_vars(len(dims))
            vnd = self._nd_visitor(dims, idx)
            input_nd = vnd.visit(ast.input)
            st_nd = vnd.visit(ast.smooth_time)
            init_nd = vnd.visit(ast.initial)
            for_clause = self._for_clause(dims, idx)
            cell = ", ".join(idx)

            prev_nd = input_nd
            for i in range(1, order + 1):
                lv_name = f"_lv{i}_{identifier}"
                self.namespace.namespace[f"__internal_lv{i}_{identifier}"] = lv_name
                self.stock_decls.append(
                    f"@variables {lv_name}(t)[{self._range_str(dims)}]"
                )
                for index, init_i in self._nd_cells(dims, idx, init_nd):
                    self.u0_entries.append(f"{lv_name}[{index}] => {init_i}")
                lv_ref = f"{lv_name}[{cell}]"
                eqs.append(
                    f"[D({lv_ref}) ~ ({prev_nd} - {lv_ref}) / "
                    f"({st_nd} / {order}) for {for_clause}]..."
                )
                prev_nd = lv_ref
            self.aux_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            eqs.append(
                f"[{identifier}[{cell}] ~ {prev_nd} for {for_clause}]..."
            )
        else:
            # Materialise the input: if it's itself a complex structure (e.g.
            # DELAY3 nested inside SMOOTH), create an intermediate variable.
            input_expr = self._materialize_input(ast.input, f"i_{identifier}", visitor, eqs)
            smooth_time_expr = visitor.visit(ast.smooth_time)
            initial_expr = visitor.visit(ast.initial)
            prev_expr = input_expr
            for i in range(1, order + 1):
                lv_name = f"_lv{i}_{identifier}"
                self.namespace.namespace[f"__internal_lv{i}_{identifier}"] = lv_name
                self.stock_decls.append(f"@variables {lv_name}(t)")
                self.u0_entries.append(f"{lv_name} => {initial_expr}")
                eqs.append(
                    f"D({lv_name}) ~ ({prev_expr} - {lv_name}) / "
                    f"({smooth_time_expr} / {order})"
                )
                prev_expr = lv_name
            self.aux_decls.append(f"@variables {identifier}(t)")
            eqs.append(f"{identifier} ~ {prev_expr}")

        return eqs

    # ------------------------------------------------------------------
    # Delay expansion
    # ------------------------------------------------------------------

    def _expand_delay(
        self,
        identifier: str,
        ast,
        visitor: JuliaASTVisitor,
        order: int,
        dims: Optional[List[Tuple[str, int]]] = None,
    ) -> List[str]:
        """Expand a DELAY(N) into *order* chained first-order pipeline levels.

        Each level ``L_i`` satisfies::

            dL_i/dt = (inflow_i - L_i * rate)
            rate    = order / delay_time
            inflow_1 = input;  inflow_i = L_{i-1} * rate  for i > 1
        """
        dims = dims or []
        eqs: List[str] = []

        if dims:
            ndim = len(dims)
            idx_vars = self._idx_vars(ndim)
            vnd = self._nd_visitor(dims, idx_vars)
            input_nd = vnd.visit(ast.input)
            delay_time_nd = vnd.visit(ast.delay_time)
            initial_nd = vnd.visit(ast.initial)
            idx_str_t = ", ".join(idx_vars)
            for_clause = self._for_clause(dims, idx_vars)
            ranges_list = [range(1, size + 1) for _, size in dims]

            stage_times, out_time = self._delay_stage_times(
                identifier, ast, delay_time_nd, delay_time_nd, order, eqs,
                dims=dims, idx=idx_str_t, for_clause=for_clause,
                ranges=ranges_list, idx_vars=idx_vars)
            prev_nd = input_nd
            for stage in range(1, order + 1):
                rate_nd = f"({order} / ({stage_times[stage - 1]}))"
                lv_name = f"_dl{stage}_{identifier}"
                self.namespace.namespace[f"__internal_dl{stage}_{identifier}"] = lv_name
                self.stock_decls.append(
                    f"@variables {lv_name}(t)[{self._range_str(dims)}]"
                )
                for idx_combo in itertools.product(*ranges_list):
                    expr_i = initial_nd
                    dt_i = delay_time_nd
                    for iv, idx in zip(idx_vars, idx_combo):
                        expr_i = expr_i.replace(iv, str(idx))
                        dt_i = dt_i.replace(iv, str(idx))
                    idx_s = ", ".join(str(v) for v in idx_combo)
                    self.u0_entries.append(
                        f"{lv_name}[{idx_s}] => {expr_i} * ({dt_i}) / {order}"
                    )
                lv_ref = f"{lv_name}[{idx_str_t}]"
                eqs.append(
                    f"[D({lv_ref}) ~ ({prev_nd} - {lv_ref} * {rate_nd}) "
                    f"for {for_clause}]..."
                )
                prev_nd = f"{lv_ref} .* {rate_nd}"
            self.aux_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            out_nd = f"_dl{order}_{identifier}[{idx_str_t}] .* ({order} / ({out_time}))"
            eqs.append(f"[{identifier}[{idx_str_t}] ~ {out_nd} for {for_clause}]...")
            return eqs

        input_expr = visitor.visit(ast.input)
        delay_time_expr = visitor.visit(ast.delay_time)
        initial_expr = visitor.visit(ast.initial)
        stage_times, out_time = self._delay_stage_times(
            identifier, ast, delay_time_expr, delay_time_expr, order, eqs)
        prev_outflow = input_expr
        for i in range(1, order + 1):
            lv_name = f"_dl{i}_{identifier}"
            self.namespace.namespace[f"__internal_dl{i}_{identifier}"] = lv_name
            self.stock_decls.append(f"@variables {lv_name}(t)")
            self.u0_entries.append(
                f"{lv_name} => {initial_expr} * {delay_time_expr} / {order}"
            )
            rate_i = f"({order} / {stage_times[i - 1]})"
            eqs.append(
                f"D({lv_name}) ~ ({prev_outflow} - {lv_name} * {rate_i})"
            )
            prev_outflow = f"{lv_name} * {rate_i}"

        self.aux_decls.append(f"@variables {identifier}(t)")
        eqs.append(f"{identifier} ~ _dl{order}_{identifier} * ({order} / {out_time})")
        return eqs

    def _delay_stage_times(self, identifier, ast, delay_time_expr, u0_delay_expr,
                           order, eqs, dims=None, idx=None, for_clause=None,
                           ranges=None, idx_vars=None):
        """Delay times of a DELAY N: one per stage, and one for the output.

        As Python PySD (DelayN), when the delay time changes stage k uses the
        delay time of k - 1 time steps ago, and the output the one of the
        previous time step; for a constant delay time, or DELAY1/DELAY3, all
        use the current one.  The past values are kept in a pipeline of states
        (exact with Euler, as DELAY FIXED), whose equations are appended to
        *eqs*.
        """
        constant = self._try_eval_as_float(delay_time_expr) is not None
        if not isinstance(ast, DelayNStructure) or constant:
            return [delay_time_expr] * order, delay_time_expr
        ts_expr = self.control_vals.get("time_step") or "time_step"
        times = [delay_time_expr]
        prev = delay_time_expr
        for k in range(1, max(order, 2)):
            name = f"_dlt{k}_{identifier}"
            self.namespace.namespace[f"__internal_dlt{k}_{identifier}"] = name
            if dims:
                self.stock_decls.append(f"@variables {name}(t)[{self._range_str(dims)}]")
                for combo in itertools.product(*ranges):
                    expr_i = u0_delay_expr
                    for iv, val in zip(idx_vars, combo):
                        expr_i = expr_i.replace(iv, str(val))
                    idx_s = ", ".join(str(v) for v in combo)
                    self.u0_entries.append(f"{name}[{idx_s}] => {expr_i}")
                ref = f"{name}[{idx}]"
                eqs.append(f"[D({ref}) ~ ({prev} - {ref}) / ({ts_expr}) for {for_clause}]...")
            else:
                self.stock_decls.append(f"@variables {name}(t)")
                self.u0_entries.append(f"{name} => {u0_delay_expr}")
                ref = name
                eqs.append(f"D({ref}) ~ ({prev} - {ref}) / ({ts_expr})")
            times.append(ref)
            prev = ref
        return times[:order], times[1]

    # ------------------------------------------------------------------
    # DELAY FIXED expansion
    # ------------------------------------------------------------------

    def _eval_fixing(self, node, reason: str) -> Optional[float]:
        """``_eval_ast_at_t0``, recording the constants it reads as fixed by
        the translation (for *reason*): ``params()`` refuses new values for
        them, which would need the model translated again."""
        saved, self._fix_reason = self._fix_reason, reason
        try:
            return self._eval_ast_at_t0(node)
        finally:
            self._fix_reason = saved

    def _eval_ast_at_t0(self, node) -> Optional[float]:
        """Evaluate an abstract-expression AST node at t=0.

        Used to resolve the initial order of SMOOTH N / DELAY N when the order
        is a time-varying expression (e.g. ``2 + STEP(1, 10)``).  Returns None
        if the value cannot be determined.

        Handles: literals, ArithmeticStructure (+−×÷), ReferenceStructure
        resolved via prescanned constants or built_elements, and common
        zero-at-t0 calls (STEP, RAMP, PULSE).
        """
        if isinstance(node, (int, float)):
            return float(node)
        if isinstance(node, ArithmeticStructure):
            args = [self._eval_ast_at_t0(a) for a in node.arguments]
            if any(v is None for v in args):
                return None
            ops = node.operators
            result = args[0]
            for op, val in zip(ops, args[1:]):
                if op == "+":
                    result += val
                elif op in ("-", "−"):
                    result -= val
                elif op in ("*", "×"):
                    result *= val
                elif op in ("/", "÷"):
                    result = result / val if val != 0 else None
                else:
                    return None
                if result is None:
                    return None
            return result
        if isinstance(node, CallStructure):
            func_name = ""
            if isinstance(node.function, ReferenceStructure):
                func_name = node.function.reference.lower()
            # Functions that are zero at t=0
            if func_name in ("step", "ramp", "pulse", "pulse train"):
                return 0.0
            if func_name in _T0_MATH:
                args = [self._eval_ast_at_t0(a) for a in node.arguments]
                if any(v is None for v in args):
                    return None
                try:
                    return float(_T0_MATH[func_name](*args))
                except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                    return None
            return None
        if isinstance(node, ReferenceStructure):
            if node.reference.lower() == "time":
                return self._eval_ast_at_t0(ReferenceStructure("initial_time"))
            julia_id = self.namespace.get(node.reference)
            if julia_id is not None and self._fix_reason:
                self._fixed_inputs.setdefault(julia_id, self._fix_reason)
            if julia_id is not None:
                val = self._try_eval_as_float(julia_id)
                if val is not None:
                    return val
            # Try to find the referenced element in the abstract section and
            # recursively evaluate its AST at t=0 (handles variables like
            # "Order Variable = 2 + STEP(1, 10)" → returns 2.0).
            # Normalize both names: lowercase, spaces→underscores.
            ref_norm = node.reference.lower().replace(" ", "_").strip()
            for elem in self.abstract_elements:
                elem_norm = elem.name.lower().replace(" ", "_").strip()
                if elem_norm == ref_norm:
                    for comp in elem.components:
                        if not isinstance(comp.ast, (int, float, ArithmeticStructure,
                                                     CallStructure, ReferenceStructure)):
                            break
                        val = self._eval_ast_at_t0(comp.ast)
                        if val is not None:
                            return val
            return None
        return None

    def _try_eval_as_float(self, expr: str) -> Optional[float]:
        """Try to evaluate a Julia expression string as a constant float.

        Checks (in order):
        1. Direct float literal
        2. Pre-scanned scalar constant values (from _prescanned_const_vals)
        3. @parameters / const declaration
        4. A simple algebraic equation ``name ~ number`` in built_elements
        Returns the float value or None if the expression is not resolvable.
        """
        expr = expr.strip()
        try:
            return float(expr)
        except ValueError:
            pass
        if self._fix_reason and re.fullmatch(r"[a-z_]\w*", expr):
            self._fixed_inputs.setdefault(expr, self._fix_reason)
        # Check pre-scanned constants (covers forward references — constants not
        # yet processed at the time this DELAY FIXED element is being built).
        if hasattr(self, "_prescanned_const_vals") and expr in self._prescanned_const_vals:
            return self._prescanned_const_vals[expr]
        for decl in self.param_decls + self.ext_const_decls:
            m = re.match(r"(?:@parameters|const)\s+(\w+)\s*=\s*([\d.eE+\-]+)", decl)
            if m and m.group(1) == expr:
                try:
                    return float(m.group(2))
                except ValueError:
                    pass
        # Check auxiliary equations: "name ~ <number>"
        if expr in self.built_elements:
            eqs, _ = self.built_elements[expr]
            for eq in eqs:
                if "~" in eq:
                    rhs = eq.split("~", 1)[1].strip()
                    rhs = rhs.split("#")[0].strip()  # strip trailing comments
                    try:
                        return float(rhs)
                    except ValueError:
                        pass
        return None

    def _drain_embedded_delays(self, visitor: "JuliaASTVisitor") -> None:
        """Process any DelayFixedStructure nodes queued by the expression visitor.

        When XMILE DELAY(x, n) appears embedded inside an arithmetic expression
        (rather than as the top-level AST of an element), the visitor queues each
        one as (name, node).  This method lifts each into a proper ODE auxiliary.
        """
        while visitor._pending_delay_fixed:
            edf_name, edf_ast = visitor._pending_delay_fixed.pop(0)
            edf_eqs = self._expand_delay_fixed(edf_name, edf_ast, visitor)
            # Store as a pseudo-element so the equations reach the final output.
            self.built_elements[edf_name] = (edf_eqs, False)

    def _expand_delay_fixed(
        self,
        identifier: str,
        ast,
        visitor: "JuliaASTVisitor",
        dims: Optional[List[Tuple[str, int]]] = None,
    ) -> List[str]:
        """Expand DELAY FIXED into an exact N-stage Euler pipeline (ODE backend)
        or a first-order ODE approximation (MTK backend / dynamic delay time).

        For the ODE backend the pipeline is exact when using Euler integration:
        each of the N = round(delay_time / time_step) stages performs one step of
        delay via a first-order ODE with averaging_time = time_step.  With the
        Euler solver, u[pipe_k](t+dt) = u[pipe_{k-1}](t), giving exact transport.

        For the MTK backend (or when delay_time cannot be evaluated at translation
        time) a single first-order ODE approximation is emitted instead.
        """
        dims = dims or []
        if dims:
            # x[Dim1] = DELAY FIXED(y[Dim1], …): the arguments are read
            # inside the loop over the variable's dimensions
            visitor = self._nd_visitor(dims, self._idx_vars(len(dims)))
        input_expr = visitor.visit(ast.input)
        delay_time_expr = visitor.visit(ast.delay_time)
        initial_expr = visitor.visit(ast.initial)

        # ---- N-stage pipeline, both backends ----
        # As Python PySD (DelayFixed), the number of stages comes from the delay
        # time at the initial time; with Euler and dt = time_step the pipeline
        # is exact.
        ts_val = self._eval_ast_at_t0(ReferenceStructure("time_step"))
        if ts_val is None:
            ts_val = 1.0  # Vensim's default TIME STEP
        reason = f"the number of stages of DELAY FIXED in {identifier}"
        self._fix_reason = reason
        try:
            dt_val = self._try_eval_as_float(delay_time_expr)
        finally:
            self._fix_reason = None
        if dt_val is None:
            dt_val = self._eval_fixing(ast.delay_time, reason)
        if dt_val is not None and ts_val is not None and ts_val > 0:
            N = max(round(dt_val / ts_val + 1e-6), 1)
            return self._expand_delay_fixed_pipeline(
                identifier, input_expr, initial_expr, N, dims
            )
        else:
            warn(
                f"DELAY FIXED for '{identifier}': delay time '{delay_time_expr}' "
                "cannot be evaluated at translation time — "
                "falling back to first-order ODE approximation."
            )

        # ---- Fallback: single first-order ODE approximation ----
        lv_name = f"_df_{identifier}"
        self.namespace.namespace[f"__internal_df_{identifier}"] = lv_name

        if dims:
            ndim = len(dims)
            idx_vars = self._idx_vars(ndim)
            vnd = self._nd_visitor(dims, idx_vars)
            input_nd = vnd.visit(ast.input)
            delay_time_nd = vnd.visit(ast.delay_time)
            initial_nd = vnd.visit(ast.initial)
            self.stock_decls.append(
                f"@variables {lv_name}(t)[{self._range_str(dims)}]"
            )
            ranges_list = [range(1, size + 1) for _, size in dims]
            for idx_combo in itertools.product(*ranges_list):
                expr_i = initial_nd
                for iv, idx in zip(idx_vars, idx_combo):
                    expr_i = expr_i.replace(iv, str(idx))
                idx_str = ", ".join(str(v) for v in idx_combo)
                self.u0_entries.append(f"{lv_name}[{idx_str}] => {expr_i}")
            self.aux_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            idx_str_t = ", ".join(idx_vars)
            for_clause = self._for_clause(dims, idx_vars)
            return [
                f"[D({lv_name}[{idx_str_t}]) ~ ({input_nd} - {lv_name}[{idx_str_t}]) / max({delay_time_nd}, eps(Float64)) "
                f"for {for_clause}]...",
                f"[{identifier}[{idx_str_t}] ~ {lv_name}[{idx_str_t}] for {for_clause}]...",
            ]

        self.stock_decls.append(f"@variables {lv_name}(t)")
        self.u0_entries.append(f"{lv_name} => {initial_expr}")
        self.aux_decls.append(f"@variables {identifier}(t)")
        return [
            f"D({lv_name}) ~ ({input_expr} - {lv_name}) / max({delay_time_expr}, eps(Float64))",
            f"{identifier} ~ {lv_name}",
        ]

    def _expand_delay_fixed_pipeline(
        self,
        identifier: str,
        input_expr: str,
        initial_expr: str,
        N: int,
        dims: Optional[List[Tuple[str, int]]] = None,
    ) -> List[str]:
        """Emit an N-stage pipeline for an exact DELAY FIXED with Euler integration.

        The stages are one array stock ``_df_pipe_<identifier>`` with the stage
        as its first index (then the variable's dimensions), so that a long
        delay (a delay time of hundreds of time steps) is one comprehension
        and not hundreds of variables.  With Euler and dt = time_step, each
        stage shifts a value by exactly one time step, giving a total
        transport delay of N * time_step.  All stages start at
        ``initial_expr``.
        """
        dims = dims or []
        ts_expr = self.control_vals.get("time_step") or "time_step"
        pipe_name = f"_df_pipe_{identifier}"
        self.namespace.namespace[f"__internal_df_pipe_{identifier}"] = pipe_name
        idx_vars = self._idx_vars(len(dims))
        cells = ", ".join(["_k"] + idx_vars)
        prev = ", ".join(["_k - 1"] + idx_vars)
        last = ", ".join([str(N)] + idx_vars)
        for_clause = ", ".join(
            ["_k in 1:" + str(N)] + ([self._for_clause(dims, idx_vars)] if dims else [])
        )
        ranges = ", ".join([f"1:{N}"] + ([self._range_str(dims)] if dims else []))
        self.stock_decls.append(f"@variables {pipe_name}(t)[{ranges}]")
        # One u0 entry per cell of the variable, for all its stages
        for idx_combo in itertools.product(*[range(1, size + 1) for _, size in dims]):
            expr_i = initial_expr
            for iv, idx in zip(idx_vars, idx_combo):
                expr_i = expr_i.replace(iv, str(idx))
            idx_s = ", ".join([f"1:{N}"] + [str(i) for i in idx_combo])
            self.u0_entries.append(f"{pipe_name}[{idx_s}] => {expr_i}")

        equations = [
            f"[D({pipe_name}[{cells}]) ~ "
            f"((_k == 1 ? {input_expr} : {pipe_name}[{prev}]) - {pipe_name}[{cells}]) "
            f"/ ({ts_expr}) for {for_clause}]..."
        ]
        if dims:
            idx_str_t = ", ".join(idx_vars)
            self.aux_decls.append(f"@variables {identifier}(t)[{self._range_str(dims)}]")
            equations.append(
                f"[{identifier}[{idx_str_t}] ~ {pipe_name}[{last}] "
                f"for {self._for_clause(dims, idx_vars)}]..."
            )
        else:
            self.aux_decls.append(f"@variables {identifier}(t)")
            equations.append(f"{identifier} ~ {pipe_name}[{N}]")
        return equations

    # ------------------------------------------------------------------
    # Trend expansion
    # ------------------------------------------------------------------

    def _expand_trend(
        self,
        identifier: str,
        ast,
        visitor: "JuliaASTVisitor",
        dims: Optional[List[Tuple[str, int]]] = None,
    ) -> List[str]:
        """Expand TREND(input, average_time, initial_trend), as Python PySD
        (``Trend``): a smooth level ``_sm`` with

            _sm(0) = input / (1 + initial_trend * average_time)
            D(_sm) = (input - _sm) / average_time
            output = ZIDZ(input - _sm, average_time * |_sm|)
        """
        return self._expand_trend_like(identifier, ast, visitor, dims, forecast=False)

    # ------------------------------------------------------------------
    # Forecast expansion
    # ------------------------------------------------------------------

    def _expand_forecast(
        self,
        identifier: str,
        ast,
        visitor: "JuliaASTVisitor",
        dims: Optional[List[Tuple[str, int]]] = None,
    ) -> List[str]:
        """Expand FORECAST(input, average_time, horizon, initial_trend), as
        Python PySD (``Forecast``): the same smooth level as TREND, with

            _sm(0) = input / (1 + initial_trend)
            output = input * (1 + ZIDZ(input - _sm, average_time * _sm) * horizon)
        """
        return self._expand_trend_like(identifier, ast, visitor, dims, forecast=True)

    def _expand_trend_like(self, identifier, ast, visitor, dims, forecast: bool) -> List[str]:
        """TREND and FORECAST, scalar or over any number of dimensions."""
        dims = dims or []
        idx = self._idx_vars(len(dims))
        vis = self._nd_visitor(dims, idx) if dims else visitor
        input_expr = vis.visit(ast.input)
        avg_time_expr = vis.visit(ast.average_time)
        initial_trend_expr = vis.visit(ast.initial_trend)

        sm_name = f"_sm_{identifier}"
        self.namespace.namespace[f"__internal_sm_{identifier}"] = sm_name
        self.needed_helpers.add("pysd_zidz")
        cell = ", ".join(idx)
        sm = f"{sm_name}[{cell}]" if dims else sm_name
        if forecast:
            start = f"{input_expr} / (1.0 + ({initial_trend_expr}))"
            out = (f"{input_expr} * (1.0 + pysd_zidz({input_expr} - {sm}, "
                   f"({avg_time_expr}) * {sm}) * ({vis.visit(ast.horizon)}))")
        else:
            start = f"{input_expr} / (1.0 + ({initial_trend_expr}) * ({avg_time_expr}))"
            out = f"pysd_zidz({input_expr} - {sm}, ({avg_time_expr}) * abs({sm}))"
        rate = f"({input_expr} - {sm}) / ({avg_time_expr})"

        if not dims:
            self.stock_decls.append(f"@variables {sm_name}(t)")
            self.u0_entries.append(f"{sm_name} => {start}")
            self.aux_decls.append(f"@variables {identifier}(t)")
            return [f"D({sm_name}) ~ {rate}", f"{identifier} ~ {out}"]

        for_clause = self._for_clause(dims, idx)
        self.stock_decls.append(f"@variables {sm_name}(t)[{self._range_str(dims)}]")
        for index, start_i in self._nd_cells(dims, idx, start):
            self.u0_entries.append(f"{sm_name}[{index}] => {start_i}")
        self.aux_decls.append(f"@variables {identifier}(t)[{self._range_str(dims)}]")
        return [
            f"[D({sm}) ~ {rate} for {for_clause}]...",
            f"[{identifier}[{cell}] ~ {out} for {for_clause}]...",
        ]

    # ------------------------------------------------------------------
    # SAMPLE IF TRUE expansion
    # ------------------------------------------------------------------

    def _expand_sample_if_true(
        self,
        identifier: str,
        ast,
        visitor: "JuliaASTVisitor",
        dims: Optional[List[Tuple[str, int]]] = None,
        ndim: int = 0,
    ) -> List[str]:
        """Expand SAMPLE IF TRUE(condition, input, initial).

        SAMPLE IF TRUE is a discrete sample-and-hold: whenever the condition
        is true the output is updated to the input; otherwise the output holds
        its previous value.

        We approximate this as an INTEG with a conditional flow whose rate is
        tied to the simulation time step so that the Euler solver updates the
        state to ``input`` within one time step when the condition is true:

            D(output) ~ ifelse(condition > 0.5,
                               (input - output) / time_step,
                               0.0)

        Here ``time_step`` refers to the Julia variable defined in the
        generated file.  With Euler integration the next step will be:

            output_new = output + dt * (input - output) / dt = input

        which is exact (one-step snap to input).
        """
        dims = dims or []
        st_name = f"_sit_{identifier}"
        self.namespace.namespace[f"__internal_sit_{identifier}"] = st_name

        ts_val = self.control_vals.get("time_step")
        ts_expr = ts_val if ts_val is not None else "time_step"

        if dims:
            idx_vars = self._idx_vars(len(dims))
            vnd = self._nd_visitor(dims, idx_vars)
            condition_nd = vnd.visit(ast.condition)
            input_nd = vnd.visit(ast.input)
            initial_nd = vnd.visit(ast.initial)
            idx_str_t = ", ".join(idx_vars)
            for_clause = self._for_clause(dims, idx_vars)
            ranges_list = [range(1, size + 1) for _, size in dims]

            self.stock_decls.append(
                f"@variables {st_name}(t)[{self._range_str(dims)}]"
            )
            for idx_combo in itertools.product(*ranges_list):
                expr_i = initial_nd
                for iv, idx in zip(idx_vars, idx_combo):
                    expr_i = expr_i.replace(iv, str(idx))
                idx_s = ", ".join(str(v) for v in idx_combo)
                self.u0_entries.append(f"{st_name}[{idx_s}] => {expr_i}")
            self.aux_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            return [
                f"[D({st_name}[{idx_str_t}]) ~ pysd_ifelse({condition_nd} > 0.5, "
                f"({input_nd} - {st_name}[{idx_str_t}]) / ({ts_expr}), 0.0) "
                f"for {for_clause}]...",
                f"[{identifier}[{idx_str_t}] ~ pysd_ifelse({condition_nd} > 0.5, "
                f"{input_nd}, {st_name}[{idx_str_t}]) for {for_clause}]...",
            ]

        condition_expr = visitor.visit(ast.condition)
        input_expr = visitor.visit(ast.input)
        initial_expr = visitor.visit(ast.initial)
        self.stock_decls.append(f"@variables {st_name}(t)")
        self.u0_entries.append(f"{st_name} => {initial_expr}")
        self.aux_decls.append(f"@variables {identifier}(t)")
        return [
            f"D({st_name}) ~ pysd_ifelse({condition_expr} > 0.5, "
            f"({input_expr} - {st_name}) / ({ts_expr}), 0.0)",
            f"{identifier} ~ pysd_ifelse({condition_expr} > 0.5, {input_expr}, {st_name})",
        ]

    # ------------------------------------------------------------------
    # ALLOCATE AVAILABLE / ALLOCATE BY PRIORITY
    # ------------------------------------------------------------------

    def _expand_allocate(
        self,
        identifier: str,
        ast,
        visitor: "JuliaASTVisitor",
        dims: Optional[List[Tuple[str, int]]] = None,
    ) -> List[str]:
        """Emit PySD.jl allocation helper calls.

        ALLOCATE BY PRIORITY → pysd_allocate_by_priority(request, priority, width, supply)
        ALLOCATE AVAILABLE   → pysd_allocate_available(request, pp, avail)

        As in Python PySD, the allocation is along the last dimension: the
        request (and priority) are vectors along their last subscript, and
        the priority profiles of ALLOCATE AVAILABLE a matrix (the allocation
        dimension by the profile, its last subscript).  Each cell of the
        result takes its element of the allocation with the last loop index;
        the other subscripts select which allocation it belongs to.
        """
        dims = dims or []
        if not dims:
            self.aux_decls.append(f"@variables {identifier}(t)")
            if isinstance(ast, AllocateAvailableStructure):
                rhs = (f"pysd_allocate_available({visitor.visit(ast.request)}, "
                       f"{visitor.visit(ast.pp)}, {visitor.visit(ast.avail)})")
            else:
                rhs = (f"pysd_allocate_by_priority({visitor.visit(ast.request)}, "
                       f"{visitor.visit(ast.priority)}, {visitor.visit(ast.width)}, "
                       f"{visitor.visit(ast.supply)})")
            return [f"{identifier} ~ {rhs}"]

        idx = self._idx_vars(len(dims))
        vnd = self._nd_visitor(dims, idx)
        last_dim = dims[-1][0]
        n = self._jl_n(last_dim)

        def subs_of(arg) -> tuple:
            if isinstance(arg, ReferenceStructure) and arg.subscripts is not None:
                return tuple(arg.subscripts.subscripts)
            return ()

        def vector(arg) -> str:
            s = subs_of(arg)
            last = s[-1] if s else last_dim
            return f"[{vnd.bound_visit(arg, {last: '_v'})} for _v in 1:{n}]"

        if isinstance(ast, AllocateAvailableStructure):
            s = subs_of(ast.pp)
            profile = s[-1] if s else None
            along = s[-2] if len(s) >= 2 else last_dim
            if profile:
                # Vensim writes the profile as its first element (pp[branch,
                # ptype] with pprofile: ptype, ppriority, pwidth, pextra): the
                # profile dimension is that element's range, by size since it
                # may have no N_ constant
                if profile not in self._subs_elems and profile in self._elem_to_range:
                    profile_range = self._elem_to_range[profile]
                else:
                    profile_range = profile
                n_profile = len(self._subs_elems.get(profile_range, [])) or self._jl_n(profile_range)
                pp = (f"[{vnd.bound_visit(ast.pp, {along: '_v', profile: '_p'})} "
                      f"for _v in 1:{n}, _p in 1:{n_profile}]")
            else:
                pp = vnd.visit(ast.pp)
            rhs = (f"pysd_allocate_available({vector(ast.request)}, {pp}, "
                   f"{vnd.visit(ast.avail)})[{idx[-1]}]")
        else:
            rhs = (f"pysd_allocate_by_priority({vector(ast.request)}, "
                   f"{vector(ast.priority)}, {vnd.visit(ast.width)}, "
                   f"{vnd.visit(ast.supply)})[{idx[-1]}]")
        self.aux_decls.append(f"@variables {identifier}(t)[{self._range_str(dims)}]")
        return [f"[{identifier}[{', '.join(idx)}] ~ {rhs} for {self._for_clause(dims, idx)}]..."]

    # ------------------------------------------------------------------
    # DataStructure (tab-file DATA variable) processing
    # ------------------------------------------------------------------

    def _process_tab_data_structure(
        self,
        elem: "AbstractElement",
        identifier: str,
        comp: "AbstractData",
    ) -> List[str]:
        """Emit _tab_val() call(s) for a Vensim DATA variable (INTERPOLATE /
        HOLD BACKWARD / LOOK FORWARD / RAW keyword).

        The generated Julia code calls ``_tab_val(key, t)`` which reads from
        the ``_tab_data`` Dict populated at runtime by ``_load_tab_data!(files)``.
        """
        real_name = elem.name
        kw = getattr(comp, "keyword", None) or "interpolate"
        method_sym = f":{kw}"  # e.g. ":interpolate", ":hold_backward"

        comp_dims = self._comp_coords(comp)  # dict dim_name -> elements
        ndim = len(comp_dims)

        if ndim == 0:
            # Scalar DATA variable
            self._tab_data_entries.append((identifier, real_name, method_sym, []))
            key = identifier
            return [f"{identifier} ~ _tab_val(\"{key}\", t)"]

        # Subscripted: collect element labels per dimension
        dim_names = list(comp_dims.keys())
        dim_elems = [comp_dims[d] for d in dim_names]  # list of label lists

        self._tab_data_entries.append((identifier, real_name, method_sym, dim_elems))

        if ndim == 1:
            n = self._subs_sizes.get(dim_names[0], len(dim_elems[0]))
            n_expr = f"N_{dim_names[0].upper().replace(' ', '_')}" if n > 0 else str(len(dim_elems[0]))
            # Build comprehension: [identifier[_i] ~ _tab_val("identifier_$(_i)", t) for _i in 1:N]...
            return [
                f"[{identifier}[_i] ~ _tab_val(\"{identifier}_$(_i)\", t) for _i in 1:{len(dim_elems[0])}]..."
            ]
        elif ndim == 2:
            n0, n1 = len(dim_elems[0]), len(dim_elems[1])
            return [
                f"[{identifier}[_i, _j] ~ _tab_val(\"{identifier}_$(_i)_$(_j)\", t) "
                f"for _i in 1:{n0}, _j in 1:{n1}]..."
            ]
        else:
            # 3D+: emit per-element equations
            n0, n1, n2 = len(dim_elems[0]), len(dim_elems[1]), len(dim_elems[2])
            return [
                f"[{identifier}[_i, _j, _k] ~ _tab_val(\"{identifier}_$(_i)_$(_j)_$(_k)\", t) "
                f"for _i in 1:{n0}, _j in 1:{n1}, _k in 1:{n2}]..."
            ]

    # ------------------------------------------------------------------
    # Subscripted inline lookup processing
    # ------------------------------------------------------------------

    def _process_subscripted_inline_lookup(
        self,
        elem: "AbstractElement",
        identifier: str,
    ) -> List[str]:
        """Emit a dispatching lookup function for subscripted inline lookup tables.

        Handles elements like::

            lookup1dim[A]((2,3),(4,7),(7,1)) ~~|
            lookup1dim[B]((3,4),(4,-1),(8,1.5))

        where each subscript combination has its own ``LookupsStructure`` data.
        For each component we register a named interpolant constant and then
        build a dispatch function that selects by integer index.

        Supports 1D and 2D subscript combinations.  Higher-D combinations are
        flattened (each component becomes an independent 0-D lookup function
        array entry) with a warning.
        """
        dims = self._element_dims(elem)
        ndim = len(dims)

        # Collect (subscript_indices_tuple, LookupsStructure) pairs.
        # For each component, resolve subscript labels to 1-based indices.
        comp_entries: List[Tuple[Tuple[int, ...], "LookupsStructure"]] = []
        for comp in elem.components:
            spec = comp.subscripts[0] if comp.subscripts else []
            indices: List[int] = []
            for k, label in enumerate(spec):
                if k < len(dims):
                    dim_name, _ = dims[k]
                    dim_elems = self._subs_elems.get(dim_name, [])
                    idx = next(
                        (i + 1 for i, e in enumerate(dim_elems)
                         if e.lower() == label.lower()),
                        None,
                    )
                    if idx is not None:
                        indices.append(idx)
                    else:
                        indices.append(1)
            comp_entries.append((tuple(indices), comp.ast))

        if ndim == 1:
            # Build array of interpolants, one per element of dim0.
            dim0_name, dim0_size = dims[0]
            # Map index → LookupsStructure; use first comp if multiple share idx
            idx_to_lkp: Dict[int, "LookupsStructure"] = {}
            for idxs, lkp in comp_entries:
                idx = idxs[0] if idxs else 1
                if idx not in idx_to_lkp:
                    idx_to_lkp[idx] = lkp

            itp_names: List[str] = []
            for i in range(1, dim0_size + 1):
                lkp = idx_to_lkp.get(i)
                itp_name = f"{identifier}_{i}_itp"
                if lkp is not None:
                    const_decl, _, _ = lookup_interpolation_code(
                        f"{identifier}_{i}", lkp.x, lkp.y, lkp.type
                    )
                    self.lookup_const_decls.append(const_decl)
                else:
                    self.lookup_const_decls.append(
                        f"const {itp_name} = LinearInterpolation([0.0], [0.0];"
                        " extrapolation_left=ExtrapolationType.Constant,"
                        " extrapolation_right=ExtrapolationType.Constant)"
                    )
                itp_names.append(itp_name)
            arr_name = f"{identifier}_itps"
            self.lookup_const_decls.append(
                f"const {arr_name} = [{', '.join(itp_names)}]"
            )
            self.lookup_func_decls.append(
                f"{identifier}(i::Integer, x::Real) = {arr_name}[clamp(i, 1, {dim0_size})](x)"
            )
            self.lookup_register_decls.append(
                f"@register_symbolic {identifier}(i::Integer, x::Real)"
            )
            self.lookup_identifiers.add(identifier)
            self._var_dims[identifier] = [dim0_name]

        elif ndim == 2:
            dim0_name, dim0_size = dims[0]
            dim1_name, dim1_size = dims[1]
            idx_to_lkp2d: Dict[Tuple[int, int], "LookupsStructure"] = {}
            for idxs, lkp in comp_entries:
                i0 = idxs[0] if len(idxs) > 0 else 1
                i1 = idxs[1] if len(idxs) > 1 else 1
                if (i0, i1) not in idx_to_lkp2d:
                    idx_to_lkp2d[(i0, i1)] = lkp

            row_lists: List[str] = []
            for i in range(1, dim0_size + 1):
                row_itp_names: List[str] = []
                for j in range(1, dim1_size + 1):
                    lkp = idx_to_lkp2d.get((i, j))
                    itp_name = f"{identifier}_{i}_{j}_itp"
                    if lkp is not None:
                        const_decl, _, _ = lookup_interpolation_code(
                            f"{identifier}_{i}_{j}", lkp.x, lkp.y, lkp.type
                        )
                        self.lookup_const_decls.append(const_decl)
                    else:
                        self.lookup_const_decls.append(
                            f"const {itp_name} = LinearInterpolation([0.0], [0.0];"
                            " extrapolation_left=ExtrapolationType.Constant,"
                            " extrapolation_right=ExtrapolationType.Constant)"
                        )
                    row_itp_names.append(itp_name)
                row_lists.append("[" + ", ".join(row_itp_names) + "]")

            arr_name = f"{identifier}_itps"
            self.lookup_const_decls.append(
                f"const {arr_name} = [{', '.join(row_lists)}]"
            )
            self.lookup_func_decls.append(
                f"{identifier}(i::Integer, j::Integer, x::Real) = "
                f"{arr_name}[clamp(i, 1, {dim0_size})][clamp(j, 1, {dim1_size})](x)"
            )
            self.lookup_register_decls.append(
                f"@register_symbolic {identifier}(i::Integer, j::Integer, x::Real)"
            )
            self.lookup_identifiers.add(identifier)
            self._var_dims[identifier] = [dim0_name, dim1_name]

        else:
            # Higher-D: emit a flat array of lookups, indexed linearly.
            warn(
                f"Subscripted inline lookup '{elem.name}' has {ndim} dimensions; "
                "only 1D and 2D are supported — flattening to 1D array."
            )
            all_itp_names: List[str] = []
            for k, (idxs, lkp) in enumerate(comp_entries, 1):
                itp_name = f"{identifier}_{k}_itp"
                const_decl, _, _ = lookup_interpolation_code(
                    f"{identifier}_{k}", lkp.x, lkp.y, lkp.type
                )
                self.lookup_const_decls.append(const_decl)
                all_itp_names.append(itp_name)
            arr_name = f"{identifier}_itps"
            total = len(all_itp_names)
            self.lookup_const_decls.append(
                f"const {arr_name} = [{', '.join(all_itp_names)}]"
            )
            self.lookup_func_decls.append(
                f"{identifier}(i::Integer, x::Real) = {arr_name}[clamp(i, 1, {total})](x)"
            )
            self.lookup_register_decls.append(
                f"@register_symbolic {identifier}(i::Integer, x::Real)"
            )
            self.lookup_identifiers.add(identifier)

        return []

    # ------------------------------------------------------------------
    # GET LOOKUPS processing
    # ------------------------------------------------------------------

    def _process_get_lookups(
        self, elem: "AbstractElement", identifier: str
    ) -> List[str]:
        """Emit a named interpolation function for external lookup data.

        For single-component (scalar) lookups, emits a runtime
        ``pysd_xlsx_read_series`` call so the Excel file is read when the
        Julia model loads.  Multi-component (subscripted) lookups fall back
        to reading at translation time via ``ExtLookup``.
        """
        comp0 = elem.components[0]
        ast0 = comp0.ast

        # ---- Single-component scalar: emit runtime Excel read ----
        if self.data_format != "json" and len(elem.components) == 1 and not self._comp_coords(comp0):
            file_expr = f'joinpath(@__DIR__, "{ast0.file}")'
            series_call = (
                f'pysd_xlsx_read_series({file_expr}, '
                f'"{ast0.tab}", "{ast0.x_row_or_col}", "{ast0.cell}")'
            )
            itp_name = f"{identifier}_itp"
            const_decl = (
                f"const {itp_name} = let (_xs, _ys) = {series_call}\n"
                f"    LinearInterpolation(_ys, _xs; "
                f"extrapolation_left=ExtrapolationType.Constant, "
                f"extrapolation_right=ExtrapolationType.Constant)\nend"
            )
            func_decl = f"{identifier}(x) = {itp_name}(x)"
            reg_decl = f"@register_symbolic {identifier}(x::Real)"
            self.lookup_const_decls.append(const_decl)
            self.lookup_func_decls.append(func_decl)
            self.lookup_register_decls.append(reg_decl)
            self.lookup_identifiers.add(identifier)
            return []

        # ---- Single-component subscripted: runtime dispatch ----
        if self.data_format != "json" and len(elem.components) == 1 and self._comp_coords(comp0):
            file_expr = f'joinpath(@__DIR__, "{ast0.file}")'
            self.lookup_const_decls.append(
                f"const {identifier}_fns = pysd_xlsx_build_lookup_dispatch("
                f'{file_expr}, "{ast0.tab}", "{ast0.x_row_or_col}", "{ast0.cell}"'
                f"{self._dispatch_n(comp0)})"
            )
            self.lookup_func_decls.append(
                f"{identifier}(i, x) = {identifier}_fns[i](x)"
            )
            self.lookup_register_decls.append(
                f"@register_symbolic {identifier}(i::Integer, x::Real)"
            )
            self.lookup_identifiers.add(identifier)
            return []

        # ---- Multi-component: emit per-component dispatch ----
        if self.data_format != "json" and len(elem.components) > 1:
            ast0 = elem.components[0].ast
            file_expr = f'joinpath(@__DIR__, "{ast0.file}")'
            y_comps = [comp for comp in elem.components
                       if isinstance(comp.ast, GetLookupsStructure)]
            y_names = [comp.ast.cell for comp in y_comps]
            if y_names:
                # Each component may produce a 1D or 2D lookup.
                # Use pysd_xlsx_build_lookup_dispatch per component to get
                # a vector of interpolations, then build a nested dispatch.
                sub_dispatch_names = []
                for k, y_name in enumerate(y_names):
                    sub_name = f"{identifier}_{k + 1}"
                    self.lookup_const_decls.append(
                        f"const {sub_name}_fns = pysd_xlsx_build_lookup_dispatch("
                        f'{file_expr}, "{ast0.tab}", "{ast0.x_row_or_col}", "{y_name}"'
                        f"{self._dispatch_n(y_comps[k])})"
                    )
                    sub_dispatch_names.append(f"{sub_name}_fns")

                fn_list = ", ".join(sub_dispatch_names)
                self.lookup_const_decls.append(
                    f"const {identifier}_fns = [{fn_list}]"
                )
                # Determine dispatch arity from subscript dimensions
                n_sub_dims = len(self._comp_coords(elem.components[0]))
                if n_sub_dims >= 2:
                    self.lookup_func_decls.append(
                        f"{identifier}(i, j, x) = (i <= length({identifier}_fns) && j <= length({identifier}_fns[i])) ? {identifier}_fns[i][j](x) : {identifier}_fns[j][i](x)"
                    )
                    self.lookup_register_decls.append(
                        f"@register_symbolic {identifier}(i::Integer, j::Integer, x::Real)"
                    )
                else:
                    self.lookup_func_decls.append(
                        f"{identifier}(i, x) = {identifier}_fns[clamp(i, 1, length({identifier}_fns))][1](x)"
                    )
                    self.lookup_register_decls.append(
                        f"@register_symbolic {identifier}(i::Integer, x::Real)"
                    )
                self.lookup_identifiers.add(identifier)
                return []

        # ---- Baked-in fallback for edge cases ----
        try:
            from pysd.py_backend.external import ExtLookup

            if len(elem.components) > 1:
                # Detect which subscript positions vary across components and
                # find the unique containing range for each such position.
                split_ranges = self._detect_split_ranges(elem.components)
                coords0 = self._comp_coords_split(comp0, split_ranges)
                final_coords: Dict[str, list] = {}
                for comp in elem.components:
                    for range_key, elem_val in self._comp_coords_split(comp, split_ranges).items():
                        if range_key not in final_coords:
                            final_coords[range_key] = self._subs_elems.get(range_key, elem_val)
            else:
                split_ranges = {}
                coords0 = self._comp_coords(comp0)
                final_coords = {k: self._subs_elems.get(k, v) for k, v in coords0.items()}

            ext = ExtLookup(
                file_name=ast0.file,
                tab=ast0.tab,
                x_row_or_col=ast0.x_row_or_col,
                cell=ast0.cell,
                coords=coords0,
                root=self.root,
                final_coords=final_coords,
                py_name=identifier,
            )

            for comp in elem.components[1:]:
                ast_i = comp.ast
                comp_coords = self._comp_coords_split(comp, split_ranges)
                ext.add(ast_i.file, ast_i.tab, ast_i.x_row_or_col, ast_i.cell, comp_coords)

            ext.initialize()

            # ext.data is an xarray DataArray with dim "lookup_dim"
            import numpy as np
            data = ext.data
            if hasattr(data, "values"):
                arr = data.values
            else:
                arr = np.asarray(data)

            xs = tuple(float(x) for x in data.coords["lookup_dim"].values)

            # For scalar lookups, data has shape (n_points,)
            if arr.ndim == 1:
                ys = tuple(float(y) for y in arr)
                const_decl, func_decl, reg_decl = lookup_interpolation_code(
                    identifier, xs, ys, "interpolate"
                )
                self.lookup_const_decls.append(const_decl)
                self.lookup_func_decls.append(func_decl)
                self.lookup_register_decls.append(reg_decl)
                if self.data_format == "json":
                    self._json_data["lookups"][identifier] = {
                        "x": list(xs), "y": list(ys),
                        "interp_type": "interpolate", "subscripts": [],
                    }
                return []
            elif arr.ndim == 2:
                # 2D: shape (n_points, n_subs).
                # Emit one lookup function per subscript element:
                #   identifier_1(x), identifier_2(x), ...
                # and a dispatch function identifier(i, x) that selects by index.
                n_subs = arr.shape[1]
                sub_func_names = []
                for k in range(n_subs):
                    col_ys = tuple(float(y) for y in arr[:, k])
                    sub_name = f"{identifier}_{k + 1}"
                    const_decl, func_decl, reg_decl = lookup_interpolation_code(
                        sub_name, xs, col_ys, "interpolate"
                    )
                    self.lookup_const_decls.append(const_decl)
                    self.lookup_func_decls.append(func_decl)
                    self.lookup_register_decls.append(reg_decl)
                    sub_func_names.append(sub_name)

                # Build a dispatch array and wrapper:
                # const identifier_fns = [identifier_1, identifier_2, ...]
                # identifier(i, x) = identifier_fns[i](x)
                fn_list = ", ".join(sub_func_names)
                self.lookup_const_decls.append(
                    f"const {identifier}_fns = [{fn_list}]"
                )
                self.lookup_func_decls.append(
                    f"{identifier}(i, x) = {identifier}_fns[i](x)"
                )
                self.lookup_register_decls.append(
                    f"@register_symbolic {identifier}(i::Integer, x::Real)"
                )
                if self.data_format == "json":
                    for k in range(n_subs):
                        sub_name = f"{identifier}_{k + 1}"
                        col_ys = tuple(float(y) for y in arr[:, k])
                        self._json_data["lookups"][sub_name] = {
                            "x": list(xs), "y": list(col_ys),
                            "interp_type": "interpolate", "subscripts": [],
                        }
                return []
            elif arr.ndim == 3:
                # 3D: shape (n_points, n_dim1, n_dim2).
                # Emit one lookup per (i, j) pair and a 2-index dispatch.
                n_dim1, n_dim2 = arr.shape[1], arr.shape[2]
                rows: List[List[str]] = []
                for i in range(n_dim1):
                    row: List[str] = []
                    for j in range(n_dim2):
                        col_ys = tuple(float(y) for y in arr[:, i, j])
                        sub_name = f"{identifier}_{i + 1}_{j + 1}"
                        const_decl, func_decl, reg_decl = lookup_interpolation_code(
                            sub_name, xs, col_ys, "interpolate"
                        )
                        self.lookup_const_decls.append(const_decl)
                        self.lookup_func_decls.append(func_decl)
                        self.lookup_register_decls.append(reg_decl)
                        if self.data_format == "json":
                            self._json_data["lookups"][sub_name] = {
                                "x": list(xs), "y": list(col_ys),
                                "interp_type": "interpolate", "subscripts": [],
                            }
                        row.append(sub_name)
                    rows.append(row)
                inner = ", ".join("[" + ", ".join(r) + "]" for r in rows)
                self.lookup_const_decls.append(
                    f"const {identifier}_fns = [{inner}]"
                )
                self.lookup_func_decls.append(
                    f"{identifier}(i, j, x) = (i <= length({identifier}_fns) && j <= length({identifier}_fns[i])) ? {identifier}_fns[i][j](x) : {identifier}_fns[j][i](x)"
                )
                self.lookup_register_decls.append(
                    f"@register_symbolic {identifier}(i::Integer, j::Integer, x::Real)"
                )
                return []
            else:  # pragma: no cover
                warn(
                    f"Subscripted GET LOOKUPS '{elem.name}' has {arr.ndim - 1} "
                    "subscript dimensions (> 2D) — only up to 2D subscripted lookups "
                    "are supported. Emitting flattened first-column lookup as approximation."
                )
                ys = tuple(float(y) for y in arr.reshape(arr.shape[0], -1)[:, 0])
                const_decl, func_decl, reg_decl = lookup_interpolation_code(
                    identifier, xs, ys, "interpolate"
                )
                self.lookup_const_decls.append(const_decl)
                self.lookup_func_decls.append(func_decl)
                self.lookup_register_decls.append(reg_decl)
                return []

        except Exception as exc:  # pragma: no cover
            warn(
                f"Could not read GET LOOKUPS for '{elem.name}': {exc} "
                "— emitting placeholder auxiliary."
            )
            self.aux_decls.append(f"@variables {identifier}(t)")
            return [f"# GET_LOOKUPS_FAILED: {identifier} ~ 0.0"]

    # ------------------------------------------------------------------
    # GET DATA processing
    # ------------------------------------------------------------------

    def _place_data_parts_1d(self, identifier: str, data_comps, names) -> Optional[List[str]]:
        """Data defined one element at a time (``x[a1]``, ``x[a3]``, ``x[a2]``,
        in any order): each part's series
        at its element's position in the variable's range, which is how the
        equations index it, whatever the order of the definitions.  Elements
        with no definition give NaN."""
        subs = [list(dc.subscripts[0]) for dc in data_comps]
        if any(len(s) != 1 or s[0] not in self._elem_to_range for s in subs):
            return None
        labels = [s[0] for s in subs]
        dim = (self._var_dims.get(identifier) or [None])[0]
        if dim not in self._subs_elems:
            dim = self._infer_parent_range(labels)
        elems = self._subs_elems.get(dim, []) if dim else []
        if not elems or any(lb not in elems for lb in labels):
            return None
        placed = ["[_ -> NaN]"] * len(elems)
        for lb, name in zip(labels, names):
            placed[elems.index(lb)] = name
        return placed

    def _keyed_dispatch(self, identifier: str, data_comps) -> Optional[tuple]:
        """Index the data definitions of a variable by the elements they fix.

        For revenue3[A, X, company] = GET DIRECT DATA(…), revenue3[B, X,
        company] = …: each definition fixes some subscripts to one element
        and leaves at most one as a range (read from the columns of its
        data), at the same position in every definition.  Returns the
        indices each definition fixes (in the variable's dimensions), the
        number of subscripts and the position of the range (or None), or
        None when the definitions do not have that shape.
        """
        subs_lists = [list(dc.subscripts[0]) for dc in data_comps]
        n_pos = len(subs_lists[0])
        if n_pos < 2 or any(len(s) != n_pos for s in subs_lists):
            return None
        dims = list(self._var_dims.get(identifier) or [])
        if len(dims) != n_pos:
            dims = []
            for p in range(n_pos):
                ranges = [s[p] for s in subs_lists if s[p] in self._subs_elems]
                dims.append(ranges[0] if ranges else self._elem_to_range.get(subs_lists[0][p]))
        range_pos = None
        keys = []
        for subs in subs_lists:
            rp = [p for p, s in enumerate(subs) if s in self._subs_elems]
            if len(rp) > 1 or (rp and range_pos is not None and rp[0] != range_pos):
                return None
            if rp:
                range_pos = rp[0]
            key = []
            for p, s in enumerate(subs):
                if p == range_pos:
                    continue
                elems = self._subs_elems.get(dims[p], []) if dims[p] else []
                if s not in elems:
                    return None
                key.append(elems.index(s) + 1)
            keys.append(tuple(key))
        if len(set(keys)) != len(keys):
            return None
        return keys, n_pos, range_pos

    def _process_get_data(
        self,
        elem: "AbstractElement",
        identifier: str,
        comp: "AbstractComponent",
    ) -> List[str]:
        """Read external time-series data and emit a time-indexed interpolation.

        Uses ``ExtData`` to load the series at translation time, then
        emits a ``LinearInterpolation`` over (time, value) pairs just
        like a lookup, but with ``t`` as the argument.
        """
        # Collect only components that carry a GetDataStructure
        data_comps = [c for c in elem.components if isinstance(c.ast, GetDataStructure)]
        if not data_comps:  # pragma: no cover  # unreachable: routing requires _has_get_data_ast=True
            self.aux_decls.append(f"@variables {identifier}(t)")
            return [f"{identifier} ~ 0.0"]

        comp0 = data_comps[0]
        ast0 = comp0.ast

        # ---- Single-component scalar: emit runtime Excel read ----
        if self.data_format != "json" and len(data_comps) == 1 and not self._comp_coords(comp0):
            file_expr = f'joinpath(@__DIR__, "{ast0.file}")'
            series_call = (
                f'pysd_xlsx_read_series({file_expr}, '
                f'"{ast0.tab}", "{ast0.time_row_or_col}", "{ast0.cell}")'
            )
            itp_name = f"{identifier}_itp"
            julia_itp = _vensim_keyword_to_itp_type(getattr(comp0, "keyword", None))
            if julia_itp == "hold_forward":
                itp_call, dir_arg = "ConstantInterpolation", ""
            elif julia_itp == "hold_backward":
                itp_call, dir_arg = "ConstantInterpolation", "dir=:right, "
            else:
                itp_call, dir_arg = "LinearInterpolation", ""
            const_decl = (
                f"const {itp_name} = let (_xs, _ys) = {series_call}\n"
                f"    {itp_call}(_ys, _xs; {dir_arg}"
                f"extrapolation_left=ExtrapolationType.Constant, "
                f"extrapolation_right=ExtrapolationType.Constant)\nend"
            )
            func_decl = f"{identifier}(x) = {itp_name}(x)"
            reg_decl = f"@register_symbolic {identifier}(x::Real)"
            self.lookup_const_decls.append(const_decl)
            self.lookup_func_decls.append(func_decl)
            self.lookup_register_decls.append(reg_decl)
            self.lookup_identifiers.add(identifier)
            return []

        # ---- Single-component subscripted: runtime dispatch ----
        if self.data_format != "json" and len(data_comps) == 1:
            file_expr = f'joinpath(@__DIR__, "{ast0.file}")'
            self.lookup_const_decls.append(
                f"const {identifier}_fns = pysd_xlsx_build_lookup_dispatch("
                f'{file_expr}, "{ast0.tab}", "{ast0.time_row_or_col}", "{ast0.cell}"'
                f"{self._dispatch_n(comp0)})"
            )
            self.lookup_func_decls.append(
                f"{identifier}(i, x) = {identifier}_fns[i](x)"
            )
            self.lookup_register_decls.append(
                f"@register_symbolic {identifier}(i::Integer, x::Real)"
            )
            self.lookup_identifiers.add(identifier)
            return []

        # ---- Multi-component: emit per-component dispatch ----
        if self.data_format != "json" and len(data_comps) > 1:
            file_expr = f'joinpath(@__DIR__, "{ast0.file}")'
            sub_dispatch_names = []
            for k, dc in enumerate(data_comps):
                sub_name = f"{identifier}_{k + 1}"
                self.lookup_const_decls.append(
                    f"const {sub_name}_fns = pysd_xlsx_build_lookup_dispatch("
                    f'{file_expr}, "{dc.ast.tab}", "{dc.ast.time_row_or_col}", "{dc.ast.cell}"'
                    f"{self._dispatch_n(dc)})"
                )
                sub_dispatch_names.append(f"{sub_name}_fns")

            keyed = self._keyed_dispatch(identifier, data_comps)
            if keyed is not None:
                entries, n_pos, range_pos = keyed
                args = [f"a{p + 1}" for p in range(n_pos)]
                key = ", ".join(a for p, a in enumerate(args) if p != range_pos)
                series = args[range_pos] if range_pos is not None else "1"
                pairs = ", ".join(
                    f"({', '.join(map(str, k))},) => {name}"
                    for k, name in zip(entries, sub_dispatch_names)
                )
                self.lookup_const_decls.append(f"const {identifier}_fns = Dict({pairs})")
                self.lookup_func_decls.append(
                    f"{identifier}({', '.join(args)}, x) = {identifier}_fns[({key},)][{series}](x)"
                )
                self.lookup_register_decls.append(
                    f"@register_symbolic {identifier}("
                    + ", ".join(f"{a}::Integer" for a in args) + ", x::Real)"
                )
                self.lookup_identifiers.add(identifier)
                return []

            n_sub_dims = len(self._comp_coords(data_comps[0]))
            placed = self._place_data_parts_1d(identifier, data_comps, sub_dispatch_names)
            fn_list = ", ".join(placed or sub_dispatch_names)
            self.lookup_const_decls.append(
                f"const {identifier}_fns = [{fn_list}]"
            )
            if n_sub_dims >= 2:
                self.lookup_func_decls.append(
                    f"{identifier}(i, j, x) = (i <= length({identifier}_fns) && j <= length({identifier}_fns[i])) ? {identifier}_fns[i][j](x) : {identifier}_fns[j][i](x)"
                )
                self.lookup_register_decls.append(
                    f"@register_symbolic {identifier}(i::Integer, j::Integer, x::Real)"
                )
            else:
                self.lookup_func_decls.append(
                    f"{identifier}(i, x) = {identifier}_fns[clamp(i, 1, length({identifier}_fns))][1](x)"
                )
                self.lookup_register_decls.append(
                    f"@register_symbolic {identifier}(i::Integer, x::Real)"
                )
            self.lookup_identifiers.add(identifier)
            return []

        # ---- Baked-in fallback for edge cases ----
        try:
            from pysd.py_backend.external import ExtData

            if len(data_comps) > 1:
                split_ranges = self._detect_split_ranges(data_comps)
                coords0 = self._comp_coords_split(comp0, split_ranges)
                final_coords: Dict[str, list] = {}
                for c in data_comps:
                    for range_key, elem_val in self._comp_coords_split(c, split_ranges).items():
                        if range_key not in final_coords:
                            final_coords[range_key] = self._subs_elems.get(range_key, elem_val)
            else:
                split_ranges = {}
                coords0 = self._comp_coords(comp0)
                final_coords = {k: self._subs_elems.get(k, v) for k, v in coords0.items()}

            # Determine interpolation type from AbstractData keyword
            julia_itp = _vensim_keyword_to_itp_type(
                getattr(comp, "keyword", None)
            )

            ext = ExtData(
                file_name=ast0.file,
                tab=ast0.tab,
                time_row_or_col=ast0.time_row_or_col,
                cell=ast0.cell,
                interp="interpolate",  # always interpolate when reading at translate time
                coords=coords0,
                root=self.root,
                final_coords=final_coords,
                py_name=identifier,
            )

            for c in data_comps[1:]:
                ai = c.ast
                comp_coords = self._comp_coords_split(c, split_ranges)
                ext.add(ai.file, ai.tab, ai.time_row_or_col, ai.cell, "interpolate", comp_coords)

            ext.initialize()

            import numpy as np
            data = ext.data
            if hasattr(data, "values"):
                arr = data.values
                time_vals = data.coords["time"].values
            else:
                arr = np.asarray(data)
                time_vals = None

            if time_vals is None:
                raise ValueError(f"No time dimension in data (shape={arr.shape})")

            xs = tuple(float(t) for t in time_vals)

            if arr.ndim == 1:
                ys = tuple(float(y) for y in arr)
                const_decl, func_decl, reg_decl = lookup_interpolation_code(
                    identifier, xs, ys, julia_itp
                )
                self.lookup_const_decls.append(const_decl)
                self.lookup_func_decls.append(func_decl)
                self.lookup_register_decls.append(reg_decl)
                if self.data_format == "json":
                    self._json_data["data"][identifier] = {
                        "time": list(xs), "values": list(ys),
                        "interp_type": julia_itp, "subscripts": [],
                    }
                return []
            elif arr.ndim == 2:
                # Subscripted time-series: shape (n_time, n_subs)
                n_subs = arr.shape[1]
                sub_func_names = []
                for k in range(n_subs):
                    col_ys = tuple(float(y) for y in arr[:, k])
                    sub_name = f"{identifier}_{k + 1}"
                    const_decl, func_decl, reg_decl = lookup_interpolation_code(
                        sub_name, xs, col_ys, julia_itp
                    )
                    self.lookup_const_decls.append(const_decl)
                    self.lookup_func_decls.append(func_decl)
                    self.lookup_register_decls.append(reg_decl)
                    if self.data_format == "json":
                        self._json_data["data"][sub_name] = {
                            "time": list(xs), "values": list(col_ys),
                            "interp_type": julia_itp, "subscripts": [],
                        }
                    sub_func_names.append(sub_name)

                fn_list = ", ".join(sub_func_names)
                self.lookup_const_decls.append(
                    f"const {identifier}_fns = [{fn_list}]"
                )
                self.lookup_func_decls.append(
                    f"{identifier}(i, x) = {identifier}_fns[i](x)"
                )
                self.lookup_register_decls.append(
                    f"@register_symbolic {identifier}(i::Integer, x::Real)"
                )
                return []
            elif arr.ndim == 3:
                # 3D time-series: shape (n_time, n_dim1, n_dim2)
                n_dim1, n_dim2 = arr.shape[1], arr.shape[2]
                rows: List[List[str]] = []
                for i in range(n_dim1):
                    row: List[str] = []
                    for j in range(n_dim2):
                        col_ys = tuple(float(y) for y in arr[:, i, j])
                        sub_name = f"{identifier}_{i + 1}_{j + 1}"
                        const_decl, func_decl, reg_decl = lookup_interpolation_code(
                            sub_name, xs, col_ys, julia_itp
                        )
                        self.lookup_const_decls.append(const_decl)
                        self.lookup_func_decls.append(func_decl)
                        self.lookup_register_decls.append(reg_decl)
                        if self.data_format == "json":
                            self._json_data["data"][sub_name] = {
                                "time": list(xs), "values": list(col_ys),
                                "interp_type": julia_itp, "subscripts": [],
                            }
                        row.append(sub_name)
                    rows.append(row)
                inner = ", ".join("[" + ", ".join(r) + "]" for r in rows)
                self.lookup_const_decls.append(
                    f"const {identifier}_fns = [{inner}]"
                )
                self.lookup_func_decls.append(
                    f"{identifier}(i, j, x) = (i <= length({identifier}_fns) && j <= length({identifier}_fns[i])) ? {identifier}_fns[i][j](x) : {identifier}_fns[j][i](x)"
                )
                self.lookup_register_decls.append(
                    f"@register_symbolic {identifier}(i::Integer, j::Integer, x::Real)"
                )
                return []
            else:  # pragma: no cover
                raise ValueError(f"Unexpected data dimensions: {arr.ndim} (shape={arr.shape})")

        except Exception as exc:  # pragma: no cover
            warn(
                f"Could not read GET DATA for '{elem.name}': {exc} "
                "— emitting placeholder auxiliary."
            )
            self.aux_decls.append(f"@variables {identifier}(t)")
            return [f"# GET_DATA_FAILED: {identifier} ~ 0.0"]

    def _expand_initial_frozen_stock(
        self,
        identifier: str,
        inner_ast,
        dims: List[Tuple[str, int]],
        ndim: int,
    ) -> List[str]:
        """Emit ``INITIAL(expr)`` as a zero-derivative stock.

        MTK evaluates the initial-condition expression at t=t0, which gives
        the correct Vensim semantics (value frozen at the initial time).

        Scalar::

            @variables x(t)
            D(x) ~ 0.0
            u0: x => expr

        1D subscripted::

            @variables x(t)[1:N]
            [D(x[_i0]) ~ 0.0 for _i0 in 1:N]...
            u0: x[i] => expr_at_i   (for i in 1..N)

        2D subscripted::

            @variables x(t)[1:N0, 1:N1]
            [D(x[_i0, _i1]) ~ 0.0 for _i0 in 1:N0, _i1 in 1:N1]...
            u0: x[i, j] => expr_at_ij
        """
        if ndim == 0:
            v = JuliaASTVisitor(
                self.namespace, self.inline_registry, self.needed_helpers,
                subs_sizes=self._subs_sizes, root=self.root,
                macro_names=self._known_macro_names,
            )
            init_expr = v.visit(inner_ast)
            self.stock_decls.append(f"@variables {identifier}(t)")
            self.u0_entries.append(f"{identifier} => {init_expr}")
            return [f"D({identifier}) ~ 0.0"]

        if ndim == 1:
            (d0, n0) = dims[0]
            idx_vars = ["_i0"]
            vnd = self._nd_visitor(dims, idx_vars)
            raw_expr = vnd.visit(inner_ast)
            self.stock_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            for i in range(1, n0 + 1):
                expr_i = raw_expr.replace("_i0", str(i))
                self.u0_entries.append(f"{identifier}[{i}] => {expr_i}")
            return [f"[D({identifier}[_i0]) ~ 0.0 for _i0 in 1:{self._jl_n(d0)}]..."]

        # ndim >= 2
        idx_vars = self._idx_vars(ndim)
        vnd = self._nd_visitor(dims, idx_vars)
        raw_expr = vnd.visit(inner_ast)
        ranges_list = [range(1, size + 1) for _, size in dims]
        self.stock_decls.append(
            f"@variables {identifier}(t)[{self._range_str(dims)}]"
        )
        for idx_combo in itertools.product(*ranges_list):
            expr_ij = raw_expr
            for iv, idx in zip(idx_vars, idx_combo):
                expr_ij = expr_ij.replace(iv, str(idx))
            idx_str = ", ".join(str(i) for i in idx_combo)
            self.u0_entries.append(f"{identifier}[{idx_str}] => {expr_ij}")
        for_clause = self._for_clause(dims, idx_vars)
        idx_str_template = ", ".join(idx_vars)
        return [
            f"[D({identifier}[{idx_str_template}]) ~ 0.0 "
            f"for {for_clause}]..."
        ]

    def _resolve_initial_value(self, inner_ast) -> Optional[str]:
        """Return the t=0 value of *inner_ast* as a Julia literal, or None.

        Handles:
        * Numeric literals
        * References to stocks (in ``u0_entries``)
        * References to parameters/constants (including GetConstantsStructure)
        * References to auxiliaries whose own equation chains back to a stock
          (one level of indirection, e.g. ``INITIAL(InflowA)`` where
          ``InflowA ~ StockA`` and StockA has a known initial condition)
        * GetConstantsStructure directly embedded in the INITIAL() argument
        """
        from pysd.builders.julia.julia_expressions_builder import format_number
        if isinstance(inner_ast, (int, float)):
            return format_number(inner_ast)
        if isinstance(inner_ast, ReferenceStructure):
            return self._resolve_ref_initial(inner_ast.reference, depth=3)
        if isinstance(inner_ast, GetConstantsStructure):
            # Try to read the constant directly
            try:
                from pysd.py_backend.external import ExtConstant
                ext = ExtConstant(
                    file_name=inner_ast.file,
                    tab=inner_ast.tab,
                    cell=inner_ast.cell,
                    coords={},
                    root=self.root,
                    final_coords={},
                    py_name="_initial_resolve",
                )
                ext.initialize()
                return _format_julia_value(ext.data)
            except Exception:
                pass
        return None

    def _resolve_ref_initial(self, ref: str, depth: int) -> Optional[str]:
        """Recursively resolve the t=0 value of a variable reference."""
        if depth < 0:
            return None
        julia_id = self.namespace.get(ref)
        if julia_id is None:
            return None
        # Check u0_entries (stocks)
        for entry in self.u0_entries:
            parts = entry.split("=>", 1)
            if len(parts) == 2 and parts[0].strip() == julia_id:
                return parts[1].strip()
        # Check param_decls (constants)
        for decl in self.param_decls:
            prefix = f"@parameters {julia_id} = "
            if decl.startswith(prefix):
                return decl[len(prefix):]
        # Follow an auxiliary equation one level deeper
        if depth > 0 and julia_id in self.built_elements:
            eqs, _ = self.built_elements[julia_id]
            for eq in eqs:
                if "~" in eq:
                    rhs = eq.split("~", 1)[1].strip()
                    # Plain number
                    try:
                        float(rhs)
                        return rhs
                    except ValueError:
                        pass
                    # Plain identifier → recurse
                    import re as _re
                    if _re.match(r"^[a-z_][a-z0-9_]*$", rhs):
                        result = self._resolve_ref_initial(rhs, depth - 1)
                        if result:
                            return result
        return None

    # ------------------------------------------------------------------
    # External constants reader
    # ------------------------------------------------------------------

    def _read_get_constants(
        self, elem: AbstractElement, identifier: str
    ) -> Optional[str]:
        """Emit a Julia expression that reads external constant data at runtime.

        For single-component elements, emits a ``pysd_xlsx_read_constant``
        call so the Excel file is read when the Julia model loads.
        Multi-component (subscripted) elements and piecewise (mixed GCS +
        literal) elements fall back to reading at translation time via
        ``ExtConstant`` and embedding the values.
        """
        import numpy as np

        comp0 = elem.components[0]
        ast0 = comp0.ast

        # ----- Single-component: emit runtime Excel read -----
        if len(elem.components) == 1 and isinstance(ast0, GetConstantsStructure):
            cell = ast0.cell
            transpose = cell.endswith('*')
            clean_cell = _cell_block(
                cell.rstrip('*'), self._comp_coords(comp0), transpose)
            file_expr = f'joinpath(@__DIR__, "{ast0.file}")'
            kw_parts = []
            if transpose:
                kw_parts.append("transpose=true")
            kw = ("; " + ", ".join(kw_parts)) if kw_parts else ""
            return (
                f'pysd_xlsx_read_constant({file_expr}, '
                f'"{ast0.tab}", "{clean_cell}"{kw})'
            )

        # ----- Multi-component 1-D: each definition at its own elements -----
        # var2[t2] = GET DIRECT CONSTANTS(…), var2[t3] = …, var2[subdim] = …:
        # the definitions are written in any order, so each one is assigned
        # to the positions of its elements in the variable's range.
        placed = self._placed_constants_1d(elem)
        if placed is not None:
            return placed

        # ----- Multi-component with 2D+ subscripts: fall back to baked-in -----
        comp0_coords = self._comp_coords(elem.components[0])
        if len(comp0_coords) >= 2 and len(elem.components) > 1:
            return self._read_get_constants_baked(elem, identifier)

        # ----- Multi-component: emit single pysd_xlsx_read_constant with vector -----
        # Build a Julia vector literal of specs: strings for range names,
        # vectors for literal values.
        specs = []
        file_expr = None
        tab = None
        transpose = False
        for comp in elem.components:
            if isinstance(comp.ast, GetConstantsStructure):
                cell = comp.ast.cell
                if cell.endswith('*'):
                    transpose = True
                clean_cell = _cell_block(
                    cell.rstrip('*'), self._comp_coords(comp), cell.endswith('*'))
                if file_expr is None:
                    file_expr = f'joinpath(@__DIR__, "{comp.ast.file}")'
                    tab = comp.ast.tab
                specs.append(f'"{clean_cell}"')
            elif isinstance(comp.ast, (int, float)):
                val = format_number(comp.ast)
                coords = self._comp_coords(comp)
                n_elems = 1
                for dim_elems in coords.values():
                    n_elems *= max(len(dim_elems), 1)
                if n_elems > 1:
                    specs.append(f"fill({val}, {n_elems})")
                else:
                    specs.append(f"[{val}]")
            else:  # pragma: no cover  # unreachable: _const_like ensures GCS or numeric only
                visitor = JuliaASTVisitor(
                    self.namespace, self.inline_registry,
                    self.needed_helpers,
                    lookup_names=self.lookup_identifiers,
                    macro_names=self._known_macro_names,
                )
                val = visitor.visit(comp.ast)
                coords = self._comp_coords(comp)
                n_elems = 1
                for dim_elems in coords.values():
                    n_elems *= max(len(dim_elems), 1)
                if n_elems > 1:
                    specs.append(f"fill({val}, {n_elems})")
                else:
                    specs.append(f"[{val}]")

        if file_expr is None:  # pragma: no cover  # unreachable: _const_like has at least one GCS
            return None
        specs_str = ", ".join(specs)
        kw_parts = []
        if transpose:
            kw_parts.append("transpose=true")
        # Multi-dimensional reshaping is handled by the equation generator
        # which uses flat indexing, so we keep the result flat here.
        kw = ("; " + ", ".join(kw_parts)) if kw_parts else ""
        return (
            f'pysd_xlsx_read_constant({file_expr}, '
            f'"{tab}", [{specs_str}]{kw})'
        )

    def _placed_constants_1d(self, elem: AbstractElement) -> Optional[str]:
        """A 1-D constant defined in parts, as a ``let`` block that fills
        each part's positions (None if the definitions are not 1-D parts of
        one range, or the JSON data format is used)."""
        if self.data_format == "json" or len(elem.components) < 2:
            return None
        parts = []
        covered: List[str] = []
        for comp in elem.components:
            coords = self._comp_coords(comp)
            if len(coords) != 1:
                return None
            elems = list(next(iter(coords.values())))
            covered.extend(elems)
            parts.append((comp, elems))
        target = next((r for r, e in self._subs_elems.items()
                       if isinstance(e, list) and sorted(e) == sorted(covered)), None)
        if target is None:
            return None
        order = self._subs_elems[target]
        lines = [f"let _v = zeros({len(order)})"]
        for comp, elems in parts:
            positions = ", ".join(str(order.index(e) + 1) for e in elems)
            if isinstance(comp.ast, GetConstantsStructure):
                cell = comp.ast.cell
                kw = "; transpose=true" if cell.endswith("*") else ""
                block = _cell_block(cell.rstrip("*"), self._comp_coords(comp), cell.endswith("*"))
                value = (f'pysd_xlsx_read_constant(joinpath(@__DIR__, "{comp.ast.file}"), '
                         f'"{comp.ast.tab}", "{block}"{kw})')
            else:
                value = format_number(comp.ast)
            lines.append(f"    _v[[{positions}]] .= {value}")
        lines += ["    _v", "end"]
        return "\n".join(lines)

    def _read_get_constants_baked(
        self, elem: "AbstractElement", identifier: str
    ) -> Optional[str]:
        """Fall back to reading constants at translation time for complex cases."""
        try:
            from pysd.py_backend.external import ExtConstant

            gcs_comps = [c for c in elem.components if isinstance(c.ast, GetConstantsStructure)]
            lit_comps = [c for c in elem.components if not isinstance(c.ast, GetConstantsStructure)]
            if gcs_comps and lit_comps:
                return self._read_get_constants_piecewise(
                    elem, identifier, gcs_comps, lit_comps
                )

            comp0 = elem.components[0]
            ast0 = comp0.ast

            if len(elem.components) > 1:
                split_ranges = self._detect_split_ranges(elem.components)
                coords0 = self._comp_coords_split(comp0, split_ranges)
                final_coords: Dict[str, list] = {}
                for comp in elem.components:
                    for range_key, elem_val in self._comp_coords_split(comp, split_ranges).items():
                        if range_key not in final_coords:
                            final_coords[range_key] = self._subs_elems.get(range_key, elem_val)
            else:  # pragma: no cover  # unreachable: baked only called when len(components) > 1
                split_ranges = {}
                coords0 = self._comp_coords(comp0)
                final_coords = {k: self._subs_elems.get(k, v) for k, v in coords0.items()}

            ext = ExtConstant(
                file_name=ast0.file,
                tab=ast0.tab,
                cell=ast0.cell,
                coords=coords0,
                root=self.root,
                final_coords=final_coords,
                py_name=identifier,
            )

            for comp in elem.components[1:]:
                ast_i = comp.ast
                comp_coords = self._comp_coords_split(comp, split_ranges)
                ext.add(ast_i.file, ast_i.tab, ast_i.cell, comp_coords)

            ext.initialize()
            return _format_julia_value(ext.data)

        except Exception as exc:  # pragma: no cover
            warn(
                f"Could not read external constant for '{elem.name}': {exc} "
                "— emitting placeholder."
            )
            return None

    def _read_get_constants_piecewise(
        self,
        elem: "AbstractElement",
        identifier: str,
        gcs_comps: List["AbstractComponent"],
        lit_comps: List["AbstractComponent"],
    ) -> Optional[str]:
        """Build a constant array from a mix of GCS and numeric-literal components.

        Vensim allows piecewise definitions such as::

            var[fuel1, fuel2, fuel3] = GET DIRECT CONSTANTS(...)
            var[electricity] = 0
            var[heat] = 0

        For 1-D subscripts the element labels are collected and ordered by their
        parent range.  For 2-D+ subscripts a numpy array of the full shape is
        built and each component fills its slice.
        """
        import numpy as np
        from pysd.py_backend.external import ExtConstant
        from pysd.builders.julia.julia_expressions_builder import format_number

        all_comps = elem.components

        # ---- Detect dimensionality ----------------------------------------
        max_ndim = max(
            (len(c.subscripts[0]) for c in all_comps if c.subscripts and c.subscripts[0]),
            default=0
        )

        if max_ndim >= 2:
            return self._read_get_constants_piecewise_nd(
                elem, identifier, gcs_comps, lit_comps
            )
        else:  # pragma: no cover  # 1-D path unreachable: routing requires 2D+ comps
        # ---- 1-D path (original logic) ------------------------------------
            split_ranges = self._detect_split_ranges(all_comps)

        # Build a map: element_label → float value
        elem_values: Dict[str, float] = {}

        for comp in lit_comps:
            val = float(comp.ast) if isinstance(comp.ast, (int, float)) else 0.0
            subs = comp.subscripts[0] if comp.subscripts else []
            for s in subs:
                if s in self._subs_elems:
                    for e in self._subs_elems[s]:
                        elem_values[e] = val
                else:
                    elem_values[s] = val

        for comp in gcs_comps:
            ast = comp.ast
            coords = self._comp_coords_split(comp, split_ranges)
            final_c = {k: self._subs_elems.get(k, v) for k, v in coords.items()}
            ext = ExtConstant(
                file_name=ast.file, tab=ast.tab, cell=ast.cell,
                coords=coords, root=self.root, final_coords=final_c,
                py_name=identifier,
            )
            ext.initialize()
            data = ext.data
            arr = data.values if hasattr(data, "values") else np.asarray(data)
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 0:
                subs = comp.subscripts[0] if comp.subscripts else []
                if subs:
                    elem_values[subs[0]] = float(arr)
            else:
                for dim_name, coord_vals in data.coords.items():
                    labels = [str(v) for v in coord_vals.values]
                    for idx, label in enumerate(labels):
                        sliced = arr.take(idx, axis=list(data.dims).index(dim_name))
                        if sliced.ndim == 0:
                            elem_values[label] = float(sliced)

        all_elems = list(elem_values.keys())
        parent_range = self._infer_parent_range(all_elems)
        if parent_range is None:
            vals = list(elem_values.values())
        else:
            ordered_elems = self._subs_elems.get(parent_range, all_elems)
            vals = [elem_values.get(e, 0.0) for e in ordered_elems]

        if len(vals) == 1:
            return format_number(vals[0])
        return "[" + ", ".join(format_number(v) for v in vals) + "]"

    def _read_get_constants_piecewise_nd(
        self,
        elem: "AbstractElement",
        identifier: str,
        gcs_comps: List["AbstractComponent"],
        lit_comps: List["AbstractComponent"],
    ) -> Optional[str]:
        """Multi-dimensional (N≥2) piecewise assembly.

        Handles the common Vensim pattern where a 2D+ constant is defined by
        a mix of GCS and literal-value components, each covering a different
        sub-range of one dimension while sharing all other dimensions.

        Example:
            x[DIM, other]
              [SUB1, other] = 0   (literal)
              [SUB2, other] = GCS (Excel data)

        The full parent dims are determined by _element_dims, then a numpy
        array is allocated and each component fills its slice.
        """
        import numpy as np
        from pysd.py_backend.external import ExtConstant

        # Determine full parent dimensions for each subscript position
        dims = self._element_dims(elem)
        if not dims:  # pragma: no cover  # routing via _read_get_constants_baked guarantees dims
            return None

        parent_dim_names = [d for d, _ in dims]
        parent_dim_elems = [self._subs_elems.get(d, []) for d in parent_dim_names]

        if any(len(e) == 0 for e in parent_dim_elems):  # pragma: no cover
            return None  # unknown dim — fall back to caller

        shape = tuple(len(e) for e in parent_dim_elems)
        full_arr = np.zeros(shape)

        def _comp_idx_arrays(comp_subs):
            """Return index arrays (one per dim) for np.ix_."""
            idx_arrs = []
            for pos, elems in enumerate(parent_dim_elems):
                s = comp_subs[pos] if pos < len(comp_subs) else None
                if s is None:
                    idx_arrs.append(np.arange(len(elems)))
                elif s in self._subs_elems:
                    # Sub-range: indices of its elements in the parent dim
                    sub_els = set(self._subs_elems[s])
                    idxs = [i for i, e in enumerate(elems) if e in sub_els]
                    idx_arrs.append(np.array(idxs, dtype=int))
                elif s in elems:
                    idx_arrs.append(np.array([elems.index(s)], dtype=int))
                else:
                    idx_arrs.append(np.arange(len(elems)))
            return idx_arrs

        # Fill literal components
        for comp in lit_comps:
            val = float(comp.ast) if isinstance(comp.ast, (int, float)) else 0.0
            comp_subs = comp.subscripts[0] if comp.subscripts else []
            idx_arrs = _comp_idx_arrays(comp_subs)
            full_arr[np.ix_(*idx_arrs)] = val

        # Fill GCS components
        for comp in gcs_comps:
            ast = comp.ast
            comp_subs = comp.subscripts[0] if comp.subscripts else []
            idx_arrs = _comp_idx_arrays(comp_subs)

            # Build coords keyed by parent dim names with the actual element lists
            coords: Dict[str, list] = {}
            for pos, (dim_name, elems) in enumerate(zip(parent_dim_names, parent_dim_elems)):
                s = comp_subs[pos] if pos < len(comp_subs) else None
                if s in self._subs_elems:
                    coords[dim_name] = self._subs_elems[s]
                elif s in elems:  # pragma: no cover  # requires ExtConstant with element-label subscript
                    coords[dim_name] = [s]
                else:  # pragma: no cover  # requires ExtConstant with unknown subscript
                    coords[dim_name] = list(elems)

            try:
                ext = ExtConstant(
                    file_name=ast.file, tab=ast.tab, cell=ast.cell,
                    coords=coords, root=self.root, final_coords=coords,
                    py_name=identifier,
                )
                ext.initialize()
                data_arr = np.asarray(
                    ext.data.values if hasattr(ext.data, "values") else ext.data,
                    dtype=float,
                )
                full_arr[np.ix_(*idx_arrs)] = data_arr
            except Exception as exc:  # pragma: no cover
                warn(
                    f"Could not read external constant for '{elem.name}' "
                    f"(component {comp_subs}): {exc}"
                )

        return _format_julia_value(full_arr)

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def _json_accumulate_constant(
        self, elem: "AbstractElement", identifier: str, julia_val: str
    ) -> None:
        """Store an external constant's value in ``_json_data["constants"]``."""
        import numpy as np
        try:
            from pysd.py_backend.external import ExtConstant
            comp0 = elem.components[0]
            coords0 = self._comp_coords(comp0)
            ext = ExtConstant(
                file_name=comp0.ast.file,
                tab=comp0.ast.tab,
                cell=comp0.ast.cell,
                coords=coords0,
                root=self.root,
                final_coords={k: self._subs_elems.get(k, v) for k, v in coords0.items()},
                py_name=identifier,
            )
            ext.initialize()
            raw = ext.data
            if hasattr(raw, "values"):
                raw = raw.values
            arr = np.asarray(raw, dtype=float)
            if arr.ndim == 0:
                values: object = float(arr)
                dims: list = []
            else:
                values = arr.tolist()
                dims = [f"dim{i}" for i in range(arr.ndim)]
            self._json_data["constants"][identifier] = {
                "dims": dims,
                "coords": {},
                "values": values,
                "units": elem.units or "",
            }
        except Exception:
            # Best-effort; fall back to the Julia literal string
            self._json_data["constants"][identifier] = {
                "dims": [], "coords": {},
                "values": julia_val,
                "units": elem.units or "",
            }

    # ------------------------------------------------------------------
    # JSON data file
    # ------------------------------------------------------------------

    def _write_data_json(self) -> Path:
        """Write accumulated external data to ``<model>_data.json``.

        Returns the path of the written file.

        Schema::

            {
              "constants": {
                "<jl_id>": {
                  "dims": [...],
                  "coords": {...},
                  "values": <scalar|list>,
                  "units": ""
                }
              },
              "lookups": {
                "<jl_id>": {
                  "x": [...],
                  "y": [...],
                  "interp_type": "interpolate",
                  "subscripts": []
                }
              },
              "data": {
                "<jl_id>": {
                  "time": [...],
                  "values": [...],
                  "interp_type": "interpolate",
                  "subscripts": []
                }
              }
            }
        """
        import json

        # Use self.path.stem (not self.model_name) so macro sections write
        # their data file next to their own .jl file.
        json_path = self.path.with_name(f"{self.path.stem}_data.json")
        json_path.write_text(
            json.dumps(self._json_data, indent=2), encoding="UTF-8"
        )
        return json_path

    # ------------------------------------------------------------------
    # Single-file build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Write the whole model as one ``.jl`` file."""
        all_eqs: List[str] = []
        for eqs, _is_ctrl in self.built_elements.values():
            all_eqs.extend(eqs)
        if self.data_format == "json":
            self._write_data_json()
        text = self._full_file_content(all_eqs)
        self.path.write_text(text, encoding="UTF-8")

    # ------------------------------------------------------------------
    # Modular build
    # ------------------------------------------------------------------

    def _build_modular(self) -> None:
        """Write main ``.jl`` + one file per Vensim view."""
        modules_dir = self.root / f"modules_{self.model_name}"
        modules_dir.mkdir(exist_ok=True)

        assigned_ids: Set[str] = set()
        include_lines: List[str] = []
        eq_var_names: List[str] = []

        base = Path(f"modules_{self.model_name}")
        self._process_views_tree(
            self.views_dict,
            base,
            self.root,
            assigned_ids,
            include_lines,
            eq_var_names,
        )

        # Variables not assigned to any view go into the main file
        leftover_eqs: List[str] = []
        for identifier, (eqs, is_ctrl) in self.built_elements.items():
            if identifier not in assigned_ids and not is_ctrl:
                leftover_eqs.extend(eqs)
                if leftover_eqs:
                    warn(
                        f"Variable '{identifier}' is not declared in any view — "
                        "added to the main module."
                    )

        if self.data_format == "json":
            self._write_data_json()
        text = self._modular_main_content(include_lines, eq_var_names, leftover_eqs)
        self.path.write_text(text, encoding="UTF-8")

    def _process_views_tree(
        self,
        tree: dict,
        current_path: Path,
        wdir: Path,
        assigned_ids: Set[str],
        include_lines: List[str],
        eq_var_names: List[str],
    ) -> None:
        """Recursively walk *tree* and write one module file per leaf view."""
        for view_name, content in tree.items():
            view_path = current_path / view_name
            if isinstance(content, set):
                # Leaf node — collect identifiers for this view
                view_ids = self._resolve_view_ids(content)
                non_ctrl_ids = [
                    vid for vid in view_ids
                    if not self.built_elements.get(vid, ([], True))[1]
                ]
                if not non_ctrl_ids:
                    continue

                module_file = wdir / view_path.with_suffix(".jl")
                module_file.parent.mkdir(parents=True, exist_ok=True)
                eq_var = _path_to_eq_var(view_path)

                module_eqs: List[str] = []
                for vid in sorted(non_ctrl_ids):
                    eqs, _ = self.built_elements.get(vid, ([], False))
                    module_eqs.extend(eqs)
                    assigned_ids.add(vid)

                self._write_module_file(module_file, eq_var, module_eqs, view_path)
                rel = module_file.relative_to(wdir)
                include_lines.append(f'include("{rel}")')
                eq_var_names.append(eq_var)
            else:
                # Intermediate node — recurse
                (wdir / view_path).mkdir(parents=True, exist_ok=True)
                self._process_views_tree(
                    content, view_path, wdir, assigned_ids, include_lines, eq_var_names
                )

    def _resolve_view_ids(self, vensim_names: set) -> List[str]:
        """Map a set of Vensim variable names to Julia identifiers."""
        result = []
        for name in vensim_names:
            julia_id = self.namespace.get(name)
            if julia_id and julia_id in self.built_elements:
                result.append(julia_id)
        return result

    def _write_module_file(
        self,
        path: Path,
        eq_var: str,
        equations: List[str],
        module_path: Path,
    ) -> None:
        # Drop the modules_<name> prefix for the display name
        display = ".".join(list(module_path.parts)[1:])
        eq_lines = ",\n    ".join(equations) if equations else ""
        text = textwrap.dedent(f"""\
            # Module {display}
            # Translated using PySD version {__version__}

            {eq_var} = Equation[
                {eq_lines}
            ]
            """)
        path.write_text(text, encoding="UTF-8")

    # ------------------------------------------------------------------
    # Content assembly helpers
    # ------------------------------------------------------------------

    def _file_header(self, extra_packages: bool = False) -> str:
        if self.backend == "mtk":
            return self._file_header_mtk(extra_packages)
        # OrdinaryDiffEq v7 split Euler into OrdinaryDiffEqLowOrderRK.
        # PySD re-exports the helper functions (pysd_*), the Excel readers
        # (pysd_xlsx_read_*) and DataInterpolations, so it is always imported.
        uses = ["OrdinaryDiffEq", "PySD", "NCDatasets", "ComponentArrays"]
        has_lookups = bool(self.lookup_const_decls)
        if has_lookups or extra_packages:
            uses.append("DataInterpolations")
        if self.data_format == "json":
            uses.append("JSON3")
        header = (
            f"# Model {self.model_name}\n"
            f"# Translated using PySD version {__version__}\n\n"
            f"using {', '.join(uses)}\n\n"
            f'check_compat(v"{PYSD_JL_VERSION}")\n\n'
        )
        if self.data_format == "json":
            json_fname = f"{self.path.stem}_data.json"
            header += (
                f'const _model_data = JSON3.read(read(joinpath(@__DIR__, "{json_fname}"), String))\n\n'
            )
        return header

    def _file_header_mtk(self, extra_packages: bool = False) -> str:
        uses = ["ModelingToolkit", "OrdinaryDiffEq", "PySD", "NCDatasets"]
        has_lookups = bool(self.lookup_const_decls)
        if has_lookups or extra_packages:
            uses.append("DataInterpolations")
        if self.data_format == "json":
            uses.append("JSON3")
        header = (
            f"# Model {self.model_name}\n"
            f"# Translated using PySD version {__version__}\n\n"
            f"using {', '.join(uses)}\n\n"
            f'check_compat(v"{PYSD_JL_VERSION}")\n\n'
            "@independent_variables t\n"
            "D = Differential(t)\n\n"
        )
        if self.data_format == "json":
            json_fname = f"{self.path.stem}_data.json"
            header += (
                f'const _model_data = JSON3.read(read(joinpath(@__DIR__, "{json_fname}"), String))\n\n'
            )
        return header

    def _helpers_block(self) -> str:
        # Helpers are provided by `using PySD` — nothing to inline.
        return ""

    def _tab_data_block(self) -> str:
        """Emit tab-file DATA variable infrastructure (only when DataStructure
        variables are present in the model).

        Generates:
          - ``const _tab_data = Dict{String, Any}()`` — runtime interpolation cache
          - ``_load_tab_data!(files)`` — reads .tab files and fills the cache
          - ``_tab_val(key, t)`` — retrieves interpolated value at time t
        """
        if not self._tab_data_entries:
            return ""

        lines = ["# Tab-file DATA variable infrastructure"]
        lines.append("const _tab_data = Dict{String, Any}()")
        lines.append("")
        lines.append("function _load_tab_data!(files::AbstractVector{<:AbstractString})")
        lines.append("    empty!(_tab_data)")
        lines.append("    for filepath in files")

        for julia_id, real_name, method_sym, dim_elems in self._tab_data_entries:
            if not dim_elems:
                # Scalar
                col = real_name
                key = julia_id
                lines.append(
                    f"        try; _ts, _vs = pysd_tab_read_series(filepath, \"{col}\"); "
                    f"_tab_data[\"{key}\"] = pysd_build_tab_itp(_vs, _ts, {method_sym}); catch; end"
                )
            elif len(dim_elems) == 1:
                for i, lbl in enumerate(dim_elems[0], start=1):
                    col = f"{real_name}[{lbl}]"
                    key = f"{julia_id}_{i}"
                    lines.append(
                        f"        try; _ts, _vs = pysd_tab_read_series(filepath, \"{col}\"); "
                        f"_tab_data[\"{key}\"] = pysd_build_tab_itp(_vs, _ts, {method_sym}); catch; end"
                    )
            elif len(dim_elems) == 2:
                for i, lbl0 in enumerate(dim_elems[0], start=1):
                    for j, lbl1 in enumerate(dim_elems[1], start=1):
                        col = f"{real_name}[{lbl0},{lbl1}]"
                        key = f"{julia_id}_{i}_{j}"
                        lines.append(
                            f"        try; _ts, _vs = pysd_tab_read_series(filepath, \"{col}\"); "
                            f"_tab_data[\"{key}\"] = pysd_build_tab_itp(_vs, _ts, {method_sym}); catch; end"
                        )
            else:
                # 3D
                for i, lbl0 in enumerate(dim_elems[0], start=1):
                    for j, lbl1 in enumerate(dim_elems[1], start=1):
                        for k, lbl2 in enumerate(dim_elems[2], start=1):
                            col = f"{real_name}[{lbl0},{lbl1},{lbl2}]"
                            key = f"{julia_id}_{i}_{j}_{k}"
                            lines.append(
                                f"        try; _ts, _vs = pysd_tab_read_series(filepath, \"{col}\"); "
                                f"_tab_data[\"{key}\"] = pysd_build_tab_itp(_vs, _ts, {method_sym}); catch; end"
                            )

        lines.append("    end")
        lines.append("end")
        lines.append("")
        lines.append("_tab_val(key::String, t::Real) = haskey(_tab_data, key) ? Float64(_tab_data[key](t)) : 0.0")
        lines.append("")

        # NC data-file infrastructure: registry + _load_nc_data!
        # Registry maps Julia identifier → (method_symbol, n_subscript_dims).
        # _load_nc_data! reads any NC file whose variable names match the registry
        # and populates _tab_data using the same integer-indexed keys as _load_tab_data!
        # so that _tab_val() works transparently for both sources.
        lines.append("const _nc_data_registry = Dict{String, Tuple{Symbol, Int}}(")
        for julia_id, _real, method_sym, dim_elems in self._tab_data_entries:
            ndims = len(dim_elems)
            lines.append(f'    "{julia_id}" => ({method_sym}, {ndims}),')
        lines.append(")")
        lines.append("")
        lines.append("function _load_nc_data!(files::AbstractVector{<:AbstractString})")
        lines.append("    for f in files")
        lines.append("        NCDatasets.Dataset(f, \"r\") do ds")
        lines.append('            "time" ∉ keys(ds) && return')
        lines.append("            ts = Float64.(ds[\"time\"][:])")
        lines.append("            for (varname, (method, nd)) in _nc_data_registry")
        lines.append("                haskey(ds, varname) || continue")
        lines.append("                try")
        # NetCDF files list time first (PySD/xarray layout); NCDatasets returns
        # the dimensions reversed, so flip back to (time, dim_1, ..., dim_n).
        lines.append("                    data = Array(ds[varname])")
        lines.append("                    data = permutedims(data, ndims(data):-1:1)")
        lines.append("                    if nd == 0")
        lines.append("                        _tab_data[varname] = pysd_build_tab_itp(Float64.(vec(data)), ts, method)")
        lines.append("                    elseif nd == 1")
        lines.append("                        for k in 1:size(data, 2)")
        lines.append("                            _tab_data[\"$(varname)_$(k)\"] = pysd_build_tab_itp(Float64.(data[:, k]), ts, method)")
        lines.append("                        end")
        lines.append("                    elseif nd == 2")
        lines.append("                        for i in 1:size(data, 2), j in 1:size(data, 3)")
        lines.append("                            _tab_data[\"$(varname)_$(i)_$(j)\"] = pysd_build_tab_itp(Float64.(data[:, i, j]), ts, method)")
        lines.append("                        end")
        lines.append("                    else")
        lines.append("                        for i in 1:size(data, 2), j in 1:size(data, 3), k in 1:size(data, 4)")
        lines.append("                            _tab_data[\"$(varname)_$(i)_$(j)_$(k)\"] = pysd_build_tab_itp(Float64.(data[:, i, j, k]), ts, method)")
        lines.append("                        end")
        lines.append("                    end")
        lines.append("                catch; end")
        lines.append("            end")
        lines.append("        end")
        lines.append("    end")
        lines.append("end")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _lookup_block(self) -> str:
        if not self.lookup_const_decls and not self._json_data.get("lookups") \
                and not self._json_data.get("data"):
            return ""
        if self.backend == "ode":
            return ""  # in params()
        lines = ["# Lookup tables"]
        emit_register = (self.backend == "mtk")
        if self.data_format == "json":
            # JSON mode: build LinearInterpolation from _model_data at startup
            for key in list(self._json_data.get("lookups", {})):
                itp_name = f"{key}_itp"
                lines.append(
                    f'const {itp_name} = LinearInterpolation('
                    f'Float64.(_model_data["lookups"]["{key}"]["y"]), '
                    f'Float64.(_model_data["lookups"]["{key}"]["x"]))'
                )
                lines.append(f"{key}(x) = {itp_name}(x)")
                if emit_register:
                    lines.append(f"@register_symbolic {key}(x::Real)")
            for key in list(self._json_data.get("data", {})):
                itp_name = f"{key}_itp"
                lines.append(
                    f'const {itp_name} = LinearInterpolation('
                    f'Float64.(_model_data["data"]["{key}"]["values"]), '
                    f'Float64.(_model_data["data"]["{key}"]["time"]))'
                )
                lines.append(f"{key}(x) = {itp_name}(x)")
                if emit_register:
                    lines.append(f"@register_symbolic {key}(x::Real)")
        else:
            for const_decl in self.lookup_const_decls:
                lines.append(const_decl)
            for func_decl in self.lookup_func_decls:
                lines.append(func_decl)
            if emit_register:
                for reg_decl in self.lookup_register_decls:
                    lines.append(reg_decl)
        return "\n".join(lines) + "\n\n"

    def _declarations_block(self) -> str:
        if self.backend == "mtk":
            return self._declarations_block_mtk()
        lines: List[str] = []
        if self.subs_const_decls:
            lines.append("# Subscript dimension sizes")
            lines.extend(self.subs_const_decls)
            lines.append("")
        return "\n".join(lines) + "\n"

    def _ode_constant_decls(self) -> List[str]:
        """The constants and external constants as ``const name = value``
        declarations (or ``# comment`` lines), in model order."""
        lines: List[str] = []
        for decl in self.param_decls:
            # "@parameters name = value" -> "const name = value"
            if decl.startswith("@parameters "):
                val_part = decl[len("@parameters "):]
                if " = " in val_part:
                    name, val = val_part.split(" = ", 1)
                    name = name.strip()
                    val = val.strip()
                    json_consts = self._json_data.get("constants", {})
                    if self.data_format == "json" and name in json_consts:
                        entry = json_consts[name]
                        if entry.get("dims"):
                            lines.append(
                                f'const {name} = pysd_safe(Float64.'
                                f'(_model_data["constants"]["{name}"]["values"]))'
                            )
                        else:
                            lines.append(
                                f'const {name} = Float64('
                                f'_model_data["constants"]["{name}"]["values"])'
                            )
                    elif "pysd_xlsx_read_constant" in val:
                        lines.append(f"const {name} = pysd_safe({val})")
                    elif "[" in name:
                        lines.append(self._ode_array_constant(name, val))
                    else:
                        lines.append("const " + val_part)
                else:  # pragma: no cover  # all @parameters entries include "= value"
                    lines.append("const " + val_part)
            else:
                lines.append(decl)
        for decl in self.ext_const_decls:
            name_eq = decl.split(" = ", 1)
            if len(name_eq) == 2 and ("pysd_xlsx_read_constant" in decl
                                      or decl.strip().startswith("const") and "[" in name_eq[1]):
                lines.append(f"{name_eq[0]} = pysd_safe({name_eq[1]})")
            else:
                lines.append(decl)
        return lines

    def _declarations_block_mtk(self) -> str:
        lines: List[str] = []
        if self.subs_const_decls:
            lines.append("# Subscript dimension sizes")
            lines.extend(self.subs_const_decls)
            lines.append("")
        if self.stock_decls or self.aux_decls:
            lines.append("# State and auxiliary variables")
            lines.extend(self.stock_decls)
            lines.extend(self.aux_decls)
            lines.append("")
        if self.param_decls:
            lines.append("# Parameters")
            lines.extend(self.param_decls)
        if self.ext_const_decls:
            lines.append("\n# External constants")
            for decl in self.ext_const_decls:
                name_eq = decl.split(" = ", 1)
                if len(name_eq) == 2 and ("pysd_xlsx_read_constant" in decl
                                          or decl.strip().startswith("const") and "[" in name_eq[1]):
                    lines.append(f"{name_eq[0]} = pysd_safe({name_eq[1]})")
                else:
                    lines.append(decl)
        return "\n".join(lines) + "\n"

    def _equations_block(self, equations: List[str]) -> str:
        if self.backend == "mtk":
            return self._equations_block_mtk(equations)
        # Collect stock names and sizes from u0_entries
        # Each entry is "var => init" or "var[idx] => init"
        stock_info: Dict[str, int] = {}  # name -> count
        for entry in self.u0_entries:
            name = entry.split("=>")[0].strip()
            base = name.split("[")[0]
            stock_info[base] = stock_info.get(base, 0) + self._u0_size(entry)

        # Build state variable index map: name -> (start_idx, size)
        stock_indices: Dict[str, int] = {}
        stock_sizes: Dict[str, int] = {}
        idx = 1
        for name, size in stock_info.items():
            stock_indices[name] = idx
            stock_sizes[name] = size
            idx += size

        # Separate ODE equations (D(var) ~ ...) from algebraic (var ~ ...)
        ode_lines = []
        alg_lines = []
        for eq in equations:
            eq = eq.strip().rstrip(",")
            if not eq or eq.startswith("#"):
                continue
            if eq.startswith("D(") or "Symbolics.scalarize" in eq:
                ode_lines.append(eq)
            elif eq.startswith("["):
                if eq.startswith("[D("):
                    ode_lines.append(eq)
                else:
                    alg_lines.append(eq)
            else:
                alg_lines.append(eq)

        # Detect stock dimensionality: declarations first (always the full
        # ranges), then the ODE equations' for-clauses.
        stock_dims: Dict[str, List[str]] = {}
        for decl in self.stock_decls:
            m_sd = re.match(r"@variables\s+(\w+)\(t\)\[([^\]]+)\]", decl.strip())
            if m_sd:
                ranges = [r.strip() for r in m_sd.group(2).split(",")]
                if len(ranges) >= 2 and all(r.startswith("1:") for r in ranges):
                    stock_dims[m_sd.group(1)] = [r[2:] for r in ranges]
        for eq in ode_lines:
            eq_s = eq.strip().rstrip(",")
            if eq_s.startswith("["):
                inner = eq_s.strip().lstrip("[").rstrip(".]")
                m = re.match(r"D\((\w+)\[([^\]]+)\]\)", inner)
                if m:
                    name = m.group(1)
                    idx_parts = [x.strip() for x in m.group(2).split(",")]
                    if name not in stock_dims:
                        # Find dims from for clause
                        ranges = re.findall(r"in\s+\d+:(\w+)", _outer_for_clause(eq_s))
                        if ranges:
                            stock_dims[name] = ranges

        # Pre-allocate auxiliary arrays
        # Scan equations for indexed assignments like "var[i] = ..."
        alloc_needed: Dict[str, List[str]] = {}  # name -> [dim1, dim2, ...]

        # Seed from @variables aux declarations — these always have correct symbolic
        # ranges, e.g. "@variables my_var(t)[1:N_REGION, 1:N_SEC_ALL]"
        for decl in self.aux_decls:
            m_vd = re.match(r"@variables\s+(\w+)\(t\)\[([^\]]+)\]", decl.strip())
            if m_vd:
                vname = m_vd.group(1)
                ranges = [r.strip() for r in m_vd.group(2).split(",")]
                dims_from_decl = []
                for spec in ranges:
                    dims_from_decl.append(spec.split(":")[1] if ":" in spec else spec)
                alloc_needed[vname] = dims_from_decl

        # First pass: scan ALL equations (LHS AND RHS) for max literal indices
        all_eq_text = "\n".join(alg_lines)
        # Module-level constants (e.g. GET DIRECT CONSTANTS arrays) are read,
        # not allocated: a local array would hide them
        module_names = set(self._const_names_for_observe(set()))
        for m in re.finditer(r"\b(\w+)\[([^\]]+)\]", all_eq_text):
            name = m.group(1)
            if name in stock_indices or name.startswith("du") or name.startswith("u"):
                continue
            if name in module_names:
                continue
            if "(" in m.group(2):
                # an index expression such as (1, 0, 0, 2)[_i1]: its commas
                # are not dimensions, and it has no literal index to size by
                continue
            indices = [x.strip() for x in m.group(2).split(",")]
            cur = alloc_needed.get(name, [])
            while len(cur) < len(indices):
                cur.append("0")
            for d, idx in enumerate(indices):
                try:
                    val = int(idx)
                    old = int(cur[d]) if cur[d].isdigit() else 0
                    cur[d] = str(max(old, val))
                except ValueError:
                    pass
            alloc_needed[name] = cur

        # Second pass: scan comprehensions for symbolic ranges (N_CONST)
        # Scan ALL equations for indexed LHS assignments
        for eq in alg_lines:
            eq_s = eq.strip().rstrip(",")

            # Comprehension: [var[i, j] ~ ... for i in 1:N, j in 1:M]...
            if eq_s.startswith("["):
                m2 = re.match(r"\[(\w+)\[", eq_s)
                if m2:
                    name = m2.group(1)
                    if name not in stock_indices:
                        all_dims = []
                        # Reconstruct dimension order from for clause
                        for m_for in re.finditer(r"in\s+(?:(\d+:\w+)|\[([^\]]+)\])", _outer_for_clause(eq_s)):
                            if m_for.group(1):
                                all_dims.append(m_for.group(1).split(":")[1])
                            elif m_for.group(2):
                                # A list of element indices ("in [3]" or
                                # "in [1, 3]"): the array needs room up to the
                                # largest one, not one row per listed element.
                                listed = [x.strip() for x in m_for.group(2).split(",")]
                                if all(x.isdigit() for x in listed):
                                    all_dims.append(str(max(int(x) for x in listed)))
                                else:
                                    all_dims.append(str(len(listed)))
                        cur = alloc_needed.get(name, [])
                        if len(all_dims) >= len(cur):
                            # Merge: symbolic N_* wins over literal; larger literal wins
                            merged = []
                            for i, new_d in enumerate(all_dims):
                                old_d = cur[i] if i < len(cur) else "0"
                                is_new_sym = not new_d.isdigit()
                                is_old_sym = not old_d.isdigit()
                                if is_old_sym:
                                    merged.append(old_d)  # keep existing symbolic
                                elif is_new_sym:
                                    merged.append(new_d)  # new symbolic wins
                                else:
                                    merged.append(str(max(int(old_d), int(new_d))))
                            alloc_needed[name] = merged
                continue

            # Individual: var[idx1, idx2] ~ expr
            m = re.match(r"(\w+)\[([^\]]+)\]\s*~", eq_s)
            if m:
                name = m.group(1)
                indices = [x.strip() for x in m.group(2).split(",")]
                if name not in stock_indices:
                    cur = alloc_needed.get(name, [])
                    n_dims = len(indices)
                    # Ensure we have enough dimensions
                    while len(cur) < n_dims:  # pragma: no cover  # first pass always pre-populates alloc_needed to exact size
                        cur.append("0")
                    for d, idx in enumerate(indices):
                        try:
                            val = int(idx)
                            old = int(cur[d]) if cur[d].isdigit() else 0  # pragma: no cover  # cur[d] always "0" or str(int)
                            cur[d] = str(max(old, val))
                        except ValueError:  # pragma: no cover  # per-index equations always have integer indices
                            pass
                    alloc_needed[name] = cur

        # Build alloc lines (shared by rhs! and observe)
        alloc_lines: List[str] = []
        for name, dims in sorted(alloc_needed.items()):
            dims = [d if d != "0" else "100" for d in dims]
            if len(dims) == 1:
                alloc_lines.append(f"    {name} = pysd_safe(zeros({dims[0]}))")
            else:
                dims_str = ", ".join(dims)
                alloc_lines.append(f"    {name} = pysd_safe(zeros({dims_str}))")

        sorted_alg = self._topo_sort_equations(alg_lines, stock_indices)
        return self._ode_model_block(
            stock_indices, stock_sizes, stock_dims, alloc_needed,
            alloc_lines, sorted_alg, ode_lines)

    # ------------------------------------------------------------------
    # ODE model: params(), model!, rhs!, observe
    # ------------------------------------------------------------------

    def _literal_arrays(self, eqs, dynamic, alloc_needed) -> Dict[str, str]:
        """Constants defined element by element with plain numbers covering
        every element (``x[1] = 0.0``, ``x[2] = 1.0``, ...): name -> the
        whole array as one Julia value (``fill(0.0, N_A)`` when all the
        numbers are the same, ``Float64[...]`` otherwise; 1-D and 2-D)."""
        sizes = {}
        for decl in self.subs_const_decls:
            m = re.match(r"const (\w+) = (\d+)", decl)
            if m:
                sizes[m.group(1)] = int(m.group(2))
        number = r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
        parts: Dict[str, Dict[Tuple[int, ...], str]] = {}
        bad: set = set()
        for lhs, _, _, converted, _ in eqs:
            if not lhs or lhs in dynamic or lhs not in alloc_needed:
                continue
            for ln in converted:
                m = re.fullmatch(rf"{lhs}\[(\d+(?:, \d+)*)\] = ({number})", ln.strip())
                if not m:
                    bad.add(lhs)
                    break
                idx = tuple(int(i) for i in m.group(1).split(", "))
                parts.setdefault(lhs, {})[idx] = m.group(2)
        out: Dict[str, str] = {}
        for lhs, cells in parts.items():
            dims = alloc_needed[lhs]
            if lhs in bad or len(dims) > 2:
                continue
            n = [int(d) if d.isdigit() else sizes.get(d) for d in dims]
            if None in n or len(cells) != math.prod(n) or any(len(i) != len(n) for i in cells):
                continue
            if any(not 1 <= i[k] <= n[k] for i in cells for k in range(len(n))):
                continue
            values = {float(v) for v in cells.values()}
            if len(values) == 1:
                out[lhs] = f"pysd_safe(fill({format_number(values.pop())}, {', '.join(dims)}))"
            elif len(n) == 1:
                out[lhs] = "pysd_safe(Float64[" + ", ".join(
                    cells[(i,)] for i in range(1, n[0] + 1)) + "])"
            else:
                rows = ["  ".join(cells[(i, j)] for j in range(1, n[1] + 1))
                        for i in range(1, n[0] + 1)]
                out[lhs] = "pysd_safe(Float64[" + "; ".join(r.replace("  ", " ") for r in rows) + "])"
        return out

    @staticmethod
    def _split_trailing_comment(text: str) -> Tuple[str, str]:
        """``code  # comment`` -> (code, "  # comment"), for a one-line
        declaration; a ``#`` inside a string is not a comment."""
        if "\n" in text:
            return text, ""
        in_str = False
        for k, ch in enumerate(text):
            if ch == '"' and (k == 0 or text[k - 1] != "\\"):
                in_str = not in_str
            elif ch == "#" and not in_str:
                return text[:k].rstrip(), "  " + text[k:]
        return text, ""

    @staticmethod
    def _external_paths(code: str) -> str:
        """Excel and data file paths through ``_file`` and Excel sheets
        through ``_sheet``, so that ``params(sheets=..., files=...)`` can
        read the inputs from other sheets or files."""
        string = r'"(?:[^"\\]|\\.)*"'
        code = re.sub(rf"joinpath\(@__DIR__, ({string})\)", r"_file(\1)", code)
        return re.sub(rf"(pysd_xlsx_\w+\()_file\(({string})\), ({string})",
                      r"\1_file(\2), _sheet(\3)", code)

    def _ode_input_decls(self) -> List[Tuple[Optional[str], str]]:
        """The model's inputs as ``(name, value)``: lookups and data
        (``value`` an anonymous function), then constants and external
        constants.  ``(None, line)`` is a line kept as it is: a comment, or
        an internal value such as an interpolation the lookup calls."""
        out: List[Tuple[Optional[str], str]] = []

        def const(decl: str, internal: bool = False) -> None:
            if decl.startswith("#"):
                out.append((None, decl))
                return
            m = re.match(r"(?:const\s+)?(\w+)\s*=\s*(.*)", decl, re.S)
            if not m:  # pragma: no cover  # every declaration is "name = value"
                out.append((None, decl))
                return
            name, value = m.group(1), m.group(2)
            if internal:
                out.append((None, f"{name} = {value}"))
            else:
                out.append((name, value))

        consts, funcs = self._lookup_decls()
        if consts or funcs:
            out.append((None, "# Lookups and data"))
        for decl in consts:
            const(decl, internal=True)
        for decl in funcs:
            m = re.match(r"(\w+)\((.*?)\)\s*=\s*(.*)", decl, re.S)
            out.append((m.group(1), f"({m.group(2)}) -> {m.group(3)}"))
        decls = self._ode_constant_decls()
        if decls:
            out.append((None, "# Constants"))
        for decl in decls:
            const(decl)
        return out

    def _lookup_decls(self) -> Tuple[List[str], List[str]]:
        """Lookup and data declarations: (``const x_itp = …``, ``x(x) = …``)."""
        if self.data_format != "json":
            return list(self.lookup_const_decls), list(self.lookup_func_decls)
        consts, funcs = [], []
        for kind, xs, ys in (("lookups", "x", "y"), ("data", "time", "values")):
            for key in self._json_data.get(kind, {}):
                consts.append(
                    f'const {key}_itp = LinearInterpolation('
                    f'Float64.(_model_data["{kind}"]["{key}"]["{ys}"]), '
                    f'Float64.(_model_data["{kind}"]["{key}"]["{xs}"]))')
                funcs.append(f"{key}(x) = {key}_itp(x)")
        return consts, funcs

    def _ode_dynamic_names(self, eqs, stock_indices) -> Set[str]:
        """Variables that change during the run: they depend on time, on a
        stock, on random numbers or on the ACTIVE INITIAL switch, directly
        or through other variables.  A variable defined in parts is dynamic
        if any part is."""
        dynamic = set(stock_indices) | {"t"}
        changed = True
        while changed:
            changed = False
            for lhs, rhs_ids, eq, _, _ in eqs:
                if lhs in dynamic:
                    continue
                if (rhs_ids & dynamic or "random" in eq or "_active_initial" in eq
                        or "pysd_active_initial" in eq or re.search(r"\bt\b", eq)):
                    dynamic.add(lhs)
                    changed = True
        return dynamic

    def _ode_model_block(self, stock_indices, stock_sizes, stock_dims,
                         alloc_needed, alloc_lines, sorted_alg, ode_lines) -> str:
        """The ODE model:

        - ``params(; sheets, files, values...)``: the lookups, data and
          constants, and the equations computed from them alone, as a
          NamedTuple; any of them can be given by name;
        - ``_state_axes``: the stocks by name and shape (ComponentArrays);
        - ``model!(d, s, p, t)``: the other equations; writes the rates of
          change into ``d`` (unless ``d`` is ``nothing``) and returns the
          auxiliaries;
        - ``rhs!`` (for the solver) and ``observe`` (every variable at a
          state and time), both over ``model!``.
        """
        eqs = []
        for eq in sorted_alg:
            converted = self._convert_eq_to_assignment(eq)
            lhs = self._extract_lhs_name(eq)
            comment = self._var_comments.get(lhs) if lhs else None
            eqs.append((lhs, self._extract_rhs_identifiers(eq), eq, converted, comment))
        dynamic = self._ode_dynamic_names(eqs, stock_indices)

        const_alloc = [ln for ln in alloc_lines
                       if re.match(r"\s*(\w+)", ln).group(1) not in dynamic]
        dyn_alloc = [ln for ln in alloc_lines
                     if re.match(r"\s*(\w+)", ln).group(1) in dynamic]
        computed, dyn_names = [], []
        for lhs, _, _, _, _ in eqs:
            if not lhs:  # pragma: no cover  # every equation the builder writes names its variable
                continue
            target = dyn_names if lhs in dynamic else computed
            if lhs not in target and lhs not in stock_indices:
                target.append(lhs)
        for name in alloc_needed:
            if name in stock_indices:  # pragma: no cover  # the allocation scan leaves stocks out
                continue
            target = dyn_names if name in dynamic else computed
            if name not in target:
                target.append(name)

        inputs = self._ode_input_decls()
        input_names = [name for name, _ in inputs if name]
        p_names = input_names + [n for n in computed if n not in input_names]
        self._ode_p_names = set(p_names)

        # params(): inputs, then what is computed from them alone
        lines = [
            '"""',
            "    params(; sheets = Dict(), files = Dict(), values...)",
            "",
            "The model's lookups, data and constants, and the variables computed",
            "from them alone, by name.  `values` gives any of them another value",
            "(`name = value`: a number or an array of the variable's size for a",
            "constant; a function, an `(x, y)` series or an interpolation for a",
            "lookup or data); what is computed from them follows.  `sheets` and",
            "`files` read the Excel inputs from other sheets or files (a Dict from",
            "the name in the model to the new one).",
            '"""',
            "function params(; sheets = Dict{String,String}(), files = Dict{String,String}(), values...)",
            "    _given = Dict{Symbol,Any}(values)",
            "    _sheets, _files = Dict{String,String}(sheets), Dict{String,String}(files)",
            "    _file(f) = pysd_file(@__DIR__, f, _files)",
            "    _sheet(s) = get(_sheets, s, s)",
            "",
        ]
        for name, value in inputs:
            value = self._external_paths(value)
            if name is None:
                lines.extend("    " + ln for ln in value.split("\n"))
                continue
            value, comment = self._split_trailing_comment(value)
            text = f"{name} = pysd_input(_given, :{name}, {value}){comment}"
            lines.extend("    " + ln for ln in text.split("\n"))
        last_part = {lhs: k for k, (lhs, *_rest) in enumerate(eqs) if lhs}
        literal = self._literal_arrays(eqs, dynamic, alloc_needed)
        const_alloc = [ln for ln in const_alloc
                       if re.match(r"\s*(\w+)", ln).group(1) not in literal]
        body: List[str] = []
        commented: set = set()
        for k, (lhs, _, _, converted, comment) in enumerate(eqs):
            if lhs in dynamic:
                continue
            if comment and lhs not in commented:
                # a variable defined in parts: its description once
                commented.add(lhs)
                body.append(f"    # {comment}")
            if lhs in literal:
                # numbers for every element: one array
                if last_part[lhs] == k:
                    body.append(f"    {lhs} = pysd_input(_given, :{lhs}, {literal[lhs]})")
                continue
            last = lhs and last_part[lhs] == k and lhs not in stock_indices
            single = (last and len(converted) == 1 and lhs not in alloc_needed
                      and converted[0].startswith(f"{lhs} = ")
                      and [e[0] for e in eqs].count(lhs) == 1)
            if single:
                # one line: the value is the default of the input
                value, trailing = self._split_trailing_comment(converted[0][len(lhs) + 3:])
                body.append(f"    {lhs} = pysd_input(_given, :{lhs}, {value}){trailing}")
                continue
            body.extend(f"    {ln}" for ln in converted)
            if last:
                body.append(f"    {lhs} = pysd_input(_given, :{lhs}, {lhs})")
        if const_alloc or body:
            if lines[-1] != "":
                lines.append("")
            lines.append("    # Computed from the constants")
            lines += const_alloc + _fold_cell_loops(_fold_number_cells(body))
        lines.append("")
        lines += _wrap_list("    p = (; ", p_names, ")")
        lines.append("    pysd_check_inputs(_given, p)")
        fixed = {k: v for k, v in self._fixed_inputs.items() if k in p_names}
        if fixed:
            # values the translation used: a new one needs a new translation
            lines.append("    pysd_check_fixed(_given, (; " + ", ".join(
                f'{k} = "{v}"' for k, v in sorted(fixed.items())) + "))")
        lines.append("    return p")
        lines += ["end", ""]

        # State: named, shaped
        axes = []
        self._ode_stock_ndims = {}
        for name in stock_indices:
            size = stock_sizes[name]
            if size == 1:
                axes.append(f"{name} = 0.0")
                self._ode_stock_ndims[name] = 0
            elif name in stock_dims and len(stock_dims[name]) >= 2:
                axes.append(f"{name} = zeros({', '.join(stock_dims[name])})")
                self._ode_stock_ndims[name] = len(stock_dims[name])
            else:
                axes.append(f"{name} = zeros({size})")
                self._ode_stock_ndims[name] = 1
        lines.append("# Stocks by name and shape; the solver sees one flat vector")
        lines += _wrap_list("const _state_axes = getaxes(ComponentArray(; ", axes, "))")
        lines.append("_named(u) = u isa ComponentArray ? u : ComponentArray(u, _state_axes)")
        lines.append("")

        lines.append("# The rates of change of the stocks (into d, unless it is nothing) and")
        lines.append("# the variables that change during the run, at the state s and time t")
        lines.append("function model!(d, s, p, t)")
        if p_names:
            lines += _wrap_list("    (; ", p_names, ") = p")
        for name in stock_indices:
            lines.append(f"    {name} = s.{name}")
        lines.append("")
        lines += dyn_alloc
        commented = set()
        dyn_lines: List[str] = []
        for lhs, _, _, converted, comment in eqs:
            if lhs not in dynamic:
                continue
            if comment and lhs not in commented:
                commented.add(lhs)
                dyn_lines.append(f"    # {comment}")
            dyn_lines.extend(f"    {ln}" for ln in converted)
        lines += _fold_cell_loops(_fold_number_cells(dyn_lines))
        if ode_lines:
            lines.append("")
            lines.append("    # Rates of change")
            lines.append("    if d !== nothing")
            rate_lines: List[str] = []
            for name in stock_indices:
                if name in stock_dims and len(stock_dims[name]) >= 2:
                    lines.append(f"        du_{name} = d.{name}")
            commented: set = set()
            for eq in ode_lines:
                m_stock = re.match(r"\[?D\((\w+)", eq.strip())
                if m_stock and m_stock.group(1) not in commented:
                    commented.add(m_stock.group(1))
                    if m_stock.group(1) in self._var_comments:
                        rate_lines.append(f"        # {self._var_comments[m_stock.group(1)]}")
                for ln in self._convert_ode_to_du(eq, stock_indices):
                    rate_lines.append("        " + self._du_to_named(ln, stock_indices, stock_sizes))
            lines += _fold_cell_loops(rate_lines)
            lines.append("    end")
        returned = list(dyn_names)
        if self._dynamic_controls():
            # Control variables that change during the run, at this time;
            # INITIAL TIME is the one the run started from
            def ctrl(expr: str) -> str:
                return re.sub(r"\b(time_step|saveper|final_time)\b", r"_ctrl_\1", expr)
            for name, default in (("initial_time", "initial_time"), ("time_step", "1.0"),
                                  ("saveper", "time_step"), ("final_time", "100.0")):
                expr = self.control_vals.get(name) or default
                lines.append(f"    _ctrl_{name} = {ctrl(expr)}")
                returned.append(f"{name} = _ctrl_{name}")
        lines += _wrap_list("    return (; ", returned, ")")
        lines += ["end", ""]

        lines += [
            "rhs!(du, u, p, t) = (model!(_named(du), _named(u), p isa NamedTuple ? p : _params, t); nothing)",
            "# the solver's parameters (SciML takes no parameters as NullParameters)",
            "_problem_p(p) = isempty(p) ? OrdinaryDiffEq.SciMLBase.NullParameters() : p",
            "",
            "# Every variable at a state and time, by name",
            "function observe(u, t; p = _params)",
            "    s = _named(u)",
            "    out = Dict{String,Any}(String(k) => v for (k, v) in pairs(p) if !(v isa Function))",
            "    for k in propertynames(s)",
            "        out[String(k)] = s[k]",
            "    end",
            "    for (k, v) in pairs(model!(nothing, s, p, t))",
            "        out[String(k)] = v",
            "    end",
        ]
        data = sorted(n for n in self._data_names if n in p_names)
        if data:
            lines.append("    # data at this time")
            for name in data:
                dims = self._var_dims.get(name, [])
                if not dims:
                    lines.append(f'    out["{name}"] = p.{name}(t)')
                else:
                    idx = [f"_i{k}" for k in range(len(dims))]
                    ranges = ", ".join(f"{i} in 1:{self._jl_n(d)}" for i, d in zip(idx, dims))
                    lines.append(f'    out["{name}"] = [p.{name}({", ".join(idx)}, t) for {ranges}]')
        if not self._dynamic_controls():
            lines.append("    # simulation control")
            for name in self._CONTROL_NAMES:
                lines.append(f'    out["{name}"] = {name}')
        lines += ["    return out", "end"]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _du_to_named(line: str, stock_indices, stock_sizes) -> str:
        """A derivative assignment by position (``du[12] = …``,
        ``du[35 - 1 + _i0] = …``) written by stock name."""
        starts = {idx: name for name, idx in stock_indices.items()}
        m = re.match(r"(\s*)du\[(\d+) - 1 \+ (.+?)\] = (.*)$", line)
        if m and int(m.group(2)) in starts:
            name = starts[int(m.group(2))]
            if stock_sizes[name] == 1:   # a one-stage pipeline is a scalar stock
                return f"{m.group(1)}d.{name} = {m.group(4)}"
            return f"{m.group(1)}d.{name}[{m.group(3)}] = {m.group(4)}"
        m = re.match(r"(\s*)du\[(\d+)\] = (.*)$", line)
        if m:
            k = int(m.group(2))
            for name, idx in stock_indices.items():
                size = stock_sizes[name]
                if idx <= k < idx + size:
                    if size == 1:
                        return f"{m.group(1)}d.{name} = {m.group(3)}"
                    return f"{m.group(1)}d.{name}[{k - idx + 1}] = {m.group(3)}"
        return line

    def _equations_block_mtk(self, equations: List[str]) -> str:
        if not equations:
            return "eqs = Equation[]\n"
        eq_lines = ",\n    ".join(equations)
        return f"eqs = Equation[\n    {eq_lines},\n]\n"

    @staticmethod
    def _ode_array_constant(name: str, val: str) -> str:
        """``const x = …`` for an array constant declared MTK-style.

        ``name`` is ``x[1:N_A, 1:N_B]`` and ``val`` the values in row-major
        order (as Vensim writes them), possibly followed by a comment.  The
        ODE backend needs a plain Julia array, which is column-major, so the
        values are reshaped with the dimensions reversed and permuted back.
        """
        base, _, spec = name.partition("[")
        sizes = [r.split(":", 1)[-1].strip() for r in spec.rstrip("]").split(",")]
        values, _, comment = val.partition(" #")
        values = values.strip()
        if values.startswith("permutedims("):
            # already an array of the right shape
            comment = f"  #{comment}" if comment else ""
            return f"const {base} = {values}{comment}"
        if values.startswith("[") and not values.startswith("Float64["):
            values = "Float64" + values
        comment = f"  #{comment}" if comment else ""
        if len(sizes) == 1:
            return f"const {base} = {values}{comment}"
        perm = ", ".join(str(k) for k in range(len(sizes), 0, -1))
        return (f"const {base} = permutedims(reshape({values}, "
                f"{', '.join(reversed(sizes))}), ({perm})){comment}")

    def _const_names_for_observe(self, already_known: set) -> List[str]:
        """Return names of module-level consts not yet in the observe Dict.

        Scans param_decls and ext_const_decls for ``@parameters name = ...``
        or ``const name = ...`` lines.  Names in *already_known* (stocks,
        alloc'd arrays, scalar aux) are skipped to avoid duplicates.
        """
        names: List[str] = []
        seen: set = set(already_known)
        for decl in self.param_decls + self.ext_const_decls:
            if decl.startswith("#"):
                continue
            m = re.match(r"(?:@parameters\s+|const\s+)(\w+)", decl)
            if m:
                name = m.group(1)
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    @staticmethod
    def _extract_lhs_name(eq: str) -> Optional[str]:
        """Extract the variable name defined by an equation."""
        eq = eq.strip().rstrip(",")
        if eq.startswith("["):
            inner = eq.strip().lstrip("[").rstrip(".]")
            m = re.match(r"(\w+)\[", inner)
            return m.group(1) if m else None
        m = re.match(r"(\w+)(?:\[.*?\])?\s*~", eq)
        return m.group(1) if m else None

    @staticmethod
    def _extract_rhs_identifiers(eq: str) -> Set[str]:
        """Extract all identifiers referenced on the RHS of an equation."""
        eq = eq.strip().rstrip(",")
        if eq.startswith("#"):
            return set()
        # Split on ~ to get RHS
        parts = eq.split(" ~ ", 1)
        if len(parts) < 2:
            parts = eq.split(" = ", 1)
        rhs = parts[-1] if len(parts) == 2 else eq
        # Find all word tokens (potential variable references)
        tokens = set(re.findall(r"\b([a-z_]\w*)\b", rhs))
        # Remove Julia keywords and numeric-like tokens
        tokens -= {"for", "in", "end", "if", "else", "elseif", "true", "false",
                    "nothing", "Float64", "Int", "sum", "min", "max", "abs",
                    "log", "exp", "sqrt", "sin", "cos", "tan", "mod", "inv",
                    "fill", "vec", "reshape", "permutedims", "clamp", "floor",
                    "prod", "maximum", "minimum", "length", "float"}
        # Remove PySD helper functions
        tokens -= {t for t in tokens if t.startswith("pysd_")}
        return tokens

    def _topo_sort_equations(
        self, equations: List[str], stock_names: dict
    ) -> List[str]:
        """Topologically sort algebraic equations so each variable is defined
        before it's used.

        Variables that are stocks (in ``stock_names``), parameters (in
        ``param_decls``/``ext_const_decls``), lookup functions, or constants
        are considered "available" and don't need to be sorted.
        """
        # Collect names that are already available (stocks, params, lookups, etc.)
        available = set(stock_names.keys())
        available.add("t")
        available.add("time_step")
        available.add("initial_time")
        available.add("final_time")
        for decl in self.param_decls:
            m = re.match(r"@parameters\s+(\w+)", decl)
            if m:
                available.add(m.group(1))
        for decl in self.ext_const_decls:
            m = re.match(r"const\s+(\w+)", decl)
            if m:
                available.add(m.group(1))
        for decl in self.subs_const_decls:
            m = re.match(r"const\s+(\w+)", decl)
            if m:
                available.add(m.group(1))
        available.update(self.lookup_identifiers)
        # Lookup function names (from func_decls like "name(x) = ...")
        for decl in self.lookup_func_decls:
            m = re.match(r"(\w+)\(", decl)
            if m:
                available.add(m.group(1))

        # Build graph: eq_index -> (lhs_name, set of dependencies)
        eq_lhs = []
        eq_deps = []
        for eq in equations:
            lhs = self._extract_lhs_name(eq)
            rhs_ids = self._extract_rhs_identifiers(eq)
            # Dependencies = RHS identifiers that are NOT available
            deps = rhs_ids - available
            eq_lhs.append(lhs)
            eq_deps.append(deps)

        # Build name -> equation indices map: a variable defined element by
        # element has one equation per part, and all of them come before
        # the equations that use it
        name_to_idx: Dict[str, List[int]] = {}
        for i, lhs in enumerate(eq_lhs):
            if lhs:
                name_to_idx.setdefault(lhs, []).append(i)

        # Kahn's algorithm for topological sort
        n = len(equations)
        in_degree = [0] * n
        dependents: List[List[int]] = [[] for _ in range(n)]

        for i in range(n):
            resolved_deps = set()
            for dep in eq_deps[i]:
                for j in name_to_idx.get(dep, ()):
                    # parts of the same variable keep their order
                    if dep == eq_lhs[i] and j > i:
                        continue
                    if j != i and j not in resolved_deps:
                        dependents[j].append(i)
                        in_degree[i] += 1
                        resolved_deps.add(j)

        from collections import deque
        queue = deque(i for i in range(n) if in_degree[i] == 0)
        sorted_order = []

        while queue:
            i = queue.popleft()
            sorted_order.append(i)
            for j in dependents[i]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)

        # Any remaining equations have circular dependencies — append them at end
        if len(sorted_order) < n:
            remaining = [i for i in range(n) if i not in set(sorted_order)]
            sorted_order.extend(remaining)

        return [equations[i] for i in sorted_order]

    def _convert_eq_to_assignment(self, eq: str) -> List[str]:
        """Convert 'var ~ expr' to 'var = expr'."""
        eq = eq.strip().rstrip(",")
        # Handle comprehension: [var[i] ~ expr for _i in 1:N]...
        if eq.startswith("["):
            # Strip exactly: outer "[", then trailing "...", then outer "]".
            # Using rstrip(".]") is wrong for multi-dim for-clauses that contain
            # list ranges like [1, 2, 3] — those brackets would also be stripped.
            _eq = eq.strip()
            if _eq.endswith("..."):
                _eq = _eq[:-3]
            if _eq.endswith("]"):
                _eq = _eq[:-1]
            inner = _eq.lstrip("[")
            # Find the outer "for" clause — the one NOT inside brackets.
            # Walk backwards to find "for" at bracket depth 0.
            for_pos = None
            depth = 0
            for i in range(len(inner) - 1, 3, -1):
                c = inner[i]
                if c in ")]":
                    depth += 1
                elif c in "([":
                    depth -= 1
                elif depth == 0 and inner[i:i+4] == "for " and inner[i-1] == " ":
                    for_pos = i
                    break
            if for_pos is not None:
                for_clause = inner[for_pos + 4:]
                body = inner[:for_pos].rstrip()
                body = body.replace(" ~ ", " = ", 1)
                return [
                    f"for {for_clause}",
                    f"    {body}",
                    "end",
                ]
            raise ValueError(
                f"Cannot convert comprehension equation to assignment: no outer "
                f"'for' clause found at depth 0 in: {eq!r}"
            )
        return [eq.replace(" ~ ", " = ", 1)]

    def _convert_ode_to_du(self, eq: str, stock_indices: dict) -> List[str]:
        """Convert 'D(var) ~ expr' to 'du[i] = expr'."""
        eq = eq.strip().rstrip(",")
        # Handle comprehension: [D(var[i]) ~ expr for i in 1:N]...
        if eq.startswith("["):
            _eq = eq.strip()
            if _eq.endswith("..."):
                _eq = _eq[:-3]
            if _eq.endswith("]"):
                _eq = _eq[:-1]
            inner = _eq.lstrip("[")
            # Find the outer "for" at bracket depth 0
            for_pos = None
            depth = 0
            for i in range(len(inner) - 1, 3, -1):
                c = inner[i]
                if c in ")]":
                    depth += 1
                elif c in "([":
                    depth -= 1
                elif depth == 0 and inner[i:i+4] == "for " and inner[i-1] == " ":
                    for_pos = i
                    break
            if for_pos is not None:
                for_clause = inner[for_pos + 4:]
                body = inner[:for_pos].rstrip()
                m_d = re.match(r"D\((\w+)\[([^\]]+)\]\)\s*~\s*(.*)", body)
                if m_d:
                    var_name = m_d.group(1)
                    idx_expr = m_d.group(2)
                    rhs = m_d.group(3)
                    # Use reshaped view for multi-dim, flat index for 1D
                    if "," in idx_expr:
                        return [
                            f"for {for_clause}",
                            f"    du_{var_name}[{idx_expr}] = {rhs}",
                            "end",
                        ]
                    else:
                        base_idx = stock_indices.get(var_name, 1)
                        return [
                            f"for {for_clause}",
                            f"    du[{base_idx} - 1 + {idx_expr}] = {rhs}",
                            "end",
                        ]
            return [eq.replace(" ~ ", " = ")]

        # Simple scalar: D(var) ~ expr or D(var[N]) ~ expr
        m = re.match(r"D\((\w+)(?:\[(\d+)\])?\)\s*~\s*(.*)", eq)
        if m:
            var_name = m.group(1)
            idx_str = m.group(2)
            expr = m.group(3)
            if idx_str:
                base_idx = stock_indices.get(var_name, 1)
                offset = int(idx_str) - 1
                return [f"du[{base_idx + offset}] = {expr}"]
            else:
                idx = stock_indices.get(var_name, 1)
                return [f"du[{idx}] = {expr}"]
        return [eq.replace(" ~ ", " = ", 1)]

    @staticmethod
    def _u0_size(entry: str) -> int:
        """Number of state cells of a u0 entry: ``x[1:N, 2] => v`` gives N
        cells (all stages of a DELAY FIXED pipeline), other entries one."""
        lhs = entry.split("=>", 1)[0]
        size = 1
        for a, b in re.findall(r"\b(\d+):(\d+)\b", lhs):
            size *= int(b) - int(a) + 1
        return size

    def _sort_u0_entries(self) -> None:
        """Order ``u0_entries`` the way the ODE state vector is read back.

        Entries of each stock are kept together (stocks in first-appearance
        order) and multi-dimensional stocks are listed in Julia's column-major
        order, so ``reshape(@view(u[a:b]), n0, n1, ...)`` in ``rhs!`` and
        ``observe`` maps every element to its own initial value.
        """
        groups: Dict[str, List[Tuple[Tuple[int, ...], str]]] = {}
        for entry in self.u0_entries:
            lhs = entry.split("=>", 1)[0].strip()
            base, _, idx = lhs.partition("[")
            try:
                key = tuple(
                    0 if ":" in i else int(i) for i in idx.rstrip("]").split(",")
                ) if idx else ()
            except ValueError:  # pragma: no cover  # u0 indices are always integer literals
                key = ()
            groups.setdefault(base, []).append((key[::-1], entry))
        self.u0_entries = [
            entry
            for items in groups.values()
            for _, entry in sorted(items, key=lambda item: item[0])
        ]

    def _u0_assignments(self, u0_entries: List[str], target: str) -> List[str]:
        """``target.stock = value`` for every stock: the model's stocks, then
        the internal ones.  A stock whose cells follow one formula is one
        line (``target.x .= 0.0``, ``target.x .= [src[i] for i in 1:14]``);
        otherwise one line per cell."""
        groups: List[Tuple[str, List[Tuple[List[str], str]]]] = []
        for entry in u0_entries:
            lhs, rhs = (x.strip() for x in entry.split("=>", 1))
            base, _, idx = lhs.partition("[")
            index = [i.strip() for i in idx.rstrip("]").split(",")] if idx else []
            if not groups or groups[-1][0] != base:
                groups.append((base, []))
            groups[-1][1].append((index, rhs))
        own = [g for g in groups if not g[0].startswith("_")]
        internal = [g for g in groups if g[0].startswith("_")]
        lines: List[str] = []
        for base, cells in own:
            lines += self._u0_stock_lines(target, base, cells)
        if internal:
            lines.append("# the translation's own stocks (DELAY FIXED stages, SAMPLE IF TRUE, ...)")
            for base, cells in internal:
                lines += self._u0_stock_lines(target, base, cells)
        return lines

    def _u0_stock_lines(self, target: str, base: str, cells) -> List[str]:
        ndims = self._ode_stock_ndims.get(base, 1)
        name = f"{target}.{base}"
        if ndims == 0:
            return [f"{name} = {cells[0][1]}"]
        pattern = self._u0_pattern(cells)
        if pattern is not None:
            kind, value, iterators = pattern
            if kind == "same":
                return [f"{name} .= {value}"]
            if kind == "numbers":
                return [f"{name} .= [{value}]"]
            comprehension = f"[{value} for {', '.join(iterators)}]"
            if ndims != len(iterators):
                comprehension = f"vec({comprehension})"
            return [f"{name} .= {comprehension}"]
        # one line per cell, by its indices (or its position in a flat stock)
        out, position = [], 1
        for index, rhs in cells:
            size = 1
            for i in index:
                if ":" in i:
                    a, _, b = i.partition(":")
                    size *= int(b) - int(a) + 1
            if ndims == len(index):
                op = " .= " if size > 1 else " = "
                out.append(f"{name}[{', '.join(index)}]{op}{rhs}")
            elif size > 1:
                out.append(f"{name}[{position}:{position + size - 1}] .= {rhs}")
            else:
                out.append(f"{name}[{position}] = {rhs}")
            position += size
        return out

    @staticmethod
    def _u0_pattern(cells):
        """The cells of a stock as one formula: ("same", value, []) when all
        are the same value, ("each", formula, iterators) when they follow one
        formula of their indices (i, j, k; the first index fastest, the order
        of the stock's cells), ("numbers", "1.0, 2.0", []) for different
        numbers of a vector; or None."""
        first = cells[0][0]
        n = len(first)
        if n == 0 or any(len(index) != n for index, _ in cells):
            return None
        ranges = {k: v for k, v in enumerate(first) if ":" in v}
        if any(any((":" in index[k]) != (k in ranges) or (k in ranges and index[k] != ranges[k])
                   for k in range(n)) for index, _ in cells):
            return None
        ints = [k for k in range(n) if k not in ranges]
        try:
            values = [[int(index[k]) for k in ints] for index, _ in cells]
        except ValueError:
            return None
        extents = [max(v[j] for v in values) for j in range(len(ints))]
        expected = [list(reversed(c)) for c in itertools.product(
            *[range(1, e + 1) for e in reversed(extents)])]
        if values != expected:
            return None
        # each integer of the formulas is the same in every cell (a number of
        # the formula) or the cell's index at some position (a loop variable)
        token = r"(?<![\w.])\d+(?![\w.])"
        number = r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
        if len({re.sub(token, "#", rhs) for _, rhs in cells}) != 1:
            if not ranges and len(ints) == 1 and all(re.fullmatch(number, rhs) for _, rhs in cells):
                return ("numbers", ", ".join(rhs for _, rhs in cells), [])
            return None
        names = ["i", "j", "k", "l"]
        if len(ints) > len(names) or any(re.search(rf"\b{v}\b", cells[0][1]) for v in names):
            names = [f"_i{j}" for j in range(len(ints))]
        loop = {k: names[j] for j, k in enumerate(ints)}
        found = [[int(x) for x in re.findall(token, rhs)] for _, rhs in cells]
        parts = re.split(token, cells[0][1])
        formula = parts[0]
        used = False
        for j in range(len(found[0])):
            column = [f[j] for f in found]
            if len(set(column)) == 1:
                text = str(column[0])
            else:
                k = next((k for k in ints
                          if all(c == v[ints.index(k)] for c, v in zip(column, values))), None)
                if k is None:
                    return None
                text, used = loop[k], True
            formula += text + parts[j + 1]
        if not used:
            return ("same", formula, [])
        iterators = [f"_ in {ranges[k]}" if k in ranges
                     else f"{loop[k]} in 1:{extents[ints.index(k)]}" for k in range(n)]
        return ("each", formula, iterators)

    def _u0_block(self) -> str:
        if self.backend == "mtk":
            return self._u0_block_mtk()
        if not self.u0_entries:
            return ("initial_state(p = _params) = Float64[]\n\n"
                    "u0 = initial_state()\n")

        # Initial values are evaluated at the initial time: Time (t) in them,
        # e.g. DELAY1(Time, d) starting at Time * d, is the initial time
        u0_entries = [
            entry if "=>" not in entry else
            entry.split("=>", 1)[0] + "=> "
            + re.sub(r"\bt\b", "initial_time", entry.split("=>", 1)[1].strip())
            for entry in self.u0_entries
        ]
        # Names of variables in an expression; names that are called
        # (abs(...), mod(...)) are functions
        var_re = r"\b([a-z_]\w*)\b(?!\s*\()"
        inputs = self._ode_p_names | {name for name, _ in self._ode_input_decls() if name}
        controls = {"time_step", "initial_time", "final_time", "saveper", "t"}
        used: set = set()
        called: set = set()      # lookups and data, called as functions
        for entry in u0_entries:
            value = entry.split("=>", 1)[1]
            used |= set(re.findall(var_re, value))
            called |= set(re.findall(r"\b([a-z_]\w*)\s*\(", value))
        from_p = sorted((used | called) & inputs)
        # Other variables (auxiliaries, stocks) are read from observe at the
        # initial time.  Initial values may depend on stocks (e.g. DELAY(stock,
        # 1) starts at the stock's initial value), so they are evaluated from
        # the state they define, starting from zeros, until it no longer
        # changes.
        from_observe = sorted(used - inputs - controls)

        # local names for the state: s and next, unless the model uses them
        taken = used | called | set(from_p) | set(from_observe)
        sv, nv = ("s", "next") if not {"s", "next"} & taken else ("_s", "_next")
        n = sum(self._u0_size(entry) for entry in u0_entries)
        lines = [
            "# The stocks at the initial time, with the constants p",
            "function initial_state(p = _params)",
        ]
        body: List[str] = []
        if from_p:
            body += _wrap_list("(; ", from_p, ") = p")
        body.append(f"{sv} = ComponentArray(zeros({n}), _state_axes)")
        if not from_observe:
            body += self._u0_assignments(u0_entries, sv)
            body.append(f"return getdata({sv})")
        else:
            n_stocks = len({e.split("=>", 1)[0].split("[")[0].strip() for e in u0_entries})
            body += [
                "# Initial values can depend on other stocks' initial values: they",
                "# are computed again until they settle (each pass settles at least",
                "# one more stock)",
                f"for _ in 1:{n_stocks + 1}",
                f"    _obs_init = observe({sv}, initial_time; p = p)",
            ]
            body += [f"    {tok} = get(_obs_init, \"{tok}\", 0.0)" for tok in from_observe]
            body.append(f"    {nv} = zero({sv})")
            body += ["    " + ln for ln in self._u0_assignments(u0_entries, nv)]
            body += [f"    isequal({nv}, {sv}) && return getdata({nv})", f"    {sv} = {nv}", "end",
                     'error("the initial values do not settle: they depend on each other in a cycle")']
        if self._ode_active_initial:
            # ACTIVE INITIAL gives its initial value while the stocks start
            body = (["_initialising[] = true", "try"] + ["    " + ln for ln in body]
                    + ["finally", "    _initialising[] = false", "end"])
        lines += ["    " + ln for ln in body] + ["end"]
        lines += ["", "u0 = initial_state()"]
        return "\n".join(lines) + "\n"

    def _u0_block_mtk(self) -> str:
        if not self.u0_entries:
            return "u0 = Pair{Num, Float64}[]\n"
        # Build param name → numeric value map from param_decls
        # "@parameters name = value" entries
        param_values: Dict[str, str] = {}
        for decl in self.param_decls:
            if decl.startswith("@parameters "):
                rest = decl[len("@parameters "):]
                m = re.match(r"(\w+)\s*=\s*(.+)", rest)
                if m:
                    param_values[m.group(1)] = m.group(2).strip()
        lines = []
        for entry in self.u0_entries:
            if "=>" in entry:
                lhs, rhs = entry.split("=>", 1)
                rhs = rhs.strip()
                m_rng = re.match(r"\s*(\w+)\[(\d+):(\d+)(.*)\]\s*$", lhs)
                if m_rng:  # all stages of a pipeline cell
                    for pname, pval in param_values.items():
                        rhs = re.sub(rf"\b{re.escape(pname)}\b", pval, rhs)
                    name, a, b, rest = m_rng.groups()
                    for k in range(int(a), int(b) + 1):
                        lines.append(f"    {name}[{k}{rest}] => {rhs},")
                    continue
                # Substitute param references with numeric values
                for pname, pval in param_values.items():
                    rhs = re.sub(rf"\b{re.escape(pname)}\b", pval, rhs)
                lines.append(f"    {lhs.strip()} => {rhs},")
            else:  # pragma: no cover  # all u0_entries use "name => expr" format
                lines.append(f"    {entry},")
        return "u0 = [\n" + "\n".join(lines) + "\n]\n"

    _CONTROL_NAMES = ("initial_time", "final_time", "time_step", "saveper")

    def _control_unknowns(self, expr: str) -> set:
        """Names in a control expression that are not module-level values."""
        module_names = set(self._const_names_for_observe(set()))
        names = set(re.findall(r"\b([a-z_]\w*)\b(?!\s*\()", expr))
        return names - set(self._CONTROL_NAMES) - module_names

    def _dynamic_controls(self) -> bool:
        """Whether a control variable depends on Time or on model variables
        (e.g. FINAL TIME = IF THEN ELSE(x > 50, Time, 100)).  The ODE
        backend then evaluates them at every step, as Python PySD."""
        return self.backend == "ode" and any(
            self._control_unknowns(self.control_vals.get(name) or "")
            for name in self._CONTROL_NAMES
        )

    def _control_block(self) -> str:
        it = self.control_vals.get("initial_time") or "0.0"
        ft = self.control_vals.get("final_time") or "100.0"
        ts = self.control_vals.get("time_step") or "1.0"
        sp = self.control_vals.get("saveper") or "time_step"
        if self._dynamic_controls():
            def at_start(expr: str, when: str, default: str) -> str:
                if self._control_unknowns(expr) <= {"t"}:
                    return re.sub(r"\bt\b", when, expr)
                return default
            return (
                "# Simulation control.  These change during the run: observe and\n"
                "# run_model evaluate them at every time; the values here are the\n"
                "# ones at the initial time (set again from the initial state below).\n"
                f"initial_time = {at_start(it, '0.0', '0.0')}\n"
                f"time_step    = {at_start(ts, 'initial_time', '1.0')}\n"
                f"saveper      = {at_start(sp, 'initial_time', 'time_step')}\n"
                f"final_time   = {at_start(ft, 'initial_time', 'initial_time')}\n"
                "tspan = (initial_time, final_time)\n"
            )
        return (
            "# Simulation control\n"
            f"initial_time = {it}\n"
            f"final_time   = {ft}\n"
            f"time_step    = {ts}\n"
            f"saveper      = {sp}\n"
            "tspan = (initial_time, final_time)\n"
        )

    def _run_function(self) -> str:
        if self.backend == "mtk":
            return self._run_function_mtk()
        ts = self.control_vals.get("time_step") or "time_step"
        has_tab = bool(self._tab_data_entries)
        tab_param = ", tab_data_files=String[], nc_data_files=String[]" if has_tab else ""
        tab_load = "{TAB_LOAD}" if has_tab else ""
        tab_lines = (
            "\n    isempty(tab_data_files) || _load_tab_data!(tab_data_files)"
            "\n    isempty(nc_data_files) || _load_nc_data!(nc_data_files)"
        )
        doc = textwrap.dedent("""\
            \"\"\"
                run_model(; params = _params, u0 = initial_state(params), ...)

            Simulate the model.  `params` are its constants, lookups and data:
            `run_model(params = params(name = value, ...))` runs it with other
            values.  The stocks start from `initial_state(params)` unless `u0`
            is given.  Other keywords go to `solve` (e.g. `saveat`).
            \"\"\"
            """)
        if self._dynamic_controls():
            return textwrap.dedent(f"""\
                let _c = observe(u0, initial_time)
                    global time_step = _c["time_step"]
                    global saveper = _c["saveper"]
                    global final_time = _c["final_time"]
                end
                tspan = (initial_time, final_time)
                prob = ODEProblem(rhs!, u0, tspan, _problem_p(_params))

                # The control variables change during the run, so (as Python
                # PySD) the model takes Euler steps of the current TIME STEP
                # until Time reaches the current FINAL TIME, and saves every
                # current SAVEPER from the initial time.
                """) + doc + textwrap.dedent(f"""\
                function run_model(; params=_params, u0=nothing{tab_param}){tab_load}
                    u = u0 === nothing ? initial_state(params) : copy(u0)
                    du = similar(u)
                    t = initial_time
                    ts = Float64[]
                    us = Vector{{typeof(u)}}()
                    while true
                        c = observe(u, t; p = params)
                        prec = c["time_step"] * 1e-5
                        elapsed = t - initial_time
                        if mod(elapsed, c["saveper"]) < prec || mod(-elapsed, c["saveper"]) < prec
                            push!(ts, t)
                            push!(us, copy(u))
                        end
                        t + prec < c["final_time"] || break
                        rhs!(du, u, params, t)
                        u .+= c["time_step"] .* du
                        t += c["time_step"]
                    end
                    return OrdinaryDiffEq.SciMLBase.DiffEqArray(us, ts)
                end
                """).replace("{TAB_LOAD}", tab_lines)
        return "prob = ODEProblem(rhs!, u0, tspan, _problem_p(_params))\n\n" + doc + textwrap.dedent(f"""\
            function run_model(; params=_params, u0=nothing, tspan=tspan, dt={ts}, solver=Euler(){tab_param}, kwargs...){tab_load}
                u0 === nothing && (u0 = initial_state(params))
                solve(ODEProblem(rhs!, u0, tspan, _problem_p(params)), solver; dt=dt, adaptive=false, kwargs...)
            end
            """).replace("{TAB_LOAD}", tab_lines)

    def _run_function_mtk(self) -> str:
        ts = self.control_vals.get("time_step") or "time_step"
        # The unknowns are matched to u0 by name (structural_simplify may
        # reorder them, and may add unknowns from algebraic loops, which start
        # at 0.0).  saveat is needed so that a system without unknowns is
        # still saved at every step.
        return textwrap.dedent(f"""\
            u0_map = Dict(Symbolics.unwrap(k) => v for (k, v) in u0)
            u0_full = isempty(unknowns(sys)) ? u0 :
                [x => get(u0_map, x, 0.0) for x in unknowns(sys)]
            prob = ODEProblem(sys, u0_full, tspan; build_initializeprob = false)

            function run_model(; u0=u0_full, tspan=tspan, dt={ts}, solver=Euler())
                prob_local = remake(prob; u0=u0, tspan=tspan)
                solve(prob_local, solver; dt=dt, saveat=tspan[1]:dt:tspan[2], adaptive=false)
            end
            """)

    def _entrypoint_block(self) -> str:
        """Generate the calls that run the model and save results.

        They only run when the file is executed (``julia model.jl``), so
        ``include("model.jl")`` defines the model without simulating it.
        """
        nc_name = f"{self.model_name}_results.nc"
        if self.backend == "mtk":
            save_call = f'save_results(sol, sys, _dim_labels, joinpath(@__DIR__, "{nc_name}"))'
        else:
            save_call = f'save_results(sol, _state_map, _dim_labels, joinpath(@__DIR__, "{nc_name}"))'
        return textwrap.dedent(f"""\
            if abspath(PROGRAM_FILE) == @__FILE__
                println("Running model…")
                sol = run_model()
                println("Saving results to {nc_name}…")
                {save_call}
                println("Done.")
            end
            """)

    def _dim_labels_block(self) -> str:
        """Emit ``const _dim_labels = Dict(...)`` for all known subscript ranges."""
        if not self._subs_elems:
            return "const _dim_labels = Dict{String,Vector{String}}()\n"
        entry_lines = []
        for dim_name in sorted(self._subs_elems):
            labels = self._subs_elems[dim_name]
            labels_jl = ", ".join(f'"{lbl}"' for lbl in labels)
            entry_lines.append(f'    "{dim_name}" => [{labels_jl}],')
        return "const _dim_labels = Dict(\n" + "\n".join(entry_lines) + "\n)\n"

    def _state_map_block(self) -> str:
        """Emit ``const _state_map`` for ODE backend.

        Each entry is ``(name, start_index, [dim_names])``.
        """
        if not self.u0_entries:
            return "const _state_map = Tuple{String,Int,Vector{String}}[]\n"

        # Parse u0_entries to collect (base_name, size) in order
        stock_order: List[str] = []
        stock_sizes: Dict[str, int] = {}
        for entry in self.u0_entries:
            lhs = entry.split("=>")[0].strip()
            base = lhs.split("[")[0]
            if base not in stock_sizes:
                stock_order.append(base)
                stock_sizes[base] = 0
            stock_sizes[base] += self._u0_size(entry)

        lines = ["const _state_map = ["]
        idx = 1
        for name in stock_order:
            size = stock_sizes[name]
            dim_names = self._var_dims.get(name, [])
            # The model's own stocks, by their subscript ranges.  Internal
            # stocks (_df_pipe_*, _sit_*, ...: the stages of a DELAY FIXED,
            # the value a SAMPLE IF TRUE holds) are not results, as in
            # Python PySD, and their cells are not laid out over ranges.
            if not name.startswith("_") and size == math.prod(
                    self._subs_sizes.get(d, 0) for d in dim_names):
                dims_jl = ", ".join(f'"{d}"' for d in dim_names)
                lines.append(f'    ("{name}", {idx}, String[{dims_jl}]),')
            idx += size
        lines.append("]")
        return "\n".join(lines) + "\n"

    def _save_results_function(self) -> str:
        """Emit _dim_labels (and _state_map for ODE) so PySD.save_results can be called."""
        parts = [self._dim_labels_block()]
        if self.backend == "ode":
            parts.append(self._state_map_block())
        return "".join(parts)

    def _system_block(self) -> str:
        if self.backend == "mtk":
            return (
                "@named sys = ODESystem(eqs, t)\n"
                "sys = structural_simplify(sys)\n"
            )
        return ""

    def _full_file_content(self, equations: List[str]) -> str:
        needs_di = bool(self.lookup_const_decls)
        equations_block = self._equations_block(equations)
        self._ode_active_initial = (self.backend == "ode"
                                    and "pysd_active_initial(" in equations_block)
        u0_block = self._u0_block()
        active_initial = ""
        if self._ode_active_initial:
            # ACTIVE INITIAL(expr, initial) is `initial` while the initial
            # values are computed and `expr` afterwards, as in Python PySD
            active_initial = (
                "# ACTIVE INITIAL: its initial value while the initial values\n"
                "# are computed, its active value afterwards\n"
                "const _initialising = Ref(false)\n"
                "_active_initial(expr, initial) = _initialising[] ? initial : expr\n\n"
            )
            equations_block = equations_block.replace("pysd_active_initial(", "_active_initial(")
            u0_block = u0_block.replace("pysd_active_initial(", "_active_initial(")
        if self.backend == "ode":
            # the constants are computed once, after the simulation control
            # (which equations may use)
            u0_block = "const _params = params()\n\n" + u0_block
        return "".join([
            self._file_header(extra_packages=needs_di),
            self._helpers_block(),
            self._tab_data_block(),
            self._lookup_block(),
            self._declarations_block(),
            self._control_block(),
            "\n",
            active_initial,
            equations_block,
            "\n",
            u0_block,
            "\n",
            self._system_block(),
            "\n",
            self._run_function(),
            self._save_results_function(),
            self._entrypoint_block(),
        ])

    def _modular_main_content(
        self,
        include_lines: List[str],
        eq_var_names: List[str],
        leftover_eqs: List[str],
    ) -> str:
        needs_di = bool(self.lookup_const_decls)
        include_block = "\n# Module includes\n" + "\n".join(include_lines) + "\n"

        leftover_block = ""
        if leftover_eqs:
            lines = ",\n    ".join(leftover_eqs)
            leftover_block = f"\n_main_eqs = Equation[\n    {lines},\n]\n"
            eq_var_names = list(eq_var_names) + ["_main_eqs"]

        if eq_var_names:
            concat = "; ".join(f"{v}..." for v in eq_var_names)
            combined = f"eqs = [{concat}]\n"
        else:
            combined = "eqs = Equation[]\n"

        return "".join([
            self._file_header(extra_packages=needs_di),
            self._helpers_block(),
            self._tab_data_block(),
            self._lookup_block(),
            self._declarations_block(),
            # Control variables (time_step, initial_time, …) must be defined
            # before the module includes so equations can reference them.
            self._control_block(),
            "\n",
            include_block,
            leftover_block,
            "\n",
            combined,
            "\n",
            self._u0_block(),
            "\n",
            self._system_block(),
            "\n",
            self._run_function(),
            self._save_results_function(),
            self._entrypoint_block(),
        ])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_NUMBER = r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER_CELL = re.compile(rf"(\s*)(\w+)\[(\d+)\] = ({_NUMBER})")


def _fold_number_cells(lines: List[str]) -> List[str]:
    """Consecutive number assignments to cells of one vector as one line:
    ``x[1] = 0.0; x[2] = 0.0`` -> ``x[1:2] .= 0.0``; ``x[3] = 0.0;
    x[5] = 0.0`` -> ``x[[3, 5]] .= 0.0``; different numbers ->
    ``x[1:3] .= [1.0, 2.0, 3.0]``."""
    out: List[str] = []
    k = 0
    while k < len(lines):
        m = _NUMBER_CELL.fullmatch(lines[k])
        if not m:
            out.append(lines[k])
            k += 1
            continue
        indent, name = m.group(1), m.group(2)
        run = [(int(m.group(3)), m.group(4))]
        j = k + 1
        while j < len(lines):
            m2 = _NUMBER_CELL.fullmatch(lines[j])
            if not m2 or m2.group(1) != indent or m2.group(2) != name:
                break
            run.append((int(m2.group(3)), m2.group(4)))
            j += 1
        idx = [i for i, _ in run]
        if len(run) == 1 or len(set(idx)) != len(idx):
            out.extend(lines[k:j])
        else:
            if idx == list(range(idx[0], idx[0] + len(idx))):
                where = f"{idx[0]}:{idx[-1]}"
            else:
                where = "[" + ", ".join(str(i) for i in idx) + "]"
            values = [v for _, v in run]
            value = values[0] if len(set(float(v) for v in values)) == 1 else \
                "[" + ", ".join(values) + "]"
            out.append(f"{indent}{name}[{where}] .= {value}")
        k = j
    return out


_CELL_LINE = re.compile(r"(\s*)([\w.]+)\[(\d+)\] = (.+)")
_INT_TOKEN = r"(?<![\w.])\d+(?![\w.])"


def _fold_cell_loops(lines: List[str]) -> List[str]:
    """Consecutive assignments to consecutive cells of one vector that
    follow one formula of the cell's index, as a loop:
    ``d.x[2] = f[2] - f[3]; d.x[3] = f[3] - f[4]`` ->
    ``for i in 2:3  d.x[i] = f[i] - f[i + 1]  end``.  Each integer of the
    formulas is the same in every line, or the index plus a constant."""
    out: List[str] = []
    k = 0
    while k < len(lines):
        m = _CELL_LINE.fullmatch(lines[k])
        if not m or " = " in m.group(4):
            out.append(lines[k])
            k += 1
            continue
        indent, name = m.group(1), m.group(2)
        run = [(int(m.group(3)), m.group(4))]
        j = k + 1
        while j < len(lines):
            m2 = _CELL_LINE.fullmatch(lines[j])
            if (not m2 or m2.group(1) != indent or m2.group(2) != name
                    or int(m2.group(3)) != run[-1][0] + 1):
                break
            run.append((int(m2.group(3)), m2.group(4)))
            j += 1
        # groups of consecutive cells with the same formula (the last cell
        # may differ, as x[n] = f[n] after x[i] = f[i] - f[i + 1])
        start = 0
        while start < len(run):
            skeleton = re.sub(_INT_TOKEN, "#", run[start][1])
            end = start + 1
            while end < len(run) and re.sub(_INT_TOKEN, "#", run[end][1]) == skeleton:
                end += 1
            group = run[start:end]
            loop = _loop_formula(group) if len(group) >= 3 else None
            if loop is None:
                out.extend(lines[k + start:k + end])
            else:
                var, formula = loop
                out += [f"{indent}for {var} in {group[0][0]}:{group[-1][0]}",
                        f"{indent}    {name}[{var}] = {formula}",
                        f"{indent}end"]
            start = end
        k = j
    return out


def _loop_formula(run) -> Optional[Tuple[str, str]]:
    """The formula of cells (index, formula) as one formula of a loop
    variable, or None."""
    texts = [f for _, f in run]
    if len({re.sub(_INT_TOKEN, "#", f) for f in texts}) != 1:
        return None
    var = "i" if not any(re.search(r"\bi\b", f) for f in texts) else "_i"
    found = [[int(x) for x in re.findall(_INT_TOKEN, f)] for f in texts]
    parts = re.split(_INT_TOKEN, texts[0])
    formula, used = parts[0], False
    for c in range(len(found[0])):
        column = [f[c] for f in found]
        offsets = {v - idx for v, (idx, _) in zip(column, run)}
        if len(set(column)) == 1:
            piece = str(column[0])
        elif len(offsets) == 1:
            off = offsets.pop()
            piece = var if off == 0 else (f"{var} + {off}" if off > 0 else f"{var} - {-off}")
            used = True
        else:
            return None
        formula += piece + parts[c + 1]
    return (var, formula) if used else None


def _wrap_list(start: str, items: List[str], end: str, width: int = 92) -> List[str]:
    """``start`` + items separated by commas + ``end``, wrapped at *width*,
    the continuation lines aligned with the first item."""
    indent = " " * len(start)
    lines, line = [], start
    for k, item in enumerate(items):
        text = item + ("," if k < len(items) - 1 else end)
        if line not in (start, indent) and len(line) + 1 + len(text) > width:
            lines.append(line.rstrip())
            line = indent
        line += ("" if line in (start, indent) else " ") + text
    if not items:
        line += end
    lines.append(line)
    return lines


def _path_to_eq_var(path: Path) -> str:
    """Convert a module path like ``modules_model/Sector A/Sub1`` to ``sector_a_sub1_eqs``."""
    # Drop the first path component (the modules_<name> directory)
    parts = list(path.parts)[1:] if len(path.parts) > 1 else list(path.parts)
    name = "_".join(parts)
    name = re.sub(r"[^a-z0-9_]", "_", name.lower())
    name = re.sub(r"_+", "_", name).strip("_")
    return f"{name}_eqs"


def _format_julia_value(data) -> str:
    """Format a Python/numpy/xarray value as a Julia literal.

    Scalars become plain number strings.
    1-D arrays become ``[v1, v2, ...]``.
    2-D arrays become ``[r1c1 r1c2; r2c1 r2c2]`` (Julia matrix literal).
    Higher-dimensional arrays are flattened to 1-D.
    """
    import numpy as np

    # xarray DataArray → plain numpy array
    if hasattr(data, "values"):
        data = data.values

    if isinstance(data, (int, float)):
        return format_number(float(data))

    arr = np.asarray(data, dtype=float)

    if arr.ndim == 0:
        return format_number(float(arr))

    if arr.ndim == 1:
        vals = ", ".join(format_number(float(v)) for v in arr)
        return f"[{vals}]"

    if arr.ndim == 2:
        rows = "; ".join(
            " ".join(format_number(float(v)) for v in row) for row in arr
        )
        return f"[{rows}]"

    # Higher dims: reshape preserving Julia column-major indexing
    # Flatten in Fortran (column-major) order so reshape(..., s0, s1, ...) in
    # Julia gives A[i,j,...] == arr[i-1,j-1,...].
    vals = ", ".join(format_number(float(v)) for v in arr.flatten(order="F"))
    shape = ", ".join(str(s) for s in arr.shape)
    return f"reshape([{vals}], {shape})"


def _vensim_keyword_to_itp_type(keyword: Optional[str]) -> str:
    """Map a Vensim DATA keyword to the ``itp_type`` used by
    :func:`lookup_interpolation_code`.

    Vensim keywords and their meanings:

    * ``None`` / ``"interpolate"`` — linear interpolation (default)
    * ``"hold_backward"``          — step function, hold previous value
      → ``ConstantInterpolation``
    * ``"look_forward"``           — step function, hold next value
      → ``ConstantInterpolation(dir=:right)``
    * ``"raw"``                    — no interpolation; approximated as linear
    """
    if keyword == "hold_backward":
        return "hold_forward"   # ConstantInterpolation (left/previous)
    if keyword == "look_forward":
        return "hold_backward"  # ConstantInterpolation(dir=:right) (right/next)
    return "interpolate"
