"""
Translates a PySD AbstractModel into a standalone Julia file that uses
ModelingToolkit.jl.  The generated file requires no PySD or Python at runtime.

Entry point::

    from pysd.builders.julia.julia_model_builder import JuliaModelBuilder
    path = JuliaModelBuilder(abstract_model).build_model()
"""
from __future__ import annotations

import itertools
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
from .namespace import JuliaNamespaceManager

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
        self.original_path = abstract_model.original_path
        self.sections = [
            JuliaSectionBuilder(section, data_format=data_format, backend=backend)
            for section in abstract_model.sections
        ]

    def build_model(self) -> Path:
        """Translate all sections and return the path to the main ``.jl`` file.

        The first section is always the main model.  Any additional sections
        are Vensim macros; each gets its own ``<macro_name>.jl`` companion file.

        Macro sections are built first so the main section knows their names
        (to suppress spurious "Unknown Vensim function" warnings) and companion
        file paths (to emit ``include()`` statements).
        """
        # Collect all macro Julia identifiers up-front so every section
        # (including other macros doing cross-reference calls) can suppress
        # "Unknown Vensim function" warnings for macro calls.
        macro_names: Set[str] = {
            re.sub(r"[^a-z0-9_]", "_", s.name.lower())
            for s in self.sections[1:]
        }

        # Build macro sections first.
        for section in self.sections[1:]:
            section._known_macro_names = macro_names
            section.build_section()

        # Build main section with knowledge of all macro names + companion paths.
        main = self.sections[0]
        main._known_macro_names = macro_names
        main._macro_companion_paths = [
            s._macro_companion_path
            for s in self.sections[1:]
            if getattr(s, "_macro_companion_path", None) is not None
        ]
        main.build_section()
        return main.path


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
        self.name: str = abstract_section.name
        self.path: Path = abstract_section.path.with_suffix(".jl")
        self.root: Path = self.path.parent
        self.model_name: str = self.path.stem
        self.split: bool = abstract_section.split
        self.views_dict: Optional[dict] = abstract_section.views_dict
        self.abstract_elements: List[AbstractElement] = list(abstract_section.elements)
        self._abstract_subscripts = abstract_section.subscripts
        # Macro parameters (populated from abstract_section.params for macro sections)
        self._macro_params: List[str] = list(abstract_section.params)
        # Set by JuliaModelBuilder before building: Julia ids of known macros
        self._known_macro_names: Set[str] = set()
        # Set by JuliaModelBuilder before building: companion paths of macro sections
        self._macro_companion_paths: List[Path] = []
        # Set by _build_macro_section: path to this section's companion .jl file
        self._macro_companion_path: Optional[Path] = None
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

    def _build_macro_section(self) -> None:
        """Generate a companion ``.jl`` file for a Vensim macro section.

        **ODE backend**: emits a Julia function ``macro_name(params...)`` that
        evaluates the macro's algebraic body and returns the output variable.
        Stateful macros (containing INTEG stocks) are flagged with a warning
        and a placeholder ``return 0.0`` is emitted.

        **MTK backend**: emits the previous-style ``Equation[]`` vector
        (``{macro_name}_eqs``) for use in ``ODESystem`` composition.
        """
        macro_jl_name = re.sub(r"[^a-z0-9_]", "_", self.name.lower())

        # Add macro params to namespace so references to them inside the macro
        # body don't produce "not found in namespace" warnings.
        for param in self._macro_params:
            clean = re.sub(r"[^a-z0-9_]", "_", param.lower())
            self.namespace.namespace[param] = clean

        # Populate namespace with element names
        for elem in self.abstract_elements:
            self.namespace.add_to_namespace(elem.name)

        # Process all elements (macros have no control elements)
        for elem in self.abstract_elements:
            identifier = self.namespace.namespace[elem.name]
            eqs = self._process_element(elem, identifier, is_control=False)
            self.built_elements[identifier] = (eqs, False)

        # Register inline lookups
        for lut_name, xs, ys, itp_type in self.inline_registry.entries:
            const_decl, func_decl, reg_decl = lookup_interpolation_code(
                lut_name, xs, ys, itp_type
            )
            self.lookup_const_decls.append(const_decl)
            self.lookup_func_decls.append(func_decl)
            self.lookup_register_decls.append(reg_decl)

        # Determine companion file path and write it
        self.path = self.path.with_name(
            f"{self.path.stem}_{macro_jl_name}.jl"
        )
        if self.data_format == "json":
            self._write_data_json()

        if self.backend == "ode":
            text = self._build_macro_ode_text(macro_jl_name)
        else:
            text = self._build_macro_mtk_text(macro_jl_name)

        self.path.write_text(text, encoding="UTF-8")
        self._macro_companion_path = self.path

    def _build_macro_ode_text(self, macro_jl_name: str) -> str:
        """Return the companion file content for a macro in ODE-backend mode.

        Generates a plain Julia function ``macro_name(params...)`` that
        evaluates the macro body and returns the output variable.
        """
        all_eqs: List[str] = []
        for eqs, _ in self.built_elements.values():
            all_eqs.extend(eqs)

        # Detect stateful macros (contain ODE equations D(x) ~ ...)
        has_ode = any(
            eq.strip().startswith("D(") or eq.strip().startswith("[D(")
            for eq in all_eqs
        )

        param_list = ", ".join(
            re.sub(r"[^a-z0-9_]", "_", p.lower()) for p in self._macro_params
        )

        if has_ode:
            warn(
                f"Macro '{macro_jl_name}' contains stocks (INTEG); "
                "the ODE backend cannot inline stateful macros. "
                "A placeholder function returning 0.0 is generated."
            )
            body = "    # Stateful macro — ODE stocks cannot be inlined; placeholder only.\n    return 0.0"
        else:
            # Collect algebraic equations and convert ~ → =
            alg_eqs = [
                eq for eq in all_eqs
                if eq.strip() and not eq.strip().startswith("D(")
                and not eq.strip().startswith("[D(")
            ]
            body_lines: List[str] = []
            for eq in alg_eqs:
                converted = self._convert_eq_to_assignment(eq.strip().rstrip(","))
                body_lines.extend(f"    {line}" for line in converted)

            # Return value: element whose Julia id matches the macro name,
            # or the last element if no match.
            return_id = macro_jl_name
            if macro_jl_name not in self.built_elements:
                ids = list(self.built_elements.keys())
                return_id = ids[-1] if ids else macro_jl_name

            if body_lines:
                body = "\n".join(body_lines) + f"\n    return {return_id}"
            else:
                body = f"    return {return_id}"

        # Build using/lookup preamble (DataInterpolations for inline lookups)
        uses: List[str] = []
        if self.lookup_const_decls:
            uses.append("DataInterpolations")
        if self.data_format == "json":
            uses.append("JSON3")
        using_line = f"using {', '.join(uses)}\n\n" if uses else ""
        lookup_block = self._lookup_block() if self.lookup_const_decls else ""

        return (
            f"# Macro {self.name}\n"
            f"# Translated using PySD version {__version__}\n\n"
            f"{using_line}"
            f"{lookup_block}"
            f"function {macro_jl_name}({param_list})\n"
            f"{body}\n"
            f"end\n"
        )

    def _build_macro_mtk_text(self, macro_jl_name: str) -> str:
        """Return the companion file content for a macro in MTK-backend mode."""
        all_eqs: List[str] = []
        for eqs, _ in self.built_elements.values():
            all_eqs.extend(eqs)

        eq_var = f"{macro_jl_name}_eqs"
        eq_lines = ",\n    ".join(all_eqs) if all_eqs else ""
        uses = ["ModelingToolkit", "Symbolics"]
        if self.lookup_const_decls:
            uses.append("DataInterpolations")
        if self.data_format == "json":
            uses.append("JSON3")
        using_line = f"using {', '.join(uses)}"

        return textwrap.dedent(f"""\
            # Macro {self.name}
            # Translated using PySD version {__version__}

            {using_line}

            {self._helpers_block()}
            {self._lookup_block()}
            {self._declarations_block()}
            {eq_var} = Equation[
                {eq_lines}
            ]
            """)

    def build_section(self) -> None:
        """Build the section, writing one or more ``.jl`` files.

        For macro sections (``type == 'macro'``) a standalone companion
        ``.jl`` file is generated containing the macro's equations as a
        Julia ``Equation`` vector named ``{macro_name}_eqs``.  The file
        is written next to the main model file.
        """
        is_macro = (self.name != "__main__")

        if is_macro:
            self._build_macro_section()
            return

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
                c.subscripts and c.subscripts[0]
                and c.subscripts[0][0] not in self._subs_elems
                and c.subscripts[0][0] in self._elem_to_range
                for c in elem.components
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

            if _has_except or _has_per_elem:
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
                    t0_val = self._eval_ast_at_t0(ast.order)
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
                    t0_val = self._eval_ast_at_t0(ast.order)
                    if t0_val is not None:
                        order = max(1, round(t0_val))
                if order is None:
                    warn(
                        f"DELAY with non-integer order for '{elem.name}'; defaulting to 3."
                    )
                    order = 3
            return self._expand_delay(identifier, ast, visitor, order=order, dims=dims)

        # ---- DELAY FIXED ------------------------------------------------
        # Approximate DELAY FIXED as a first-order ODE delay (same formula
        # as DELAY1 with order=1).  True fixed delays need a DDE solver
        # that ModelingToolkit/OrdinaryDiffEq does not support, so this is
        # the best we can do in the MTK ODE framework.
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
            return self._expand_trend(identifier, ast, visitor)

        # ---- FORECAST ---------------------------------------------------
        if isinstance(ast, ForecastStructure):
            return self._expand_forecast(identifier, ast, visitor)

        # ---- SAMPLE IF TRUE ---------------------------------------------
        if isinstance(ast, SampleIfTrueStructure):
            return self._expand_sample_if_true(identifier, ast, visitor, dims=dims, ndim=ndim)

        # ---- ALLOCATE AVAILABLE / ALLOCATE BY PRIORITY ------------------
        if isinstance(ast, (AllocateAvailableStructure, AllocateByPriorityStructure)):
            return self._expand_allocate(identifier, ast, visitor)

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

            final0 = [i for i in covered0 if all((i, j) not in excluded for j in covered1)]
            final1 = covered1  # column coverage doesn't change

            if not final0 or not final1:
                continue

            if isinstance(comp.ast, IntegStructure):
                # Stock component — emit per-pair D(identifier[i,j]) ODE equations.
                for i0 in final0:
                    for i1 in final1:
                        if (i0, i1) in excluded:  # pragma: no cover  # unreachable: final0 only contains rows unexcluded for all cols
                            continue
                        vis_ij = JuliaASTVisitor(
                            self.namespace, self.inline_registry, self.needed_helpers,
                            active_subs={dim0_name: str(i0), dim1_name: str(i1)},
                            var_dims=self._var_dims, subs_sizes=self._subs_sizes,
                            subs_elems=self._subs_elems, lookup_names=self._lookup_func_names,
                            root=self.root, macro_names=self._known_macro_names,
                        )
                        flow_expr = vis_ij.visit(comp.ast.flow)
                        init_expr = vis_ij.visit(comp.ast.initial)
                        self.u0_entries.append(f"{identifier}[{i0}, {i1}] => {init_expr}")
                        equations.append(f"D({identifier}[{i0}, {i1}]) ~ {flow_expr}")
            else:
                # Auxiliary component — use comprehension over remaining index ranges.
                # Check if all remaining rows still cover the full column range
                # (so we can use a range expression rather than an explicit list).
                full_col_range = list(range(1, len(dim1_elems) + 1))
                use_full_cols = final1 == full_col_range

                vnd = self._nd_visitor(dims, ["_i0", "_i1"])
                rhs_expr = vnd.visit(comp.ast)

                row_str = (
                    f"1:{self._jl_n(dim0_name)}"
                    if final0 == list(range(1, len(dim0_elems) + 1))
                    else "[" + ", ".join(str(i) for i in final0) + "]"
                )
                col_str = (
                    f"1:{self._jl_n(dim1_name)}"
                    if use_full_cols
                    else "[" + ", ".join(str(j) for j in final1) + "]"
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

            # Apply exclusion to dim0; keep dim1 and dim2 as-is
            final0 = [
                i for i in covered0
                if not all((i, j, k) in excluded for j in covered1 for k in covered2)
            ]
            final1 = covered1
            final2 = covered2

            if not final0 or not final1 or not final2:
                continue

            vnd = self._nd_visitor(dims, ["_i0", "_i1", "_i2"])
            rhs_expr = vnd.visit(comp.ast)

            def _idx_str(indices: List[int], dim_elems: List[str], dim_name: str) -> str:
                full = list(range(1, len(dim_elems) + 1))
                if indices == full:
                    return f"1:{self._jl_n(dim_name)}"
                return "[" + ", ".join(str(i) for i in indices) + "]"

            d0_str = _idx_str(final0, dim0_elems, dim0_name)
            d1_str = _idx_str(final1, dim1_elems, dim1_name)
            d2_str = _idx_str(final2, dim2_elems, dim2_name)

            equations.append(
                f"[{identifier}[_i0, _i1, _i2] ~ {rhs_expr} "
                f"for _i0 in {d0_str}, _i1 in {d1_str}, _i2 in {d2_str}]..."
            )

        if not is_control:
            self.aux_decls.append(
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
            # Subscripted SMOOTH — each internal level is an array.
            (d0, n0) = dims[0]
            vnd = self._nd_visitor(dims, ["_i0"])
            input_nd = vnd.visit(ast.input)
            st_nd = vnd.visit(ast.smooth_time)
            init_nd = vnd.visit(ast.initial)

            prev_nd = input_nd
            for i in range(1, order + 1):
                lv_name = f"_lv{i}_{identifier}"
                self.namespace.namespace[f"__internal_lv{i}_{identifier}"] = lv_name
                self.stock_decls.append(
                    f"@variables {lv_name}(t)[{self._range_str(dims)}]"
                )
                for idx in range(1, n0 + 1):
                    init_i = init_nd.replace("_i0", str(idx))
                    self.u0_entries.append(f"{lv_name}[{idx}] => {init_i}")
                lv_ref = f"{lv_name}[_i0]"
                eqs.append(
                    f"[D({lv_ref}) ~ ({prev_nd} - {lv_ref}) / "
                    f"({st_nd} / {order}) for _i0 in 1:{self._jl_n(d0)}]..."
                )
                prev_nd = lv_ref
            self.aux_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            eqs.append(
                f"[{identifier}[_i0] ~ {prev_nd} for _i0 in 1:{self._jl_n(d0)}]..."
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

            rate_nd = f"({order} / ({delay_time_nd}))"
            prev_nd = input_nd
            for stage in range(1, order + 1):
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
            eqs.append(f"[{identifier}[{idx_str_t}] ~ {prev_nd} for {for_clause}]...")
            return eqs

        input_expr = visitor.visit(ast.input)
        delay_time_expr = visitor.visit(ast.delay_time)
        initial_expr = visitor.visit(ast.initial)
        rate_expr = f"({order} / {delay_time_expr})"
        prev_outflow = input_expr
        for i in range(1, order + 1):
            lv_name = f"_dl{i}_{identifier}"
            self.namespace.namespace[f"__internal_dl{i}_{identifier}"] = lv_name
            self.stock_decls.append(f"@variables {lv_name}(t)")
            self.u0_entries.append(
                f"{lv_name} => {initial_expr} * {delay_time_expr} / {order}"
            )
            eqs.append(
                f"D({lv_name}) ~ ({prev_outflow} - {lv_name} * {rate_expr})"
            )
            prev_outflow = f"{lv_name} * {rate_expr}"

        self.aux_decls.append(f"@variables {identifier}(t)")
        eqs.append(f"{identifier} ~ {prev_outflow}")
        return eqs

    # ------------------------------------------------------------------
    # DELAY FIXED expansion
    # ------------------------------------------------------------------

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
            return None
        if isinstance(node, ReferenceStructure):
            julia_id = self.namespace.get(node.reference)
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
        input_expr = visitor.visit(ast.input)
        delay_time_expr = visitor.visit(ast.delay_time)
        initial_expr = visitor.visit(ast.initial)

        # ---- Attempt N-stage pipeline (ODE backend, constant delay_time) ----
        if self.backend == "ode":
            ts_str = self.control_vals.get("time_step")
            ts_val = float(ts_str) if ts_str is not None else None
            if ts_val is None:
                try:
                    ts_val = float(ts_str or "1.0")
                except (ValueError, TypeError):  # pragma: no cover  # ts_str is None → "1.0" always valid
                    ts_val = None
            dt_val = self._try_eval_as_float(delay_time_expr)
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

        # ---- MTK / fallback: single first-order ODE approximation ----
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
        """Emit N pipeline stages for an exact DELAY FIXED with Euler integration.

        With Euler solver and dt = time_step, each stage shifts a value by exactly
        one time step, giving a total transport delay of N * time_step.
        All stages are initialised to ``initial_expr``.
        """
        dims = dims or []
        equations: List[str] = []
        ts_expr = self.control_vals.get("time_step") or "time_step"

        if dims:
            ndim = len(dims)
            idx_vars = self._idx_vars(ndim)
            for_clause = self._for_clause(dims, idx_vars)
            idx_str_t = ", ".join(idx_vars)
            ranges_list = [range(1, size + 1) for _, size in dims]

            prev_expr_template = input_expr  # pipe_0 = input
            for k in range(1, N + 1):
                pipe_name = f"_df_pipe_{k}_{identifier}"
                self.namespace.namespace[f"__internal_df_pipe_{k}_{identifier}"] = pipe_name
                self.stock_decls.append(
                    f"@variables {pipe_name}(t)[{self._range_str(dims)}]"
                )
                # Inline the initial_expr directly — it's already the Julia expression
                for idx_combo in itertools.product(*ranges_list):
                    expr_i = initial_expr
                    for iv, idx in zip(idx_vars, idx_combo):
                        expr_i = expr_i.replace(iv, str(idx))
                    idx_s = ", ".join(str(v) for v in idx_combo)
                    self.u0_entries.append(f"{pipe_name}[{idx_s}] => {expr_i}")

                if k < N:
                    # Intermediate stage: D(pipe_k) ~ (prev - pipe_k) / time_step
                    equations.append(
                        f"[D({pipe_name}[{idx_str_t}]) ~ "
                        f"({prev_expr_template.replace(idx_str_t, idx_str_t)} - {pipe_name}[{idx_str_t}]) / ({ts_expr}) "
                        f"for {for_clause}]..."
                    )
                else:
                    # Last stage feeds the output identifier directly
                    self.aux_decls.append(
                        f"@variables {identifier}(t)[{self._range_str(dims)}]"
                    )
                    equations.append(
                        f"[D({pipe_name}[{idx_str_t}]) ~ "
                        f"({prev_expr_template} - {pipe_name}[{idx_str_t}]) / ({ts_expr}) "
                        f"for {for_clause}]..."
                    )
                    equations.append(
                        f"[{identifier}[{idx_str_t}] ~ {pipe_name}[{idx_str_t}] for {for_clause}]..."
                    )

                prev_expr_template = f"{pipe_name}[{idx_str_t}]"
            return equations

        # ---- Scalar case ----
        prev_expr = input_expr
        for k in range(1, N + 1):
            pipe_name = f"_df_pipe_{k}_{identifier}"
            self.namespace.namespace[f"__internal_df_pipe_{k}_{identifier}"] = pipe_name
            self.stock_decls.append(f"@variables {pipe_name}(t)")
            self.u0_entries.append(f"{pipe_name} => {initial_expr}")
            equations.append(
                f"D({pipe_name}) ~ ({prev_expr} - {pipe_name}) / ({ts_expr})"
            )
            prev_expr = pipe_name

        # Output variable is the last pipeline stage
        self.aux_decls.append(f"@variables {identifier}(t)")
        equations.append(f"{identifier} ~ {prev_expr}")
        return equations

    # ------------------------------------------------------------------
    # Trend expansion
    # ------------------------------------------------------------------

    def _expand_trend(
        self,
        identifier: str,
        ast,
        visitor: "JuliaASTVisitor",
    ) -> List[str]:
        """Expand TREND(input, average_time, initial_trend) into an ODE.

        Introduces a smooth level ``_sm_{identifier}`` that tracks the
        exponential moving average of the input:

            D(_sm) ~ (input - _sm) / average_time

        Then the trend (fractional growth rate) is:

            output ~ (input - _sm) / (average_time * _sm)

        The smooth level is initialised so that at t=0 the output equals
        ``initial_trend``:

            _sm(0) = input(0) / (1 + initial_trend * average_time)

        We use the simpler ``input(0)`` approximation (same as PySD's
        Trend stateful initialisation) and rely on the model's initial
        conditions to provide a consistent starting point.
        """
        input_expr = visitor.visit(ast.input)
        avg_time_expr = visitor.visit(ast.average_time)
        initial_trend_expr = visitor.visit(ast.initial_trend)

        sm_name = f"_sm_{identifier}"
        self.namespace.namespace[f"__internal_sm_{identifier}"] = sm_name
        self.stock_decls.append(f"@variables {sm_name}(t)")
        # u0: _sm = input / (1 + initial_trend * average_time)
        # We approximate the initial input as the initial_trend expression;
        # a better approximation requires evaluating the input at t0.
        # Use the same formula as PySD: sm0 = input0 (the Trend stateful
        # initialises its smooth to input/1 when initial_trend is given).
        # We store the initial as a formula that Julia will evaluate at t=0.
        self.u0_entries.append(
            f"{sm_name} => {input_expr} / (1.0 + ({initial_trend_expr}) * ({avg_time_expr}))"
        )

        self.aux_decls.append(f"@variables {identifier}(t)")
        return [
            f"D({sm_name}) ~ ({input_expr} - {sm_name}) / ({avg_time_expr})",
            (
                f"{identifier} ~ ifelse(iszero({sm_name}), {initial_trend_expr}, "
                f"({input_expr} - {sm_name}) / (({avg_time_expr}) * {sm_name}))"
            ),
        ]

    # ------------------------------------------------------------------
    # Forecast expansion
    # ------------------------------------------------------------------

    def _expand_forecast(
        self,
        identifier: str,
        ast,
        visitor: "JuliaASTVisitor",
    ) -> List[str]:
        """Expand FORECAST(input, average_time, horizon) = input*(1 + TREND*horizon).

        FORECAST internally computes a TREND and projects it forward by
        *horizon*.  We expand it inline, introducing the same internal
        smooth level as ``_expand_trend``.
        """
        input_expr = visitor.visit(ast.input)
        avg_time_expr = visitor.visit(ast.average_time)
        horizon_expr = visitor.visit(ast.horizon)
        initial_trend_expr = visitor.visit(ast.initial_trend)

        sm_name = f"_sm_{identifier}"
        self.namespace.namespace[f"__internal_sm_{identifier}"] = sm_name
        self.stock_decls.append(f"@variables {sm_name}(t)")
        self.u0_entries.append(
            f"{sm_name} => {input_expr} / (1.0 + ({initial_trend_expr}) * ({avg_time_expr}))"
        )

        # trend = (input - sm) / (avg_time * sm)
        # forecast = input * (1 + trend * horizon)
        self.aux_decls.append(f"@variables {identifier}(t)")
        return [
            f"D({sm_name}) ~ ({input_expr} - {sm_name}) / ({avg_time_expr})",
            (
                f"{identifier} ~ {input_expr} * (1.0 + "
                f"ifelse(iszero({sm_name}), {initial_trend_expr}, "
                f"({input_expr} - {sm_name}) / (({avg_time_expr}) * {sm_name})) "
                f"* ({horizon_expr}))"
            ),
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
    ) -> List[str]:
        """Emit PySD.jl allocation helper calls.

        ALLOCATE BY PRIORITY → pysd_allocate_by_priority(request, priority, width, supply)
        ALLOCATE AVAILABLE   → pysd_allocate_available(request, pp, avail)

        Both helpers implement the exact Vensim algorithm (not proportional
        approximation).  They work on concrete Julia vectors at solve time and
        are compatible with the ODE backend.
        """
        self.aux_decls.append(f"@variables {identifier}(t)")
        if isinstance(ast, AllocateAvailableStructure):
            request_expr = visitor.visit(ast.request)
            pp_expr = visitor.visit(ast.pp)
            avail_expr = visitor.visit(ast.avail)
            rhs = f"pysd_allocate_available({request_expr}, {pp_expr}, {avail_expr})"
        else:
            # AllocateByPriorityStructure
            request_expr = visitor.visit(ast.request)
            priority_expr = visitor.visit(ast.priority)
            width_expr = visitor.visit(ast.width)
            supply_expr = visitor.visit(ast.supply)
            rhs = (
                f"pysd_allocate_by_priority("
                f"{request_expr}, {priority_expr}, {width_expr}, {supply_expr})"
            )

        return [f"{identifier} ~ {rhs}"]

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
                f'{file_expr}, "{ast0.tab}", "{ast0.x_row_or_col}", "{ast0.cell}")'
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
            y_names = []
            for comp in elem.components:
                if isinstance(comp.ast, GetLookupsStructure):
                    y_names.append(comp.ast.cell)
            if y_names:
                # Each component may produce a 1D or 2D lookup.
                # Use pysd_xlsx_build_lookup_dispatch per component to get
                # a vector of interpolations, then build a nested dispatch.
                sub_dispatch_names = []
                for k, y_name in enumerate(y_names):
                    sub_name = f"{identifier}_{k + 1}"
                    self.lookup_const_decls.append(
                        f"const {sub_name}_fns = pysd_xlsx_build_lookup_dispatch("
                        f'{file_expr}, "{ast0.tab}", "{ast0.x_row_or_col}", "{y_name}")'
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
                f'{file_expr}, "{ast0.tab}", "{ast0.time_row_or_col}", "{ast0.cell}")'
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
                    f'{file_expr}, "{dc.ast.tab}", "{dc.ast.time_row_or_col}", "{dc.ast.cell}")'
                )
                sub_dispatch_names.append(f"{sub_name}_fns")

            fn_list = ", ".join(sub_dispatch_names)
            self.lookup_const_decls.append(
                f"const {identifier}_fns = [{fn_list}]"
            )
            n_sub_dims = len(self._comp_coords(data_comps[0]))
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
            clean_cell = cell.rstrip('*')
            file_expr = f'joinpath(@__DIR__, "{ast0.file}")'
            kw_parts = []
            if transpose:
                kw_parts.append("transpose=true")
            kw = ("; " + ", ".join(kw_parts)) if kw_parts else ""
            return (
                f'pysd_xlsx_read_constant({file_expr}, '
                f'"{ast0.tab}", "{clean_cell}"{kw})'
            )

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
                clean_cell = cell.rstrip('*')
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

        Example (pymedeas world model):
            materials_for_o_m_per_capacity_installed_res_elec[RES_ELEC, materials]
              [RES_ELEC_DISPATCHABLE, materials] = 0   (literal)
              [RES_ELEC_VARIABLE,     materials] = GCS (Excel data)

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
        uses = ["OrdinaryDiffEq", "PySD", "NCDatasets"]
        has_lookups = bool(self.lookup_const_decls)
        if has_lookups or extra_packages:
            uses.append("DataInterpolations")
        if self.data_format == "json":
            uses.append("JSON3")
        header = (
            f"# Model {self.model_name}\n"
            f"# Translated using PySD version {__version__}\n\n"
            f"using {', '.join(uses)}\n\n"
            f'check_compat(v"0.1.0")\n\n'
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
            f'check_compat(v"0.1.0")\n\n'
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
        lines.append("                    data = Array(ds[varname])")
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
        if self.param_decls:
            lines.append("# Parameters")
            for decl in self.param_decls:
                # Convert "@parameters name = value" to "const name = value"
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
                        else:
                            lines.append("const " + val_part)
                    else:  # pragma: no cover  # all @parameters entries include "= value"
                        lines.append("const " + val_part)
                elif decl.startswith("#"):
                    lines.append(decl)
                else:  # pragma: no cover  # param_decls only contains @parameters or # prefixed entries
                    lines.append(decl)
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
        if not equations:
            return (
                "function rhs!(du, u, p, t)\nend\n\n"
                "function observe(u, t)\n    return Dict{String,Any}()\nend\n"
            )

        # Collect stock names and sizes from u0_entries
        # Each entry is "var => init" or "var[idx] => init"
        stock_info: Dict[str, int] = {}  # name -> count
        for entry in self.u0_entries:
            name = entry.split("=>")[0].strip()
            base = name.split("[")[0]
            stock_info[base] = stock_info.get(base, 0) + 1

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

        # Detect stock dimensionality from ODE equations
        stock_dims: Dict[str, List[str]] = {}
        for eq in ode_lines:
            eq_s = eq.strip().rstrip(",")
            if eq_s.startswith("["):
                inner = eq_s.strip().lstrip("[").rstrip(".]")
                m = re.match(r"D\((\w+)\[([^\]]+)\]\)", inner)
                if m:
                    name = m.group(1)
                    idx_parts = [x.strip() for x in m.group(2).split(",")]
                    if name not in stock_dims or len(idx_parts) > len(stock_dims[name]):
                        # Find dims from for clause
                        ranges = re.findall(r"in\s+\d+:(\w+)", eq_s)
                        if ranges:
                            stock_dims[name] = ranges

        # Build state-variable unpacking lines (shared by rhs! and observe)
        state_lines: List[str] = []
        for name in stock_indices:
            idx = stock_indices[name]
            size = stock_sizes[name]
            if size == 1:
                state_lines.append(f"    {name} = u[{idx}]")
            elif name in stock_dims and len(stock_dims[name]) >= 2:
                dims = stock_dims[name]
                dims_str = ", ".join(dims)
                state_lines.append(
                    f"    {name} = reshape(@view(u[{idx}:{idx + size - 1}]), {dims_str})"
                )
            else:
                state_lines.append(f"    {name} = @view u[{idx}:{idx + size - 1}]")

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
        for m in re.finditer(r"\b(\w+)\[([^\]]+)\]", all_eq_text):
            name = m.group(1)
            if name in stock_indices or name.startswith("du") or name.startswith("u"):
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
                        for m_for in re.finditer(r"in\s+(?:(\d+:\w+)|\[([^\]]+)\])", eq_s):
                            if m_for.group(1):
                                all_dims.append(m_for.group(1).split(":")[1])
                            elif m_for.group(2):
                                all_dims.append(str(len(m_for.group(2).split(","))))
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

        # Build aux assignment lines (shared by rhs! and observe), collecting names
        sorted_alg = self._topo_sort_equations(alg_lines, stock_indices)
        aux_assign_lines: List[str] = []
        scalar_aux_names: List[str] = []
        seen_aux: set = set(alloc_needed.keys())
        for eq in sorted_alg:
            if "Symbolics.scalarize" in eq or ".~" in eq:  # pragma: no cover  # scalarize/.~ equations route to ode_lines at line 4208, never appear in alg_lines
                continue
            converted = self._convert_eq_to_assignment(eq)
            # Collect scalar aux variable name from first converted line
            if converted:
                m_lhs = re.match(r"\s*(\w+)\s*=", converted[0])
                if m_lhs:
                    vname = m_lhs.group(1)
                    if vname in self._var_comments:
                        aux_assign_lines.append(f"    # {self._var_comments[vname]}")
                    if vname not in stock_indices and vname not in seen_aux:
                        seen_aux.add(vname)
                        scalar_aux_names.append(vname)
            aux_assign_lines.extend(f"    {line}" for line in converted)

        # ------------------------------------------------------------------ #
        # rhs!(du, u, p, t)                                                  #
        # ------------------------------------------------------------------ #
        func_lines = ["function rhs!(du, u, p, t)"]
        func_lines.append("    # State variables")
        func_lines.extend(state_lines)
        func_lines.append("")
        func_lines.append("    # Auxiliaries")
        if alloc_lines:
            func_lines.extend(alloc_lines)
            func_lines.append("")
        func_lines.extend(aux_assign_lines)

        # Create reshaped views of du for multi-dimensional stocks
        func_lines.append("")
        func_lines.append("    # Derivatives")
        for name in stock_indices:
            if name in stock_dims and len(stock_dims[name]) >= 2:
                idx = stock_indices[name]
                size = stock_sizes[name]
                dims = stock_dims[name]
                dims_str = ", ".join(dims)
                func_lines.append(
                    f"    du_{name} = reshape(@view(du[{idx}:{idx + size - 1}]), {dims_str})"
                )
        _emitted_stock_comments: set = set()
        for eq in ode_lines:
            if "Symbolics.scalarize" in eq or ".~" in eq:  # pragma: no cover  # these forms only appear in alg_lines, not ode_lines
                continue
            # Prepend comment for the stock variable (once per stock)
            m_stock = re.match(r"\[?D\((\w+)", eq.strip())
            if m_stock:
                sname = m_stock.group(1)
                if sname in self._var_comments and sname not in _emitted_stock_comments:
                    func_lines.append(f"    # {self._var_comments[sname]}")
                    _emitted_stock_comments.add(sname)
            converted = self._convert_ode_to_du(eq, stock_indices)
            for line in converted:
                func_lines.append(f"    {line}")

        func_lines.append("    return nothing")
        func_lines.append("end")

        # ------------------------------------------------------------------ #
        # observe(u, t) — reconstruct every variable at a given state/time   #
        # ------------------------------------------------------------------ #
        obs_lines = ["function observe(u, t)"]
        obs_lines.append("    # State variables")
        obs_lines.extend(state_lines)
        obs_lines.append("")
        obs_lines.append("    # Auxiliaries")
        if alloc_lines:
            obs_lines.extend(alloc_lines)
            obs_lines.append("")
        obs_lines.extend(aux_assign_lines)
        obs_lines.append("")
        # Module-level consts (INITIAL, GET CONSTANTS, etc.) are in scope inside
        # observe because they are module globals — just reference them by name.
        const_names = self._const_names_for_observe(set(stock_indices) | seen_aux)
        obs_lines.append("")
        obs_lines.append("    return Dict{String,Any}(")
        for name in list(stock_indices.keys()) + list(alloc_needed.keys()) + scalar_aux_names + const_names:
            obs_lines.append(f'        "{name}" => {name},')
        obs_lines.append("    )")
        obs_lines.append("end")

        return "\n".join(func_lines) + "\n\n" + "\n".join(obs_lines) + "\n"

    def _equations_block_mtk(self, equations: List[str]) -> str:
        if not equations:
            return "eqs = Equation[]\n"
        eq_lines = ",\n    ".join(equations)
        return f"eqs = Equation[\n    {eq_lines},\n]\n"

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

        # Build name -> equation index map
        name_to_idx: Dict[str, int] = {}
        for i, lhs in enumerate(eq_lhs):
            if lhs and lhs not in name_to_idx:
                name_to_idx[lhs] = i

        # Kahn's algorithm for topological sort
        n = len(equations)
        in_degree = [0] * n
        dependents: List[List[int]] = [[] for _ in range(n)]

        for i in range(n):
            resolved_deps = set()
            for dep in eq_deps[i]:
                if dep in name_to_idx:
                    j = name_to_idx[dep]
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

    def _u0_block(self) -> str:
        if self.backend == "mtk":
            return self._u0_block_mtk()
        if not self.u0_entries:
            return "u0 = Float64[]\n"

        # Check if any u0 values reference non-constant expressions
        needs_init_fn = False
        for entry in self.u0_entries:
            if "=>" in entry:
                rhs = entry.split("=>", 1)[1].strip()
                # If RHS contains variable references (not just numbers/params)
                tokens = set(re.findall(r"\b([a-z_]\w*)\b", rhs))
                # Remove known constants/params
                for t in list(tokens):
                    if any(f" {t} =" in d or f" {t}[" in d
                           for d in self.param_decls + self.ext_const_decls):
                        tokens.discard(t)
                if tokens - {"time_step", "initial_time", "final_time", "t"}:
                    needs_init_fn = True
                    break

        if needs_init_fn:
            # Collect all auxiliary identifiers referenced in u0 expressions that
            # are not module-level constants.  Use observe(zeros, initial_time) to
            # evaluate them at t=0 so stocks depending on auxiliaries initialise
            # correctly (e.g. TREND smooth stocks that depend on a dynamic input).
            dynamic_tokens: set = set()
            for entry in self.u0_entries:
                if "=>" in entry:
                    rhs = entry.split("=>", 1)[1].strip()
                    tokens = set(re.findall(r"\b([a-z_]\w*)\b", rhs))
                    for tok in list(tokens):
                        if any(f" {tok} =" in d or f" {tok}[" in d
                               for d in self.param_decls + self.ext_const_decls):
                            tokens.discard(tok)
                    dynamic_tokens |= tokens - {"time_step", "initial_time", "final_time", "t"}

            n = len(self.u0_entries)
            lines = [f"u0 = let _obs_init = observe(zeros(Float64, {n}), initial_time)"]
            for tok in sorted(dynamic_tokens):
                lines.append(f"    {tok} = get(_obs_init, \"{tok}\", 0.0)")
            lines.append("    Float64[")
            for entry in self.u0_entries:
                if "=>" in entry:
                    lhs, rhs = entry.split("=>", 1)
                    lines.append(f"        {rhs.strip()},  # {lhs.strip()}")
                else:  # pragma: no cover  # all u0_entries use "name => expr" format
                    lines.append(f"        {entry},")
            lines.append("    ]")
            lines.append("end")
            return "\n".join(lines) + "\n"
        else:
            lines = []
            for entry in self.u0_entries:
                if "=>" in entry:
                    lhs, rhs = entry.split("=>", 1)
                    lines.append(f"    {rhs.strip()},  # {lhs.strip()}")
                else:  # pragma: no cover  # all u0_entries use "name => expr" format
                    lines.append(f"    {entry},")
            return "u0 = Float64[\n" + "\n".join(lines) + "\n]\n"

    def _u0_block_mtk(self) -> str:
        if not self.u0_entries:
            return "u0 = []\n"
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
                # Substitute param references with numeric values
                for pname, pval in param_values.items():
                    rhs = re.sub(rf"\b{re.escape(pname)}\b", pval, rhs)
                lines.append(f"    {lhs.strip()} => {rhs},")
            else:  # pragma: no cover  # all u0_entries use "name => expr" format
                lines.append(f"    {entry},")
        return "u0 = [\n" + "\n".join(lines) + "\n]\n"

    def _control_block(self) -> str:
        it = self.control_vals.get("initial_time") or "0.0"
        ft = self.control_vals.get("final_time") or "100.0"
        ts = self.control_vals.get("time_step") or "1.0"
        return (
            "# Simulation control\n"
            f"initial_time = {it}\n"
            f"final_time   = {ft}\n"
            f"time_step    = {ts}\n"
            "tspan = (initial_time, final_time)\n"
        )

    def _run_function(self) -> str:
        if self.backend == "mtk":
            return self._run_function_mtk()
        ts = self.control_vals.get("time_step") or "time_step"
        has_tab = bool(self._tab_data_entries)
        tab_param = ", tab_data_files=String[], nc_data_files=String[]" if has_tab else ""
        tab_load = (
            "\n    isempty(tab_data_files) || _load_tab_data!(tab_data_files)"
            "\n    isempty(nc_data_files) || _load_nc_data!(nc_data_files)"
        ) if has_tab else ""
        return textwrap.dedent(f"""\
            prob = ODEProblem(rhs!, u0, tspan)

            function run_model(; u0=u0, tspan=tspan, dt={ts}, solver=Euler(){tab_param}){tab_load}
                prob_local = remake(prob; u0=u0, tspan=tspan)
                solve(prob_local, solver; dt=dt, saveat=tspan[1]:dt:tspan[2], adaptive=false)
            end
            """)

    def _run_function_mtk(self) -> str:
        ts = self.control_vals.get("time_step") or "time_step"
        return textwrap.dedent(f"""\
            u0_dict = Dict(x => v for (x, v) in zip(unknowns(sys), u0))
            u0_full = [get(u0_dict, x, 0.0) for x in unknowns(sys)]
            prob = ODEProblem(sys, u0_full, tspan; build_initializeprob = false)

            function run_model(; u0=u0_full, tspan=tspan, dt={ts}, solver=Euler())
                prob_local = remake(prob; u0=u0, tspan=tspan)
                solve(prob_local, solver; dt=dt, saveat=tspan[1]:dt:tspan[2], adaptive=false)
            end
            """)

    def _entrypoint_block(self) -> str:
        """Generate the top-level calls that run the model and save results."""
        nc_name = f"{self.model_name}_results.nc"
        if self.backend == "mtk":
            save_call = f'save_results(sol, sys, _dim_labels, joinpath(@__DIR__, "{nc_name}"))'
        else:
            save_call = f'save_results(sol, _state_map, _dim_labels, joinpath(@__DIR__, "{nc_name}"))'
        return textwrap.dedent(f"""\
            println("Running model…")
            sol = run_model()
            println("Saving results to {nc_name}…")
            {save_call}
            println("Done.")
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
            stock_sizes[base] += 1

        lines = ["const _state_map = ["]
        idx = 1
        for name in stock_order:
            size = stock_sizes[name]
            dim_names = self._var_dims.get(name, [])
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

    def _macro_includes_block(self) -> str:
        """Return ``include(...)`` statements for macro companion files."""
        if not self._macro_companion_paths:
            return ""
        lines = [
            f'include(joinpath(@__DIR__, "{p.name}"))'
            for p in self._macro_companion_paths
        ]
        return "\n# Macro companion files\n" + "\n".join(lines) + "\n"

    def _full_file_content(self, equations: List[str]) -> str:
        needs_di = bool(self.lookup_const_decls)
        return "".join([
            self._file_header(extra_packages=needs_di),
            self._helpers_block(),
            self._tab_data_block(),
            self._lookup_block(),
            self._macro_includes_block(),
            self._declarations_block(),
            self._control_block(),
            "\n",
            self._equations_block(equations),
            "\n",
            self._u0_block(),
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
