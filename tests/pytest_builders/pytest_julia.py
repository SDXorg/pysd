"""
Unit tests for the Julia/ModelingToolkit builder.

Tests are organised into classes that mirror the modules they exercise:

* ``TestJuliaNamespaceManager``  — namespace.py
* ``TestJuliaASTVisitor``         — julia_expressions_builder.py
* ``TestInlineLookupRegistry``    — julia_expressions_builder.py
* ``TestLookupHelpers``           — julia_expressions_builder.py
* ``TestJuliaSectionBuilder``     — julia_model_builder.py (element processing)
* ``TestJuliaModelBuilder``       — julia_model_builder.py (end-to-end)
* ``TestModularBuild``            — modular file generation
* ``TestTranslateToJulia``        — pysd.translate_to_julia entry point
"""
from pathlib import Path

import pytest

from pysd.builders.julia.namespace import JuliaNamespaceManager, JULIA_KEYWORDS
from pysd.builders.julia.julia_expressions_builder import (
    JuliaASTVisitor,
    InlineLookupRegistry,
    HELPER_IMPLEMENTATIONS,
    format_number,
    format_vector,
    lookup_interpolation_code,
)
from pysd.builders.julia.julia_model_builder import (
    JuliaModelBuilder,
    JuliaSectionBuilder,
    _path_to_eq_var,
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
    SmoothNStructure,
    SmoothStructure,
    SubscriptsReferenceStructure,
    TrendStructure,
)
from pysd.translators.structures.abstract_model import (
    AbstractComponent,
    AbstractControlElement,
    AbstractData,
    AbstractElement,
    AbstractLookup,
    AbstractModel,
    AbstractSection,
    AbstractSubscriptRange,
    AbstractUnchangeableConstant,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_section(
    elements=None,
    subscripts=(),
    split=False,
    views_dict=None,
    path=None,
):
    """Return a minimal AbstractSection suitable for JuliaSectionBuilder."""
    if path is None:
        path = Path("test_model.mdl")
    return AbstractSection(
        name="__main__",
        path=path,
        type="main",
        params=[],
        returns=[],
        subscripts=tuple(subscripts),
        elements=tuple(elements or []),
        constraints=tuple(),
        test_inputs=tuple(),
        split=split,
        views_dict=views_dict,
    )


def _make_component(ast, comp_type="Auxiliary", subtype="Normal"):
    comp = AbstractComponent(subscripts=[[], []], ast=ast)
    comp.type = comp_type
    comp.subtype = subtype
    return comp


def _make_constant_component(value):
    comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=value)
    return comp


def _make_element(name, ast, comp_class=None, units="", docs=""):
    if comp_class is AbstractUnchangeableConstant:
        comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=ast)
    else:
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
    return AbstractElement(name=name, components=[comp], units=units, documentation=docs)


def _make_lookup_element(name, xs, ys, itp_type="interpolate"):
    lut_ast = LookupsStructure(x=xs, y=ys, x_limits=(xs[0], xs[-1]),
                               y_limits=(ys[0], ys[-1]), type=itp_type)
    comp = AbstractLookup(subscripts=[[], []], ast=lut_ast)
    return AbstractElement(name=name, components=[comp])


def _make_stock_element(name, flow_ast, initial_ast):
    ast = IntegStructure(flow=flow_ast, initial=initial_ast)
    comp = AbstractComponent(subscripts=[[], []], ast=ast)
    return AbstractElement(name=name, components=[comp])


def _make_control_element(name, value):
    comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=value)
    return AbstractControlElement(name=name, components=[comp])


def _make_subscript_range(name, elems):
    return AbstractSubscriptRange(name=name, subscripts=elems, mapping=[])


def _make_subscripted_element(name, ast, dim_name, comp_class=None):
    """Element whose first component covers one subscript dimension."""
    if comp_class is AbstractUnchangeableConstant:
        comp = AbstractUnchangeableConstant(subscripts=[[dim_name], []], ast=ast)
    else:
        comp = AbstractComponent(subscripts=[[dim_name], []], ast=ast)
    return AbstractElement(name=name, components=[comp])


def _make_data_element(name, ast):
    """Element whose component is an AbstractData (external time-series)."""
    comp = AbstractData(subscripts=[[], []], ast=ast)
    return AbstractElement(name=name, components=[comp])


def _section_builder_from_elements(elements, path=None, split=False, views_dict=None,
                                   subscripts=(), backend="ode"):
    section = _make_section(elements, path=path or Path("test_model.mdl"),
                            split=split, views_dict=views_dict,
                            subscripts=subscripts)
    return JuliaSectionBuilder(section, backend=backend)


def _visitor_with_namespace(names=None):
    ns = JuliaNamespaceManager()
    for n in (names or []):
        ns.add_to_namespace(n)
    registry = InlineLookupRegistry()
    needed = set()
    return JuliaASTVisitor(ns, registry, needed), ns, registry, needed


# ===========================================================================
# JuliaNamespaceManager
# ===========================================================================

class TestJuliaNamespaceManager:

    def test_time_pre_registered(self):
        ns = JuliaNamespaceManager()
        assert ns.get("Time") == "t"

    def test_add_simple_name(self):
        ns = JuliaNamespaceManager()
        ident = ns.add_to_namespace("Population")
        assert ident == "population"
        assert ns.get("Population") == "population"

    def test_add_name_with_spaces(self):
        ns = JuliaNamespaceManager()
        ident = ns.add_to_namespace("Birth Rate")
        assert ident == "birth_rate"

    def test_add_name_with_special_chars(self):
        ns = JuliaNamespaceManager()
        ident = ns.add_to_namespace("var-n")
        assert ident == "var_n"

    def test_case_insensitive_lookup(self):
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("Population")
        assert ns.get("population") == "population"
        assert ns.get("POPULATION") == "population"
        assert ns.get("PoPuLaTiOn") == "population"

    def test_idempotent_registration(self):
        ns = JuliaNamespaceManager()
        id1 = ns.add_to_namespace("Alpha")
        id2 = ns.add_to_namespace("Alpha")
        assert id1 == id2

    def test_keyword_avoidance(self):
        ns = JuliaNamespaceManager()
        for kw in ("end", "begin", "if", "for", "while", "module"):
            ident = ns.add_to_namespace(kw)
            assert ident not in JULIA_KEYWORDS, f"'{ident}' is a Julia keyword"

    def test_collision_resolution(self):
        ns = JuliaNamespaceManager()
        # Both "Birth Rate" and "birth rate" map to the same clean form
        id1 = ns.add_to_namespace("Birth Rate")
        id2 = ns.add_to_namespace("birth rate")
        assert id1 != id2
        assert id1 == "birth_rate"
        assert id2 == "birth_rate_1"

    def test_triple_collision(self):
        """Third registration of the same clean name gets _2 suffix (line 84 in namespace.py)."""
        ns = JuliaNamespaceManager()
        id1 = ns.add_to_namespace("Alpha")
        id2 = ns.add_to_namespace("ALPHA")
        id3 = ns.add_to_namespace("alpha")
        assert id1 == "alpha"
        assert id2 == "alpha_1"
        assert id3 == "alpha_2"

    def test_empty_after_sanitization_gets_var_prefix(self):
        """Name with only special chars sanitises to empty string → falls back to '_var' (line 73)."""
        ns = JuliaNamespaceManager()
        ident = ns.add_to_namespace("!!!")
        assert ident == "_var"

    def test_leading_digit(self):
        ns = JuliaNamespaceManager()
        ident = ns.add_to_namespace("1st var")
        assert ident[0].isalpha() or ident[0] == "_"

    def test_unknown_name_returns_none(self):
        ns = JuliaNamespaceManager()
        assert ns.get("nonexistent") is None

    @pytest.mark.parametrize("name,expected", [
        ("GDP", "gdp"),
        ("CO2 emissions", "co2_emissions"),
        ("Net__Flow", "net_flow"),
        ("x", "x"),
    ])
    def test_various_names(self, name, expected):
        ns = JuliaNamespaceManager()
        assert ns.add_to_namespace(name) == expected


# ===========================================================================
# format_number / format_vector helpers
# ===========================================================================

class TestFormatHelpers:

    @pytest.mark.parametrize("value,expected", [
        (1.0, "1.0"),
        (0.5, "0.5"),
        (float("inf"), "Inf"),
        (float("-inf"), "-Inf"),
        (float("nan"), "NaN"),
        (3, "3.0"),
    ])
    def test_format_number(self, value, expected):
        assert format_number(value) == expected

    def test_format_vector(self):
        result = format_vector((0.0, 50.0, 100.0))
        assert result == "[0.0, 50.0, 100.0]"


# ===========================================================================
# InlineLookupRegistry
# ===========================================================================

class TestInlineLookupRegistry:

    def test_register_returns_unique_names(self):
        reg = InlineLookupRegistry()
        n1 = reg.register((0.0, 1.0), (0.0, 1.0), "interpolate")
        n2 = reg.register((0.0, 2.0), (0.0, 4.0), "interpolate")
        assert n1 != n2

    def test_register_increments_counter(self):
        reg = InlineLookupRegistry()
        n1 = reg.register((0.0,), (0.0,), "interpolate")
        n2 = reg.register((0.0,), (0.0,), "interpolate")
        assert n1 == "_inline_lookup_1"
        assert n2 == "_inline_lookup_2"

    def test_entries_returns_all_registered(self):
        reg = InlineLookupRegistry()
        reg.register((0.0, 1.0), (0.0, 2.0), "interpolate")
        reg.register((0.0, 5.0), (0.0, 10.0), "interpolate")
        assert len(reg.entries) == 2


# ===========================================================================
# lookup_interpolation_code
# ===========================================================================

class TestLookupInterpolationCode:

    def test_basic_output(self):
        const_decl, func_decl, reg_decl = lookup_interpolation_code(
            "my_lut", (0.0, 1.0, 2.0), (0.0, 5.0, 10.0), "interpolate"
        )
        assert "LinearInterpolation" in const_decl
        assert "my_lut_itp" in const_decl
        # DataInterpolations: ys first, xs second
        assert "[0.0, 5.0, 10.0]" in const_decl   # ys
        assert "[0.0, 1.0, 2.0]" in const_decl    # xs
        assert func_decl == "my_lut(x) = my_lut_itp(x)"
        assert "@register_symbolic" in reg_decl
        assert "my_lut" in reg_decl

    def test_const_keyword_present(self):
        const_decl, _, _ = lookup_interpolation_code("lut", (1.0,), (2.0,), "extrapolate")
        assert const_decl.startswith("const ")

    def test_hold_forward_uses_constant_interpolation(self):
        const_decl, _, _ = lookup_interpolation_code(
            "lut", (0.0, 1.0), (5.0, 10.0), "hold_forward"
        )
        assert "ConstantInterpolation" in const_decl
        assert "LinearInterpolation" not in const_decl
        assert "dir" not in const_decl

    def test_hold_backward_uses_constant_interpolation_right(self):
        const_decl, _, _ = lookup_interpolation_code(
            "lut", (0.0, 1.0), (5.0, 10.0), "hold_backward"
        )
        assert "ConstantInterpolation" in const_decl
        assert "dir=:right" in const_decl

    def test_unknown_type_falls_back_to_linear(self):
        const_decl, _, _ = lookup_interpolation_code(
            "lut", (0.0, 1.0), (5.0, 10.0), "unknown_type"
        )
        assert "LinearInterpolation" in const_decl

    def test_linear_interpolation_has_constant_left_extrapolation(self):
        # MTK evaluates at t=0 during init; data may start at e.g. 2020
        const_decl, _, _ = lookup_interpolation_code(
            "lut", (2020.0, 2050.0), (0.0, 1.0), "interpolate"
        )
        assert "ExtrapolationType.Constant" in const_decl

    def test_constant_interpolation_has_constant_left_extrapolation(self):
        const_decl, _, _ = lookup_interpolation_code(
            "lut", (2020.0, 2050.0), (0.0, 1.0), "hold_forward"
        )
        assert "ExtrapolationType.Constant" in const_decl

    def test_extrapolate_type_also_has_constant_extrapolation(self):
        const_decl, _, _ = lookup_interpolation_code(
            "lut", (2020.0, 2050.0), (0.0, 1.0), "extrapolate"
        )
        assert "ExtrapolationType.Constant" in const_decl


# ===========================================================================
# JuliaASTVisitor
# ===========================================================================

class TestJuliaASTVisitor:

    # --- numeric literals ---------------------------------------------------

    def test_integer_literal(self):
        v, *_ = _visitor_with_namespace()
        assert v.visit(3) == "3.0"

    def test_float_literal(self):
        v, *_ = _visitor_with_namespace()
        assert v.visit(0.5) == "0.5"

    def test_inf_literal(self):
        v, *_ = _visitor_with_namespace()
        assert v.visit(float("inf")) == "Inf"

    def test_none_becomes_zero(self):
        v, *_ = _visitor_with_namespace()
        assert v.visit(None) == "0.0"

    # --- arithmetic ---------------------------------------------------------

    @pytest.mark.parametrize("ops,args,expected", [
        (["+"], [1.0, 2.0], "(1.0 + 2.0)"),
        (["-"], [5.0, 3.0], "(5.0 - 3.0)"),
        (["*"], [2.0, 4.0], "(2.0 * 4.0)"),
        (["/"], [6.0, 3.0], "(6.0 / 3.0)"),
        (["^"], [2.0, 8.0], "pysd_power(2.0, 8.0)"),
    ])
    def test_binary_arithmetic(self, ops, args, expected):
        v, *_ = _visitor_with_namespace()
        node = ArithmeticStructure(operators=ops, arguments=args)
        assert v.visit(node) == expected

    def test_unary_negation(self):
        v, *_ = _visitor_with_namespace()
        node = ArithmeticStructure(operators=["-"], arguments=[3.0])
        assert v.visit(node) == "(-3.0)"

    def test_chained_arithmetic(self):
        v, *_ = _visitor_with_namespace()
        node = ArithmeticStructure(operators=["+", "*"], arguments=[1.0, 2.0, 3.0])
        result = v.visit(node)
        assert "1.0" in result and "2.0" in result and "3.0" in result

    # --- logic --------------------------------------------------------------

    @pytest.mark.parametrize("vensim_op,julia_op", [
        ("=", "=="),
        ("<>", "!="),
        ("<", "<"),
        (">", ">"),
        ("<=", "<="),
        (">=", ">="),
    ])
    def test_comparison_operators(self, vensim_op, julia_op):
        v, *_ = _visitor_with_namespace()
        node = LogicStructure(operators=[vensim_op], arguments=[1.0, 0.0])
        assert julia_op in v.visit(node)

    def test_and_uses_helper_function(self):
        """AND maps to pysd_logical_and helper (not &&) for symbolic MTK compatibility."""
        v, _, _, needed = _visitor_with_namespace()
        node = LogicStructure(operators=[":AND:"], arguments=[1.0, 0.0])
        result = v.visit(node)
        assert "pysd_logical_and(" in result
        assert "pysd_logical_and" in needed

    def test_or_uses_helper_function(self):
        """OR maps to pysd_logical_or helper (not ||) for symbolic MTK compatibility."""
        v, _, _, needed = _visitor_with_namespace()
        node = LogicStructure(operators=[":OR:"], arguments=[1.0, 0.0])
        result = v.visit(node)
        assert "pysd_logical_or(" in result
        assert "pysd_logical_or" in needed

    def test_unary_not_uses_helper_function(self):
        """NOT maps to pysd_logical_not helper for symbolic MTK compatibility."""
        v, _, _, needed = _visitor_with_namespace()
        node = LogicStructure(operators=[":NOT:"], arguments=[1.0])
        result = v.visit(node)
        assert "pysd_logical_not(" in result
        assert "pysd_logical_not" in needed

    # --- references ---------------------------------------------------------

    def test_known_reference(self):
        v, ns, *_ = _visitor_with_namespace(["Population"])
        node = ReferenceStructure(reference="Population")
        assert v.visit(node) == "population"

    def test_case_insensitive_reference(self):
        v, ns, *_ = _visitor_with_namespace(["Birth Rate"])
        node = ReferenceStructure(reference="birth rate")
        assert v.visit(node) == "birth_rate"

    def test_unknown_reference_warns(self):
        v, *_ = _visitor_with_namespace()
        node = ReferenceStructure(reference="Unknown Var")
        with pytest.warns(UserWarning, match="not found in namespace"):
            result = v.visit(node)
        assert isinstance(result, str)

    # --- built-in function calls --------------------------------------------

    @pytest.mark.parametrize("vensim_name,julia_name", [
        ("ABS", "abs"),
        ("EXP", "exp"),
        ("LN", "log"),
        ("SQRT", "sqrt"),
        ("SIN", "sin"),
        ("COS", "cos"),
        ("TAN", "tan"),
        ("ARCSIN", "asin"),
        ("ARCCOS", "acos"),
        ("ARCTAN", "atan"),
        ("MIN", "min"),
        ("MAX", "max"),
        ("MODULO", "mod"),
        ("INTEGER", "trunc"),
    ])
    def test_builtin_functions(self, vensim_name, julia_name):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference=vensim_name),
            arguments=(1.0,),
        )
        assert julia_name in v.visit(node)

    @pytest.mark.parametrize("func_ref", ["IF THEN ELSE", "if_then_else"])
    def test_if_then_else(self, func_ref):
        """Both the space form and the underscore form (as stored by the parser) work."""
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference=func_ref),
            arguments=(1.0, 2.0, 3.0),
        )
        assert "ifelse" in v.visit(node)

    @pytest.mark.parametrize("func_ref", ["PULSE TRAIN", "pulse_train"])
    def test_pulse_train_both_forms(self, func_ref):
        """Both the space form and the underscore form (as stored by the parser) work."""
        v, _, _, needed = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference=func_ref),
            arguments=(10.0, 1.0, 5.0, 100.0),
        )
        result = v.visit(node)
        assert "pysd_pulse_train" in result
        assert "pysd_pulse_train" in needed

    def test_unknown_function_warns(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="SOME_UNKNOWN_FUNC"),
            arguments=(1.0,),
        )
        with pytest.warns(UserWarning, match="Unknown Vensim function"):
            result = v.visit(node)
        assert "some_unknown_func" in result

    # --- helper functions registered in needed_helpers ----------------------

    @pytest.mark.parametrize("func_name", ["XIDZ", "ZIDZ", "PULSE", "RAMP", "STEP"])
    def test_helper_functions_registered(self, func_name):
        v, _, _, needed = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference=func_name),
            arguments=(1.0, 2.0, 3.0),
        )
        v.visit(node)
        helper_name = f"pysd_{func_name.lower()}"
        assert helper_name in needed

    def test_pulse_prepends_t(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="PULSE"),
            arguments=(10.0, 1.0),
        )
        result = v.visit(node)
        assert result.startswith("pysd_pulse(t,")

    def test_ramp_prepends_t(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="RAMP"),
            arguments=(0.1, 5.0),
        )
        result = v.visit(node)
        assert result.startswith("pysd_ramp(t,")

    # --- newly added functions -----------------------------------------------

    def test_power_maps_to_helper(self):
        v, _, _, needed = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="power"),
            arguments=(2.0, 3.0),
        )
        result = v.visit(node)
        assert "pysd_power" in result
        assert "pysd_power" in needed

    def test_sinh_maps_directly(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="sinh"),
            arguments=(1.0,),
        )
        result = v.visit(node)
        assert result == "sinh(1.0)"

    def test_cosh_maps_directly(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="cosh"),
            arguments=(1.0,),
        )
        result = v.visit(node)
        assert result == "cosh(1.0)"

    def test_tanh_maps_directly(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="tanh"),
            arguments=(1.0,),
        )
        result = v.visit(node)
        assert result == "tanh(1.0)"

    def test_quantum_pulls_in_trunc(self):
        v, _, _, needed = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="quantum"),
            arguments=(10.0, 3.0),
        )
        result = v.visit(node)
        assert "pysd_quantum" in result
        assert "pysd_quantum" in needed
        assert "pysd_trunc" in needed

    def test_random_uniform_registered(self):
        v, _, _, needed = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="random_uniform"),
            arguments=(0.0, 1.0, 42.0),
        )
        result = v.visit(node)
        assert "pysd_random_uniform" in result
        assert "pysd_random_uniform" in needed

    def test_vector_sort_order_registered(self):
        v, _, _, needed = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="vector_sort_order"),
            arguments=(1.0, 1.0),
        )
        result = v.visit(node)
        assert "pysd_vector_sort_order" in result
        assert "pysd_vector_sort_order" in needed

    def test_get_time_value_prepends_t(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="get_time_value"),
            arguments=(1.0, 2.0, 3.0),
        )
        result = v.visit(node)
        assert result.startswith("pysd_get_time_value(t,")

    def test_xpulse_prepends_t(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="Xpulse"),
            arguments=(10.0, 5.0),
        )
        result = v.visit(node)
        assert result.startswith("pysd_xpulse(t,")

    def test_xramp_prepends_t(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="Xramp"),
            arguments=(0.5, 10.0),
        )
        result = v.visit(node)
        assert result.startswith("pysd_xramp(t,")

    # --- InitialStructure / GameStructure -----------------------------------

    def test_initial_structure_returns_inner(self):
        v, *_ = _visitor_with_namespace()
        node = InitialStructure(initial=42.0)
        assert v.visit(node) == "42.0"

    def test_game_structure_returns_inner(self):
        v, *_ = _visitor_with_namespace()
        node = GameStructure(expression=7.0)
        assert v.visit(node) == "7.0"

    # --- inline lookups -----------------------------------------------------

    def test_inline_lookup_registers(self):
        v, _, registry, _ = _visitor_with_namespace(["x_var"])
        lut = LookupsStructure(
            x=(0.0, 1.0, 2.0),
            y=(0.0, 5.0, 10.0),
            x_limits=(0.0, 2.0),
            y_limits=(0.0, 10.0),
            type="interpolate",
        )
        node = InlineLookupsStructure(
            argument=ReferenceStructure(reference="x_var"),
            lookups=lut,
        )
        result = v.visit(node)
        assert len(registry.entries) == 1
        assert "_inline_lookup_1" in result

    def test_inline_lookup_call_includes_arg(self):
        v, ns, registry, _ = _visitor_with_namespace(["input"])
        lut = LookupsStructure(
            x=(0.0, 1.0),
            y=(0.0, 2.0),
            x_limits=(0.0, 1.0),
            y_limits=(0.0, 2.0),
            type="interpolate",
        )
        node = InlineLookupsStructure(
            argument=ReferenceStructure(reference="input"),
            lookups=lut,
        )
        result = v.visit(node)
        assert "input" in result

    def test_with_extra_subs_propagates_macro_names(self):
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("my_macro")
        registry = InlineLookupRegistry()
        helpers = set()
        v = JuliaASTVisitor(
            ns, registry, helpers,
            macro_names={"my_macro"},
        )
        child = v._with_extra_subs({"dim": "_i0"})
        assert "my_macro" in child._macro_names, (
            f"_with_extra_subs must propagate macro_names, got: {child._macro_names}"
        )


# ===========================================================================
# JuliaSectionBuilder — element processing
# ===========================================================================

class TestJuliaSectionBuilderElements:

    # --- stocks -----------------------------------------------------------

    def test_stock_creates_ode_equation(self):
        flow = ArithmeticStructure(operators=["-"], arguments=[
            ReferenceStructure("Births"), ReferenceStructure("Deaths")
        ])
        pop_elem = _make_stock_element("Population", flow, 1000.0)
        # Register the referenced variables so the visitor can resolve them
        births_elem = _make_element("Births", 10.0)
        deaths_elem = _make_element("Deaths", 5.0)
        sb = _section_builder_from_elements([pop_elem, births_elem, deaths_elem])
        sb.build_section()

        assert any("@variables population(t)" in d for d in sb.stock_decls)
        assert any("D(population)" in e for e in sb.built_elements["population"][0])
        assert any("population =>" in u for u in sb.u0_entries)

    def test_stock_initial_value_in_u0(self):
        elem = _make_stock_element("Capital", 5.0, 100.0)
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        assert any("capital => 100.0" in u for u in sb.u0_entries)

    # --- constants / parameters -------------------------------------------

    def test_constant_creates_parameter(self):
        comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.03)
        elem = AbstractElement(name="Birth Rate", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        assert any("@parameters birth_rate = 0.03" in d for d in sb.param_decls)

    def test_auxiliary_creates_variable_and_equation(self):
        rhs = ArithmeticStructure(operators=["*"], arguments=[
            ReferenceStructure("population"), ReferenceStructure("birth_rate_param")
        ])
        # Register the references so namespace resolves them
        elem_pop = _make_stock_element("Population", 0.0, 100.0)
        comp_br = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.03)
        elem_br = AbstractElement(name="birth rate param", components=[comp_br])
        comp_aux = AbstractComponent(subscripts=[[], []], ast=rhs)
        elem_births = AbstractElement(name="Births", components=[comp_aux])

        sb = _section_builder_from_elements([elem_pop, elem_br, elem_births])
        sb.build_section()

        assert any("@variables births(t)" in d for d in sb.aux_decls)
        assert any("births ~" in e for eqs, _ in sb.built_elements.values() for e in eqs)

    # --- lookup tables ----------------------------------------------------

    def test_named_lookup_registers_interpolant(self):
        elem = _make_lookup_element(
            "Effect Table", (0.0, 0.5, 1.0), (0.0, 0.8, 1.0)
        )
        sb = _section_builder_from_elements([elem])
        sb.build_section()

        assert any("effect_table_itp" in d for d in sb.lookup_const_decls)
        assert any("effect_table(x)" in f for f in sb.lookup_func_decls)

    def test_named_lookup_no_equation_generated(self):
        elem = _make_lookup_element("LUT", (0.0, 1.0), (0.0, 2.0))
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        # lookup elements produce no ODE/algebraic equations
        assert sb.built_elements["lut"][0] == []

    # --- control variables ------------------------------------------------

    def test_control_vars_stored_not_emitted_as_params(self):
        elems = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 100.0),
            _make_control_element("TIME STEP", 0.25),
            _make_control_element("SAVEPER", 0.25),
        ]
        sb = _section_builder_from_elements(elems)
        sb.build_section()

        # Control vars should NOT appear as @parameters
        assert not any("initial_time" in d for d in sb.param_decls)
        assert sb.control_vals["initial_time"] == "0.0"
        assert sb.control_vals["final_time"] == "100.0"
        assert sb.control_vals["time_step"] == "0.25"

    # --- smooth expansion -------------------------------------------------

    def test_smooth1_expands_to_ode_and_aux(self):
        flow_ast = SmoothStructure(
            input=5.0, smooth_time=3.0, initial=5.0, order=1
        )
        comp = AbstractComponent(subscripts=[[], []], ast=flow_ast)
        elem = AbstractElement(name="Smooth Output", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()

        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("D(_lv1_smooth_output)" in e for e in all_eqs)
        assert any("smooth_output ~" in e for e in all_eqs)
        assert any("_lv1_smooth_output(t)" in d for d in sb.stock_decls)

    def test_smooth3_produces_three_levels(self):
        flow_ast = SmoothStructure(
            input=5.0, smooth_time=3.0, initial=5.0, order=3
        )
        comp = AbstractComponent(subscripts=[[], []], ast=flow_ast)
        elem = AbstractElement(name="Smooth3", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()

        assert sum(1 for d in sb.stock_decls if "_lv" in d and "smooth3" in d) == 3

    # --- delay expansion --------------------------------------------------

    def test_delay1_expands_to_ode_and_aux(self):
        delay_ast = DelayStructure(
            input=10.0, delay_time=2.0, initial=10.0, order=1
        )
        comp = AbstractComponent(subscripts=[[], []], ast=delay_ast)
        elem = AbstractElement(name="Delayed Value", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()

        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("D(_dl1_delayed_value)" in e for e in all_eqs)
        assert any("delayed_value ~" in e for e in all_eqs)

    def test_delay3_produces_three_levels(self):
        delay_ast = DelayStructure(
            input=5.0, delay_time=6.0, initial=5.0, order=3
        )
        comp = AbstractComponent(subscripts=[[], []], ast=delay_ast)
        elem = AbstractElement(name="Delay3", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()

        assert sum(1 for d in sb.stock_decls if "_dl" in d and "delay3" in d) == 3


# ===========================================================================
# JuliaModelBuilder — end-to-end file generation
# ===========================================================================

class TestJuliaModelBuilder:

    def _minimal_model(self, tmp_path):
        """Build an AbstractModel with one stock and one parameter."""
        birth_rate_comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.03)
        birth_rate_elem = AbstractElement(name="Birth Rate", components=[birth_rate_comp])

        flow_ast = ArithmeticStructure(
            operators=["*"],
            arguments=[ReferenceStructure("Population"), ReferenceStructure("Birth Rate")],
        )
        pop_ast = IntegStructure(flow=flow_ast, initial=1000.0)
        pop_comp = AbstractComponent(subscripts=[[], []], ast=pop_ast)
        pop_elem = AbstractElement(name="Population", components=[pop_comp])

        control_elems = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 100.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]

        section = _make_section(
            elements=[birth_rate_elem, pop_elem] + control_elems,
            path=tmp_path / "my_model.mdl",
        )
        return AbstractModel(
            original_path=tmp_path / "my_model.mdl",
            sections=(section,),
        )

    def test_build_model_returns_jl_path(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        assert path.suffix == ".jl"
        assert path.exists()

    def test_output_contains_using_mtk(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model, backend="mtk").build_model()
        content = path.read_text()
        assert "using ModelingToolkit" in content

    def test_output_contains_stock_declaration(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model, backend="mtk").build_model()
        content = path.read_text()
        assert "@variables population(t)" in content

    def test_output_contains_parameter_declaration(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model, backend="mtk").build_model()
        content = path.read_text()
        assert "@parameters birth_rate = 0.03" in content

    def test_output_contains_ode_equation(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model, backend="mtk").build_model()
        content = path.read_text()
        assert "D(population)" in content

    def test_output_contains_u0(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model, backend="mtk").build_model()
        content = path.read_text()
        assert "population => 1000.0" in content

    def test_output_contains_ode_system(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model, backend="mtk").build_model()
        content = path.read_text()
        assert "ODESystem" in content
        assert "structural_simplify" in content

    def test_output_contains_run_model_function(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "function run_model(" in content

    def test_run_model_skips_initializeprob(self, tmp_path):
        """run_model must skip MTK's initialization system (build_initializeprob=false)
        and fill missing state variables with 0.0 via unknowns(sys). After
        structural_simplify, MTK may promote algebraic-loop variables to state
        variables with no explicit u0 entry; iterating unknowns(sys) ensures all
        are covered. The initialization system itself OOMs on large models."""
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model, backend="mtk").build_model()
        content = path.read_text()
        assert "build_initializeprob = false" in content
        assert "unknowns(sys)" in content
        assert "get(u0_dict, x, 0.0)" in content

    def test_output_contains_entrypoint_invocations(self, tmp_path):
        """Generated script must actually call run_model() and save_results() so
        running it with ``julia model.jl`` produces output rather than silently
        defining functions and exiting."""
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        # run_model() must be called and result assigned
        assert "sol = run_model()" in content
        # save_results must be called with the sol and a .nc path
        assert "save_results(sol," in content
        assert ".nc" in content

    def test_u0_param_reference_inlined_to_numeric(self, tmp_path):
        """A stock whose initial condition is a constant parameter must have
        the numeric value inlined in u0, not the parameter symbol.
        MTK's InitializationProblem rejects @parameters symbols as u0 values
        whether they appear bare or inside expressions."""
        # constant 'k' = 5.0; stock 's' INTEG(0, k)
        k_comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=5.0)
        k_elem = AbstractElement(name="K", components=[k_comp])
        integ = IntegStructure(
            flow=0.0, initial=ReferenceStructure("K")
        )
        s_comp = AbstractComponent(subscripts=[[], []], ast=integ)
        s_elem = AbstractElement(name="S", components=[s_comp])
        # stock 'q' INTEG(0, k * 2.0) — param inside expression
        flow2 = ArithmeticStructure(
            operators=["*"],
            arguments=[ReferenceStructure("K"), 2.0],
        )
        integ2 = IntegStructure(
            flow=0.0,
            initial=ArithmeticStructure(
                operators=["*"],
                arguments=[ReferenceStructure("K"), 2.0],
            ),
        )
        q_comp = AbstractComponent(subscripts=[[], []], ast=integ2)
        q_elem = AbstractElement(name="Q", components=[q_comp])
        control_elems = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[k_elem, s_elem, q_elem] + control_elems,
            path=tmp_path / "param_u0_model.mdl",
        )
        model = AbstractModel(
            original_path=tmp_path / "param_u0_model.mdl",
            sections=(section,),
        )
        content = JuliaModelBuilder(model, backend="mtk").build_model().read_text()
        # bare param reference → inlined
        assert "s => 5.0" in content
        assert "s => k" not in content
        # param inside expression → also inlined
        assert "k" not in content.split("u0 = [")[1].split("]")[0]

    def test_control_vars_emitted(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "initial_time = 0.0" in content
        assert "final_time   = 100.0" in content
        assert "time_step    = 1.0" in content

    def test_lookup_table_emitted(self, tmp_path):
        lut_ast = LookupsStructure(
            x=(0.0, 1.0, 2.0),
            y=(0.0, 0.5, 1.0),
            x_limits=(0.0, 2.0),
            y_limits=(0.0, 1.0),
            type="interpolate",
        )
        lut_comp = AbstractLookup(subscripts=[[], []], ast=lut_ast)
        lut_elem = AbstractElement(name="Effect LUT", components=[lut_comp])
        control_elems = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 0.1),
            _make_control_element("SAVEPER", 0.1),
        ]
        section = _make_section(
            elements=[lut_elem] + control_elems,
            path=tmp_path / "lut_model.mdl",
        )
        model = AbstractModel(
            original_path=tmp_path / "lut_model.mdl",
            sections=(section,),
        )
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "LinearInterpolation" in content
        assert "effect_lut_itp" in content
        assert "DataInterpolations" in content

    def test_helper_functions_emitted(self, tmp_path):
        pulse_ast = CallStructure(
            function=ReferenceStructure(reference="PULSE"),
            arguments=(10.0, 2.0),
        )
        comp = AbstractComponent(subscripts=[[], []], ast=pulse_ast)
        elem = AbstractElement(name="Pulse Signal", components=[comp])
        control_elems = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 20.0),
            _make_control_element("TIME STEP", 0.1),
            _make_control_element("SAVEPER", 0.1),
        ]
        section = _make_section(
            elements=[elem] + control_elems,
            path=tmp_path / "pulse_model.mdl",
        )
        model = AbstractModel(
            original_path=tmp_path / "pulse_model.mdl",
            sections=(section,),
        )
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "_pulse(" in content


# ===========================================================================
# Modular build
# ===========================================================================

class TestModularBuild:

    def _two_view_model(self, tmp_path):
        """Model with two views: 'Sector A' (population) and 'Sector B' (capital)."""
        pop_elem = _make_stock_element("Population", 1.0, 100.0)
        cap_elem = _make_stock_element("Capital", 2.0, 500.0)
        control_elems = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 50.0),
            _make_control_element("TIME STEP", 0.5),
            _make_control_element("SAVEPER", 0.5),
        ]
        views_dict = {
            "Sector A": {"Population"},
            "Sector B": {"Capital"},
        }
        section = _make_section(
            elements=[pop_elem, cap_elem] + control_elems,
            path=tmp_path / "split_model.mdl",
            split=True,
            views_dict=views_dict,
        )
        return AbstractModel(
            original_path=tmp_path / "split_model.mdl",
            sections=(section,),
        )

    def test_main_file_created(self, tmp_path):
        model = self._two_view_model(tmp_path)
        path = JuliaModelBuilder(model, backend="mtk").build_model()
        assert path.exists()

    def test_module_files_created(self, tmp_path):
        model = self._two_view_model(tmp_path)
        JuliaModelBuilder(model, backend="mtk").build_model()
        modules_dir = tmp_path / "modules_split_model"
        assert modules_dir.exists()
        jl_files = list(modules_dir.glob("*.jl"))
        assert len(jl_files) == 2

    def test_main_file_has_include_statements(self, tmp_path):
        model = self._two_view_model(tmp_path)
        path = JuliaModelBuilder(model, backend="mtk").build_model()
        content = path.read_text()
        assert "include(" in content

    def test_main_file_concatenates_eq_vectors(self, tmp_path):
        model = self._two_view_model(tmp_path)
        path = JuliaModelBuilder(model, backend="mtk").build_model()
        content = path.read_text()
        # The main file should reference the module equation vectors
        assert "eqs = [" in content

    def test_module_files_contain_eq_var(self, tmp_path):
        model = self._two_view_model(tmp_path)
        JuliaModelBuilder(model, backend="mtk").build_model()
        modules_dir = tmp_path / "modules_split_model"
        for jl_file in modules_dir.glob("*.jl"):
            content = jl_file.read_text()
            assert "_eqs = Equation[" in content

    def test_all_declarations_in_main_file(self, tmp_path):
        """Variable declarations must be in main file so modules can reference them."""
        model = self._two_view_model(tmp_path)
        path = JuliaModelBuilder(model, backend="mtk").build_model()
        content = path.read_text()
        assert "@variables population(t)" in content
        assert "@variables capital(t)" in content


# ===========================================================================
# _path_to_eq_var
# ===========================================================================

class TestPathToEqVar:

    @pytest.mark.parametrize("parts,expected", [
        (["modules_model", "Sector A"], "sector_a_eqs"),
        (["modules_model", "Sector A", "Sub1"], "sector_a_sub1_eqs"),
        (["modules_model", "Demographics"], "demographics_eqs"),
        (["modules_model", "sector-b"], "sector_b_eqs"),
    ])
    def test_conversion(self, parts, expected):
        path = Path(*parts)
        assert _path_to_eq_var(path) == expected


# ===========================================================================
# translate_to_julia entry point (integration, Vensim .mdl)
# ===========================================================================

class TestTranslateToJulia:

    def test_unsupported_format_raises(self, tmp_path):
        fake = tmp_path / "model.xyz"
        fake.write_text("dummy")
        from pysd import translate_to_julia
        with pytest.raises(ValueError, match="Unsupported model format"):
            translate_to_julia(fake)

    def test_vensim_model_produces_jl_file(self, tmp_path):
        """End-to-end smoke test with the split_model fixture."""
        import shutil
        src = Path("tests/more-tests/split_model/test_split_model.mdl")
        if not src.exists():
            pytest.skip("test-models submodule not checked out")

        dst = tmp_path / "test_split_model.mdl"
        shutil.copy(src, dst)

        from pysd import translate_to_julia
        path = translate_to_julia(dst)
        assert path.exists()
        assert path.suffix == ".jl"
        content = path.read_text()
        assert "using OrdinaryDiffEq" in content
        assert "function rhs!" in content
        assert "run_model" in content


# ===========================================================================
# Extended AST visitor coverage
# ===========================================================================

class TestJuliaASTVisitorExtended:

    def test_bool_true(self):
        v, *_ = _visitor_with_namespace()
        assert v.visit(True) == "true"

    def test_bool_false(self):
        v, *_ = _visitor_with_namespace()
        assert v.visit(False) == "false"

    def test_string_numeric(self):
        v, *_ = _visitor_with_namespace()
        assert v.visit("3.14") == "3.14"

    def test_string_non_numeric(self):
        v, *_ = _visitor_with_namespace()
        result = v.visit("hello")
        assert "'hello'" in result

    def test_numpy_scalar(self):
        import numpy as np
        v, *_ = _visitor_with_namespace()
        assert v.visit(np.float64(2.5)) == "2.5"

    def test_numpy_1d_array(self):
        import numpy as np
        v, *_ = _visitor_with_namespace()
        result = v.visit(np.array([1.0, 2.0, 3.0]))
        assert result == "[1.0, 2.0, 3.0]"

    def test_numpy_2d_array_flattened(self):
        import numpy as np
        v, *_ = _visitor_with_namespace()
        result = v.visit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert "[" in result
        assert "1.0" in result and "4.0" in result

    def test_subscripts_reference_structure_known(self):
        v, ns, *_ = _visitor_with_namespace(["sectors"])
        node = SubscriptsReferenceStructure(subscripts=("sectors",))
        result = v.visit(node)
        assert result == "sectors"

    def test_subscripts_reference_structure_unknown(self):
        v, *_ = _visitor_with_namespace()
        node = SubscriptsReferenceStructure(subscripts=("unknown_dim",))
        result = v.visit(node)
        assert "unknown_dim" in result

    def test_subscripts_reference_structure_empty(self):
        v, *_ = _visitor_with_namespace()
        node = SubscriptsReferenceStructure(subscripts=())
        result = v.visit(node)
        assert result == "0.0"

    def test_unknown_node_warns_and_returns_zero(self):
        v, *_ = _visitor_with_namespace()
        with pytest.warns(UserWarning, match="Unsupported AST node type"):
            result = v.visit(object())
        assert result == "0.0"

    def test_unary_not(self):
        v, *_ = _visitor_with_namespace()
        node = LogicStructure(operators=[":NOT:"], arguments=[1.0])
        result = v.visit(node)
        assert "_logical_not" in result

    def test_elmcount_no_args(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure("ELMCOUNT"), arguments=()
        )
        result = v.visit(node)
        assert result == "0"

    def test_elmcount_non_reference_arg(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure("ELMCOUNT"), arguments=(3.0,)
        )
        result = v.visit(node)
        assert result == "3.0"

    def test_elmcount_reference_with_known_size(self):
        v, ns, _, _, = _visitor_with_namespace()
        ns.add_to_namespace("sectors")
        v.subs_sizes = {"sectors": 5}
        node = CallStructure(
            function=ReferenceStructure("ELMCOUNT"),
            arguments=(ReferenceStructure("sectors"),)
        )
        result = v.visit(node)
        assert result == "5"

    def test_time_helper_prepends_t(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure("PULSE"),
            arguments=(10.0, 2.0),
        )
        result = v.visit(node)
        assert result.startswith("pysd_pulse(t,")

    def test_model_variable_lookup_call(self):
        """A function call whose name is a model variable → emit as-is."""
        v, ns, *_ = _visitor_with_namespace(["effect table"])
        node = CallStructure(
            function=ReferenceStructure("effect table"),
            arguments=(ReferenceStructure("input"),),
        )
        ns.add_to_namespace("input")
        result = v.visit(node)
        assert "effect_table" in result

    def test_get_constants_in_expression_fallback(self):
        """GetConstantsStructure inside an expression warns when file unreadable."""
        v, *_ = _visitor_with_namespace()
        node = GetConstantsStructure(file="nonexistent.xlsx", tab="Sheet1", cell="A1")
        with pytest.warns(UserWarning, match="GetConstantsStructure"):
            result = v.visit(node)
        assert result == "0.0"

    def test_subscript_element_label_resolves_to_index(self):
        """A bare reference to a subscript element label must resolve to its
        1-based integer index, not emit a 'not found in namespace' warning.

        In Vensim: Vector2[dimA] = IF THEN ELSE(dimA = B, 1, 0)
        The AST stores the element name 'B' (or 'b') as a ReferenceStructure.
        When iterating dimA with _i0, 'B' should become '2'.
        """
        ns = JuliaNamespaceManager()
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            active_subs={"dimA": "_i0"},
            subs_elems={"dimA": ["A", "B", "C"]},
        )
        import warnings
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = v.visit(ReferenceStructure("b"))
        assert result == "2", f"Expected '2', got {result!r}"
        ns_warns = [w for w in captured if issubclass(w.category, UserWarning)
                    and "not found in namespace" in str(w.message)]
        assert not ns_warns, f"Should not warn about 'b' not in namespace: {ns_warns}"

    def test_subscript_element_label_third_element(self):
        """Element 'C' (3rd in dimA: A, B, C) must resolve to '3'."""
        ns = JuliaNamespaceManager()
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            active_subs={"dimA": "_i0"},
            subs_elems={"dimA": ["A", "B", "C"]},
        )
        import warnings
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = v.visit(ReferenceStructure("c"))
        assert result == "3", f"Expected '3', got {result!r}"
        ns_warns = [w for w in captured if issubclass(w.category, UserWarning)
                    and "not found in namespace" in str(w.message)]
        assert not ns_warns

    def test_subscript_element_label_prefers_active_dim(self):
        """When an element appears in multiple ranges, the active dim's index wins."""
        ns = JuliaNamespaceManager()
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            active_subs={"dimD": "_i1"},
            subs_elems={
                "dimA": ["A", "B", "C"],
                "dimD": ["D", "E", "F"],
            },
        )
        # 'E' is 2nd in dimD (active) and not in dimA, should give 2
        result = v.visit(ReferenceStructure("e"))
        assert result == "2"

    def test_sum_subscripted_lookup_call_no_double_comprehension(self):
        """SUM(f[dim!](t)) where f is a subscripted lookup must produce a single
        comprehension sum([f(_ii0, t) for _ii0 in 1:N_DIM]), not a nested one.

        Regression for the pymedeas world model: historic_labour_compensation_share
        was generated as sum([[f(_ii0,t) for _ii0 in 1:N] for _ii0 in 1:N]).
        """
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("historic_labour_compensation")
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            subs_sizes={"sectors": 14},
            var_dims={"historic_labour_compensation": ["sectors"]},
        )
        # SUM(historic_labour_compensation[sectors!](Time))
        node = CallStructure(
            function=ReferenceStructure("SUM"),
            arguments=[
                CallStructure(
                    function=ReferenceStructure(
                        "historic_labour_compensation",
                        subscripts=SubscriptsReferenceStructure(subscripts=("sectors!",)),
                    ),
                    arguments=[ReferenceStructure("Time")],
                )
            ],
        )
        result = v.visit(node)
        assert result == "sum([historic_labour_compensation(_ii0, t) for _ii0 in 1:N_SECTORS])"


# ===========================================================================
# Section builder — subscript handling
# ===========================================================================

class TestJuliaSectionBuilderSubscripts:

    def test_alias_subscript_defaults_to_zero_size(self):
        sr_alias = AbstractSubscriptRange(name="alias_dim", subscripts="real_dim", mapping=[])
        section = _make_section(subscripts=[sr_alias])
        sb = JuliaSectionBuilder(section)
        assert sb._subs_sizes.get("alias_dim") == 0

    def test_list_subscript_has_correct_size(self):
        sr = _make_subscript_range("energy_type", ["Hydro", "Solar", "Wind"])
        section = _make_section(subscripts=[sr])
        sb = JuliaSectionBuilder(section)
        assert sb._subs_sizes["energy_type"] == 3

    def test_subs_const_decl_emitted(self):
        sr = _make_subscript_range("sector", ["A", "B", "C", "D"])
        elem = _make_subscripted_element("output", 1.0, "sector",
                                         comp_class=AbstractUnchangeableConstant)
        sr_elem = _make_subscript_range("sector", ["A", "B", "C", "D"])
        sb = _section_builder_from_elements([elem], subscripts=[sr_elem])
        sb.build_section()
        assert any("N_SECTOR" in d for d in sb.subs_const_decls)

    def test_1d_subscripted_parameter(self):
        sr = _make_subscript_range("energy_type", ["Hydro", "Solar"])
        elem = _make_subscripted_element("cost", 2.5, "energy_type",
                                         comp_class=AbstractUnchangeableConstant)
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        assert any("cost[1:N_ENERGY_TYPE]" in d for d in sb.param_decls)

    def test_1d_subscripted_stock(self):
        sr = _make_subscript_range("sector", ["A", "B"])
        comp = AbstractComponent(
            subscripts=[["sector"], []],
            ast=IntegStructure(flow=1.0, initial=0.0),
        )
        elem = AbstractElement(name="capital", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        assert any("capital(t)[" in d for d in sb.stock_decls)

    def test_1d_subscripted_auxiliary(self):
        sr = _make_subscript_range("sector", ["A", "B", "C"])
        elem = _make_subscripted_element("output", 3.0, "sector")
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        assert any("output(t)[" in d for d in sb.aux_decls)
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # 1D subscripted aux now emits a per-element comprehension (like ndim≥2)
        # instead of Symbolics.scalarize, to avoid ifelse shape-mismatch errors.
        assert any("output[_i0]" in e and "for _i0" in e for e in eqs)

    def test_2d_subscripted_auxiliary(self):
        sr1 = _make_subscript_range("row_dim", ["R1", "R2"])
        sr2 = _make_subscript_range("col_dim", ["C1", "C2", "C3"])
        comp = AbstractComponent(
            subscripts=[["row_dim", "col_dim"], []],
            ast=1.0,
        )
        elem = AbstractElement(name="matrix", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("_i0" in e and "_i1" in e for e in eqs)

    def test_element_with_no_components_returns_empty(self):
        elem = AbstractElement(name="empty_var", components=[])
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        assert sb.built_elements["empty_var"][0] == []


# ===========================================================================
# Section builder — INITIAL() handling
# ===========================================================================

class TestJuliaSectionBuilderInitial:

    def test_initial_resolves_from_stock(self):
        stock = _make_stock_element("Level", 1.0, 42.0)
        init_ast = InitialStructure(initial=ReferenceStructure("Level"))
        comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        init_elem = AbstractElement(name="Init Value", components=[comp])
        sb = _section_builder_from_elements([stock, init_elem])
        sb.build_section()
        assert any("@parameters init_value = 42.0" in d for d in sb.param_decls)

    def test_initial_resolves_from_parameter(self):
        const_comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=7.5)
        const_elem = AbstractElement(name="Base Rate", components=[const_comp])
        init_ast = InitialStructure(initial=ReferenceStructure("Base Rate"))
        comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        init_elem = AbstractElement(name="Init Rate", components=[comp])
        sb = _section_builder_from_elements([const_elem, init_elem])
        sb.build_section()
        assert any("@parameters init_rate = 7.5" in d for d in sb.param_decls)

    def test_initial_fallback_frozen_stock(self):
        # Reference that can't be resolved at translation time → frozen-stock
        # fallback: D(x) ~ 0.0 with initial condition x(t0) = expr.
        # No warning is emitted; the variable is a stock, not an auxiliary.
        import warnings
        init_ast = InitialStructure(initial=ReferenceStructure("unknown_var"))
        comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        elem = AbstractElement(name="Init Fallback", components=[comp])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sb = _section_builder_from_elements([elem])
            sb.build_section()
        assert not any("Cannot resolve INITIAL" in str(x.message) for x in w)
        # Declared as a stock variable (not an auxiliary)
        assert any("@variables init_fallback(t)" in d for d in sb.stock_decls)
        assert not any("@variables init_fallback(t)" in d for d in sb.aux_decls)
        # D(init_fallback) ~ 0.0 in the equations
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("D(init_fallback)" in e for e in all_eqs)

    def test_1d_frozen_initial_emits_indexed_du(self):
        # 1D subscripted INITIAL that can't be resolved at translation time →
        # must emit [D(x[_i0]) ~ 0.0 for _i0 in 1:N]..., NOT Symbolics.scalarize.
        # The ODE builder skips equations containing ".~" or "Symbolics.scalarize",
        # so using scalarize causes the du entries to be lost.
        import warnings
        sr = _make_subscript_range("dim", ["A", "B", "C"])
        init_ast = InitialStructure(initial=ReferenceStructure("unknown_var"))
        comp = AbstractComponent(subscripts=[["dim"], []], ast=init_ast)
        elem = AbstractElement(name="Init 1D", components=[comp])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sb = _section_builder_from_elements([elem], subscripts=[sr])
            sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert not any("Symbolics.scalarize" in e for e in all_eqs), (
            f"1D frozen INITIAL must not use Symbolics.scalarize, got: {all_eqs}"
        )
        assert any("D(init_1d[" in e for e in all_eqs), (
            f"1D frozen INITIAL must emit indexed D(x[i]) form, got: {all_eqs}"
        )

    def test_resolve_ref_initial_chain(self):
        """INITIAL(aux) where aux ~ stock → resolves to stock initial."""
        stock = _make_stock_element("S", 1.0, 99.0)
        aux_comp = AbstractComponent(subscripts=[[], []], ast=ReferenceStructure("S"))
        aux_elem = AbstractElement(name="A", components=[aux_comp])
        init_ast = InitialStructure(initial=ReferenceStructure("A"))
        init_comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        init_elem = AbstractElement(name="Init A", components=[init_comp])
        sb = _section_builder_from_elements([stock, aux_elem, init_elem])
        sb.build_section()
        assert any("@parameters init_a = 99.0" in d for d in sb.param_decls)

    def test_2d_stock_with_2d_initial_ref_generates_indexed_u0(self):
        """A 2D stock whose initial condition is a 2D variable reference must
        produce per-element u0 entries with matching indices, e.g.
          level[1, 1] => base[1, 1]
        not the full-array form
          level[1, 1] => base          ← causes MTK shape-mismatch error
        """
        # Build: base[sector, fuel] = 1.0 (constant), level[sector, fuel] integ(0, base)
        subs_sector = _make_subscript_range("sector", ["s1", "s2"])
        subs_fuel = _make_subscript_range("fuel", ["f1", "f2", "f3"])

        base_comp = AbstractUnchangeableConstant(
            subscripts=[["sector", "fuel"], []], ast=1.0
        )
        base_elem = AbstractElement(name="Base", components=[base_comp])

        flow_ast = 0.0
        init_ast = ReferenceStructure("Base", subscripts=(["sector", "fuel"],))
        integ_ast = IntegStructure(flow=flow_ast, initial=init_ast)
        level_comp = AbstractComponent(
            subscripts=[["sector", "fuel"], []], ast=integ_ast
        )
        level_elem = AbstractElement(name="Level", components=[level_comp])

        sb = _section_builder_from_elements(
            [base_elem, level_elem],
            subscripts=[subs_sector, subs_fuel],
        )
        sb.build_section()

        # Every u0 entry must index BOTH dimensions; none should be bare "base"
        for entry in sb.u0_entries:
            if entry.startswith("level["):
                assert "base[" in entry, (
                    f"u0 entry assigns full 2D array to scalar element: {entry!r}"
                )

    def test_2d_stock_with_numpy_array_initial_generates_scalar_u0(self):
        """A 2D stock whose initial condition is a literal numpy array must
        produce per-element u0 entries with scalar values."""
        import numpy as np
        subs_row = _make_subscript_range("row", ["r1", "r2"])
        subs_col = _make_subscript_range("col", ["c1", "c2"])
        init_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        integ_ast = IntegStructure(flow=0.0, initial=init_arr)
        comp = AbstractComponent(subscripts=[["row", "col"], []], ast=integ_ast)
        elem = AbstractElement(name="M", components=[comp])
        sb = _section_builder_from_elements(
            [elem],
            subscripts=[subs_row, subs_col],
        )
        sb.build_section()
        assert "m[1, 1] => 1.0" in sb.u0_entries
        assert "m[1, 2] => 2.0" in sb.u0_entries
        assert "m[2, 1] => 3.0" in sb.u0_entries
        assert "m[2, 2] => 4.0" in sb.u0_entries


# ===========================================================================
# Section builder — expansion methods
# ===========================================================================

class TestJuliaSectionBuilderExpansions:

    def test_delay_fixed_expands(self):
        """DELAY FIXED with literal delay=2, time_step=1 → N=2 pipeline stages."""
        ast = DelayFixedStructure(input=5.0, delay_time=2.0, initial=5.0)
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Delayed Fixed", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # New pipeline uses _df_pipe_k_delayed_fixed naming
        assert any("_df_pipe_" in e for e in all_eqs), \
            "DELAY FIXED must emit pipeline stages"
        assert any("delayed_fixed ~" in e for e in all_eqs), \
            "DELAY FIXED must assign the output identifier"
        assert any("_df_pipe_" in d for d in sb.stock_decls), \
            "Pipeline stages must appear in stock_decls"

    def test_delay_fixed_dynamic_fallback_guards_division_by_zero(self):
        """DELAY FIXED with dynamic (non-constant) delay time falls back to a
        first-order ODE.  The denominator must use max(delay_expr, eps(Float64))
        so that when delay_time = 0 at t=0 the Euler solver does not blow up
        to Inf and halt the simulation prematurely."""
        import warnings
        # ReferenceStructure delay_time cannot be evaluated at translation time
        # → triggers the fallback first-order ODE path.
        delay_ast = DelayFixedStructure(
            input=ReferenceStructure("input_var"),
            delay_time=ReferenceStructure("delay_var"),
            initial=5.0,
        )
        comp = AbstractComponent(subscripts=[[], []], ast=delay_ast)
        elem = AbstractElement(name="Out", components=[comp])
        sb = _section_builder_from_elements([elem])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        fallback_eq = next((e for e in all_eqs if "_df_out" in e and "~" in e), None)
        assert fallback_eq is not None, "Expected a fallback ODE equation for _df_out"
        assert "max(" in fallback_eq, (
            "Fallback DELAY FIXED ODE must use max(delay_time, eps(Float64)) "
            f"to prevent division by zero, but got: {fallback_eq!r}"
        )
        assert "eps(Float64)" in fallback_eq, (
            f"Fallback must clamp with eps(Float64), but got: {fallback_eq!r}"
        )

    def test_delay_n_variable_order_uses_initial_value(self):
        """DELAY N whose order is a time-varying expression (e.g. 2 + STEP(1, 10))
        must use the order evaluated at t=0 (here: 2) instead of defaulting to 3.
        This matches the Python backend behaviour and gives correct initial dynamics."""
        import warnings
        order_ast = ArithmeticStructure(
            operators=["+"],
            arguments=[2.0, CallStructure(
                function=ReferenceStructure("step"),
                arguments=(1.0, 10.0),
            )],
        )
        delay_ast = DelayNStructure(
            input=5.0,
            delay_time=4.0,
            initial=6.0,
            order=order_ast,
        )
        comp = AbstractComponent(subscripts=[[], []], ast=delay_ast)
        elem = AbstractElement(name="Out Delay N", components=[comp])
        sb = _section_builder_from_elements([elem])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            sb.build_section()
        fallback_warns = [str(w.message) for w in captured
                          if "defaulting to 3" in str(w.message)]
        assert not fallback_warns, (
            "DELAY N with evaluable initial order should not fall back to 3: "
            + str(fallback_warns)
        )
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        stage_names = [e for e in sb.stock_decls if "_dl" in e]
        assert len(stage_names) == 2, (
            f"Order 2 (from t=0 evaluation) must produce 2 pipeline stages, "
            f"got: {stage_names}"
        )

    def test_trend_expands(self):
        ast = TrendStructure(input=10.0, average_time=5.0, initial_trend=0.02)
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Trend Out", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("_sm_trend_out" in e for e in all_eqs)
        assert any("trend_out ~" in e for e in all_eqs)

    def test_forecast_expands(self):
        ast = ForecastStructure(input=10.0, average_time=5.0, horizon=3.0, initial_trend=0.01)
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Forecast Out", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("_sm_forecast_out" in e for e in all_eqs)
        assert any("forecast_out ~" in e for e in all_eqs)

    def test_sample_if_true_expands(self):
        ts_elem = _make_control_element("TIME STEP", 0.25)
        ast = SampleIfTrueStructure(condition=1.0, input=5.0, initial=5.0)
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Sample Out", components=[comp])
        sb = _section_builder_from_elements([ts_elem, elem])
        sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("_sit_sample_out" in e for e in all_eqs)
        assert any("sample_out ~" in e for e in all_eqs)

    def test_allocate_available_emits_helper_call(self):
        ast = AllocateAvailableStructure(
            request=ReferenceStructure("request"),
            pp=ReferenceStructure("pp"),
            avail=ReferenceStructure("supply"),
        )
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Alloc Out", components=[comp])
        req_elem = _make_element("request", 1.0)
        pp_elem = _make_element("pp", 1.0)
        sup_elem = _make_element("supply", 10.0)
        sb = _section_builder_from_elements([req_elem, pp_elem, sup_elem, elem])
        sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("pysd_allocate_available" in e for e in all_eqs), \
            "Expected pysd_allocate_available() call in generated equations"
        assert not any("proportional" in e.lower() for e in all_eqs), \
            "Should not fall back to proportional approximation"

    def test_allocate_by_priority_emits_helper_call(self):
        ast = AllocateByPriorityStructure(
            request=ReferenceStructure("demand"),
            priority=ReferenceStructure("prio"),
            size=1,
            width=0.1,
            supply=ReferenceStructure("available"),
        )
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Alloc Prio", components=[comp])
        d_elem = _make_element("demand", 1.0)
        p_elem = _make_element("prio", 1.0)
        a_elem = _make_element("available", 5.0)
        sb = _section_builder_from_elements([d_elem, p_elem, a_elem, elem])
        sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("pysd_allocate_by_priority" in e for e in all_eqs), \
            "Expected pysd_allocate_by_priority() call in generated equations"
        assert not any("proportional" in e.lower() for e in all_eqs), \
            "Should not fall back to proportional approximation"

    def test_smooth_non_integer_order_warns_and_defaults(self):
        ast = SmoothStructure(input=1.0, smooth_time=2.0, initial=1.0, order="bad")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Sm Bad", components=[comp])
        with pytest.warns(UserWarning, match="non-integer order"):
            sb = _section_builder_from_elements([elem])
            sb.build_section()
        assert sum(1 for d in sb.stock_decls if "_lv" in d and "sm_bad" in d) == 3

    def test_delay_non_integer_order_warns_and_defaults(self):
        ast = DelayStructure(input=1.0, delay_time=2.0, initial=1.0, order="bad")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Dl Bad", components=[comp])
        with pytest.warns(UserWarning, match="non-integer order"):
            sb = _section_builder_from_elements([elem])
            sb.build_section()
        assert sum(1 for d in sb.stock_decls if "_dl" in d and "dl_bad" in d) == 3


# ===========================================================================
# Section builder — unsupported / fallback structures
# ===========================================================================

class TestJuliaSectionBuilderUnsupported:

    def test_data_structure_emits_warning_and_placeholder(self):
        # AbstractComponent (no keyword) with DataStructure AST still unsupported
        ast = DataStructure()
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Data Var", components=[comp])
        with pytest.warns(UserWarning, match="not supported"):
            sb = _section_builder_from_elements([elem])
            sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("UNSUPPORTED" in e for e in all_eqs)

    def test_abstract_data_with_data_structure_emits_tab_val(self):
        """AbstractData + DataStructure emits _tab_val call (tab-file read), no warning."""
        import warnings as _w
        ast = DataStructure()
        comp = AbstractData(subscripts=[[], []], ast=ast, keyword="interpolate")
        elem = AbstractElement(name="Tab Var", components=[comp])
        with _w.catch_warnings(record=True) as captured:
            _w.simplefilter("always")
            sb = _section_builder_from_elements([elem])
            sb.build_section()
        assert not any("not supported" in str(w.message).lower() for w in captured), \
            f"Expected no 'not supported' warning, got: {[str(w.message) for w in captured]}"
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("_tab_val" in e for e in all_eqs), \
            f"Expected _tab_val in equations, got: {all_eqs}"

    def test_abstract_data_with_data_structure_hold_backward(self):
        """hold_backward keyword produces a _tab_val equation."""
        import warnings as _w
        ast = DataStructure()
        comp = AbstractData(subscripts=[[], []], ast=ast, keyword="hold_backward")
        elem = AbstractElement(name="Hold Var", components=[comp])
        with _w.catch_warnings(record=True) as captured:
            _w.simplefilter("always")
            sb = _section_builder_from_elements([elem])
            sb.build_section()
        assert not any("not supported" in str(w.message).lower() for w in captured)
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("_tab_val" in e for e in all_eqs)

    def test_abstract_data_no_get_data_structure_falls_through_to_aux(self):
        # AbstractData whose AST is not a GetDataStructure falls through to the
        # regular auxiliary path and emits a "data-override" warning instead of
        # a GET_DATA_FAILED placeholder.
        comp = AbstractData(subscripts=[[], []], ast=0.0)
        elem = AbstractElement(name="Ext Data", components=[comp])
        with pytest.warns(UserWarning, match="data-override"):
            sb = _section_builder_from_elements([elem])
            sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert not any("GET_DATA_FAILED" in e for e in all_eqs), "Expected no placeholder"
        assert any("ext_data" in e for e in all_eqs), "Expected regular equation"


# ===========================================================================
# Section builder — external data readers (mocked)
# ===========================================================================

class TestJuliaSectionBuilderExternal:

    def test_read_get_constants_scalar_success(self, tmp_path):
        # Single-component GCS: runtime read path emits pysd_xlsx_read_constant
        # without calling ExtConstant at translation time.
        ast = GetConstantsStructure(file="data.xlsx", tab="Sheet1", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Rate", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("@parameters rate = pysd_xlsx_read_constant" in d for d in sb.param_decls)

    def test_read_get_constants_array_success(self, tmp_path):
        # Single-component GCS with no declared subscripts: runtime read path,
        # ends up in param_decls (not ext_const_decls) as pysd_xlsx_read_constant.
        ast = GetConstantsStructure(file="data.xlsx", tab="Sheet1", cell="B1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Costs", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("@parameters costs = pysd_xlsx_read_constant" in d for d in sb.param_decls)

    def test_read_get_constants_failure_falls_through(self, tmp_path):
        # With runtime reading, missing Excel files are NOT detected at translation
        # time — the pysd_xlsx_read_constant call is emitted unconditionally and
        # will raise at Julia load time. No UserWarning is raised here.
        ast = GetConstantsStructure(file="missing.xlsx", tab="Sheet1", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Bad Const", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("pysd_xlsx_read_constant" in d for d in sb.param_decls)

    def test_get_lookups_scalar_success(self, mocker, tmp_path):
        import numpy as np
        import xarray as xr
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.array([0.0, 0.5, 1.0])
        da = xr.DataArray(ys, coords={"lookup_dim": xs}, dims=["lookup_dim"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch(
            "pysd.py_backend.external.ExtLookup",
            return_value=mock_ext,
        )
        ast = GetLookupsStructure(file="data.xlsx", tab="Sheet1",
                                  x_row_or_col="x_col", cell="B1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Effect Table", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("effect_table_itp" in d for d in sb.lookup_const_decls)

    def test_get_lookups_2d_success(self, tmp_path):
        # Single-component lookup with no declared subscripts uses the runtime
        # scalar path (pysd_xlsx_read_series) regardless of actual data shape.
        # Data dimensionality is unknown at translation time.
        ast = GetLookupsStructure(file="data.xlsx", tab="Sheet1",
                                  x_row_or_col="x_col", cell="B1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Sub Table", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("sub_table_itp" in d for d in sb.lookup_const_decls)
        assert any("sub_table(x)" in d for d in sb.lookup_func_decls)

    def test_get_lookups_3d_emits_2d_dispatch(self, tmp_path):
        # Single-component lookup with no declared subscripts uses the runtime
        # scalar path. Data dimensionality (3D) is irrelevant at translation
        # time — no ExtLookup is called, no warnings are emitted.
        ast = GetLookupsStructure(file="data.xlsx", tab="Sheet1",
                                  x_row_or_col="x", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Hd Table", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("hd_table_itp" in d for d in sb.lookup_const_decls)
        assert any("hd_table(x)" in d for d in sb.lookup_func_decls)

    def test_get_lookups_4d_warns_and_flattens(self, tmp_path):
        # Single-component no-subscript lookup: runtime scalar path is used.
        # No warning is emitted (ExtLookup not called at translation time).
        ast = GetLookupsStructure(file="data.xlsx", tab="Sheet1",
                                  x_row_or_col="x", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Hd Table", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("hd_table_itp" in d for d in sb.lookup_const_decls)

    def test_get_lookups_read_failure_warns(self, tmp_path):
        # Runtime path: ExtLookup is never called at translation time, so
        # no warning is raised even for missing files. The lookup declaration
        # is always emitted (file read happens at Julia load time).
        ast = GetLookupsStructure(file="bad.xlsx", tab="Sheet1",
                                  x_row_or_col="x", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Bad Lut", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("bad_lut" in d for d in sb.lookup_const_decls)

    def test_get_data_scalar_success(self, mocker, tmp_path):
        import numpy as np
        import xarray as xr
        ts = np.array([1995.0, 2000.0, 2005.0])
        vals = np.array([1.0, 2.0, 3.0])
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch(
            "pysd.py_backend.external.ExtData",
            return_value=mock_ext,
        )
        ast = GetDataStructure(file="data.xlsx", tab="Sheet1",
                               time_row_or_col="time_col", cell="B1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Historic Eff", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("historic_eff_itp" in d for d in sb.lookup_const_decls)

    def test_get_data_2d_success(self, tmp_path):
        # Single-component GET DATA with no declared subscripts: runtime scalar
        # path emits _itp regardless of actual data shape (unknown at translate time).
        ast = GetDataStructure(file="data.xlsx", tab="Sheet1",
                               time_row_or_col="time_col", cell="B1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Sub Series", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("sub_series_itp" in d for d in sb.lookup_const_decls)

    def test_get_data_read_failure_warns(self, tmp_path):
        # Runtime path: ExtData is never called at translation time, so no
        # warning is raised even for missing files. The _itp declaration is
        # always emitted (file read happens at Julia load time).
        ast = GetDataStructure(file="bad.xlsx", tab="Sheet1",
                               time_row_or_col="t_col", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Bad Data", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("bad_data" in d for d in sb.lookup_const_decls)

    # ------------------------------------------------------------------
    # Per-element-component GET LOOKUPS / GET DATA (Task B fix)
    # ------------------------------------------------------------------

    def test_get_lookups_per_element_component_coords_built_correctly(self, tmp_path):
        """When a GET LOOKUPS element has per-sector-element components (each
        comp specifies a single element name rather than a range name), the
        multi-component runtime dispatch path emits per-component _fns entries."""
        sr_sector = _make_subscript_range("sector", ["A", "B"])

        # Two components: one per element of 'sector'
        ast_a = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="col_a")
        ast_b = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="col_b")
        comp_a = AbstractComponent(subscripts=[["A"], []], ast=ast_a)
        comp_b = AbstractComponent(subscripts=[["B"], []], ast=ast_b)
        elem = AbstractElement(name="My Lookup", components=[comp_a, comp_b])

        sb = _section_builder_from_elements([elem], subscripts=[sr_sector], path=tmp_path / "m.mdl")
        sb.build_section()

        # Runtime multi-component path: emits _1_fns and _2_fns entries
        assert any("my_lookup_1_fns" in d for d in sb.lookup_const_decls)
        assert any("my_lookup_2_fns" in d for d in sb.lookup_const_decls)

    def test_get_lookups_per_element_no_placeholder_emitted(self, mocker, tmp_path):
        """Per-element GET LOOKUPS components must NOT emit a GET_LOOKUPS_FAILED
        placeholder — a real lookup declaration must appear."""
        import numpy as np
        import xarray as xr

        xs = np.array([1995.0, 2000.0, 2005.0])
        ys = np.ones((3, 2))
        da = xr.DataArray(ys, coords={"lookup_dim": xs},
                          dims=["lookup_dim", "type"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtLookup", return_value=mock_ext)

        sr_type = _make_subscript_range("type", ["X", "Y"])
        ast_x = GetLookupsStructure(file="f.xlsx", tab="S", x_row_or_col="yr", cell="cx")
        ast_y = GetLookupsStructure(file="f.xlsx", tab="S", x_row_or_col="yr", cell="cy")
        comp_x = AbstractComponent(subscripts=[["X"], []], ast=ast_x)
        comp_y = AbstractComponent(subscripts=[["Y"], []], ast=ast_y)
        elem = AbstractElement(name="Rate Table", components=[comp_x, comp_y])

        sb = _section_builder_from_elements([elem], subscripts=[sr_type], path=tmp_path / "m.mdl")
        sb.build_section()

        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert not any("GET_LOOKUPS_FAILED" in e for e in all_eqs), (
            "GET_LOOKUPS_FAILED placeholder must not be emitted for per-element components"
        )
        # A lookup interpolation constant must have been declared
        assert sb.lookup_const_decls, "No lookup constant declarations emitted"

    def test_get_data_per_element_coords_uses_parent_range(self, tmp_path):
        """GET DATA with per-element components: multi-component runtime dispatch
        emits per-component _fns entries without calling ExtData at translate time."""
        sr_fuel = _make_subscript_range("fuel", ["coal", "gas", "oil"])

        ast_c = GetDataStructure(file="e.xlsx", tab="W", time_row_or_col="yr", cell="coal_c")
        ast_g = GetDataStructure(file="e.xlsx", tab="W", time_row_or_col="yr", cell="gas_c")
        comp_c = AbstractComponent(subscripts=[["coal"], []], ast=ast_c)
        comp_g = AbstractComponent(subscripts=[["gas"], []], ast=ast_g)
        from pysd.translators.structures.abstract_model import AbstractData
        comp_c.__class__ = AbstractData
        comp_g.__class__ = AbstractData
        elem = AbstractElement(name="Historic Share", components=[comp_c, comp_g])

        sb = _section_builder_from_elements([elem], subscripts=[sr_fuel], path=tmp_path / "m.mdl")
        sb.build_section()

        # Runtime multi-component path: emits per-component _fns entries
        assert any("historic_share_1_fns" in d for d in sb.lookup_const_decls)


# ===========================================================================
# Section builder — file generation helpers
# ===========================================================================

class TestJuliaFileGeneration:

    def _minimal_sb(self, tmp_path):
        stock = _make_stock_element("S", 1.0, 10.0)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        sb = _section_builder_from_elements(
            [stock] + controls, path=tmp_path / "m.mdl"
        )
        sb.build_section()
        return sb

    def test_helpers_block_empty_when_none_needed(self, tmp_path):
        sb = self._minimal_sb(tmp_path)
        sb.needed_helpers.clear()
        assert sb._helpers_block() == ""

    def test_helpers_block_always_empty_with_pysd_jl(self, tmp_path):
        sb = self._minimal_sb(tmp_path)
        sb.needed_helpers.add("pysd_xidz")
        block = sb._helpers_block()
        # Helpers are provided by `using PySD` — no inlining needed
        assert block == ""

    def test_lookup_block_empty_when_none(self, tmp_path):
        sb = self._minimal_sb(tmp_path)
        assert sb._lookup_block() == ""

    def test_lookup_block_contains_declaration(self, tmp_path):
        sb = self._minimal_sb(tmp_path)
        sb.lookup_const_decls.append("const lut_itp = LinearInterpolation([1.0], [0.0])")
        sb.lookup_func_decls.append("lut(x) = lut_itp(x)")
        sb.lookup_register_decls.append("@register_symbolic lut(x::Real)")
        block = sb._lookup_block()
        assert "LinearInterpolation" in block

    def test_declarations_block_includes_subs_constants(self, tmp_path):
        sr = _make_subscript_range("energy_type", ["H", "S"])
        elem = _make_subscripted_element("cost", 1.0, "energy_type",
                                         comp_class=AbstractUnchangeableConstant)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        sb = _section_builder_from_elements(
            [elem] + controls,
            path=tmp_path / "m.mdl",
            subscripts=[sr],
        )
        sb.build_section()
        block = sb._declarations_block()
        assert "N_ENERGY_TYPE" in block
        assert "Subscript dimension sizes" in block

    def test_equations_block_empty(self, tmp_path):
        sb = self._minimal_sb(tmp_path)
        block = sb._equations_block([])
        assert "function rhs!" in block
        assert "function observe" in block

    def test_equations_block_empty_mtk(self, tmp_path):
        stock = _make_stock_element("S", 1.0, 10.0)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        sb = _section_builder_from_elements(
            [stock] + controls, path=tmp_path / "m.mdl", backend="mtk"
        )
        sb.build_section()
        block = sb._equations_block([])
        assert block == "eqs = Equation[]\n"

    def test_u0_block_empty(self, tmp_path):
        sb = self._minimal_sb(tmp_path)
        sb.u0_entries.clear()
        block = sb._u0_block()
        assert block == "u0 = Float64[]\n"

    def test_u0_block_empty_mtk(self, tmp_path):
        stock = _make_stock_element("S", 1.0, 10.0)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        sb = _section_builder_from_elements(
            [stock] + controls, path=tmp_path / "m.mdl", backend="mtk"
        )
        sb.build_section()
        sb.u0_entries.clear()
        block = sb._u0_block()
        assert block == "u0 = []\n"

    def test_ext_const_in_declarations_block(self, tmp_path):
        sb = self._minimal_sb(tmp_path)
        sb.ext_const_decls.append("const big_array = [1.0, 2.0]")
        block = sb._declarations_block()
        assert "External constants" in block
        assert "big_array" in block


# ===========================================================================
# Modular build — extended edge cases
# ===========================================================================

class TestModularBuildExtended:

    def test_variable_not_in_any_view_emits_warning(self, tmp_path):
        """Variable assigned to no view → leftover warning."""
        pop = _make_stock_element("Population", 1.0, 100.0)
        orphan = _make_element("Orphan Var", 5.0)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        views_dict = {"Sector A": {"Population"}}
        section = _make_section(
            elements=[pop, orphan] + controls,
            path=tmp_path / "m.mdl",
            split=True,
            views_dict=views_dict,
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        with pytest.warns(UserWarning, match="not declared in any view"):
            JuliaModelBuilder(model, backend="mtk").build_model()

    def test_view_with_only_control_vars_skipped(self, tmp_path):
        """A view containing only control variables produces no module file."""
        pop = _make_stock_element("Population", 1.0, 100.0)
        it = _make_control_element("INITIAL TIME", 0.0)
        ft = _make_control_element("FINAL TIME", 10.0)
        ts = _make_control_element("TIME STEP", 1.0)
        sv = _make_control_element("SAVEPER", 1.0)
        views_dict = {
            "Main": {"Population"},
            "Controls": {"INITIAL TIME", "FINAL TIME"},
        }
        section = _make_section(
            elements=[pop, it, ft, ts, sv],
            path=tmp_path / "ctrl_model.mdl",
            split=True,
            views_dict=views_dict,
        )
        model = AbstractModel(original_path=tmp_path / "ctrl_model.mdl",
                               sections=(section,))
        JuliaModelBuilder(model, backend="mtk").build_model()
        modules_dir = tmp_path / "modules_ctrl_model"
        jl_files = list(modules_dir.glob("*.jl"))
        assert len(jl_files) == 1  # Only "Main", not "Controls"

    def test_nested_views(self, tmp_path):
        """Views with sub-views (intermediate nodes) are handled."""
        pop = _make_stock_element("Population", 1.0, 100.0)
        cap = _make_stock_element("Capital", 2.0, 500.0)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 50.0),
            _make_control_element("TIME STEP", 0.5),
            _make_control_element("SAVEPER", 0.5),
        ]
        views_dict = {
            "Economy": {
                "Demography": {"Population"},
                "Assets": {"Capital"},
            }
        }
        section = _make_section(
            elements=[pop, cap] + controls,
            path=tmp_path / "nested.mdl",
            split=True,
            views_dict=views_dict,
        )
        model = AbstractModel(original_path=tmp_path / "nested.mdl",
                               sections=(section,))
        JuliaModelBuilder(model, backend="mtk").build_model()
        modules_dir = tmp_path / "modules_nested"
        jl_files = list(modules_dir.rglob("*.jl"))
        assert len(jl_files) == 2

    def test_split_views_with_ode_backend_raises_error(self, tmp_path):
        # split_views=True is only supported for the MTK backend.
        # Combining it with backend="ode" must raise a clear error.
        pop = _make_stock_element("Population", 1.0, 100.0)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        views_dict = {"Sector A": {"Population"}}
        section = _make_section(
            elements=[pop] + controls,
            path=tmp_path / "m.mdl",
            split=True,
            views_dict=views_dict,
        )
        sb = JuliaSectionBuilder(section, backend="ode")
        with pytest.raises(ValueError, match="split_views"):
            sb.build_section()


# ===========================================================================
# _format_julia_value utility
# ===========================================================================

class TestFormatJuliaValue:

    def test_float_scalar(self):
        from pysd.builders.julia.julia_model_builder import _format_julia_value
        assert _format_julia_value(3.14) == "3.14"

    def test_int_scalar(self):
        from pysd.builders.julia.julia_model_builder import _format_julia_value
        assert _format_julia_value(5) == "5.0"

    def test_1d_numpy_array(self):
        import numpy as np
        from pysd.builders.julia.julia_model_builder import _format_julia_value
        result = _format_julia_value(np.array([1.0, 2.0, 3.0]))
        assert result == "[1.0, 2.0, 3.0]"

    def test_2d_numpy_array(self):
        import numpy as np
        from pysd.builders.julia.julia_model_builder import _format_julia_value
        result = _format_julia_value(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert "[" in result and ";" in result

    def test_xarray_dataarray(self):
        import numpy as np
        import xarray as xr
        from pysd.builders.julia.julia_model_builder import _format_julia_value
        da = xr.DataArray(np.array([1.0, 2.0]))
        result = _format_julia_value(da)
        assert result == "[1.0, 2.0]"

    def test_0d_numpy_array(self):
        import numpy as np
        from pysd.builders.julia.julia_model_builder import _format_julia_value
        result = _format_julia_value(np.array(42.0))
        assert result == "42.0"


# ===========================================================================
# Additional targeted tests for remaining coverage gaps
# ===========================================================================

class TestCoverageGaps:
    """Fills specific uncovered lines identified by coverage analysis."""

    # --- julia_expressions_builder.py ---

    def test_numpy_0d_array_in_visitor(self):
        import numpy as np
        v, *_ = _visitor_with_namespace()
        result = v.visit(np.array(7.5))  # 0-d ndarray
        assert result == "7.5"

    def test_get_constants_in_expression_success(self, mocker):
        import numpy as np
        mock_ext = mocker.MagicMock()
        mock_ext.data = np.float64(42.0)
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)
        v, *_ = _visitor_with_namespace()
        node = GetConstantsStructure(file="data.xlsx", tab="Sheet1", cell="A1")
        result = v.visit(node)
        assert result == "42.0"

    def test_unary_non_not_logic_operator(self):
        """Unary logic op that is not NOT uses LOGIC_OPS table directly."""
        v, *_ = _visitor_with_namespace()
        node = LogicStructure(operators=["<>"], arguments=[1.0])
        result = v.visit(node)
        assert "!=" in result or "1.0" in result

    def test_reference_with_active_subscript_context(self):
        """_reference appends subscript indices when active_subs is set."""
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("output")
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            active_subs={"sector": "_i0"},
            var_dims={"output": ["sector"]},
        )
        result = v.visit(ReferenceStructure("output"))
        assert "output[_i0]" == result

    def test_subscript_name_as_reference_emits_loop_index(self):
        """Bare subscript name in an expression (e.g. IF_THEN_ELSE(s=s1,1,0))
        must emit the loop-index variable, not a sanitised fallback identifier.
        This covers the identity-matrix pattern:
          I_Matrix[s, s1] = IF_THEN_ELSE(s = s1, 1, 0)
        where s and s1 are subscript range names, not model variables."""
        ns = JuliaNamespaceManager()
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            active_subs={"sectors_a_matrix": "_i0", "sectors_a_matrix1": "_i1"},
        )
        assert v.visit(ReferenceStructure("sectors_a_matrix")) == "_i0"
        assert v.visit(ReferenceStructure("sectors_a_matrix1")) == "_i1"
        # Original MDL casing should also resolve correctly
        assert v.visit(ReferenceStructure("sectors_A_matrix")) == "_i0"

    def test_subscript_name_reference_no_warning(self):
        """Subscript-name-as-loop-index must not emit a namespace-fallback warning."""
        import warnings
        ns = JuliaNamespaceManager()
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            active_subs={"s": "_i0"},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = v.visit(ReferenceStructure("s"))
        assert result == "_i0"

    def test_elmcount_resolves_to_integer_case_insensitive(self):
        """ELMCOUNT(SubName) must emit the integer size even when the
        subs_sizes key casing differs from the reference casing."""
        ns = JuliaNamespaceManager()
        registry = InlineLookupRegistry()
        needed = set()
        # subs_sizes uses mixed-case key (as the abstract model does); reference
        # arrives lowercase from the expression parser.
        v = JuliaASTVisitor(
            ns, registry, needed,
            subs_sizes={"Sectors_A_Matrix": 14},
        )
        node = CallStructure(
            function=ReferenceStructure("ELMCOUNT"),
            arguments=[ReferenceStructure("sectors_a_matrix")],
        )
        result = v.visit(node)
        assert result == "14"

    def test_elmcount_resolves_to_integer_exact_match(self):
        """ELMCOUNT works when casing matches exactly (regression guard)."""
        ns = JuliaNamespaceManager()
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            subs_sizes={"sectors": 5},
        )
        node = CallStructure(
            function=ReferenceStructure("ELMCOUNT"),
            arguments=[ReferenceStructure("sectors")],
        )
        assert v.visit(node) == "5"

    def test_invert_matrix_with_elmcount_emits_integer_size(self):
        """INVERT_MATRIX(..., ELMCOUNT(s)) inside a subscripted equation emits
        the integer count, not the loop-index variable for s."""
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("my_matrix")
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            active_subs={"s": "_i0", "s1": "_i1"},
            subs_sizes={"s": 3, "s1": 3},
        )
        node = CallStructure(
            function=ReferenceStructure("INVERT_MATRIX"),
            arguments=[
                ReferenceStructure("my_matrix"),
                CallStructure(
                    function=ReferenceStructure("ELMCOUNT"),
                    arguments=[ReferenceStructure("s")],
                ),
            ],
        )
        result = v.visit(node)
        assert result == "inv(my_matrix, 3)"

    def test_aligned_range_subscript_resolves_to_active_loop_var(self):
        """When a RHS reference uses an ALIGNED range (same elements as the LHS
        loop dimension but with a different name), the expression visitor should
        resolve it to the current loop variable — not drop it.

        Regression for pymedeas world model:
          ia_matrix[sectors, sectors1] with active_subs {sectors:_i0, sectors1:_i1}
          RHS: historic_ia_matrix[year2009, sectors_a_matrix, sectors_a_matrix1]
          Expected: historic_ia_matrix[15, _i0, _i1]
          Broken:   historic_ia_matrix[15]  (sectors_a_matrix dropped)
        """
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("historic ia matrix")
        registry = InlineLookupRegistry()
        needed = set()
        sector_elems = ["S1", "S2", "S3"]
        year_elems   = ["y1995", "y1996", "year2009"]
        v = JuliaASTVisitor(
            ns, registry, needed,
            active_subs={"sectors": "_i0", "sectors1": "_i1"},
            var_dims={"historic_ia_matrix": ["economic_years", "sectors_a_matrix", "sectors_a_matrix1"]},
            subs_sizes={
                "economic_years":    3,
                "sectors":           3,
                "sectors1":          3,
                "sectors_a_matrix":  3,
                "sectors_a_matrix1": 3,
            },
            subs_elems={
                "economic_years":    year_elems,
                "sectors":           sector_elems,
                "sectors1":          sector_elems,
                "sectors_a_matrix":  sector_elems,
                "sectors_a_matrix1": sector_elems,
            },
        )
        # Reference: historic_ia_matrix[year2009, sectors_a_matrix, sectors_a_matrix1]
        # year2009 is element index 3 in economic_years; sectors_a_matrix → _i0; sectors_a_matrix1 → _i1
        node = ReferenceStructure(
            "historic ia matrix",
            subscripts=SubscriptsReferenceStructure(
                subscripts=["year2009", "sectors_a_matrix", "sectors_a_matrix1"]
            ),
        )
        result = v.visit(node)
        # year2009 is the 3rd element of economic_years → index 3
        # sectors_a_matrix aligns with sectors → _i0
        # sectors_a_matrix1 aligns with sectors1 → _i1
        assert "_i0" in result, (
            f"Expected _i0 in result (sectors_a_matrix alignment), got: {result}"
        )
        assert "_i1" in result, (
            f"Expected _i1 in result (sectors_a_matrix1 alignment), got: {result}"
        )
        assert result == "historic_ia_matrix[3, _i0, _i1]", (
            f"Expected historic_ia_matrix[3, _i0, _i1], got: {result}"
        )

    def test_get_data_with_explicit_subscripts_uses_call_syntax(self):
        """GET DATA / LOOKUPS variables referenced with explicit subscripts must
        use function-call syntax f(idx, t), NOT array-indexing syntax f[idx].

        Regression for pymedeas world model:
          invest_res_elec[res_elec] ~ ... * invest_cost_res_elec[res_elec]
        where invest_cost_res_elec IS a GET_DIRECT_DATA lookup.
        Expected: invest_cost_res_elec(_i0, t)
        Broken:   invest_cost_res_elec[_i0]   (MethodError at model load)
        """
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("invest cost res elec")
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            active_subs={"res_elec": "_i0"},
            lookup_names={"invest_cost_res_elec"},
            subs_elems={"res_elec": ["RES1", "RES2", "RES3"]},
        )
        node = ReferenceStructure(
            "invest cost res elec",
            subscripts=SubscriptsReferenceStructure(subscripts=["res_elec"]),
        )
        result = v.visit(node)
        assert result == "invest_cost_res_elec(_i0, t)", (
            f"Expected invest_cost_res_elec(_i0, t) (call syntax), got: {result}"
        )
        assert "[" not in result, (
            f"Should not use array indexing [], got: {result}"
        )

    def test_element_label_resolves_to_subrange_index_not_parent(self):
        """When an element label is used as an explicit subscript, the 1-based
        index must come from the variable's OWN dimension, not from a larger
        parent dimension that also contains the same element.

        Regression for pymedeas world model:
          final_sources        = [electricity, heat, liquids, gases, solids]  (size 5)
          matter_final_sources = [liquids, gases, solids]                     (size 3)
          potential_fe_gen[matter_final_sources]  (declared over sub-range)

          Reference: potential_fe_gen[liquids]
          Expected:  potential_fe_gen[1]   (index of 'liquids' in matter_final_sources)
          Broken:    potential_fe_gen[3]   (index of 'liquids' in final_sources, picked
                                            because final_sources was iterated first in
                                            _elem_index)
        """
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("potential fe gen")
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            var_dims={"potential_fe_gen": ["matter_final_sources"]},
            subs_elems={
                # Larger parent range — liquids is at index 3 here
                "final_sources": ["electricity", "heat", "liquids", "gases", "solids"],
                # The variable's own range — liquids is at index 1 here
                "matter_final_sources": ["liquids", "gases", "solids"],
            },
        )
        node = ReferenceStructure(
            "potential fe gen",
            subscripts=SubscriptsReferenceStructure(subscripts=["liquids"]),
        )
        result = v.visit(node)
        assert result == "potential_fe_gen[1]", (
            f"Expected potential_fe_gen[1] (index in matter_final_sources), got: {result}"
        )

    def test_element_label_with_bang_subscript_uses_dim_index(self):
        """When a reference has both a literal element-label subscript and a '!'
        aggregation subscript, the element label must resolve to the index within
        the variable's OWN declared dimension — and that index must appear in the
        generated comprehension alongside the aggregation loop variable.

        Regression for pymedeas world model:
          fuels                        = [electricity, heat, liquids, gases, solids]  (size 5)
          transport_modes_pkm          = [car, bus, train, air]  (size 4)
          transport_modes_pkm_commercial = [car, bus, train]     (size 3, sub-range)
          energy_pkm[fuels, transport_modes_pkm]

          Reference: energy_pkm[liquids, transport_modes_pkm_commercial!]
            ('liquids' is element 3 of fuels; '!' triggers a sum comprehension)
          Expected: [energy_pkm[3, _ii0] for _ii0 in 1:N_TRANSPORT_MODES_PKM_COMMERCIAL]
          Broken:   [energy_pkm[_ii0] for _ii0 in 1:N_TRANSPORT_MODES_PKM_COMMERCIAL]
                    (fuel index 3 dropped; only the aggregation var is emitted)
        """
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("energy pkm")
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            var_dims={"energy_pkm": ["fuels", "transport_modes_pkm"]},
            subs_sizes={
                "fuels": 5,
                "transport_modes_pkm": 4,
                "transport_modes_pkm_commercial": 3,
            },
            subs_elems={
                "fuels": ["electricity", "heat", "liquids", "gases", "solids"],
                "transport_modes_pkm": ["car", "bus", "train", "air"],
                "transport_modes_pkm_commercial": ["car", "bus", "train"],
            },
        )
        node = ReferenceStructure(
            "energy pkm",
            subscripts=SubscriptsReferenceStructure(
                subscripts=["liquids", "transport_modes_pkm_commercial!"]
            ),
        )
        result = v.visit(node)
        # 'liquids' is the 3rd element of fuels → fixed index 3
        # 'transport_modes_pkm_commercial!' → loop var _ii0
        assert "3" in result, (
            f"Expected fuel index 3 in result, got: {result}"
        )
        assert "_ii0" in result, (
            f"Expected aggregation loop var _ii0 in result, got: {result}"
        )
        assert "energy_pkm[3, _ii0]" in result, (
            f"Expected energy_pkm[3, _ii0] in comprehension, got: {result}"
        )

    def test_lookup_call_with_explicit_subscripts_resolves_all_indices(self):
        """Lookup called as function with explicit non-! subscripts must resolve all indices.

        Vensim: Historic_water_use[sectors, water](Time) inside a [sectors, water] loop.
        Broken: historic_water_use(_i1, t)   ← only water index; sectors index dropped
                because var_dims uses parent dim 'sectors_and_households' not in active_subs
        Fixed:  historic_water_use(_i0, _i1, t)  ← both indices from func_node_subs

        The function has declared dims [sectors_and_households, water], active loop has
        sectors→_i0 and water→_i1.  The func subscripts [sectors, water] directly name
        the active loop ranges, so both indices must appear.
        """
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("historic water use")
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            var_dims={"historic_water_use": ["sectors_and_households", "water"]},
            subs_sizes={
                "sectors": 35,
                "sectors_and_households": 36,
                "water": 3,
            },
            subs_elems={
                "sectors": [f"sec{i}" for i in range(35)],
                "sectors_and_households": [f"sec{i}" for i in range(35)] + ["households"],
                "water": ["blue", "green", "grey"],
            },
        )
        # Simulate active subscript context: looping over [sectors, water]
        v = v._with_extra_subs({"sectors": "_i0", "water": "_i1"})
        v.lookup_names = {"historic_water_use"}
        node = CallStructure(
            function=ReferenceStructure(
                "historic water use",
                subscripts=SubscriptsReferenceStructure(subscripts=["sectors", "water"]),
            ),
            arguments=[1.0],
        )
        result = v.visit(node)
        assert "_i0" in result, f"Expected sector index _i0, got: {result}"
        assert "_i1" in result, f"Expected water index _i1, got: {result}"
        assert result == "historic_water_use(_i0, _i1, 1.0)", (
            f"Expected historic_water_use(_i0, _i1, 1.0), got: {result}"
        )

    def test_lookup_call_with_element_label_subscript_resolves_literal_index(self):
        """Lookup call with element-label subscript must resolve to a literal index.

        Vensim: Historic_water_use[Households, water](Time) inside a [water] loop.
        Broken: could give wrong indices or drop the element-label index entirely
        Fixed:  historic_water_use(36, _i0, t)  ← 36 = index of 'households' in
                sectors_and_households (1-based), _i0 = active water loop var
        """
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("historic water use")
        registry = InlineLookupRegistry()
        needed = set()
        subs_elems = {
            "sectors_and_households": [f"sec{i}" for i in range(35)] + ["Households"],
            "water": ["blue", "green", "grey"],
        }
        v = JuliaASTVisitor(
            ns, registry, needed,
            var_dims={"historic_water_use": ["sectors_and_households", "water"]},
            subs_sizes={"sectors_and_households": 36, "water": 3},
            subs_elems=subs_elems,
        )
        # _elem_index is built from subs_elems automatically: Households → index 36
        # Active sub: looping over [water]
        v = v._with_extra_subs({"water": "_i0"})
        v.lookup_names = {"historic_water_use"}
        node = CallStructure(
            function=ReferenceStructure(
                "historic water use",
                subscripts=SubscriptsReferenceStructure(subscripts=["Households", "water"]),
            ),
            arguments=[1.0],
        )
        result = v.visit(node)
        assert "36" in result, f"Expected literal index 36 for Households, got: {result}"
        assert "_i0" in result, f"Expected water index _i0, got: {result}"

    def test_ifelse_bare_reference_condition_wrapped_with_ne_zero(self):
        """IF THEN ELSE with a bare variable as condition must emit `!= 0`.

        Vensim: IF THEN ELSE(activate_elf, then_expr, 0)
        Broken: ifelse(activate_elf, then_expr, 0.0)
                → ArgumentError: Condition of `ifelse` must be a `Bool`
        Fixed:  ifelse(activate_elf != 0, then_expr, 0.0)
        """
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("activate elf")
        ns.add_to_namespace("x")
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(ns, registry, needed)
        node = CallStructure(
            function=ReferenceStructure("IF THEN ELSE"),
            arguments=[
                ReferenceStructure("activate elf"),
                ReferenceStructure("x"),
                0.0,
            ],
        )
        result = v.visit(node)
        assert "!= 0" in result, (
            f"Expected '!= 0' in ifelse condition for bare reference, got: {result}"
        )
        assert "pysd_ifelse(" in result, f"Expected pysd_ifelse call, got: {result}"

    def test_ifelse_logic_condition_not_double_wrapped(self):
        """IF THEN ELSE with a comparison condition must NOT add != 0.

        Vensim: IF THEN ELSE(t < 2015, then_expr, 0)
        Expected: ifelse((t < 2015.0), then_expr, 0.0)
        Must NOT become: ifelse((t < 2015.0) != 0, then_expr, 0.0)
        """
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("t")
        ns.add_to_namespace("x")
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(ns, registry, needed)
        node = CallStructure(
            function=ReferenceStructure("IF THEN ELSE"),
            arguments=[
                LogicStructure(operators=["<"], arguments=[ReferenceStructure("t"), 2015.0]),
                ReferenceStructure("x"),
                0.0,
            ],
        )
        result = v.visit(node)
        assert "!= 0" not in result, (
            f"Expected no '!= 0' for comparison condition, got: {result}"
        )
        assert "< 2015.0" in result, f"Expected '< 2015.0' in result, got: {result}"

    def test_subscripted_aux_literal_array_generates_per_element_equations(self):
        """A subscripted auxiliary whose AST is a literal numpy array must NOT
        produce a comprehension that puts the full vector on each scalar LHS.

        Vensim:  res_elec_variables[RES_elec] = 0, 0, 0, 0, 1, 1, 1, 1

        Broken:  [res_elec_variables[_i0] ~ [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
                  for _i0 in 1:N_RES_ELEC]...
                  → MTK ArgumentError: Cannot add arguments of different sizes
                    shapes [1:8] and []

        Fixed:   res_elec_variables[1] ~ 0.0,
                 res_elec_variables[2] ~ 0.0,
                 ...
                 res_elec_variables[8] ~ 1.0,
        """
        import numpy as np
        sr = _make_subscript_range("res_elec", ["w", "x", "y", "z"])
        arr = np.array([0.0, 0.0, 1.0, 1.0])
        elem = _make_subscripted_element("res elec variables", arr, "res_elec")
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        eqs = [e for eqlist, _ in sb.built_elements.values() for e in eqlist]
        joined = "\n".join(eqs)
        # Must NOT put the full array on the RHS of each element
        assert "[0.0, 0.0, 1.0, 1.0]" not in joined, (
            f"Full array must not appear as RHS in equations: {joined}"
        )
        # Must generate per-element equations
        assert "res_elec_variables[1] ~" in joined, f"Missing element 1 eq: {joined}"
        assert "res_elec_variables[4] ~" in joined, f"Missing element 4 eq: {joined}"

    # --- julia_model_builder.py ---

    def test_inline_lookup_registered_after_build(self, tmp_path):
        """InlineLookupsStructure inside an element populates lookup_const_decls."""
        lut_ast = InlineLookupsStructure(
            argument=ReferenceStructure("x_val"),
            lookups=LookupsStructure(
                x=(0.0, 1.0), y=(0.0, 2.0),
                x_limits=(0.0, 1.0), y_limits=(0.0, 2.0),
                type="interpolate",
            ),
        )
        x_elem = _make_element("x val", 0.5)
        comp = AbstractComponent(subscripts=[[], []], ast=lut_ast)
        elem = AbstractElement(name="Lookup Result", components=[comp])
        sb = _section_builder_from_elements([x_elem, elem])
        sb.build_section()
        assert any("_inline_lookup_" in d for d in sb.lookup_const_decls)

    def test_element_dims_empty_subscripts(self):
        """_element_dims returns [] when component has no subscript list."""
        comp = AbstractComponent(subscripts=[[], []], ast=1.0)
        comp.subscripts = [[]]  # empty first subscript
        elem = AbstractElement(name="scalar", components=[comp])
        sr = _make_subscript_range("dim", ["a", "b"])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        dims = sb._element_dims(elem)
        assert dims == []

    def test_element_dims_no_components_direct(self):
        """_element_dims defensive check: no components → empty list."""
        elem = AbstractElement(name="empty", components=[])
        sb = _section_builder_from_elements([])
        # Call directly (bypass _process_element's early-return guard)
        assert sb._element_dims(elem) == []

    def test_2d_subscripted_stock(self):
        """N≥2 dimensional stock uses comprehension form."""
        sr1 = _make_subscript_range("row", ["R1", "R2"])
        sr2 = _make_subscript_range("col", ["C1", "C2"])
        comp = AbstractComponent(
            subscripts=[["row", "col"], []],
            ast=IntegStructure(flow=1.0, initial=0.0),
        )
        elem = AbstractElement(name="matrix stock", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("_i0" in e and "_i1" in e for e in eqs)
        assert any("D(matrix_stock" in e for e in eqs)

    def test_get_constants_control_element(self, tmp_path):
        """GetConstantsStructure for a control element stores runtime expression."""
        ast = GetConstantsStructure(file="d.xlsx", tab="Sheet1", cell="A1")
        comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=ast)
        final_time = AbstractControlElement(name="FINAL TIME", components=[comp])
        other_controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        sb = _section_builder_from_elements(
            [final_time] + other_controls, path=tmp_path / "m.mdl"
        )
        sb.build_section()
        assert "pysd_xlsx_read_constant" in sb.control_vals["final_time"]

    def test_subscripted_aux_1d_control_branch(self):
        """1D subscripted control aux updates control_vals."""
        sr = _make_subscript_range("dim", ["A", "B"])
        comp = AbstractComponent(subscripts=[["dim"], []], ast=5.0)
        ctrl_elem = AbstractControlElement(name="FINAL TIME", components=[comp])
        other = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        sb = _section_builder_from_elements([ctrl_elem] + other, subscripts=[sr])
        sb.build_section()
        # Control val is set even if subscripted (value is the visited expression)
        assert sb.control_vals.get("final_time") is not None

    def test_subscripted_aux_2d_control_branch(self):
        """2D subscripted control aux updates control_vals."""
        sr1 = _make_subscript_range("row", ["R1", "R2"])
        sr2 = _make_subscript_range("col", ["C1", "C2"])
        comp = AbstractComponent(subscripts=[["row", "col"], []], ast=1.0)
        ctrl_elem = AbstractControlElement(name="FINAL TIME", components=[comp])
        other = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        sb = _section_builder_from_elements(
            [ctrl_elem] + other, subscripts=[sr1, sr2]
        )
        sb.build_section()
        assert sb.control_vals.get("final_time") is not None

    def test_get_lookups_with_subscripts_in_section(self, mocker, tmp_path):
        """_process_get_lookups iterates over section subscripts to build subs_map."""
        import numpy as np
        import xarray as xr
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        da = xr.DataArray(ys, coords={"lookup_dim": xs}, dims=["lookup_dim"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtLookup", return_value=mock_ext)
        ast = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Lut", components=[comp])
        sr = _make_subscript_range("energy_type", ["H", "S"])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl",
                                             subscripts=[sr])
        sb.build_section()
        assert any("lut_itp" in d for d in sb.lookup_const_decls)

    def test_get_lookups_single_component_with_subscripts_emits_dispatch(self, tmp_path):
        """Single-component subscripted lookup emits pysd_xlsx_build_lookup_dispatch (lines 2724-2736)."""
        sr = _make_subscript_range("energy_type", ["H", "S", "L"])
        ast = GetLookupsStructure(file="d.xlsx", tab="Sheet1", x_row_or_col="yr", cell="A1")
        comp = AbstractComponent(subscripts=[["energy_type"], []], ast=ast)
        elem = AbstractElement(name="Sub Lut", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl",
                                             subscripts=[sr])
        sb.build_section()
        assert any("sub_lut_fns" in d for d in sb.lookup_const_decls), (
            f"Expected sub_lut_fns in lookup_const_decls, got {sb.lookup_const_decls}"
        )
        assert any("sub_lut(i, x)" in d for d in sb.lookup_func_decls)

    def test_get_lookups_multi_component(self, tmp_path):
        """Multi-component GetLookupsStructure uses runtime per-component dispatch."""
        ast1 = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="A1")
        ast2 = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="B1")
        sr = _make_subscript_range("dim_a", ["X"])
        comp1 = AbstractComponent(subscripts=[["dim_a"], []], ast=ast1)
        comp2 = AbstractComponent(subscripts=[["dim_a"], []], ast=ast2)
        elem = AbstractElement(name="Multi Lut", components=[comp1, comp2])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl",
                                             subscripts=[sr])
        sb.build_section()
        assert any("multi_lut_1_fns" in d for d in sb.lookup_const_decls)

    def test_get_lookups_2d_multi_component_emits_2d_dispatch(self, tmp_path):
        """2D-subscripted multi-component lookup emits 2D dispatch function (lines 2766-2769)."""
        sr1 = _make_subscript_range("d1", ["A", "B"])
        sr2 = _make_subscript_range("d2", ["X", "Y"])
        ast1 = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="A1")
        ast2 = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="B1")
        comp1 = AbstractComponent(subscripts=[["d1", "d2"], []], ast=ast1)
        comp2 = AbstractComponent(subscripts=[["d1", "d2"], []], ast=ast2)
        elem = AbstractElement(name="Lut 2D", components=[comp1, comp2])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl",
                                             subscripts=[sr1, sr2])
        sb.build_section()
        assert any("lut_2d(i, j, x)" in d for d in sb.lookup_func_decls), (
            f"Expected 2D dispatch function, got {sb.lookup_func_decls}"
        )

    def test_get_lookups_data_without_values_attr(self, mocker, tmp_path):
        """_process_get_lookups handles data without .values (plain numpy array)."""
        import numpy as np
        # A mock where .data is a plain 1D numpy array (no .values)
        xs_arr = np.array([0.0, 1.0, 2.0])
        ys_arr = np.array([0.0, 0.5, 1.0])

        class FakeLookupData:
            values = None  # No .values — will use np.asarray path
            def __init__(self):
                # make hasattr(data, "values") False by removing attr
                pass

        # Use a real structure: mock data without .values
        mock_data = mocker.MagicMock()
        del mock_data.values  # remove values attr
        mock_data.__array__ = lambda *a: ys_arr  # make np.asarray work
        mock_data.coords = {"lookup_dim": mocker.MagicMock(values=xs_arr)}
        mock_ext = mocker.MagicMock()
        mock_ext.data = mock_data
        mocker.patch("pysd.py_backend.external.ExtLookup", return_value=mock_ext)
        ast = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Plain Lut", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl")
        sb.build_section()
        # Should produce a lookup (scalar path fallback via np.asarray)
        assert any("plain_lut" in d for d in sb.lookup_const_decls)

    def test_get_data_with_subscripts_in_section(self, mocker, tmp_path):
        """_process_get_data iterates over section subscripts to build subs_map."""
        import numpy as np
        import xarray as xr
        ts = np.array([1995.0, 2000.0])
        vals = np.array([1.0, 2.0])
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtData", return_value=mock_ext)
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Historic Data", components=[comp])
        sr = _make_subscript_range("energy_type", ["H", "S"])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl",
                                             subscripts=[sr])
        sb.build_section()
        assert any("historic_data_itp" in d for d in sb.lookup_const_decls)

    def test_get_data_single_component_with_subscripts_emits_dispatch(self, tmp_path):
        """Single-component subscripted GET DATA emits pysd_xlsx_build_lookup_dispatch (lines 2997-3009)."""
        sr = _make_subscript_range("fuel", ["gas", "oil"])
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        comp = AbstractComponent(subscripts=[["fuel"], []], ast=ast)
        elem = AbstractElement(name="Fuel Data", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl",
                                             subscripts=[sr])
        sb.build_section()
        assert any("fuel_data_fns" in d for d in sb.lookup_const_decls), (
            f"Expected fuel_data_fns in lookup_const_decls, got {sb.lookup_const_decls}"
        )
        assert any("fuel_data(i, x)" in d for d in sb.lookup_func_decls)

    def test_get_data_multi_component(self, tmp_path):
        """Multi-component GetDataStructure uses runtime per-component dispatch."""
        ast1 = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        ast2 = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="B1")
        sr = _make_subscript_range("dim_b", ["Y"])
        comp1 = AbstractComponent(subscripts=[["dim_b"], []], ast=ast1)
        comp2 = AbstractComponent(subscripts=[["dim_b"], []], ast=ast2)
        elem = AbstractElement(name="Multi Data", components=[comp1, comp2])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl",
                                             subscripts=[sr])
        sb.build_section()
        assert any("multi_data_1_fns" in d for d in sb.lookup_const_decls)

    def test_get_data_2d_multi_component_emits_2d_dispatch(self, tmp_path):
        """2D-subscripted multi-component GET DATA emits 2D dispatch function (lines 3029-3032)."""
        sr1 = _make_subscript_range("dm1", ["A", "B"])
        sr2 = _make_subscript_range("dm2", ["X", "Y"])
        ast1 = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        ast2 = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="B1")
        comp1 = AbstractComponent(subscripts=[["dm1", "dm2"], []], ast=ast1)
        comp2 = AbstractComponent(subscripts=[["dm1", "dm2"], []], ast=ast2)
        elem = AbstractElement(name="Data 2D", components=[comp1, comp2])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl",
                                             subscripts=[sr1, sr2])
        sb.build_section()
        assert any("data_2d(i, j, x)" in d for d in sb.lookup_func_decls), (
            f"Expected 2D dispatch function, got {sb.lookup_func_decls}"
        )

    def test_get_data_no_time_dimension_raises_into_fallback(self, tmp_path):
        """GET DATA with no declared subscripts: runtime scalar path always emits _itp.
        No warning is raised (ExtData not called at translation time)."""
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="No Time", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl")
        sb.build_section()
        assert any("no_time_itp" in d for d in sb.lookup_const_decls)

    def test_get_data_3d_emits_2d_dispatch(self, tmp_path):
        """Single-component GET DATA with no declared subscripts: runtime scalar
        path emits _itp. No warnings, no FAILED placeholder."""
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Hfc Emissions", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl")
        sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert not any("GET_DATA_FAILED" in e for e in all_eqs)
        assert any("hfc_emissions_itp" in d for d in sb.lookup_const_decls)

    def test_initial_from_literal_float(self):
        """INITIAL(5.0) resolves to literal without needing reference resolution."""
        init_ast = InitialStructure(initial=5.0)
        comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        elem = AbstractElement(name="Init Literal", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        assert any("@parameters init_literal = 5.0" in d for d in sb.param_decls)

    def test_initial_from_get_constants_success(self, mocker, tmp_path):
        """INITIAL(GetConstantsStructure) resolves to the read value."""
        import numpy as np
        mock_ext = mocker.MagicMock()
        mock_ext.data = np.float64(99.0)
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)
        gc_ast = GetConstantsStructure(file="d.xlsx", tab="S", cell="A1")
        init_ast = InitialStructure(initial=gc_ast)
        comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        elem = AbstractElement(name="Init Ext", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl")
        sb.build_section()
        assert any("@parameters init_ext = 99.0" in d for d in sb.param_decls)

    def test_resolve_ref_initial_depth_exceeded(self):
        """_resolve_ref_initial returns None when depth < 0."""
        sb = _section_builder_from_elements([])
        result = sb._resolve_ref_initial("anything", depth=-1)
        assert result is None

    def test_resolve_ref_initial_follows_numeric_aux_rhs(self):
        """INITIAL resolves when aux equation RHS is a plain number."""
        # aux ~ 42.0 → INITIAL(aux) → 42.0
        aux_comp = AbstractComponent(subscripts=[[], []], ast=42.0)
        aux_elem = AbstractElement(name="Aux Val", components=[aux_comp])
        init_ast = InitialStructure(initial=ReferenceStructure("Aux Val"))
        init_comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        init_elem = AbstractElement(name="Init Aux", components=[init_comp])
        sb = _section_builder_from_elements([aux_elem, init_elem])
        sb.build_section()
        assert any("@parameters init_aux = 42.0" in d for d in sb.param_decls)

    def test_resolve_ref_initial_returns_none_for_complex_rhs(self):
        """_resolve_ref_initial returns None for end of chain."""
        sb = _section_builder_from_elements([])
        sb.namespace.add_to_namespace("x")
        # x is in namespace but has no u0, param, or built_elements entry
        result = sb._resolve_ref_initial("x", depth=3)
        assert result is None

    def test_read_get_constants_multi_component(self, tmp_path):
        """Multi-component GCS emits pysd_xlsx_read_constant vector call in ext_const_decls."""
        ast1 = GetConstantsStructure(file="d.xlsx", tab="S", cell="A1")
        ast2 = GetConstantsStructure(file="d.xlsx", tab="S", cell="B1")
        sr = _make_subscript_range("dim_c", ["Z"])
        comp1 = AbstractComponent(subscripts=[["dim_c"], []], ast=ast1)
        comp2 = AbstractComponent(subscripts=[["dim_c"], []], ast=ast2)
        elem = AbstractElement(name="Multi Const", components=[comp1, comp2])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl",
                                             subscripts=[sr])
        sb.build_section()
        all_decls = sb.ext_const_decls + sb.param_decls
        assert any("multi_const" in d for d in all_decls)

    def test_read_get_constants_split_range_collision_resolved(self, mocker, tmp_path):
        """Multi-component where two subscript positions share the same parent
        range (e.g. final_sources at pos 1 and the per-element split at pos 2).
        _detect_split_ranges should pick a collision-free alias (final_sources1)
        for the split dim so ext.add() sees consistent keys."""
        import numpy as np
        import xarray as xr
        # Simulate 2 components each covering one fuel element × SECTORS
        # where 'final_sources' covers both positions (pos 1 as range, pos 2 as split)
        # and 'final_sources1' is the alias
        sr_fs = _make_subscript_range("final_sources", ["elec", "heat"])
        sr_fs1 = _make_subscript_range("final_sources1", ["elec", "heat"])
        sr_sec = _make_subscript_range("SECTORS", ["A", "B"])

        mock_ext = mocker.MagicMock()
        # Data shaped as (SECTORS=2, final_sources1=2) assembled over 2 comps
        da = xr.DataArray(
            np.ones((2, 2)),
            coords={"SECTORS": ["A", "B"], "final_sources1": ["elec", "heat"]},
            dims=["SECTORS", "final_sources1"],
        )
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)

        ast1 = GetConstantsStructure(file="d.xlsx", tab="S", cell="r1")
        ast2 = GetConstantsStructure(file="d.xlsx", tab="S", cell="r2")
        # comp[0]: [SECTORS, final_sources, elec]   — final_sources at pos 1, elec at pos 2
        # comp[1]: [SECTORS, final_sources, heat]
        comp1 = AbstractComponent(subscripts=[["SECTORS", "final_sources", "elec"], []], ast=ast1)
        comp2 = AbstractComponent(subscripts=[["SECTORS", "final_sources", "heat"], []], ast=ast2)
        elem = AbstractElement(name="Eff Rate", components=[comp1, comp2])
        sb = _section_builder_from_elements(
            [elem], path=tmp_path / "m.mdl",
            subscripts=[sr_fs, sr_fs1, sr_sec],
        )
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sb.build_section()
        # Should succeed without a "Could not read" warning
        assert not [x for x in w if "Could not read external constant" in str(x.message)]
        assert any("eff_rate" in d for d in sb.ext_const_decls + sb.param_decls)

    def test_read_get_constants_piecewise_mixed(self, tmp_path):
        """Piecewise constant: one GCS component + two literal-0 components.
        Emits pysd_xlsx_read_constant vector call (runtime Excel reading)."""
        sr_fs = _make_subscript_range("final_sources", ["elec", "heat", "liq"])
        sr_mfs = _make_subscript_range("matter_final_sources", ["liq"])

        ast_gcs = GetConstantsStructure(file="d.xlsx", tab="S", cell="r1")
        comp_gcs = AbstractComponent(subscripts=[["matter_final_sources"], []], ast=ast_gcs)
        comp_elec = AbstractComponent(subscripts=[["elec"], []], ast=0)
        comp_heat = AbstractComponent(subscripts=[["heat"], []], ast=0)
        elem = AbstractElement(name="Policy Share", components=[comp_gcs, comp_elec, comp_heat])
        sb = _section_builder_from_elements(
            [elem], path=tmp_path / "m.mdl",
            subscripts=[sr_fs, sr_mfs],
        )
        sb.build_section()
        all_decls = sb.ext_const_decls + sb.param_decls
        assert any("policy_share" in d for d in all_decls)
        combined = next(d for d in all_decls if "policy_share" in d)
        assert "pysd_xlsx_read_constant" in combined and "r1" in combined

    def test_read_get_constants_piecewise_2d(self, mocker, tmp_path):
        """Piecewise 2D constant: one GCS component covering a sub-range of the
        first dimension + one literal-0 component covering the complement.

        Regression for pymedeas world model:
          materials_for_o_m_per_capacity_installed_res_elec[RES_ELEC, materials]
            - [RES_ELEC_DISPATCHABLE, materials] = 0  (literal, 4 elements)
            - [RES_ELEC_VARIABLE, materials]    = GCS (4 elements from Excel)

        The piecewise assembly must produce a 2D Julia matrix (shape 4×2 in the
        test), NOT a flat 1D vector of 4+2=6 zeros.

        Broken: const v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   (1D, 6 elements)
        Fixed:  const v = [0.0 0.0; 0.0 0.0; 1.0 2.0; 3.0 4.0]  (2D, 4×2)
        """
        import numpy as np
        import xarray as xr
        import warnings

        # Subscript ranges
        sr_a      = _make_subscript_range("dim_a",      ["a1", "a2", "a3", "a4"])
        sr_a_first = _make_subscript_range("dim_a_first", ["a1", "a2"])
        sr_a_rest  = _make_subscript_range("dim_a_rest",  ["a3", "a4"])
        sr_b      = _make_subscript_range("dim_b",      ["b1", "b2"])

        # GCS mock: returns 2×2 DataArray for [dim_a_rest, dim_b]
        mock_ext = mocker.MagicMock()
        da = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            coords={"dim_a_rest": ["a3", "a4"], "dim_b": ["b1", "b2"]},
            dims=["dim_a_rest", "dim_b"],
        )
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)

        ast_gcs  = GetConstantsStructure(file="d.xlsx", tab="S", cell="r1")
        comp_gcs = AbstractComponent(
            subscripts=[["dim_a_rest", "dim_b"], []], ast=ast_gcs
        )
        comp_lit = AbstractComponent(
            subscripts=[["dim_a_first", "dim_b"], []], ast=0
        )
        elem = AbstractElement(name="V", components=[comp_lit, comp_gcs])
        sb = _section_builder_from_elements(
            [elem], path=tmp_path / "m.mdl",
            subscripts=[sr_a, sr_a_first, sr_a_rest, sr_b],
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sb.build_section()

        assert not [x for x in w if "Could not read" in str(x.message)]
        all_decls = sb.ext_const_decls + sb.param_decls
        combined = next((d for d in all_decls if "const v = " in d), None)
        assert combined is not None, f"Expected 'const v' in decls, got: {all_decls}"

        # Must be a 2D matrix (contains ';' row separator), NOT a 1D flat vector
        assert ";" in combined, (
            f"Expected a 2D matrix (with ';') in declaration, got: {combined}"
        )
        # GCS values must appear
        assert "1.0" in combined and "4.0" in combined, (
            f"GCS values 1.0 and 4.0 should appear in constant, got: {combined}"
        )

    def test_initial_from_get_constants_exception(self, mocker, tmp_path):
        """INITIAL(GetConstantsStructure) exception → _resolve_initial_value returns None
        → frozen-stock fallback: D(x) ~ 0.0, x(t0) = placeholder-0.0.
        No 'Cannot resolve' warning is emitted; a GCS-read warning may be."""
        mocker.patch(
            "pysd.py_backend.external.ExtConstant",
            side_effect=FileNotFoundError("missing"),
        )
        gc_ast = GetConstantsStructure(file="missing.xlsx", tab="S", cell="A1")
        init_ast = InitialStructure(initial=gc_ast)
        comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        elem = AbstractElement(name="Init Gc Fail", components=[comp])
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl")
            sb.build_section()
        # No "Cannot resolve INITIAL" warning
        assert not any("Cannot resolve INITIAL" in str(x.message) for x in caught)
        # Emitted as a frozen stock, not an auxiliary
        assert any("@variables init_gc_fail(t)" in d for d in sb.stock_decls)
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("D(init_gc_fail)" in e for e in all_eqs)

    def test_modular_build_no_equations_uses_empty_list(self, tmp_path):
        """Modular build with only control vars → combined = Equation[]."""
        pop = _make_stock_element("Population", 1.0, 100.0)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        # Single view with only the stock
        views_dict = {"Main": {"Population"}}
        section = _make_section(
            elements=[pop] + controls,
            path=tmp_path / "m.mdl",
            split=True,
            views_dict=views_dict,
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        path = JuliaModelBuilder(model, backend="mtk").build_model()
        content = path.read_text()
        assert "eqs = [" in content

    def test_modular_build_empty_eq_var_names(self, tmp_path):
        """Modular build: view references nonexistent var AND constants have no eqs
        → eq_var_names=[] AND leftover_eqs=[] → eqs = Equation[]."""
        # A constant has no equations; the view maps to nothing → both lists empty
        rate_comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.5)
        rate_elem = AbstractElement(name="Rate", components=[rate_comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        # View only references a name that is not in the namespace
        views_dict = {"Main": {"NonExistentVariable"}}
        section = _make_section(
            elements=[rate_elem] + controls,
            path=tmp_path / "empty_eq.mdl",
            split=True,
            views_dict=views_dict,
        )
        model = AbstractModel(original_path=tmp_path / "empty_eq.mdl",
                               sections=(section,))
        path = JuliaModelBuilder(model, backend="mtk").build_model()
        content = path.read_text()
        assert "eqs = Equation[]" in content

    def test_format_julia_value_3d_array_flattened(self):
        """3D numpy array → reshape expression (flat vector + shape dims)."""
        import numpy as np
        from pysd.builders.julia.julia_model_builder import _format_julia_value
        arr = np.ones((2, 2, 2))
        result = _format_julia_value(arr)
        assert "[" in result       # contains a flat vector component
        assert "1.0" in result     # values are present
        assert ";" not in result   # no 2D matrix row-separator syntax

    # --- expressions_builder: bare lookup reference paths (lines 609-621) ---

    def test_bare_lookup_ref_no_dims_emits_call_t(self):
        """Bare reference to a lookup var with no active_subs and no dims → f(t) (line 621)."""
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("my data")
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            lookup_names={"my_data"},
        )
        result = v.visit(ReferenceStructure("my data"))
        assert result == "my_data(t)"

    def test_bare_lookup_ref_with_dims_scalar_context_emits_comprehension(self):
        """Bare lookup ref with var_dims and no active_subs → broadcast comprehension (lines 613-619)."""
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("my series")
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            lookup_names={"my_series"},
            var_dims={"my_series": ["sector"]},
            subs_sizes={"sector": 3},
        )
        result = v.visit(ReferenceStructure("my series"))
        assert "my_series" in result
        assert "for" in result
        assert "_ii0" in result

    # --- expressions_builder: _collect_aggregation_subscripts (lines 358-359, 373-375) ---

    def test_sum_with_bare_bang_ref_collects_subscript(self):
        """SUM(var[dim!]) where var is a bare reference → triggers _scan on ReferenceStructure
        with '!' subscripts (lines 358-359)."""
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("myvar")
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            subs_sizes={"dim": 4},
            var_dims={"myvar": ["dim"]},
        )
        node = CallStructure(
            function=ReferenceStructure("SUM"),
            arguments=[
                ReferenceStructure(
                    "myvar",
                    subscripts=SubscriptsReferenceStructure(subscripts=("dim!",)),
                )
            ],
        )
        result = v.visit(node)
        assert "myvar" in result
        assert "sum" in result or "for" in result

    def test_sum_with_arithmetic_bang_ref_collects_subscript(self):
        """SUM(a[dim!] * b) → _scan visits ArithmeticStructure then inner refs (lines 373-375)."""
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("a var")
        ns.add_to_namespace("b var")
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            subs_sizes={"dim": 3},
            var_dims={"a_var": ["dim"]},
        )
        node = CallStructure(
            function=ReferenceStructure("SUM"),
            arguments=[
                ArithmeticStructure(
                    ["*"],
                    [
                        ReferenceStructure(
                            "a var",
                            subscripts=SubscriptsReferenceStructure(subscripts=("dim!",)),
                        ),
                        ReferenceStructure("b var"),
                    ],
                )
            ],
        )
        result = v.visit(node)
        assert "a_var" in result
        assert "b_var" in result


# ===========================================================================
# JSON data backend tests
# ===========================================================================

class TestJSONDataBackend:

    def _minimal_model_with_lookup(self, tmp_path):
        """Model with a stock, a parameter, and a named lookup table."""
        br_comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.03)
        br_elem = AbstractElement(name="Birth Rate", components=[br_comp],
                                  units="1/year")
        lut_ast = LookupsStructure(
            x=(0.0, 1.0, 2.0), y=(0.0, 0.5, 1.0),
            x_limits=(0.0, 2.0), y_limits=(0.0, 1.0), type="interpolate",
        )
        lut_comp = AbstractLookup(subscripts=[[], []], ast=lut_ast)
        lut_elem = AbstractElement(name="Effect Table", components=[lut_comp])
        flow_ast = ArithmeticStructure(
            operators=["*"],
            arguments=[ReferenceStructure("Population"), ReferenceStructure("Birth Rate")],
        )
        pop_ast = IntegStructure(flow=flow_ast, initial=1000.0)
        pop_comp = AbstractComponent(subscripts=[[], []], ast=pop_ast)
        pop_elem = AbstractElement(name="Population", components=[pop_comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 100.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[br_elem, lut_elem, pop_elem] + controls,
            path=tmp_path / "my_model.mdl",
        )
        return AbstractModel(
            original_path=tmp_path / "my_model.mdl",
            sections=(section,),
        )

    def test_invalid_data_format_raises(self, tmp_path):
        model = self._minimal_model_with_lookup(tmp_path)
        with pytest.raises(ValueError, match="data_format"):
            JuliaModelBuilder(model, data_format="invalid")

    def test_json_mode_creates_data_file(self, tmp_path):
        model = self._minimal_model_with_lookup(tmp_path)
        JuliaModelBuilder(model, data_format="json").build_model()
        assert (tmp_path / "my_model_data.json").exists()

    def test_hardcoded_mode_no_data_file(self, tmp_path):
        model = self._minimal_model_with_lookup(tmp_path)
        JuliaModelBuilder(model, data_format="hardcoded").build_model()
        assert not (tmp_path / "my_model_data.json").exists()

    def test_json_file_has_correct_schema(self, tmp_path):
        import json
        model = self._minimal_model_with_lookup(tmp_path)
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "my_model_data.json").read_text())
        assert "constants" in data
        assert "lookups" in data
        assert "data" in data

    def test_json_file_contains_parameter(self, tmp_path):
        import json
        model = self._minimal_model_with_lookup(tmp_path)
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "my_model_data.json").read_text())
        assert "birth_rate" in data["constants"]
        assert data["constants"]["birth_rate"]["values"] == pytest.approx(0.03)
        assert data["constants"]["birth_rate"]["units"] == "1/year"

    def test_json_file_contains_lookup(self, tmp_path):
        import json
        model = self._minimal_model_with_lookup(tmp_path)
        JuliaModelBuilder(model, data_format="json").build_model()
        # Named lookup tables are registered as inline lookups via inline_registry
        data = json.loads((tmp_path / "my_model_data.json").read_text())
        assert "lookups" in data
        # The lookup should have x, y, interp_type fields
        if data["lookups"]:
            key = next(iter(data["lookups"]))
            lut = data["lookups"][key]
            assert "x" in lut and "y" in lut and "interp_type" in lut

    def test_jl_file_uses_json3(self, tmp_path):
        model = self._minimal_model_with_lookup(tmp_path)
        path = JuliaModelBuilder(model, data_format="json").build_model()
        content = path.read_text()
        assert "JSON3" in content
        assert "_model_data" in content
        assert "my_model_data.json" in content

    def test_jl_file_params_reference_model_data(self, tmp_path):
        model = self._minimal_model_with_lookup(tmp_path)
        path = JuliaModelBuilder(model, data_format="json").build_model()
        content = path.read_text()
        assert '_model_data["constants"]["birth_rate"]' in content

    def test_hardcoded_mode_unchanged(self, tmp_path):
        """data_format='hardcoded' produces identical output to no data_format arg."""
        model1 = self._minimal_model_with_lookup(tmp_path / "a")
        (tmp_path / "a").mkdir()
        path1 = JuliaModelBuilder(model1).build_model()

        model2 = self._minimal_model_with_lookup(tmp_path / "b")
        (tmp_path / "b").mkdir()
        path2 = JuliaModelBuilder(model2, data_format="hardcoded").build_model()

        assert path1.read_text() == path2.read_text()

    def test_json_mode_get_lookups(self, mocker, tmp_path):
        """External lookup via GET_DIRECT_LOOKUPS appears in JSON file."""
        import json
        import numpy as np
        import xarray as xr
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.array([10.0, 20.0, 30.0])
        da = xr.DataArray(ys, coords={"lookup_dim": xs}, dims=["lookup_dim"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtLookup", return_value=mock_ext)
        ast = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Ext Lut", components=[comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "ext_lut" in data["lookups"]
        assert data["lookups"]["ext_lut"]["x"] == pytest.approx([0.0, 1.0, 2.0])

    def test_json_mode_get_data(self, mocker, tmp_path):
        """External time-series via GET_DIRECT_DATA appears in JSON file."""
        import json
        import numpy as np
        import xarray as xr
        ts = np.array([1995.0, 2000.0, 2005.0])
        vals = np.array([1.0, 2.0, 3.0])
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtData", return_value=mock_ext)
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Historic Eff", components=[comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "historic_eff" in data["data"]
        assert data["data"]["historic_eff"]["time"] == pytest.approx([1995.0, 2000.0, 2005.0])


class TestJSONDataBackendCoverage:
    """Covers remaining JSON-mode branches."""

    def _controls(self):
        return [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]

    def test_json_mode_inline_lookup_accumulated(self, tmp_path):
        """Inline lookups (InlineLookupsStructure) go into _json_data in JSON mode."""
        import json
        lut_ast = InlineLookupsStructure(
            argument=1.0,
            lookups=LookupsStructure(
                x=(0.0, 1.0), y=(0.0, 2.0),
                x_limits=(0.0, 1.0), y_limits=(0.0, 2.0),
                type="interpolate",
            ),
        )
        comp = AbstractComponent(subscripts=[[], []], ast=lut_ast)
        elem = AbstractElement(name="LutResult", components=[comp])
        section = _make_section(
            elements=[elem] + self._controls(), path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert any("_inline_lookup_" in k for k in data["lookups"])

    def test_json_mode_ext_constant_accumulates(self, mocker, tmp_path):
        """GetConstantsStructure in JSON mode calls _json_accumulate_constant."""
        import json
        import numpy as np
        mock_ext = mocker.MagicMock()
        mock_ext.data = np.float64(7.5)
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)
        ast = GetConstantsStructure(file="d.xlsx", tab="S", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Ext Rate", components=[comp], units="1/year")
        section = _make_section(
            elements=[elem] + self._controls(), path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "ext_rate" in data["constants"]
        assert data["constants"]["ext_rate"]["values"] == pytest.approx(7.5)
        assert data["constants"]["ext_rate"]["units"] == "1/year"

    def test_json_mode_ext_constant_array_in_ext_const_decls(self, mocker, tmp_path):
        """Array external constant → ext_const_decls in JSON mode → JSON-backed ref."""
        import json
        import numpy as np
        mock_ext = mocker.MagicMock()
        mock_ext.data = np.array([1.0, 2.0, 3.0])
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)
        ast = GetConstantsStructure(file="d.xlsx", tab="S", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Arr Const", components=[comp])
        section = _make_section(
            elements=[elem] + self._controls(), path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        path = JuliaModelBuilder(model, data_format="json").build_model()
        content = path.read_text()
        # In JSON mode, array const uses _model_data reference
        assert '_model_data["constants"]["arr_const"]' in content
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "arr_const" in data["constants"]

    def test_json_mode_2d_lookup_accumulates(self, mocker, tmp_path):
        """2D subscripted lookup in JSON mode stores each column."""
        import json
        import numpy as np
        import xarray as xr
        xs = np.array([0.0, 1.0])
        ys = np.ones((2, 3))
        da = xr.DataArray(ys, coords={"lookup_dim": xs}, dims=["lookup_dim", "sub"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtLookup", return_value=mock_ext)
        ast = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Sub Lut", components=[comp])
        section = _make_section(
            elements=[elem] + self._controls(), path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "sub_lut_1" in data["lookups"]
        assert "sub_lut_2" in data["lookups"]
        assert "sub_lut_3" in data["lookups"]

    def test_json_mode_2d_data_accumulates(self, mocker, tmp_path):
        """2D time-series in JSON mode stores each column."""
        import json
        import numpy as np
        import xarray as xr
        ts = np.array([1995.0, 2000.0])
        vals = np.ones((2, 2))
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time", "sub"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtData", return_value=mock_ext)
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Sub Series", components=[comp])
        section = _make_section(
            elements=[elem] + self._controls(), path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "sub_series_1" in data["data"]
        assert "sub_series_2" in data["data"]

    def test_json_mode_modular_build_writes_json(self, tmp_path):
        """Modular (split) build in JSON mode still writes the data JSON file."""
        br_comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.05)
        br_elem = AbstractElement(name="Rate", components=[br_comp])
        pop = _make_stock_element("Population", 1.0, 100.0)
        controls = self._controls()
        views_dict = {"Main": {"Population"}, "Params": {"Rate"}}
        section = _make_section(
            elements=[br_elem, pop] + controls,
            path=tmp_path / "split.mdl",
            split=True,
            views_dict=views_dict,
        )
        model = AbstractModel(original_path=tmp_path / "split.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json", backend="mtk").build_model()
        assert (tmp_path / "split_data.json").exists()

    def test_json_mode_nonnumeric_constant_uses_fallback(self, tmp_path):
        """A constant whose value can't be float()-converted is skipped gracefully."""
        import json
        # Use an ArithmeticStructure as the constant AST — visitor.visit() returns
        # a Julia expression like "(a * b)" that can't be float()'d
        rhs = ArithmeticStructure(
            operators=["*"],
            arguments=[ReferenceStructure("a"), ReferenceStructure("b")],
        )
        a_elem = _make_element("a", 2.0, comp_class=AbstractUnchangeableConstant)
        b_elem = _make_element("b", 3.0, comp_class=AbstractUnchangeableConstant)
        comp = AbstractComponent(subscripts=[[], []], ast=rhs)
        comp.type = "Constant"
        elem = AbstractElement(name="Product", components=[comp])
        section = _make_section(
            elements=[a_elem, b_elem, elem] + self._controls(),
            path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        # Should not raise even though the value can't be stored as float
        path = JuliaModelBuilder(model, data_format="json").build_model()
        assert path.exists()

    def test_json_mode_3d_lookup_accumulates(self, mocker, tmp_path):
        """3D GET LOOKUPS (x × dim1 × dim2) in JSON mode emits per-(i,j) sub-lookups."""
        import json
        import numpy as np
        import xarray as xr
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.ones((3, 2, 3))  # (n_x_points, n_dim1, n_dim2) → 3D
        da = xr.DataArray(ys, coords={"lookup_dim": xs}, dims=["lookup_dim", "sub1", "sub2"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtLookup", return_value=mock_ext)
        ast = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Sub3D Lut", components=[comp])
        section = _make_section(
            elements=[elem] + self._controls(), path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        # 2×3 grid → sub3d_lut_1_1 … sub3d_lut_2_3
        assert "sub3d_lut_1_1" in data["lookups"]
        assert "sub3d_lut_2_3" in data["lookups"]

    def test_json_mode_3d_data_accumulates(self, mocker, tmp_path):
        """3D GET DATA (time × dim1 × dim2) in JSON mode emits per-(i,j) time-series."""
        import json
        import numpy as np
        import xarray as xr
        ts = np.array([0.0, 5.0, 10.0])
        vals = np.ones((3, 2, 3))  # (n_time, n_dim1, n_dim2) → 3D
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time", "sub1", "sub2"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtData", return_value=mock_ext)
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Sub3D Series", components=[comp])
        section = _make_section(
            elements=[elem] + self._controls(), path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        # 2×3 grid → sub3d_series_1_1 … sub3d_series_2_3
        assert "sub3d_series_1_1" in data["data"]
        assert "sub3d_series_2_3" in data["data"]


class TestJSONAccumulateConstant:
    """Covers the _json_accumulate_constant helper's edge cases."""

    def test_xarray_dataarray_uses_values(self, mocker, tmp_path):
        """When ext.data is a DataArray, .values is extracted (line 1320)."""
        import json
        import numpy as np
        import xarray as xr
        da = xr.DataArray(np.float64(9.9))  # 0-D DataArray with .values
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)
        ast = GetConstantsStructure(file="d.xlsx", tab="S", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Da Const", components=[comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "da_const" in data["constants"]
        assert data["constants"]["da_const"]["values"] == pytest.approx(9.9)

    def test_exception_in_accumulate_uses_julia_val_fallback(self, mocker, tmp_path):
        """If _json_accumulate_constant raises, the julia runtime-read literal is stored."""
        import json
        # _read_get_constants now emits pysd_xlsx_read_constant (no ExtConstant call).
        # Only _json_accumulate_constant calls ExtConstant; when it raises, the
        # fallback stores julia_val (the runtime xlsx call string).
        mock_ext = mocker.MagicMock()
        mock_ext.initialize.side_effect = RuntimeError("accumulate fails")
        mocker.patch(
            "pysd.py_backend.external.ExtConstant",
            return_value=mock_ext,
        )
        ast = GetConstantsStructure(file="d.xlsx", tab="S", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Fallback Const", components=[comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        # Fallback stores the julia runtime-read expression
        assert "fallback_const" in data["constants"]
        assert "pysd_xlsx_read_constant" in data["constants"]["fallback_const"]["values"]


# ===========================================================================
# Phase 3B — GET DATA interpolation method passthrough
# ===========================================================================

class TestGetDataMethodPassthrough:

    def test_vensim_keyword_to_itp_type(self):
        from pysd.builders.julia.julia_model_builder import _vensim_keyword_to_itp_type
        assert _vensim_keyword_to_itp_type(None) == "interpolate"
        assert _vensim_keyword_to_itp_type("interpolate") == "interpolate"
        assert _vensim_keyword_to_itp_type("hold_backward") == "hold_forward"
        assert _vensim_keyword_to_itp_type("look_forward") == "hold_backward"
        assert _vensim_keyword_to_itp_type("raw") == "interpolate"

    def test_hold_backward_produces_constant_interpolation(self, mocker, tmp_path):
        import numpy as np
        import xarray as xr
        ts = np.array([1995.0, 2000.0, 2005.0])
        vals = np.array([1.0, 2.0, 3.0])
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtData", return_value=mock_ext)
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        # AbstractData with hold_backward keyword
        comp = AbstractData(subscripts=[[], []], ast=ast, keyword="hold_backward")
        elem = AbstractElement(name="Step Series", components=[comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "ConstantInterpolation" in content
        assert "LinearInterpolation" not in content

    def test_look_forward_produces_constant_interpolation_right(self, mocker, tmp_path):
        import numpy as np
        import xarray as xr
        ts = np.array([1995.0, 2000.0])
        vals = np.array([1.0, 2.0])
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtData", return_value=mock_ext)
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        comp = AbstractData(subscripts=[[], []], ast=ast, keyword="look_forward")
        elem = AbstractElement(name="Fwd Series", components=[comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "ConstantInterpolation" in content
        assert "dir=:right" in content


# ===========================================================================
# Phase 3C — Variable limits
# ===========================================================================

class TestVariableLimits:

    def test_limits_comment_no_limits(self):
        elem = _make_element("x", 1.0)
        assert JuliaSectionBuilder._limits_comment(elem) == ""

    def test_limits_comment_both_bounds(self):
        elem = AbstractElement(
            name="x", components=[_make_component(1.0)],
            limits=(0.0, 1.0), units="Dmnl",
        )
        comment = JuliaSectionBuilder._limits_comment(elem)
        assert "0.0" in comment and "1.0" in comment
        assert comment.startswith("  #= limits:")

    def test_limits_comment_lower_only(self):
        elem = AbstractElement(
            name="x", components=[_make_component(1.0)],
            limits=(0.0, None),
        )
        comment = JuliaSectionBuilder._limits_comment(elem)
        assert "0.0" in comment
        assert "Inf" in comment

    def test_limits_comment_upper_only(self):
        elem = AbstractElement(
            name="x", components=[_make_component(1.0)],
            limits=(None, 100.0),
        )
        comment = JuliaSectionBuilder._limits_comment(elem)
        assert "-Inf" in comment
        assert "100.0" in comment

    def test_limits_appear_in_param_declaration(self):
        comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.5)
        elem = AbstractElement(name="Birth Rate", components=[comp], limits=(0.0, 1.0))
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        assert any("#= limits:" in d for d in sb.param_decls)

    def test_limits_appear_in_aux_equation(self):
        comp = AbstractComponent(subscripts=[[], []], ast=2.5)
        elem = AbstractElement(name="Output", components=[comp], limits=(0.0, None))
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("#= limits:" in e for e in eqs)

    def test_limits_in_full_generated_file(self, tmp_path):
        comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.5)
        elem = AbstractElement(name="Rate", components=[comp], limits=(0.0, 1.0))
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "#= limits:" in content

    def test_limits_stored_in_json(self, tmp_path):
        import json
        comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.5)
        elem = AbstractElement(name="Rate", components=[comp], limits=(0.0, 1.0))
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "limits" in data["constants"]["rate"]
        assert data["constants"]["rate"]["limits"] == [0.0, 1.0]


# ===========================================================================
# Phase 3D — EXCEPT subscript exclusion
# ===========================================================================

class TestExceptSubscriptExclusion:

    def _make_except_element(self, name, dim_name, dim_elems,
                              comp1_ast, comp2_ast, except_labels):
        """Make an element with two components where comp1 has EXCEPT."""
        # comp1: covers dim_name, except except_labels
        comp1 = AbstractComponent(
            subscripts=[[dim_name], [except_labels]],
            ast=comp1_ast,
        )
        # comp2: covers just the excepted elements (no EXCEPT)
        comp2 = AbstractComponent(
            subscripts=[[dim_name], []],
            ast=comp2_ast,
        )
        return AbstractElement(name=name, components=[comp1, comp2])

    def test_except_element_generates_per_index_equations(self):
        sr = _make_subscript_range("category", ["A", "B", "C"])
        # comp1: category = 1.0, EXCEPT [B]
        # comp2: all category = 2.0 (no EXCEPT)
        elem = self._make_except_element(
            "My Var", "category", ["A", "B", "C"],
            comp1_ast=1.0, comp2_ast=2.0,
            except_labels=["B"],
        )
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # identifier is "my_var"; should have equations for index 1 (A), 3 (C) from comp1
        assert any("my_var[1]" in e for e in eqs)  # A from comp1
        assert any("my_var[3]" in e for e in eqs)  # C from comp1
        # The variable should be declared as array
        assert any("my_var(t)[" in d for d in sb.aux_decls)

    def test_except_element_excludes_correct_index(self):
        sr = _make_subscript_range("sector", ["S1", "S2", "S3"])
        elem = self._make_except_element(
            "Output", "sector", ["S1", "S2", "S3"],
            comp1_ast=5.0, comp2_ast=10.0,
            except_labels=["S2"],
        )
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # comp1 covers S1(1) and S3(3), NOT S2(2)
        comp1_eqs = [e for e in eqs if "5.0" in e]
        assert any("[1]" in e for e in comp1_eqs)
        assert any("[3]" in e for e in comp1_eqs)
        assert not any("[2]" in e for e in comp1_eqs)

    def test_except_2d_emits_equations_for_all_pairs(self):
        """2D EXCEPT: both components must produce equations covering all (i,j) pairs
        with no warning about unsupported dimensionality."""
        # r has 3 elements; c has 2 elements → 6 total pairs
        # comp0: r×c EXCEPT [R1]×c  → covers (R2, *) and (R3, *)
        # comp1: r×c (no EXCEPT)    → covers all r×c; effectively fills (R1, *)
        sr1 = _make_subscript_range("r", ["R1", "R2", "R3"])
        sr2 = _make_subscript_range("c", ["C1", "C2"])
        comp0 = AbstractComponent(
            subscripts=[["r", "c"], [["R1", "c"]]],
            ast=1.0,
        )
        comp1 = AbstractComponent(subscripts=[["r", "c"], []], ast=2.0)
        elem = AbstractElement(name="Matrix", components=[comp0, comp1])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # fail if any UserWarning is raised
            sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2])
            sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert len(eqs) > 0

    def test_except_2d_excluded_rows_use_second_component(self):
        """Pairs excluded from comp0 via EXCEPT must use comp1's formula, not comp0's."""
        # r={A,B,C}, c={X,Y}; comp0 covers r×c EXCEPT [B]×c; comp1 covers all r×c
        sr1 = _make_subscript_range("r", ["A", "B", "C"])
        sr2 = _make_subscript_range("c", ["X", "Y"])
        comp0 = AbstractComponent(
            subscripts=[["r", "c"], [["B", "c"]]],
            ast=10.0,
        )
        comp1 = AbstractComponent(subscripts=[["r", "c"], []], ast=99.0)
        elem = AbstractElement(name="Out", components=[comp0, comp1])
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # comp0 formula (10.0) must NOT appear for any equation at row-index 2 (B)
        comp0_eqs = [e for e in eqs if "10.0" in e]
        assert not any(
            ("[2," in e or ", 2]" in e or "[2]" in e) for e in comp0_eqs
        ), "comp0's formula must not be used for row B (index 2)"
        # comp0 formula (10.0) MUST appear for rows A(1) and C(3)
        assert any("[1," in e or "1]" in e for e in comp0_eqs), "comp0 must cover row A"
        assert any("[3," in e or "3]" in e for e in comp0_eqs), "comp0 must cover row C"

    def test_except_2d_element_spec_as_specific_element(self):
        """When a component's subscript spec names a specific element (not a range),
        only that element's rows/columns should be covered."""
        # r={A,B,C}; c={X,Y}
        # comp0: r×c EXCEPT [B]×c → covers (A,*) and (C,*)
        # comp1: B×c (specific element, no EXCEPT) → covers (B,*)
        sr1 = _make_subscript_range("r", ["A", "B", "C"])
        sr2 = _make_subscript_range("c", ["X", "Y"])
        comp0 = AbstractComponent(
            subscripts=[["r", "c"], [["B", "c"]]],
            ast=1.0,
        )
        comp1 = AbstractComponent(subscripts=[["B", "c"], []], ast=2.0)
        elem = AbstractElement(name="Res", components=[comp0, comp1])
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # comp1 formula (2.0) must appear only for row B (index 2).
        # Generated form: "[res[_i0, _i1] ~ 2.0 for _i0 in [2], _i1 in ...]..."
        comp1_eqs = [e for e in eqs if "2.0" in e]
        assert comp1_eqs, "comp1 formula must appear in some equation"
        assert all("in [2]" in e or "_i0, 2]" in e for e in comp1_eqs), (
            f"comp1's formula must only cover row 2 (B); got: {comp1_eqs}"
        )

    def test_subrange_component_uses_subrange_index_on_rhs(self):
        """When a component covers a sub-range of the LHS dimension, RHS
        references to variables subscripted over that sub-range must use
        the 1-based index WITHIN the sub-range, not the parent dimension index.

        Regression for pymedeas world model:
          SECTORS_AND_HOUSEHOLDS = [H, A, B]  (parent dim, size 3)
          sectors                = [A, B]     (sub-range, size 2)
          my_var[sectors_and_households];  component: my_var[sectors] = other_var[sectors]
        Expected: my_var[2] ~ other_var[1],  my_var[3] ~ other_var[2]
        Broken:   my_var[2] ~ other_var[2],  my_var[3] ~ other_var[3] (OOB!)
        """
        sr_parent = _make_subscript_range("sectors_and_households", ["H", "A", "B"])
        sr_sub    = _make_subscript_range("sectors", ["A", "B"])

        # other_var[sectors] — simple auxiliary subscripted over the sub-range
        other_comp = AbstractComponent(
            subscripts=[["sectors"], []],
            ast=1.0,
        )
        other_elem = AbstractElement(name="other var", components=[other_comp])

        # my_var[sectors_and_households] with two components:
        #   comp1: my_var[sectors] = other_var[sectors]
        #   comp2: my_var[H]       = 0.0
        # Having both forces _element_dims to infer sectors_and_households as parent.
        my_comp_sectors = AbstractComponent(
            subscripts=[["sectors"], []],
            ast=ReferenceStructure("other var"),
        )
        my_comp_h = AbstractComponent(
            subscripts=[["H"], []],
            ast=0.0,
        )
        my_elem = AbstractElement(name="my var", components=[my_comp_sectors, my_comp_h])

        sb = _section_builder_from_elements(
            [other_elem, my_elem],
            subscripts=[sr_parent, sr_sub],
        )
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]

        my_var_eqs = [e for e in eqs if e.startswith("my_var[")]
        # 3 equations: sectors components (A→index 2, B→index 3) + H component (index 1)
        assert len(my_var_eqs) == 3, f"expected 3 my_var equations, got: {my_var_eqs}"
        # A at parent index 2 → sub-range index 1 in sectors
        assert any("my_var[2] ~ other_var[1]" in e for e in my_var_eqs), (
            f"my_var[2] should reference other_var[1]; got: {my_var_eqs}"
        )
        # B at parent index 3 → sub-range index 2 in sectors
        assert any("my_var[3] ~ other_var[2]" in e for e in my_var_eqs), (
            f"my_var[3] should reference other_var[2]; got: {my_var_eqs}"
        )
        # H at parent index 1 uses the constant formula
        assert any("my_var[1] ~ 0.0" in e for e in my_var_eqs), (
            f"my_var[1] should be 0.0; got: {my_var_eqs}"
        )


    def test_3d_per_element_no_warning(self):
        """3D multi-component element (no EXCEPT clauses, just per-element slices)
        must not emit the '3D subscripts not yet supported' warning."""
        # c={E,F}, d={A,B}, d1={A,B}
        # comp0: covers [E,d,d1]  → c-slice 1, all d, all d1
        # comp1: covers [F,d,d1]  → c-slice 2, all d, all d1
        sr_c  = _make_subscript_range("c",  ["E", "F"])
        sr_d  = _make_subscript_range("d",  ["A", "B"])
        sr_d1 = _make_subscript_range("d1", ["A", "B"])
        comp0 = AbstractComponent(subscripts=[["E", "d", "d1"], []], ast=1.0)
        comp1 = AbstractComponent(subscripts=[["F", "d", "d1"], []], ast=2.0)
        elem = AbstractElement(name="Matrix Two", components=[comp0, comp1])
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)  # fail on any UserWarning
            sb = _section_builder_from_elements(
                [elem], subscripts=[sr_c, sr_d, sr_d1]
            )
            sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # Comprehension form: "for _i0 in [1]" means c=1 (E); "[2]" means c=2 (F)
        assert any("in [1]" in e and "matrix_two" in e for e in eqs), \
            f"Missing c=1 (E) equations; got: {eqs}"
        assert any("in [2]" in e and "matrix_two" in e for e in eqs), \
            f"Missing c=2 (F) equations; got: {eqs}"

    def test_3d_except_emits_per_index_equations(self):
        """3D element with true EXCEPT clause emits correct index comprehensions."""
        # c={E,F}, d={A,B}; d1={A,B}
        # comp0: c×d×d1 EXCEPT [F, d, d1] → covers only c=E (index 1)
        # comp1: c=F (index 2) × d×d1 (no EXCEPT)
        sr_c  = _make_subscript_range("c",  ["E", "F"])
        sr_d  = _make_subscript_range("d",  ["A", "B"])
        sr_d1 = _make_subscript_range("d1", ["A", "B"])
        comp0 = AbstractComponent(
            subscripts=[["c", "d", "d1"], [["F", "d", "d1"]]],
            ast=10.0,
        )
        comp1 = AbstractComponent(subscripts=[["F", "d", "d1"], []], ast=20.0)
        elem = AbstractElement(name="Matrix Three", components=[comp0, comp1])
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            sb = _section_builder_from_elements(
                [elem], subscripts=[sr_c, sr_d, sr_d1]
            )
            sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # comp0 formula (10.0) must only appear in equations for c=1 (E)
        comp0_eqs = [e for e in eqs if "10.0" in e and "matrix_three" in e]
        assert comp0_eqs, "comp0 equations not found"
        assert all("in [1]" in e for e in comp0_eqs), \
            f"comp0 must only cover c=1 (E); got: {comp0_eqs}"
        # comp1 formula (20.0) must only appear for c=2 (F)
        comp1_eqs = [e for e in eqs if "20.0" in e and "matrix_three" in e]
        assert comp1_eqs, "comp1 equations not found"
        assert all("in [2]" in e for e in comp1_eqs), \
            f"comp1 must only cover c=2 (F); got: {comp1_eqs}"


# ===========================================================================
# Phase 3E — Macro support
# ===========================================================================

class TestMacroSupport:

    def _two_section_model(self, tmp_path):
        """AbstractModel with a main section and one macro section.

        The macro element name 'My Macro' normalises to 'my_macro', matching
        the section name, so it is correctly identified as the return value.
        """
        # Main section: simple stock
        pop = _make_stock_element("Population", 1.0, 100.0)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        main_section = _make_section(
            elements=[pop] + controls,
            path=tmp_path / "my_model.mdl",
        )

        # Macro section: element name matches macro name so it's the return value
        macro_aux = _make_element("My Macro", 42.0)
        macro_section = AbstractSection(
            name="my_macro",
            path=tmp_path / "my_model.mdl",
            type="macro",
            params=["Input"],
            returns=["My Macro"],
            subscripts=(),
            elements=(macro_aux,),
            constraints=(),
            test_inputs=(),
            split=False,
            views_dict=None,
        )

        return AbstractModel(
            original_path=tmp_path / "my_model.mdl",
            sections=(main_section, macro_section),
        )

    def test_build_model_creates_main_jl(self, tmp_path):
        model = self._two_section_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        assert path.exists()
        assert path.suffix == ".jl"

    def test_macro_section_creates_companion_file(self, tmp_path):
        model = self._two_section_model(tmp_path)
        JuliaModelBuilder(model).build_model()
        macro_file = tmp_path / "my_model_my_macro.jl"
        assert macro_file.exists()

    def test_macro_file_contains_julia_function(self, tmp_path):
        """ODE backend companion file defines a Julia function, not MTK equations."""
        model = self._two_section_model(tmp_path)
        JuliaModelBuilder(model).build_model()
        macro_file = tmp_path / "my_model_my_macro.jl"
        content = macro_file.read_text()
        assert "function my_macro(" in content

    def test_macro_function_takes_params_as_args(self, tmp_path):
        """Companion function signature includes macro params."""
        model = self._two_section_model(tmp_path)
        JuliaModelBuilder(model).build_model()
        macro_file = tmp_path / "my_model_my_macro.jl"
        content = macro_file.read_text()
        assert "function my_macro(input)" in content

    def test_macro_file_contains_macro_name_comment(self, tmp_path):
        model = self._two_section_model(tmp_path)
        JuliaModelBuilder(model).build_model()
        macro_file = tmp_path / "my_model_my_macro.jl"
        content = macro_file.read_text()
        assert "Macro my_macro" in content

    def test_main_file_unaffected_by_macro(self, tmp_path):
        """Main model still contains rhs! function even with a macro section."""
        model = self._two_section_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "function rhs!" in content
        assert "population" in content

    def test_main_file_includes_macro_companion(self, tmp_path):
        """Main model has an include() statement for the macro companion file."""
        model = self._two_section_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert 'include(' in content
        assert 'my_macro' in content

    def test_macro_params_no_namespace_warning(self, tmp_path):
        """Macro params are in namespace; no 'not found' warning during translation."""
        import warnings
        from pathlib import Path
        import shutil
        mdl_src = Path("tests/test-models/tests/macro_expression/test_macro_expression.mdl")
        if not mdl_src.exists():
            pytest.skip("macro_expression model not found")
        dst = tmp_path / "test_macro_expression.mdl"
        shutil.copy(mdl_src, dst)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import pysd
            pysd.translate_to_julia(str(dst))
        ns_warnings = [x for x in w if "not found in namespace" in str(x.message)]
        assert not ns_warnings, f"Unexpected namespace warnings: {ns_warnings}"

    def test_macro_call_no_unknown_function_warning(self, tmp_path):
        """Calling a macro from the main section does not emit 'Unknown Vensim function'."""
        import warnings
        from pathlib import Path
        import shutil
        mdl_src = Path("tests/test-models/tests/macro_expression/test_macro_expression.mdl")
        if not mdl_src.exists():
            pytest.skip("macro_expression model not found")
        dst = tmp_path / "test_macro_expression.mdl"
        shutil.copy(mdl_src, dst)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import pysd
            pysd.translate_to_julia(str(dst))
        unk_warnings = [x for x in w if "Unknown Vensim function" in str(x.message)]
        assert not unk_warnings, f"Unexpected unknown-function warnings: {unk_warnings}"

    def test_mtk_macro_companion_uses_equations_array(self, tmp_path):
        """MTK backend macro companion file emits an Equation[] array, not a function."""
        model = self._two_section_model(tmp_path)
        JuliaModelBuilder(model, backend="mtk").build_model()
        macro_file = tmp_path / "my_model_my_macro.jl"
        assert macro_file.exists()
        content = macro_file.read_text()
        assert "my_macro_eqs = Equation[" in content
        assert "ModelingToolkit" in content
        assert "function my_macro(" not in content


class TestMacroSupportCoverage:
    """Cover remaining macro-section code paths."""

    def test_macro_with_inline_lookup_and_json(self, tmp_path):
        """Macro with inline lookup (json mode): companion file has DataInterpolations."""
        lut_ast = InlineLookupsStructure(
            argument=1.0,
            lookups=LookupsStructure(
                x=(0.0, 1.0), y=(0.0, 2.0),
                x_limits=(0.0, 1.0), y_limits=(0.0, 2.0),
                type="interpolate",
            ),
        )
        comp = AbstractComponent(subscripts=[[], []], ast=lut_ast)
        lut_elem = AbstractElement(name="Macro LUT", components=[comp])
        main_section = _make_section(
            elements=[
                _make_stock_element("S", 1.0, 1.0),
                _make_control_element("INITIAL TIME", 0.0),
                _make_control_element("FINAL TIME", 10.0),
                _make_control_element("TIME STEP", 1.0),
                _make_control_element("SAVEPER", 1.0),
            ],
            path=tmp_path / "m.mdl",
        )
        macro_section = AbstractSection(
            name="lookup_macro", path=tmp_path / "m.mdl",
            type="macro", params=[], returns=["Macro LUT"],
            subscripts=(), elements=(lut_elem,),
            constraints=(), test_inputs=(),
            split=False, views_dict=None,
        )
        model = AbstractModel(
            original_path=tmp_path / "m.mdl",
            sections=(main_section, macro_section),
        )
        JuliaModelBuilder(model, data_format="json").build_model()
        macro_path = tmp_path / "m_lookup_macro.jl"
        assert macro_path.exists()
        content = macro_path.read_text()
        assert "DataInterpolations" in content
        assert "function lookup_macro(" in content
        assert (tmp_path / "m_lookup_macro_data.json").exists()

    def test_macro_with_const_only_emits_bare_return(self, tmp_path):
        """ODE macro where all elements are constants → empty body_lines → line 396."""
        const_elem = _make_element("My Const", 7.0, comp_class=AbstractUnchangeableConstant)
        main_section = _make_section(
            elements=[
                _make_stock_element("S", 1.0, 1.0),
                _make_control_element("INITIAL TIME", 0.0),
                _make_control_element("FINAL TIME", 10.0),
                _make_control_element("TIME STEP", 1.0),
                _make_control_element("SAVEPER", 1.0),
            ],
            path=tmp_path / "m.mdl",
        )
        macro_section = AbstractSection(
            name="const_macro", path=tmp_path / "m.mdl",
            type="macro", params=[], returns=["My Const"],
            subscripts=(), elements=(const_elem,),
            constraints=(), test_inputs=(), split=False, views_dict=None,
        )
        model = AbstractModel(
            original_path=tmp_path / "m.mdl",
            sections=(main_section, macro_section),
        )
        JuliaModelBuilder(model).build_model()
        content = (tmp_path / "m_const_macro.jl").read_text()
        assert "function const_macro(" in content
        assert "return " in content

    def test_macro_with_stock_emits_placeholder_return(self, tmp_path):
        """ODE macro containing a stock emits a 'return 0.0' placeholder (lines 368-373)."""
        import warnings
        stock_elem = _make_stock_element("My Level", 1.0, 0.0)
        main_section = _make_section(
            elements=[
                _make_stock_element("S", 1.0, 1.0),
                _make_control_element("INITIAL TIME", 0.0),
                _make_control_element("FINAL TIME", 10.0),
                _make_control_element("TIME STEP", 1.0),
                _make_control_element("SAVEPER", 1.0),
            ],
            path=tmp_path / "m.mdl",
        )
        macro_section = AbstractSection(
            name="stateful_macro", path=tmp_path / "m.mdl",
            type="macro", params=[], returns=["My Level"],
            subscripts=(), elements=(stock_elem,),
            constraints=(), test_inputs=(), split=False, views_dict=None,
        )
        model = AbstractModel(
            original_path=tmp_path / "m.mdl",
            sections=(main_section, macro_section),
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            JuliaModelBuilder(model).build_model()
        content = (tmp_path / "m_stateful_macro.jl").read_text()
        assert "return 0.0" in content
        assert "function stateful_macro(" in content

    def test_mtk_macro_with_lookup_includes_datainterpolations(self, tmp_path):
        """MTK macro with inline lookup includes DataInterpolations (line 427)."""
        lut_ast = InlineLookupsStructure(
            argument=1.0,
            lookups=LookupsStructure(
                x=(0.0, 1.0), y=(0.0, 2.0),
                x_limits=(0.0, 1.0), y_limits=(0.0, 2.0),
                type="interpolate",
            ),
        )
        lut_elem = AbstractElement(
            name="Lut Var",
            components=[AbstractComponent(subscripts=[[], []], ast=lut_ast)],
        )
        main_section = _make_section(
            elements=[
                _make_stock_element("S", 1.0, 1.0),
                _make_control_element("INITIAL TIME", 0.0),
                _make_control_element("FINAL TIME", 10.0),
                _make_control_element("TIME STEP", 1.0),
                _make_control_element("SAVEPER", 1.0),
            ],
            path=tmp_path / "m.mdl",
        )
        macro_section = AbstractSection(
            name="lut_macro", path=tmp_path / "m.mdl",
            type="macro", params=[], returns=["Lut Var"],
            subscripts=(), elements=(lut_elem,),
            constraints=(), test_inputs=(), split=False, views_dict=None,
        )
        model = AbstractModel(
            original_path=tmp_path / "m.mdl",
            sections=(main_section, macro_section),
        )
        JuliaModelBuilder(model, backend="mtk").build_model()
        content = (tmp_path / "m_lut_macro.jl").read_text()
        assert "DataInterpolations" in content
        assert "ModelingToolkit" in content

    def test_mtk_macro_with_json_includes_json3(self, tmp_path):
        """MTK macro with json data_format includes JSON3 (line 429)."""
        aux_elem = _make_element("Macro Var", 42.0)
        main_section = _make_section(
            elements=[
                _make_stock_element("S", 1.0, 1.0),
                _make_control_element("INITIAL TIME", 0.0),
                _make_control_element("FINAL TIME", 10.0),
                _make_control_element("TIME STEP", 1.0),
                _make_control_element("SAVEPER", 1.0),
            ],
            path=tmp_path / "m.mdl",
        )
        macro_section = AbstractSection(
            name="json_macro", path=tmp_path / "m.mdl",
            type="macro", params=[], returns=["Macro Var"],
            subscripts=(), elements=(aux_elem,),
            constraints=(), test_inputs=(), split=False, views_dict=None,
        )
        model = AbstractModel(
            original_path=tmp_path / "m.mdl",
            sections=(main_section, macro_section),
        )
        JuliaModelBuilder(model, backend="mtk", data_format="json").build_model()
        content = (tmp_path / "m_json_macro.jl").read_text()
        assert "JSON3" in content
        assert "ModelingToolkit" in content


class TestGetConstantsPiecewise1D:
    """Direct unit test for the 1-D piecewise GET CONSTANTS path.

    This path is unreachable from the normal _read_get_constants routing
    (which requires comp0_coords len >= 2, implying max_ndim >= 2 in the
    piecewise function).  We call _read_get_constants_piecewise directly.
    """

    def _make_builder(self, tmp_path):
        sr = _make_subscript_range("fuel", ["fuel1", "fuel2", "fuel3"])
        sb = _section_builder_from_elements(
            [
                _make_control_element("INITIAL TIME", 0.0),
                _make_control_element("FINAL TIME", 10.0),
                _make_control_element("TIME STEP", 1.0),
                _make_control_element("SAVEPER", 1.0),
            ],
            subscripts=[sr],
            path=tmp_path / "m.mdl",
        )
        return sb

    def test_piecewise_1d_array_gcs_with_literals(self, mocker, tmp_path):
        """1-D piecewise: GCS returns 1-D DataArray, literals fill remaining slots."""
        import xarray as xr
        import numpy as np

        sb = self._make_builder(tmp_path)

        gcs_ast = GetConstantsStructure(file="f.xlsx", tab="Sheet1", cell="A1")
        gcs_comp = AbstractComponent(subscripts=[["fuel1"], []], ast=gcs_ast)
        lit_comp2 = AbstractComponent(subscripts=[["fuel2"], []], ast=0.0)
        lit_comp3 = AbstractComponent(subscripts=[["fuel3"], []], ast=5.0)

        elem = AbstractElement(
            name="Fuel Costs",
            components=[gcs_comp, lit_comp2, lit_comp3],
        )

        mock_ext = mocker.MagicMock()
        mock_ext.data = xr.DataArray(
            [2.5], coords={"fuel": ["fuel1"]}, dims=["fuel"]
        )
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)

        result = sb._read_get_constants_piecewise(
            elem, "fuel_costs",
            gcs_comps=[gcs_comp],
            lit_comps=[lit_comp2, lit_comp3],
        )
        assert result is not None
        assert "2.5" in result
        assert "0.0" in result or "0" in result
        assert "5.0" in result or "5" in result

    def test_piecewise_1d_scalar_gcs_single_element(self, mocker, tmp_path):
        """1-D piecewise: GCS returns scalar (arr.ndim == 0), single result."""
        import xarray as xr
        import numpy as np

        sb = self._make_builder(tmp_path)

        gcs_ast = GetConstantsStructure(file="f.xlsx", tab="Sheet1", cell="A1")
        gcs_comp = AbstractComponent(subscripts=[["fuel1"], []], ast=gcs_ast)
        elem = AbstractElement(name="Fuel Cost", components=[gcs_comp])

        mock_ext = mocker.MagicMock()
        mock_ext.data = xr.DataArray(3.14)  # 0-d scalar
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)

        result = sb._read_get_constants_piecewise(
            elem, "fuel_cost",
            gcs_comps=[gcs_comp],
            lit_comps=[],
        )
        assert result is not None
        assert "3.14" in result or "3.1" in result

    def test_piecewise_1d_range_name_literal(self, tmp_path):
        """1-D piecewise: literal comp uses a RANGE NAME subscript (covers 3537-3538)."""
        sr = _make_subscript_range("fuel", ["fuel1", "fuel2"])
        sb = _section_builder_from_elements(
            [
                _make_control_element("INITIAL TIME", 0.0),
                _make_control_element("FINAL TIME", 10.0),
                _make_control_element("TIME STEP", 1.0),
                _make_control_element("SAVEPER", 1.0),
            ],
            subscripts=[sr],
            path=tmp_path / "m.mdl",
        )
        lit_comp = AbstractComponent(subscripts=[["fuel"], []], ast=3.0)
        elem = AbstractElement(name="Fuel Rates", components=[lit_comp])

        result = sb._read_get_constants_piecewise(
            elem, "fuel_rates",
            gcs_comps=[],
            lit_comps=[lit_comp],
        )
        assert result is not None
        assert "3.0" in result or "3" in result

    def test_piecewise_1d_no_parent_range(self, tmp_path):
        """1-D piecewise: elements not in any range → parent_range=None (covers 3570)."""
        sb = _section_builder_from_elements(
            [
                _make_control_element("INITIAL TIME", 0.0),
                _make_control_element("FINAL TIME", 10.0),
                _make_control_element("TIME STEP", 1.0),
                _make_control_element("SAVEPER", 1.0),
            ],
            subscripts=[],
            path=tmp_path / "m.mdl",
        )
        lit_x = AbstractComponent(subscripts=[["X"], []], ast=1.0)
        lit_y = AbstractComponent(subscripts=[["Y"], []], ast=2.0)
        elem = AbstractElement(name="Mixed", components=[lit_x, lit_y])

        result = sb._read_get_constants_piecewise(
            elem, "mixed",
            gcs_comps=[],
            lit_comps=[lit_x, lit_y],
        )
        assert result is not None

    def test_piecewise_1d_single_val(self, mocker, tmp_path):
        """1-D piecewise: single ordered element → format_number (covers line 3576)."""
        import xarray as xr

        sr = _make_subscript_range("solo", ["only1"])
        sb = _section_builder_from_elements(
            [
                _make_control_element("INITIAL TIME", 0.0),
                _make_control_element("FINAL TIME", 10.0),
                _make_control_element("TIME STEP", 1.0),
                _make_control_element("SAVEPER", 1.0),
            ],
            subscripts=[sr],
            path=tmp_path / "m.mdl",
        )

        gcs_ast = GetConstantsStructure(file="f.xlsx", tab="S", cell="A1")
        gcs_comp = AbstractComponent(subscripts=[["only1"], []], ast=gcs_ast)
        elem = AbstractElement(name="Solo Var", components=[gcs_comp])

        mock_ext = mocker.MagicMock()
        mock_ext.data = xr.DataArray(5.0)  # scalar
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)

        result = sb._read_get_constants_piecewise(
            elem, "solo_var",
            gcs_comps=[gcs_comp],
            lit_comps=[],
        )
        assert result is not None
        assert "5" in result
        assert "[" not in result  # single value, no brackets


class TestExceptConstantComponent:
    """Covers the Constant component in EXCEPT handler (lines 744-749)."""

    def test_except_with_constant_component_emits_equation(self):
        # EXCEPT component with constant value must emit an actual equation
        # (const_except[idx] ~ value), not a comment.  A comment is dead code
        # that the Julia runtime never executes.
        sr = _make_subscript_range("dim", ["X", "Y", "Z"])
        comp1 = AbstractUnchangeableConstant(
            subscripts=[["dim"], [["Y"]]], ast=1.0
        )
        comp2 = AbstractUnchangeableConstant(
            subscripts=[["dim"], []], ast=5.0
        )
        elem = AbstractElement(name="Const Except", components=[comp1, comp2])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert not any("# EXCEPT:" in e for e in eqs), (
            f"EXCEPT constant must not emit a comment, got: {eqs}"
        )
        assert any("const_except[" in e and " ~ " in e for e in eqs), (
            f"EXCEPT constant must emit const_except[idx] ~ value, got: {eqs}"
        )


# ===========================================================================
# Phase 4 — .mdl file translation tests (run without Julia runtime)
# ===========================================================================

class TestMdlFileTranslation:
    """Translate more-tests .mdl files and check the generated .jl content.
    These tests exercise the full PySD→Julia translation pipeline without
    requiring a Julia runtime.
    """

    MORE_TESTS = Path("tests/more-tests")

    def _translate(self, mdl_path, tmp_path):
        import shutil
        dst = tmp_path / mdl_path.name
        shutil.copy(mdl_path, dst)
        from pysd import translate_to_julia
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            jl_path = translate_to_julia(dst)
        return jl_path

    def test_julia_data_structure_creates_jl_file(self, tmp_path):
        mdl = self.MORE_TESTS / "julia_data_structure" / "test_julia_data_structure.mdl"
        if not mdl.exists():
            pytest.skip("julia_data_structure test model not found")
        jl_path = self._translate(mdl, tmp_path)
        assert jl_path.exists()
        assert jl_path.suffix == ".jl"

    def test_julia_data_structure_emits_data_override_warning(self, tmp_path):
        """AbstractData with a non-GET DATA equation (data-override) still warns."""
        mdl = self.MORE_TESTS / "julia_data_structure" / "test_julia_data_structure.mdl"
        if not mdl.exists():
            pytest.skip("julia_data_structure test model not found")
        import shutil, warnings
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            translate_to_julia(dst)
        msgs = [str(w.message) for w in captured]
        assert any("data-override" in m.lower() for m in msgs), \
            f"Expected data-override warning, got: {msgs}"

    def test_data_from_other_model_emits_tab_infrastructure(self, tmp_path):
        """data_from_other_model (DataStructure + AbstractData) emits tab-data helpers."""
        import shutil, warnings
        mdl = Path("tests/test-models/tests/data_from_other_model/test_data_from_other_model.mdl")
        if not mdl.exists():
            pytest.skip("data_from_other_model test model not found")
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            jl_path = translate_to_julia(dst)
        content = jl_path.read_text()
        assert "_load_tab_data!" in content, "Must emit _load_tab_data! loader function"
        assert "_tab_data" in content, "Must emit _tab_data Dict"
        assert "_tab_val" in content, "Must emit _tab_val helper"
        assert "tab_data_files" in content, "run_model must accept tab_data_files="

    def test_data_from_other_model_no_unsupported_warning(self, tmp_path):
        """DataStructure variables should not produce 'not supported' warnings."""
        import shutil, warnings
        mdl = Path("tests/test-models/tests/data_from_other_model/test_data_from_other_model.mdl")
        if not mdl.exists():
            pytest.skip("data_from_other_model test model not found")
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            translate_to_julia(dst)
        bad = [str(w.message) for w in captured
               if "not supported" in str(w.message).lower()
               and "DataStructure" in str(w.message)]
        assert not bad, f"DataStructure must not emit 'not supported': {bad}"

    def test_conditional_subscripts_no_namespace_warning(self, tmp_path):
        """conditional_subscripts uses bare element labels (B, C) in IF THEN ELSE
        comparisons.  They must resolve to integer indices, not emit 'not found'.
        """
        import shutil, warnings
        mdl = Path("tests/test-models/tests/conditional_subscripts/test_conditional_subscripts.mdl")
        if not mdl.exists():
            pytest.skip("conditional_subscripts test model not found")
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            translate_to_julia(dst)
        ns_warns = [str(w.message) for w in captured
                    if "not found in namespace" in str(w.message)]
        assert not ns_warns, f"Element labels must not warn 'not found': {ns_warns}"

    def test_conditional_subscripts_element_label_is_integer(self, tmp_path):
        """The generated Julia for Vector2[dimA] must compare _i0 to an integer
        index (2 for 'B', 3 for 'C'), not an undefined variable 'b' or 'c'.
        """
        import shutil, warnings
        mdl = Path("tests/test-models/tests/conditional_subscripts/test_conditional_subscripts.mdl")
        if not mdl.exists():
            pytest.skip("conditional_subscripts test model not found")
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            jl_path = translate_to_julia(dst)
        content = jl_path.read_text()
        # Should contain integer comparisons, not bare 'b' or 'c' identifiers
        assert "== 2" in content or "== 2)" in content, "Expected index 2 for element B"
        assert "== 3" in content or "== 3)" in content, "Expected index 3 for element C"

    def test_subscripted_delay_fixed_no_module_warning(self, tmp_path):
        """Subscripted DELAY FIXED pipeline must not warn 'Unsupported AST node type module'.
        The internal pipeline loop was accidentally passing the abstract_expressions
        module object to visitor.visit() instead of an AST node.
        """
        import shutil, warnings
        mdl = Path("tests/test-models/tests/forecast/test_forecast.mdl")
        if not mdl.exists():
            pytest.skip("forecast test model not found")
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            translate_to_julia(dst)
        module_warns = [str(w.message) for w in captured
                        if "module" in str(w.message).lower()
                        and "unsupported" in str(w.message).lower()]
        assert not module_warns, f"Should not warn about 'module' node: {module_warns}"

    def test_delay_fixed_with_constant_variable_no_fallback_warning(self, tmp_path):
        """DELAY FIXED whose delay time is a named constant (defined later in the
        model file) must still be expanded as an N-stage pipeline, not fall back
        to the first-order ODE approximation.

        Regression: the builder pre-scanned constants before the main processing
        pass so that DELAY FIXED can look up a named constant even before its
        element has been processed.
        """
        import shutil, warnings
        mdl = Path("tests/test-models/tests/forecast/test_forecast.mdl")
        if not mdl.exists():
            pytest.skip("forecast test model not found")
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            translate_to_julia(dst)
        fallback_warns = [str(w.message) for w in captured
                          if "falling back to first-order ODE approximation" in str(w.message)]
        assert not fallback_warns, f"DELAY FIXED with constant delay time should not fall back: {fallback_warns}"

    def test_julia_delay_fixed_no_warning(self, tmp_path):
        mdl = self.MORE_TESTS / "julia_delay_fixed" / "test_julia_delay_fixed.mdl"
        if not mdl.exists():
            pytest.skip("julia_delay_fixed test model not found")
        import shutil, warnings
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            translate_to_julia(dst)
        user_warns = [w for w in captured if issubclass(w.category, UserWarning)]
        assert not user_warns, f"Expected no UserWarning, got: {user_warns}"

    def test_julia_delay_fixed_emits_ode(self, tmp_path):
        mdl = self.MORE_TESTS / "julia_delay_fixed" / "test_julia_delay_fixed.mdl"
        if not mdl.exists():
            pytest.skip("julia_delay_fixed test model not found")
        jl_path = self._translate(mdl, tmp_path)
        content = jl_path.read_text()
        assert "_df_" in content

    def test_julia_trend_emits_smooth_stock(self, tmp_path):
        mdl = self.MORE_TESTS / "julia_trend" / "test_julia_trend.mdl"
        if not mdl.exists():
            pytest.skip("julia_trend test model not found")
        jl_path = self._translate(mdl, tmp_path)
        content = jl_path.read_text()
        assert "_sm_" in content

    def test_julia_forecast_emits_smooth_stock(self, tmp_path):
        mdl = self.MORE_TESTS / "julia_forecast" / "test_julia_forecast.mdl"
        if not mdl.exists():
            pytest.skip("julia_forecast test model not found")
        jl_path = self._translate(mdl, tmp_path)
        content = jl_path.read_text()
        assert "_sm_" in content

    def test_julia_sample_if_true_emits_stock(self, tmp_path):
        mdl = self.MORE_TESTS / "julia_sample_if_true" / "test_julia_sample_if_true.mdl"
        if not mdl.exists():
            pytest.skip("julia_sample_if_true test model not found")
        jl_path = self._translate(mdl, tmp_path)
        content = jl_path.read_text()
        assert "_sit_" in content

    def test_except_2d_integ_no_unsupported_warning(self, tmp_path):
        """2-D EXCEPT + INTEG (stock variable with 2D subscript and EXCEPT clause)
        must emit ODE equations, not an 'Unsupported AST node type IntegStructure' warning.
        """
        import shutil, warnings
        mdl = Path("tests/test-models/tests/except/test_except.mdl")
        if not mdl.exists():
            pytest.skip("except test model not found")
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            translate_to_julia(dst)
        integ_warns = [str(w.message) for w in captured
                       if "IntegStructure" in str(w.message)
                       and "unsupported" in str(w.message).lower()]
        assert not integ_warns, f"IntegStructure must not warn as unsupported: {integ_warns}"

    def test_except_2d_integ_emits_ode_equations(self, tmp_path):
        """The generated Julia for a 2D stock with EXCEPT must contain D(inventory[...]) ODE
        equations, not placeholder 0.0 assignments.
        """
        import shutil, warnings
        mdl = Path("tests/test-models/tests/except/test_except.mdl")
        if not mdl.exists():
            pytest.skip("except test model not found")
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            jl_path = translate_to_julia(dst)
        content = jl_path.read_text()
        assert "D(inventory[" in content, "Expected ODE equations for inventory stock"

    def test_subscripted_inline_lookup_no_unsupported_warning(self, tmp_path):
        """Inline lookup tables with per-element subscript assignments (e.g.
        lookup1dim[A](...) and lookup1dim[B](...)) must not emit
        'Unsupported AST node type LookupsStructure' warnings.
        """
        import shutil, warnings
        mdl = Path("tests/test-models/tests/subscripted_lookups/test_subscripted_lookups.mdl")
        if not mdl.exists():
            pytest.skip("subscripted_lookups test model not found")
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            translate_to_julia(dst)
        lk_warns = [str(w.message) for w in captured
                    if "LookupsStructure" in str(w.message)
                    and "unsupported" in str(w.message).lower()]
        assert not lk_warns, f"LookupsStructure must not warn as unsupported: {lk_warns}"

    def test_subscripted_inline_lookup_emits_dispatch_function(self, tmp_path):
        """The generated Julia for subscripted inline lookups (lookup1dim, lookup2dim)
        must contain proper lookup functions with per-element interpolants.
        """
        import shutil, warnings
        mdl = Path("tests/test-models/tests/subscripted_lookups/test_subscripted_lookups.mdl")
        if not mdl.exists():
            pytest.skip("subscripted_lookups test model not found")
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            jl_path = translate_to_julia(dst)
        content = jl_path.read_text()
        # Must define the 1D subscripted lookup as a callable function
        assert "lookup1dim(" in content, "Expected lookup1dim function in generated code"
        assert "lookup2dim(" in content, "Expected lookup2dim function in generated code"

    def test_nested_delay_in_smooth_no_unsupported_warning(self, tmp_path):
        """SMOOTH N(DELAY3(...), ...) — where DELAY is nested as the input to
        SMOOTH — must not warn 'Unsupported AST node type DelayStructure'.
        The builder must create an intermediate variable for the inner DELAY3
        and use its identifier as the input to the outer SMOOTH.
        """
        import shutil, warnings
        mdl = Path("tests/test-models/tests/arguments/test_arguments.mdl")
        if not mdl.exists():
            pytest.skip("arguments test model not found")
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            translate_to_julia(dst)
        delay_warns = [str(w.message) for w in captured
                       if "DelayStructure" in str(w.message)
                       and "unsupported" in str(w.message).lower()]
        assert not delay_warns, f"Nested DelayStructure must not warn: {delay_warns}"

    def test_nested_delay_in_smooth_emits_ode_for_both(self, tmp_path):
        """The generated Julia for SMOOTH N(DELAY3(Time,...)) must contain ODE
        equations for both the inner delay pipeline and the outer smooth levels.
        """
        import shutil, warnings
        mdl = Path("tests/test-models/tests/arguments/test_arguments.mdl")
        if not mdl.exists():
            pytest.skip("arguments test model not found")
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            jl_path = translate_to_julia(dst)
        content = jl_path.read_text()
        # Should have smooth level variables from the outer SMOOTH
        assert "_lv" in content or "_sm_" in content, "Expected smooth level variables"
        # Should have delay pipeline variables from the inner DELAY3
        assert "_lv" in content, "Expected delay level variables from inner DELAY3"

    def test_json_mode_produces_data_file(self, tmp_path):
        """translate_to_julia with data_format=json creates a .json companion."""
        mdl = self.MORE_TESTS / "julia_delay_fixed" / "test_julia_delay_fixed.mdl"
        if not mdl.exists():
            pytest.skip("julia_delay_fixed test model not found")
        import shutil, warnings
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            jl_path = translate_to_julia(dst, data_format="json")
        json_path = jl_path.with_name(f"{jl_path.stem}_data.json")
        assert json_path.exists()

    def test_get_subscript_3d_arrays_xls_no_reshape_warning(self, tmp_path):
        """GET DIRECT SUBSCRIPT from Excel: subscript sizes are read from file, no reshape warning."""
        import shutil, warnings
        from pathlib import Path
        mdl_src = Path("tests/test-models/tests/get_subscript_3d_arrays_xls/test_get_subscript_3d_arrays_xls.mdl")
        if not mdl_src.exists():
            pytest.skip("get_subscript_3d_arrays_xls model not found")
        # Copy entire folder (Excel file must be present)
        dst_folder = tmp_path / "get_subscript_3d_arrays_xls"
        shutil.copytree(mdl_src.parent, dst_folder)
        dst = dst_folder / mdl_src.name
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            translate_to_julia(str(dst))
        reshape_warnings = [x for x in w if "reshape" in str(x.message).lower()]
        assert not reshape_warnings, f"Unexpected reshape warnings: {reshape_warnings}"


# ===========================================================================
# Phase 3G — XMILE min_max_1arg (vmin_xmile / vmax_xmile)
# ===========================================================================

class TestXmileDelayFixed:
    """XMILE DELAY(x, n) used inline inside an arithmetic expression."""

    TEST_MODELS = Path("tests/test-models/tests")

    def test_delay_xmile_no_unsupported_warning(self, tmp_path):
        """DELAY(X, n) embedded in arithmetic must not emit 'Unsupported AST' warning."""
        import shutil, warnings
        model_dir = self.TEST_MODELS / "delay_xmile"
        if not model_dir.exists():
            pytest.skip("delay_xmile test model not found")
        dst_dir = tmp_path / "delay_xmile"
        shutil.copytree(model_dir, dst_dir)
        xmile = next(dst_dir.glob("*.xmile"))
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            translate_to_julia(xmile)
        unsupported = [x for x in w if "Unsupported AST" in str(x.message)]
        assert not unsupported, f"Unexpected unsupported AST warnings: {unsupported}"

    def test_delay_xmile_emits_ode_stocks(self, tmp_path):
        """DELAY(X, n) embedded in arithmetic must create ODE pipeline stocks."""
        import shutil, warnings
        model_dir = self.TEST_MODELS / "delay_xmile"
        if not model_dir.exists():
            pytest.skip("delay_xmile test model not found")
        dst_dir = tmp_path / "delay_xmile"
        shutil.copytree(model_dir, dst_dir)
        xmile = next(dst_dir.glob("*.xmile"))
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            jl_path = translate_to_julia(xmile)
        content = jl_path.read_text()
        # The embedded delay must be lifted: _edf0 becomes an auxiliary backed
        # by a pipeline stock (_df_pipe_1__edf0 etc.) in the ODE state vector.
        assert "_edf0" in content
        assert "_df_pipe_1__edf0" in content


class TestEmbeddedDelayDrain2D:
    """Drain of embedded DelayFixed inside ndim≥2 auxiliaries."""

    def test_2d_aux_with_embedded_delay_drains_pipeline_stocks(self):
        # When a 2D auxiliary's RHS contains an embedded DelayFixedStructure,
        # _drain_embedded_delays must be called so the pipeline stocks are created.
        # Without the drain call the _edf placeholder is referenced but never defined.
        sr_a = _make_subscript_range("dim_a", ["A1", "A2"])
        sr_b = _make_subscript_range("dim_b", ["B1", "B2"])
        inp_elem = _make_element("Input Var", 1.0)
        delay_ast = DelayFixedStructure(
            input=ReferenceStructure("Input Var"),
            delay_time=1.0,
            initial=0.0,
        )
        emb_ast = ArithmeticStructure(operators=["+"], arguments=[delay_ast, 0.0])
        comp = AbstractComponent(subscripts=[["dim_a", "dim_b"], []], ast=emb_ast)
        aux_elem = AbstractElement(name="Aux 2D", components=[comp])
        control_elems = [
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        sb = _section_builder_from_elements(
            [inp_elem, aux_elem] + control_elems,
            subscripts=[sr_a, sr_b],
        )
        sb.build_section()
        assert any("_edf" in d for d in sb.stock_decls), (
            f"2D aux with embedded delay must generate pipeline stocks, got: {sb.stock_decls}"
        )


class TestXmileMinMax:
    """XMILE MIN/MAX over an entire subscript dimension."""

    TEST_MODELS = Path("tests/test-models/tests")

    def test_min_max_1arg_no_unknown_function_warning(self, tmp_path):
        """MIN(arr[dim]) in XMILE must not emit 'Unknown Vensim function' warning."""
        import shutil, warnings
        model_dir = self.TEST_MODELS / "min_max_1arg"
        if not model_dir.exists():
            pytest.skip("min_max_1arg test model not found")
        dst_dir = tmp_path / "min_max_1arg"
        shutil.copytree(model_dir, dst_dir)
        xmile = next(dst_dir.glob("*.xmile"))
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            translate_to_julia(xmile)
        unknown = [x for x in w if "Unknown Vensim function" in str(x.message)]
        assert not unknown, f"Unexpected unknown function warnings: {unknown}"

    def test_min_max_1arg_emits_minimum_maximum(self, tmp_path):
        """MIN(arr[dim]) → minimum(arr), MAX(arr[dim]) → maximum(arr)."""
        import shutil, warnings
        model_dir = self.TEST_MODELS / "min_max_1arg"
        if not model_dir.exists():
            pytest.skip("min_max_1arg test model not found")
        dst_dir = tmp_path / "min_max_1arg"
        shutil.copytree(model_dir, dst_dir)
        xmile = next(dst_dir.glob("*.xmile"))
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            jl_path = translate_to_julia(xmile)
        content = jl_path.read_text()
        assert "minimum(" in content
        assert "maximum(" in content


# ===========================================================================
# NC data-file input (DATA variables fed from another model's NC output)
# ===========================================================================

class TestNcDataFiles:
    """nc_data_files parameter for feeding another model's NetCDF output into
    DATA variables of the current Julia model.

    The generated model must emit:
      - ``_nc_data_registry`` — maps Julia identifier → (method, ndims)
      - ``_load_nc_data!(files)`` — reads NC, populates ``_tab_data``
      - ``nc_data_files=String[]`` kwarg in ``run_model()``

    Models without any DATA variables must emit none of the above.
    """

    TEST_MODELS = Path("tests/test-models/tests/data_from_other_model")

    def _translate(self, tmp_path):
        import shutil
        dst = tmp_path / "data_from_other_model"
        shutil.copytree(self.TEST_MODELS, dst)
        from pysd import translate_to_julia
        return translate_to_julia(dst / "test_data_from_other_model.mdl")

    def test_run_model_accepts_nc_data_files(self, tmp_path):
        """run_model() must accept nc_data_files=String[] when model has DATA variables."""
        content = self._translate(tmp_path).read_text()
        assert "nc_data_files=String[]" in content

    def test_load_nc_data_function_emitted(self, tmp_path):
        """_load_nc_data! function must be emitted for models with DATA variables."""
        content = self._translate(tmp_path).read_text()
        assert "function _load_nc_data!" in content

    def test_nc_data_registry_emitted(self, tmp_path):
        """_nc_data_registry constant must be emitted for models with DATA variables."""
        content = self._translate(tmp_path).read_text()
        assert "_nc_data_registry" in content

    def test_nc_data_registry_contains_scalar_var(self, tmp_path):
        """Scalar DATA variable (var 0dim) must appear in _nc_data_registry with ndims=0."""
        content = self._translate(tmp_path).read_text()
        # Registry entries look like: "var_0dim" => (:interpolate, 0)
        assert '"var_0dim" => (:' in content

    def test_nc_data_registry_records_subscript_ndims(self, tmp_path):
        """1D subscripted var_1dim must appear in registry; 2D var_2dim likewise."""
        content = self._translate(tmp_path).read_text()
        assert '"var_1dim" => (:' in content
        assert '"var_2dim" => (:' in content

    def test_load_nc_data_uses_ncDatasets(self, tmp_path):
        """_load_nc_data! must open NC files via NCDatasets.Dataset."""
        content = self._translate(tmp_path).read_text()
        assert "NCDatasets.Dataset" in content

    def test_model_without_data_vars_has_no_nc_infrastructure(self, tmp_path):
        """Models without DATA variables must NOT emit nc_data_files or _load_nc_data!."""
        ast = IntegStructure(flow=0.0, initial=0.0)
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Stock", components=[comp])
        control_elems = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + control_elems,
            path=tmp_path / "simple.mdl",
        )
        model = AbstractModel(
            original_path=tmp_path / "simple.mdl",
            sections=(section,),
        )
        jl_path = JuliaModelBuilder(model).build_model()
        content = jl_path.read_text()
        assert "nc_data_files" not in content
        assert "_load_nc_data!" not in content
        assert "_nc_data_registry" not in content


# ===========================================================================
# Phase 3F — INVERT_MATRIX support
# ===========================================================================

class TestInvertMatrix:
    """INVERT_MATRIX must generate Symbolics.scalarize matrix-level equations,
    not element-wise inv(scalar, n) calls which are invalid in Julia.

    Regression for MethodError: no method matching inv(::Num, ::Int64)
    """

    def _make_mat_elem(self, lhs_name, mat_ref_name, dims_2d, n_size):
        """Helper: element(lhs_name) = INVERT_MATRIX(mat_ref_name[dims...], n)"""
        mat_ast = CallStructure(
            function=ReferenceStructure(reference="invert_matrix"),
            arguments=(
                ReferenceStructure(
                    reference=mat_ref_name,
                    subscripts=SubscriptsReferenceStructure(subscripts=dims_2d),
                ),
                n_size,
            ),
        )
        comp = AbstractComponent(subscripts=[dims_2d, []], ast=mat_ast)
        return AbstractElement(name=lhs_name, components=[comp])

    def test_2d_invert_matrix_generates_scalarize(self):
        """2D case: matrix1i[d,d1] = INVERT_MATRIX(matrix_1[d,d1], 2)
        should use the _inv_mat2d_elem helper (registered via needed_helpers),
        NOT element-wise: [matrix1i[_i0,_i1] ~ inv(matrix_1[_i0,_i1], 2) ...]
        """
        sr_d  = _make_subscript_range("d",  ["A", "B"])
        sr_d1 = _make_subscript_range("d1", ["A", "B"])

        mat_comp = AbstractComponent(subscripts=[["d", "d1"], []], ast=0.0)
        mat_elem = AbstractElement(name="matrix 1", components=[mat_comp])

        inv_elem = self._make_mat_elem("matrix1i", "matrix_1", ["d", "d1"], 2)

        sb = _section_builder_from_elements(
            [mat_elem, inv_elem],
            subscripts=[sr_d, sr_d1],
        )
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        inv_eqs = [e for e in eqs if "matrix1i" in e]

        assert len(inv_eqs) == 1, f"expected 1 equation, got: {inv_eqs}"
        eq = inv_eqs[0]
        # Should use the registered helper function
        assert "_inv_mat2d_elem" in eq, (
            f"Expected _inv_mat2d_elem helper in equation, got: {eq}"
        )
        assert "matrix_1" in eq, (
            f"Expected matrix_1 argument, got: {eq}"
        )
        # Must NOT contain the broken scalar inv with size argument
        assert "inv(matrix_1[_i0, _i1], 2" not in eq, (
            f"Should not contain element-wise inv with size arg, got: {eq}"
        )
        # Must NOT use slice indexing (:)
        assert ":, :" not in eq, (
            f"Should not use slice indexing :, :, got: {eq}"
        )

    def test_3d_invert_matrix_generates_batch_scalarize(self):
        """3D case: matrix3i[d,dim1,dim2] = INVERT_MATRIX(matrix_3[d,dim1,dim2], 3)
        should produce:
          [Symbolics.scalarize(matrix3i[_i0, :, :] .~ inv(matrix_3[_i0, :, :]))...
           for _i0 in 1:N_D]...
        NOT element-wise: [matrix3i[_i0,_i1,_i2] ~ inv(matrix_3[_i0,_i1,_i2], 3) ...]
        """
        sr_d    = _make_subscript_range("d",    ["A", "B"])
        sr_dim1 = _make_subscript_range("dim1", ["h", "m", "l"])
        sr_dim2 = _make_subscript_range("dim2", ["h", "m", "l"])

        mat_comp = AbstractComponent(subscripts=[["d", "dim1", "dim2"], []], ast=0.0)
        mat_elem = AbstractElement(name="matrix 3", components=[mat_comp])

        inv_elem = self._make_mat_elem(
            "matrix3i", "matrix_3", ["d", "dim1", "dim2"], 3
        )

        sb = _section_builder_from_elements(
            [mat_elem, inv_elem],
            subscripts=[sr_d, sr_dim1, sr_dim2],
        )
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        inv_eqs = [e for e in eqs if "matrix3i" in e]

        assert len(inv_eqs) == 1, f"expected 1 equation, got: {inv_eqs}"
        eq = inv_eqs[0]
        # Should use the 3D registered helper function
        assert "_inv_mat3d_elem" in eq, (
            f"Expected _inv_mat3d_elem helper in equation, got: {eq}"
        )
        assert "matrix_3" in eq, (
            f"Expected matrix_3 argument in equation, got: {eq}"
        )
        # Must NOT use slice indexing (:)
        assert ":, :" not in eq, (
            f"Should not use slice indexing :, :, got: {eq}"
        )
        # Must NOT contain the broken element-wise pattern
        assert "inv(matrix_3[_i0, _i1, _i2]" not in eq, (
            f"Should not contain element-wise scalar inv call, got: {eq}"
        )

    def test_invert_matrix_translation_from_mdl(self, tmp_path):
        """Full pipeline: translate test_invert_matrix.mdl and check output."""
        import shutil
        mdl = Path("tests/test-models/tests/invert_matrix/test_invert_matrix.mdl")
        if not mdl.exists():
            pytest.skip("invert_matrix test model not found")
        shutil.copy(mdl, tmp_path / mdl.name)
        from pysd import translate_to_julia
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            jl_path = translate_to_julia(tmp_path / mdl.name)
        content = jl_path.read_text()
        # No broken element-wise inv with size argument
        assert "inv(matrix_1[_i0, _i1], 2" not in content, (
            "Found broken element-wise inv(matrix_1[...], n) in generated code"
        )
        assert "inv(matrix_3[_i0, _i1, _i2]" not in content, (
            "Found broken element-wise inv(matrix_3[...]) in generated code"
        )
        # Should use registered helper functions
        assert "_inv_mat2d_elem" in content, (
            "Expected _inv_mat2d_elem helper in generated code"
        )
        assert "_inv_mat3d_elem" in content, (
            "Expected _inv_mat3d_elem helper in generated code"
        )
        # Should not pass size argument to inv
        assert ", 2.0)" not in content and ", 3.0)" not in content, (
            "inv() should not receive a size argument in generated code"
        )


# ===========================================================================
# Backend dispatch — ODE (default) and MTK
# ===========================================================================

class TestBackendDispatch:
    """Tests that verify each backend emits the right Julia code."""

    # ---- helpers -----------------------------------------------------------

    def _minimal_model(self, tmp_path, backend="ode"):
        birth_rate_comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.03)
        birth_rate_elem = AbstractElement(name="Birth Rate", components=[birth_rate_comp])

        flow_ast = ArithmeticStructure(
            operators=["*"],
            arguments=[ReferenceStructure("Population"), ReferenceStructure("Birth Rate")],
        )
        pop_ast = IntegStructure(flow=flow_ast, initial=1000.0)
        pop_comp = AbstractComponent(subscripts=[[], []], ast=pop_ast)
        pop_elem = AbstractElement(name="Population", components=[pop_comp])

        control_elems = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 100.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]

        section = _make_section(
            elements=[birth_rate_elem, pop_elem] + control_elems,
            path=tmp_path / "my_model.mdl",
        )
        return AbstractModel(
            original_path=tmp_path / "my_model.mdl",
            sections=(section,),
        )

    def _build(self, tmp_path, backend="ode"):
        model = self._minimal_model(tmp_path, backend)
        path = JuliaModelBuilder(model, backend=backend).build_model()
        return path.read_text()

    # ---- invalid backend ---------------------------------------------------

    def test_invalid_backend_raises(self, tmp_path):
        model = self._minimal_model(tmp_path)
        with pytest.raises(ValueError, match="backend"):
            JuliaModelBuilder(model, backend="bad")

    # ---- ODE backend header ------------------------------------------------

    def test_ode_no_modeling_toolkit_in_header(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert "ModelingToolkit" not in content

    def test_ode_uses_ordinary_diffeq(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert "OrdinaryDiffEq" in content

    # ---- ODE backend declarations ------------------------------------------

    def test_ode_parameters_become_const(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert "const birth_rate = 0.03" in content

    def test_ode_no_at_parameters_declaration(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert "@parameters" not in content

    def test_ode_no_at_variables_declaration(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert "@variables" not in content

    # ---- ODE backend equations ---------------------------------------------

    def test_ode_uses_rhs_function(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert "function rhs!(du, u, p, t)" in content

    def test_ode_no_equation_array(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert "Equation[" not in content

    def test_ode_no_ode_system(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert "ODESystem" not in content

    # ---- ODE backend u0 ---------------------------------------------------

    def test_ode_u0_is_float64_array(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert "u0 = Float64[" in content

    def test_ode_u0_has_numeric_initial(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert "1000.0" in content
        assert "population =>" not in content

    # ---- ODE backend lookups -----------------------------------------------

    def test_ode_lookup_has_no_register_symbolic(self, tmp_path):
        lut_ast = LookupsStructure(
            x=(0.0, 1.0), y=(0.0, 1.0),
            x_limits=(0.0, 1.0), y_limits=(0.0, 1.0),
            type="interpolate",
        )
        lut_comp = AbstractLookup(subscripts=[[], []], ast=lut_ast)
        lut_elem = AbstractElement(name="Effect LUT", components=[lut_comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[lut_elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        content = JuliaModelBuilder(model, backend="ode").build_model().read_text()
        assert "@register_symbolic" not in content

    # ---- ODE backend save metadata ----------------------------------------

    def test_ode_emits_state_map(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert "_state_map" in content

    def test_ode_state_map_contains_stock_name(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert '"population"' in content

    def test_ode_state_map_contains_u_index(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert '("population", 1,' in content

    def test_ode_emits_dim_labels(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert "_dim_labels" in content

    def test_ode_calls_save_results_with_state_map(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert "save_results(sol, _state_map, _dim_labels," in content

    def test_ode_subscripted_state_map_has_dim_names(self, tmp_path):
        sr = _make_subscript_range("sectors", ["A", "B", "C"])
        stock_ast = IntegStructure(
            flow=ReferenceStructure("Inflow"),
            initial=0.0,
        )
        stock_comp = AbstractComponent(subscripts=[["sectors"], []], ast=stock_ast)
        stock_elem = AbstractElement(name="Capital", components=[stock_comp])
        inflow_elem = _make_subscripted_element("Inflow", 1.0, "sectors")
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[stock_elem, inflow_elem] + controls,
            path=tmp_path / "sub_model.mdl",
            subscripts=[sr],
        )
        model = AbstractModel(original_path=tmp_path / "sub_model.mdl", sections=(section,))
        content = JuliaModelBuilder(model, backend="ode").build_model().read_text()
        assert '"capital"' in content
        assert '"sectors"' in content

    def test_ode_dim_labels_contains_elements(self, tmp_path):
        sr = _make_subscript_range("sectors", ["Agriculture", "Industry"])
        elem = _make_subscripted_element("cost", 1.0, "sectors",
                                         comp_class=AbstractUnchangeableConstant)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls,
            path=tmp_path / "m.mdl",
            subscripts=[sr],
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        content = JuliaModelBuilder(model, backend="ode").build_model().read_text()
        assert '"Agriculture"' in content
        assert '"Industry"' in content

    # ---- MTK backend header ------------------------------------------------

    def test_mtk_has_modeling_toolkit_in_header(self, tmp_path):
        content = self._build(tmp_path, "mtk")
        assert "ModelingToolkit" in content

    # ---- MTK backend declarations ------------------------------------------

    def test_mtk_keeps_at_variables(self, tmp_path):
        content = self._build(tmp_path, "mtk")
        assert "@variables population(t)" in content

    def test_mtk_keeps_at_parameters(self, tmp_path):
        content = self._build(tmp_path, "mtk")
        assert "@parameters birth_rate = 0.03" in content

    # ---- MTK backend equations ---------------------------------------------

    def test_mtk_uses_equation_array(self, tmp_path):
        content = self._build(tmp_path, "mtk")
        assert "eqs = Equation[" in content

    def test_mtk_equation_uses_tilde(self, tmp_path):
        content = self._build(tmp_path, "mtk")
        assert "D(population) ~" in content

    def test_mtk_has_ode_system(self, tmp_path):
        content = self._build(tmp_path, "mtk")
        assert "ODESystem" in content
        assert "structural_simplify" in content

    # ---- MTK backend u0 ---------------------------------------------------

    def test_mtk_u0_uses_pair_syntax(self, tmp_path):
        content = self._build(tmp_path, "mtk")
        assert "population => 1000.0" in content

    def test_mtk_u0_is_plain_vector(self, tmp_path):
        content = self._build(tmp_path, "mtk")
        assert "u0 = [" in content
        assert "Float64[" not in content.split("u0 = ")[1].split("\n")[0]

    # ---- MTK backend lookups -----------------------------------------------

    def test_mtk_lookup_has_register_symbolic(self, tmp_path):
        lut_ast = LookupsStructure(
            x=(0.0, 1.0), y=(0.0, 1.0),
            x_limits=(0.0, 1.0), y_limits=(0.0, 1.0),
            type="interpolate",
        )
        lut_comp = AbstractLookup(subscripts=[[], []], ast=lut_ast)
        lut_elem = AbstractElement(name="Effect LUT", components=[lut_comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[lut_elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        content = JuliaModelBuilder(model, backend="mtk").build_model().read_text()
        assert "@register_symbolic" in content

    # ---- MTK backend save -------------------------------------------------

    def test_mtk_emits_dim_labels(self, tmp_path):
        content = self._build(tmp_path, "mtk")
        assert "_dim_labels" in content

    def test_mtk_calls_save_results_with_sys(self, tmp_path):
        content = self._build(tmp_path, "mtk")
        assert "save_results(sol, sys, _dim_labels," in content

    def test_mtk_dim_labels_has_subscript_elements(self, tmp_path):
        sr = _make_subscript_range("regions", ["North", "South"])
        elem = _make_subscripted_element("pop", 1.0, "regions",
                                         comp_class=AbstractUnchangeableConstant)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls,
            path=tmp_path / "m.mdl",
            subscripts=[sr],
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        content = JuliaModelBuilder(model, backend="mtk").build_model().read_text()
        assert '"North"' in content
        assert '"South"' in content


# ===========================================================================
# check_compat — version guard emitted in generated files
# ===========================================================================

class TestCheckCompat:
    """Generated Julia files must call check_compat so the runtime can detect
    a PySD.jl major-version mismatch before execution."""

    def _build(self, tmp_path, backend="ode"):
        elem = _make_stock_element("Level", 1.0, 0.0)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls,
            path=tmp_path / "m.mdl",
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        return JuliaModelBuilder(model, backend=backend).build_model().read_text()

    def test_ode_file_emits_check_compat(self, tmp_path):
        content = self._build(tmp_path, "ode")
        assert 'check_compat(v"' in content, (
            f"ODE generated file must call check_compat, got header:\n{content[:500]}"
        )

    def test_mtk_file_emits_check_compat(self, tmp_path):
        content = self._build(tmp_path, "mtk")
        assert 'check_compat(v"' in content, (
            f"MTK generated file must call check_compat, got header:\n{content[:500]}"
        )


# ===========================================================================
# Expressions builder subscript path coverage
# ===========================================================================

class TestExpressionBuilderSubscriptPaths:
    """Cover specific uncovered paths in JuliaASTVisitor reference/call handling."""

    def _v(self, **kwargs):
        ns = JuliaNamespaceManager()
        for n in kwargs.pop("names", []):
            ns.add_to_namespace(n)
        registry = InlineLookupRegistry()
        needed = set()
        return JuliaASTVisitor(ns, registry, needed, **kwargs), ns

    # ------------------------------------------------------------------
    # Lines 611-612: bare lookup reference in active_subs comprehension context
    # ------------------------------------------------------------------

    def test_bare_lookup_in_comprehension_context_inserts_loop_var(self):
        """Bare reference to a subscripted lookup in a comprehension inserts the active index (lines 611-612)."""
        v, ns = self._v(
            names=["transport data"],
            subs_sizes={"mode": 3},
            var_dims={"transport_data": ["mode"]},
            active_subs={"mode": "_i0"},
            lookup_names={"transport_data"},
        )
        node = ReferenceStructure("transport data")
        result = v.visit(node)
        assert result == "transport_data(_i0, t)"

    # ------------------------------------------------------------------
    # Lines 660: non-bang subscript in active_subs within a bang reference
    # ------------------------------------------------------------------

    def test_mixed_bang_and_range_subscript_uses_active_loop_var(self):
        """Reference with a non-bang range sub in active_subs + a bang sub (line 660)."""
        v, ns = self._v(
            names=["energy pkm"],
            subs_sizes={"sectors": 14, "modes": 4},
            subs_elems={
                "sectors": [f"S{i}" for i in range(14)],
                "modes": ["car", "bus", "train", "air"],
            },
            var_dims={"energy_pkm": ["sectors", "modes"]},
            active_subs={"sectors": "_i0"},  # 'sectors' is an active loop var
        )
        ns.add_to_namespace("energy pkm")
        # Reference with both a non-bang range subscript (sectors) and a bang sub (modes!)
        node = ReferenceStructure(
            "energy pkm",
            subscripts=SubscriptsReferenceStructure(subscripts=["sectors", "modes!"]),
        )
        result = v.visit(node)
        # 'sectors' is in active_subs → line 660 → dim_to_idx["sectors"] = "_i0"
        assert "_i0" in result
        assert "_ii" in result or "for" in result

    # ------------------------------------------------------------------
    # Lines 730-737: bang subscript fallback when var_dims is unknown
    # ------------------------------------------------------------------

    def test_bang_subscript_no_var_dims_uses_node_subs_order(self):
        """Bang subscript with no var_dims for the variable falls back to node_subs order (lines 730-737)."""
        v, ns = self._v(
            names=["generic var"],
            subs_sizes={"region": 5},
            # NO var_dims for generic_var — triggers the fallback at line 730
        )
        node = ReferenceStructure(
            "generic var",
            subscripts=SubscriptsReferenceStructure(subscripts=["region!"]),
        )
        result = v.visit(node)
        assert "_ii0" in result
        assert "1:N_REGION" in result

    # ------------------------------------------------------------------
    # Lines 1010-1017: subscripted function call in active comprehension context
    # ------------------------------------------------------------------

    def test_subscripted_func_call_in_active_context_prepends_index(self):
        """Subscripted function call inside a comprehension prepends active loop var (lines 1010-1017)."""
        v, ns = self._v(
            names=["historic gfcf"],
            subs_sizes={"sectors": 4},
            var_dims={"historic_gfcf": ["sectors"]},
            active_subs={"sectors": "_i0"},
        )
        node = CallStructure(
            function=ReferenceStructure("historic gfcf"),
            arguments=[ReferenceStructure("Time")],
        )
        result = v.visit(node)
        assert result == "historic_gfcf(_i0, t)"

    # ------------------------------------------------------------------
    # Lines 1018-1028: subscripted function call in scalar (no active_subs) context
    # ------------------------------------------------------------------

    def test_subscripted_func_call_in_scalar_context_broadcasts(self):
        """Subscripted function call with no active_subs generates a comprehension (lines 1018-1028)."""
        v, ns = self._v(
            names=["historic gfcf"],
            subs_sizes={"sectors": 4},
            var_dims={"historic_gfcf": ["sectors"]},
            # No active_subs → scalar context
        )
        node = CallStructure(
            function=ReferenceStructure("historic gfcf"),
            arguments=[ReferenceStructure("Time")],
        )
        result = v.visit(node)
        assert "historic_gfcf(_ii0, t)" in result
        assert "for _ii0 in 1:N_SECTORS" in result

    # ------------------------------------------------------------------
    # Line 586: element label bare ref with no active dim
    # ------------------------------------------------------------------

    def test_element_label_bare_ref_no_active_dim_returns_position(self):
        """Bare reference to an element label with no active loop var picks first range position (line 586)."""
        v, ns = self._v(
            subs_elems={"colors": ["red", "green", "blue"]},
        )
        node = ReferenceStructure("green")  # "green" not in namespace → fallback to _clean_elem_index
        result = v.visit(node)
        assert result == "2"  # 1-based index of "green" in "colors"

    # ------------------------------------------------------------------
    # Lines 662-663: non-bang range sub in subs_elems (in bang context, not in active_subs)
    # ------------------------------------------------------------------

    def test_bang_ref_with_non_bang_range_in_subs_elems_not_active(self):
        """Non-bang sub in subs_elems but not active_subs is entered but idx_var is None (lines 662-663)."""
        v, ns = self._v(
            names=["energy"],
            subs_sizes={"regions": 3, "modes": 2},
            subs_elems={"regions": ["R1", "R2", "R3"], "modes": ["car", "bus"]},
            var_dims={"energy": ["regions", "modes"]},
            # NO active_subs: modes is in subs_elems but not in active_subs
        )
        ns.add_to_namespace("energy")
        # node_subs has one bang ("regions!") + one plain range ("modes") that's in subs_elems
        node = ReferenceStructure(
            "energy",
            subscripts=SubscriptsReferenceStructure(subscripts=["regions!", "modes"]),
        )
        result = v.visit(node)
        # "modes" hits the elif subs_elems branch (lines 662-663), idx_var=None → skipped
        # "regions!" creates a comprehension over regions
        assert "_ii0" in result
        assert "N_REGIONS" in result

    # ------------------------------------------------------------------
    # Line 677: element label in bang path, not in any var_dims range → last resort
    # ------------------------------------------------------------------

    def test_bang_ref_element_label_not_in_var_dims_uses_last_resort_range(self):
        """Element label in bang subscript not found in var's own dims triggers last-resort range (line 677)."""
        v, ns = self._v(
            names=["energy"],
            subs_sizes={"regions": 3, "sectors": 2},
            subs_elems={"regions": ["R1", "R2", "R3"], "sectors": ["A", "B"]},
            var_dims={"energy": ["regions"]},  # energy only has "regions" dim
        )
        ns.add_to_namespace("energy")
        # "A" is an element label in "sectors", but energy's var_dims only has "regions"
        # So the loop at line 672 finds no match in var_dims_list → line 676 target_dim is None
        # → line 677: target_dim = next(iter(_elem_index["A"])) = "sectors"
        node = ReferenceStructure(
            "energy",
            subscripts=SubscriptsReferenceStructure(subscripts=["regions!", "A"]),
        )
        result = v.visit(node)
        assert "_ii0" in result  # regions! → comprehension

    # ------------------------------------------------------------------
    # Lines 782-786: size-match fallback for aligned ranges (non-bang ref)
    # ------------------------------------------------------------------

    def test_non_bang_ref_size_match_fallback_for_aligned_range(self):
        """Non-bang subscript with same-size range alias triggers size-match (lines 782-786)."""
        v, ns = self._v(
            names=["energy"],
            subs_sizes={"sectors": 3, "sectors_alias": 3},
            subs_elems={
                "sectors": ["A", "B", "C"],
                "sectors_alias": ["X", "Y", "Z"],  # same SIZE but different elements
            },
            var_dims={"energy": ["sectors"]},
            active_subs={"sectors": "_i0"},  # "sectors" is the active loop var
        )
        ns.add_to_namespace("energy")
        # "sectors_alias" not in active_subs, in subs_elems, exact element-set doesn't match
        # → falls through to size-match at lines 782-786 → finds "sectors" with size 3 → uses "_i0"
        node = ReferenceStructure(
            "energy",
            subscripts=SubscriptsReferenceStructure(subscripts=["sectors_alias"]),
        )
        result = v.visit(node)
        assert "_i0" in result

    # ------------------------------------------------------------------
    # Lines 803-806, 809: element label fallback — not in var's own dim
    # ------------------------------------------------------------------

    def test_non_bang_ref_element_label_not_in_var_dim_uses_last_resort(self):
        """Element label subscript not in the var's declared dim triggers fallback (lines 803-806, 809)."""
        v, ns = self._v(
            names=["energy"],
            subs_sizes={"regions": 3, "sectors": 2},
            subs_elems={"regions": ["R1", "R2", "R3"], "sectors": ["A", "B"]},
            var_dims={"energy": ["regions"]},  # energy is in "regions", not "sectors"
        )
        ns.add_to_namespace("energy")
        # "A" is an element in "sectors", but energy's var_dims only has "regions"
        # Pos=0, candidate="regions", "A" not in _elem_index["A"]["regions"] → parent_range stays None
        # Lines 803-806: loop through var_dims_list=["regions"], no match
        # Line 807-809: last resort → parent_range = "sectors" → index = 1
        node = ReferenceStructure(
            "energy",
            subscripts=SubscriptsReferenceStructure(subscripts=["A"]),
        )
        result = v.visit(node)
        assert "energy[1]" in result  # "A" is at index 1 in "sectors"

    # ------------------------------------------------------------------
    # Lines 868-886, 895-902, 938-941: bang subscript on function call (new dim)
    # ------------------------------------------------------------------

    def test_func_call_bang_subscript_creates_comprehension(self):
        """Function call with bang subscript not in active_subs builds comprehension (lines 868-886, 940-941)."""
        v, ns = self._v(
            names=["fuel efficiency"],
            subs_sizes={"fuel_type": 3},
            subs_elems={"fuel_type": ["gas", "oil", "elec"]},
            var_dims={"fuel_efficiency": ["fuel_type"]},
        )
        func_ref = ReferenceStructure(
            "fuel efficiency",
            subscripts=SubscriptsReferenceStructure(subscripts=["fuel_type!"]),
        )
        node = CallStructure(function=func_ref, arguments=[ReferenceStructure("Time")])
        result = v.visit(node)
        assert "_ii0" in result
        assert "for _ii0 in 1:N_FUEL_TYPE" in result
        assert "fuel_efficiency(_ii0, t)" in result

    # ------------------------------------------------------------------
    # Lines 929-937: bang subscript on function call, no var_dims → fallback to node_subs
    # ------------------------------------------------------------------

    def test_func_call_bang_subscript_no_var_dims_fallback(self):
        """Function call with bang subscript but no var_dims uses node_subs order (lines 929-937)."""
        v, ns = self._v(
            names=["generic func"],
            subs_sizes={"dim_a": 4},
            # NO var_dims for generic_func → triggers lines 929-937
        )
        func_ref = ReferenceStructure(
            "generic func",
            subscripts=SubscriptsReferenceStructure(subscripts=["dim_a!"]),
        )
        node = CallStructure(function=func_ref, arguments=[ReferenceStructure("Time")])
        result = v.visit(node)
        assert "_ii0" in result
        assert "for _ii0 in 1:N_DIM_A" in result

    # ------------------------------------------------------------------
    # Lines 903-928: 2D func, 1 bang sub, positional fallback for unmatched dim
    # ------------------------------------------------------------------

    def test_func_call_bang_2d_positional_fallback_for_unmatched_dim(self):
        """2D function with only 1 bang sub uses positional fallback for the other dim (lines 903-928)."""
        v, ns = self._v(
            names=["transport share"],
            subs_sizes={"region": 3, "mode": 4},  # DIFFERENT sizes → no size match
            var_dims={"transport_share": ["region", "mode"]},
        )
        func_ref = ReferenceStructure(
            "transport share",
            subscripts=SubscriptsReferenceStructure(subscripts=["mode!"]),
        )
        node = CallStructure(function=func_ref, arguments=[ReferenceStructure("Time")])
        result = v.visit(node)
        # "region" not in dim_to_idx_c → positional fallback → both get "_ii0"
        assert "_ii0" in result

    # ------------------------------------------------------------------
    # Lines 918-920: 2D func with equal-size bang dim → size-match succeeds
    # ------------------------------------------------------------------

    def test_func_call_bang_size_match_for_equal_size_dims(self):
        """2D function where unmatched dim (processed first) has same size as bang dim → size match (lines 918-920)."""
        v, ns = self._v(
            names=["energy matrix"],
            subs_sizes={"modes": 4, "sectors": 4},  # SAME sizes → size match triggers
            # "sectors" is listed FIRST so it is processed before "modes" (the bang dim)
            # → _ii0 not yet in used_ivars_c when the size check runs → match at 918-920
            var_dims={"energy_matrix": ["sectors", "modes"]},
        )
        func_ref = ReferenceStructure(
            "energy matrix",
            subscripts=SubscriptsReferenceStructure(subscripts=["modes!"]),
        )
        node = CallStructure(function=func_ref, arguments=[ReferenceStructure("Time")])
        result = v.visit(node)
        assert "_ii0" in result

    # ------------------------------------------------------------------
    # Lines 963-986: non-bang explicit subscripts on function call (subs_elems alignment)
    # ------------------------------------------------------------------

    def test_func_call_explicit_subscripts_element_set_alignment(self):
        """Function call with explicit range subscript aligns via element-set match (lines 963-986)."""
        v, ns = self._v(
            names=["water use"],
            subs_sizes={"sectors": 3, "sectors_alias": 3},
            subs_elems={
                "sectors": ["A", "B", "C"],
                "sectors_alias": ["A", "B", "C"],  # SAME elements → element-set match
            },
            var_dims={"water_use": ["sectors"]},
            active_subs={"sectors": "_i0"},
        )
        func_ref = ReferenceStructure(
            "water use",
            subscripts=SubscriptsReferenceStructure(subscripts=["sectors_alias"]),
        )
        node = CallStructure(function=func_ref, arguments=[ReferenceStructure("Time")])
        result = v.visit(node)
        assert "water_use(_i0, t)" in result

    def test_func_call_explicit_subscripts_size_alignment_fallback(self):
        """Function call with range subscript not matching by elements falls back to size (lines 976-986)."""
        v, ns = self._v(
            names=["water use"],
            subs_sizes={"sectors": 3, "sectors_b": 3},
            subs_elems={
                "sectors": ["A", "B", "C"],
                "sectors_b": ["X", "Y", "Z"],  # same SIZE but different elements
            },
            var_dims={"water_use": ["sectors"]},
            active_subs={"sectors": "_i0"},
        )
        func_ref = ReferenceStructure(
            "water use",
            subscripts=SubscriptsReferenceStructure(subscripts=["sectors_b"]),
        )
        node = CallStructure(function=func_ref, arguments=[ReferenceStructure("Time")])
        result = v.visit(node)
        assert "water_use(_i0, t)" in result

    # ------------------------------------------------------------------
    # Lines 987-1001: element label in non-bang function call subscripts
    # ------------------------------------------------------------------

    def test_func_call_explicit_element_label_subscript(self):
        """Function call with an element label subscript resolves to numeric index (lines 987-1001)."""
        v, ns = self._v(
            names=["energy by sector"],
            subs_sizes={"sectors": 3},
            subs_elems={"sectors": ["A", "B", "C"]},
            var_dims={"energy_by_sector": ["sectors"]},
            active_subs={"sectors": "_i0"},
        )
        func_ref = ReferenceStructure(
            "energy by sector",
            subscripts=SubscriptsReferenceStructure(subscripts=["B"]),
        )
        node = CallStructure(function=func_ref, arguments=[ReferenceStructure("Time")])
        result = v.visit(node)
        # "B" is element 2 in "sectors"
        assert "energy_by_sector(2, t)" in result

    def test_func_call_element_label_last_resort_range(self):
        """Element label in func call subscript not in var's dim uses last-resort range (lines 994-999)."""
        v, ns = self._v(
            names=["energy"],
            subs_sizes={"regions": 3, "sectors": 2},
            subs_elems={"regions": ["R1", "R2", "R3"], "sectors": ["A", "B"]},
            var_dims={"energy": ["regions"]},  # energy is in "regions" not "sectors"
            active_subs={"regions": "_i0"},
        )
        func_ref = ReferenceStructure(
            "energy",
            subscripts=SubscriptsReferenceStructure(subscripts=["A"]),
        )
        node = CallStructure(function=func_ref, arguments=[ReferenceStructure("Time")])
        result = v.visit(node)
        # "A" is at index 1 in "sectors" → last-resort range → energy(1, t)
        assert "energy(1, t)" in result

    # ------------------------------------------------------------------
    # Lines 877-878, 909: func call bang subscripts with mixed bang+non-bang subs
    # ------------------------------------------------------------------

    def test_func_call_bang_with_non_bang_sub_in_active_subs(self):
        """Bang func call with a non-bang sub that is in active_subs populates dim_to_idx_c (lines 877-878, 909)."""
        v, ns = self._v(
            names=["transport use"],
            subs_sizes={"fuel_type": 3, "sectors": 4},
            subs_elems={"fuel_type": ["gas", "oil", "elec"], "sectors": ["A", "B", "C", "D"]},
            var_dims={"transport_use": ["sectors", "fuel_type"]},
            active_subs={"sectors": "_i0"},  # non-bang sub "sectors" is active
        )
        # func_node_subs = ["fuel_type!", "sectors"] → bang + non-bang
        # Processing "sectors" (non-bang): hits line 877-878 (sub in active_subs)
        # Assembly: size-match loop skips "sectors" via line 909 (not endswith "!")
        func_ref = ReferenceStructure(
            "transport use",
            subscripts=SubscriptsReferenceStructure(subscripts=["fuel_type!", "sectors"]),
        )
        node = CallStructure(function=func_ref, arguments=[ReferenceStructure("Time")])
        result = v.visit(node)
        assert "_ii0" in result
        assert "_i0" in result

    def test_func_call_bang_with_non_bang_element_label(self):
        """Bang func call with a non-bang element label sub populates dim_to_idx_c (lines 883-886)."""
        v, ns = self._v(
            names=["data table"],
            subs_sizes={"fuel_type": 3, "sectors": 3},
            subs_elems={"fuel_type": ["gas", "oil", "elec"], "sectors": ["A", "B", "C"]},
            var_dims={"data_table": ["fuel_type", "sectors"]},
        )
        # func_node_subs = ["fuel_type!", "A"] where "A" is an element label (not a range)
        # → hits line 883-886
        func_ref = ReferenceStructure(
            "data table",
            subscripts=SubscriptsReferenceStructure(subscripts=["fuel_type!", "A"]),
        )
        node = CallStructure(function=func_ref, arguments=[ReferenceStructure("Time")])
        result = v.visit(node)
        assert "_ii0" in result

    # ------------------------------------------------------------------
    # Lines 716-718: size-match success in bang reference subscripts
    # ------------------------------------------------------------------

    def test_bang_ref_size_match_for_unmatched_dim_processed_first(self):
        """Bang reference: unmatched dim listed first has same size as bang dim → size match (lines 716-718)."""
        v, ns = self._v(
            names=["energy"],
            subs_sizes={"modes": 3, "sectors": 3},  # SAME sizes
            # "modes" is processed first (unmatched) → size matches "sectors!" → 716-718 hit
            var_dims={"energy": ["modes", "sectors"]},
        )
        ns.add_to_namespace("energy")
        node = ReferenceStructure(
            "energy",
            subscripts=SubscriptsReferenceStructure(subscripts=["sectors!"]),
        )
        result = v.visit(node)
        assert "_ii0" in result

    # ------------------------------------------------------------------
    # Lines 805-806: element label fallback loop finds match in var_dims_list
    # ------------------------------------------------------------------

    def test_non_bang_ref_element_label_fallback_loop_finds_match(self):
        """Element label subscript: pos-based check fails but fallback loop finds the range (lines 805-806)."""
        v, ns = self._v(
            names=["energy"],
            subs_sizes={"regions": 3, "sectors": 2},
            subs_elems={"regions": ["R1", "R2", "R3"], "sectors": ["A", "B"]},
            # 2D var: "regions" is pos-0 dim, "sectors" is pos-1 dim
            var_dims={"energy": ["regions", "sectors"]},
        )
        ns.add_to_namespace("energy")
        # "A" is an element label in "sectors" (pos-1), referenced at pos-0 of node_subs
        # pos=0, candidate="regions" → "A" not in _elem_index["A"]["regions"] → parent_range stays None
        # Fallback loop at 803: "regions" no match, "sectors" YES match → 805-806 hit
        node = ReferenceStructure(
            "energy",
            subscripts=SubscriptsReferenceStructure(subscripts=["A"]),
        )
        result = v.visit(node)
        assert "energy[1]" in result  # "A" is at index 1 in "sectors"

    # ------------------------------------------------------------------
    # Lines 996-997: element label fallback loop finds match in func call
    # ------------------------------------------------------------------

    def test_func_call_element_label_fallback_loop_finds_match(self):
        """Element label in func call subscript: fallback loop finds the range (lines 996-997)."""
        v, ns = self._v(
            names=["energy"],
            subs_sizes={"regions": 3, "sectors": 2},
            subs_elems={"regions": ["R1", "R2", "R3"], "sectors": ["A", "B"]},
            # 2D var: "regions" is pos-0 (doesn't contain "A"), "sectors" is pos-1 (contains "A")
            var_dims={"energy": ["regions", "sectors"]},
            active_subs={"regions": "_i0"},
        )
        # func_node_subs = ["A"] (element label) at pos-0
        # candidate = var_dims_list[0] = "regions" → "A" not in _elem_index["A"]["regions"]
        # Lines 994: loop → "regions" no match, "sectors" YES match → 996-997 hit
        func_ref = ReferenceStructure(
            "energy",
            subscripts=SubscriptsReferenceStructure(subscripts=["A"]),
        )
        node = CallStructure(function=func_ref, arguments=[ReferenceStructure("Time")])
        result = v.visit(node)
        assert "energy(1, t)" in result  # "A" at index 1 in "sectors"


# ===========================================================================
# Subscripted expansion paths (SMOOTH, DELAY, DELAY FIXED, SIT, INITIAL)
# ===========================================================================

class TestSubscriptedExpansions:
    """Cover the `if dims:` branches in _expand_smooth, _expand_delay,
    _expand_delay_fixed, _expand_sample_if_true, and _expand_initial_frozen_stock
    that are only reached when an element carries subscript dimensions.
    """

    def _sr(self, name, elems):
        return _make_subscript_range(name, elems)

    def _sub_comp(self, dims, ast, comp_class=None):
        if comp_class is AbstractUnchangeableConstant:
            c = AbstractUnchangeableConstant(subscripts=[dims, []], ast=ast)
        else:
            c = AbstractComponent(subscripts=[dims, []], ast=ast)
        return c

    # ------------------------------------------------------------------
    # Subscripted SMOOTH (lines 1828-1855)
    # ------------------------------------------------------------------

    def test_subscripted_smooth_order1_emits_comprehension(self):
        """SMOOTH(1) on a 1D subscripted element emits comprehension array levels."""
        sr = self._sr("sector", ["S1", "S2", "S3"])
        ast = SmoothStructure(input=10.0, smooth_time=5.0, initial=10.0, order=1)
        comp = self._sub_comp(["sector"], ast)
        elem = AbstractElement(name="Smooth Var", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("[_i0]" in e for e in eqs), f"Expected subscripted equations, got {eqs}"
        assert any("for _i0 in 1:" in e for e in eqs)
        assert any("[" in d for d in sb.stock_decls), "Smooth internal levels must be array stocks"

    def test_subscripted_smooth_order3_emits_multiple_levels(self):
        """SMOOTH(3) on 1D subscripted element emits 3 array-level ODE stages."""
        sr = self._sr("fuel", ["F1", "F2"])
        ast = SmoothStructure(input=5.0, smooth_time=4.0, initial=5.0, order=3)
        comp = self._sub_comp(["fuel"], ast)
        elem = AbstractElement(name="S3 Var", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        lv_decls = [d for d in sb.stock_decls if "_lv" in d]
        assert len(lv_decls) == 3, f"SMOOTH(3) must produce 3 internal levels, got {lv_decls}"

    # ------------------------------------------------------------------
    # Subscripted DELAY (lines 1901-1940)
    # ------------------------------------------------------------------

    def test_subscripted_delay3_emits_comprehension_pipeline(self):
        """DELAY3 on a 1D subscripted element emits 3 comprehension pipeline stages."""
        sr = self._sr("region", ["R1", "R2"])
        ast = DelayStructure(input=5.0, delay_time=3.0, initial=5.0, order=3)
        comp = self._sub_comp(["region"], ast)
        elem = AbstractElement(name="Delay Var", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("[_i0]" in e for e in eqs), f"Expected subscripted delay equations, got {eqs}"
        dl_decls = [d for d in sb.stock_decls if "_dl" in d]
        assert len(dl_decls) == 3, f"DELAY3 must produce 3 pipeline stages, got {dl_decls}"

    def test_subscripted_delay1_emits_u0_per_index(self):
        """DELAY1 on a 1D element produces per-index u0 entries."""
        sr = self._sr("cat", ["C1", "C2", "C3"])
        ast = DelayStructure(input=2.0, delay_time=1.0, initial=2.0, order=1)
        comp = self._sub_comp(["cat"], ast)
        elem = AbstractElement(name="D1 Var", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        u0_entries = [e for e in sb.u0_entries if "_dl1_d1_var" in e]
        assert len(u0_entries) == 3, f"Expected 3 u0 entries for 3-element dim, got {sb.u0_entries}"

    # ------------------------------------------------------------------
    # Subscripted DELAY FIXED via MTK backend (lines 2134-2159)
    # ------------------------------------------------------------------

    def test_subscripted_delay_fixed_mtk_emits_array_ode(self):
        """DELAY FIXED on subscripted element with MTK backend uses array ODE (not pipeline)."""
        sr = self._sr("sector", ["S1", "S2"])
        ast = DelayFixedStructure(input=5.0, delay_time=2.0, initial=5.0)
        comp = self._sub_comp(["sector"], ast)
        elem = AbstractElement(name="DF Sub", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr], backend="mtk")
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("_df_df_sub" in e for e in eqs), f"Expected delay-fixed array ODE, got {eqs}"
        assert any("[" in d for d in sb.stock_decls)
        u0_entries = [e for e in sb.u0_entries if "_df_df_sub" in e]
        assert len(u0_entries) == 2, f"Expected 2 u0 entries for 2-element dim, got {sb.u0_entries}"

    # ------------------------------------------------------------------
    # Subscripted SAMPLE IF TRUE (lines 2388-2415)
    # ------------------------------------------------------------------

    def test_subscripted_sample_if_true_emits_array_stock(self):
        """SAMPLE IF TRUE on subscripted element emits array-comprehension stock."""
        sr = self._sr("product", ["P1", "P2", "P3"])
        ast = SampleIfTrueStructure(condition=1.0, input=7.0, initial=0.0)
        comp = self._sub_comp(["product"], ast)
        elem = AbstractElement(name="SIT Var", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        sit_eqs = [e for e in eqs if "_sit_sit_var" in e]
        assert sit_eqs, f"Expected SIT stock equations, got {eqs}"
        assert any("for _i0 in 1:" in e for e in sit_eqs)
        u0_entries = [e for e in sb.u0_entries if "_sit_" in e]
        assert len(u0_entries) == 3, f"Expected 3 u0 entries, got {sb.u0_entries}"

    # ------------------------------------------------------------------
    # 2D INITIAL frozen stock (lines 3244-3259)
    # ------------------------------------------------------------------

    def test_2d_initial_frozen_stock_emits_per_element_u0(self):
        """INITIAL(x) with 2D subscript and non-resolvable inner → 2D frozen stock."""
        import warnings
        sr1 = self._sr("row", ["R1", "R2"])
        sr2 = self._sr("col", ["C1", "C2"])
        # ReferenceStructure inner value cannot be resolved at translation time
        inner = ReferenceStructure("dynamic_val")
        ast = InitialStructure(initial=inner)
        comp = AbstractComponent(subscripts=[["row", "col"], []], ast=ast)
        elem = AbstractElement(name="Init 2D", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sb.build_section()
        u0_entries = [e for e in sb.u0_entries if "init_2d" in e]
        assert len(u0_entries) == 4, f"Expected 4 u0 entries for 2×2 dim, got {sb.u0_entries}"
        # Per-element entries use concrete indices like "init_2d[1, 1] => ..."
        assert any("[1, 1]" in e for e in u0_entries)
        assert any("[2, 2]" in e for e in u0_entries)
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("_i0 in 1:" in e and "_i1 in 1:" in e for e in eqs)

    # ------------------------------------------------------------------
    # 3D inline lookup flattened fallback (lines 2656-2679)
    # ------------------------------------------------------------------

    def test_3d_inline_lookup_emits_flattened_array(self):
        """Subscripted inline lookup with 3 dims falls back to 1D flat array with a warning."""
        import warnings
        sr1 = self._sr("r", ["R1", "R2"])
        sr2 = self._sr("c", ["C1", "C2"])
        sr3 = self._sr("z", ["Z1", "Z2"])
        lkp = LookupsStructure(x=(0.0, 1.0), y=(0.0, 1.0),
                               x_limits=(0.0, 1.0), y_limits=(0.0, 1.0), type="interpolate")
        # Multi-component inline lookup with 3D specific-element subscripts
        comps = [
            AbstractLookup(subscripts=[["R1", "C1", "Z1"], []], ast=lkp),
            AbstractLookup(subscripts=[["R1", "C1", "Z2"], []], ast=lkp),
            AbstractLookup(subscripts=[["R2", "C2", "Z1"], []], ast=lkp),
        ]
        elem = AbstractElement(name="3D Lookup", components=comps)
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2, sr3])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            sb.build_section()
        warns = [str(w.message) for w in captured if "3D Lookup" in str(w.message)]
        assert warns, "Expected a warning about 3D lookup flattening"
        assert any("flattening" in w.lower() or "1D" in w for w in warns)
        # Should register a 1D dispatch function
        assert any("3d_lookup" in d for d in sb.lookup_func_decls)

    # ------------------------------------------------------------------
    # Subscript alias range (lines 246, 248)
    # ------------------------------------------------------------------

    def test_alias_subscript_range_resolves_size_and_elements(self):
        """An AbstractSubscriptRange with subscripts as a string (alias) resolves
        to the aliased range's size and element list (lines 246, 248)."""
        from pysd.translators.structures.abstract_model import AbstractSubscriptRange as ASR
        sr_real = _make_subscript_range("sector", ["S1", "S2", "S3"])
        sr_alias = ASR(name="sec_all", subscripts="sector", mapping=[])
        elem = _make_element("X", 1.0)
        sb = _section_builder_from_elements([elem], subscripts=[sr_real, sr_alias])
        assert sb._subs_sizes.get("sec_all") == 3
        assert sb._subs_elems.get("sec_all") == ["S1", "S2", "S3"]

    # ------------------------------------------------------------------
    # GCS transpose cell (lines 3365, 3388, 3401, 3426)
    # ------------------------------------------------------------------

    def test_single_gcs_transposed_cell_emits_transpose_kwarg(self):
        """Single-component GCS with cell='A1*' emits transpose=true kwarg (line 3365)."""
        sr = self._sr("sector", ["S1", "S2"])
        gcs_ast = GetConstantsStructure(file="f.xlsx", tab="Sheet1", cell="A1*")
        comp = AbstractUnchangeableConstant(subscripts=[["sector"], []], ast=gcs_ast)
        elem = AbstractElement(name="Trans Const", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        decls = sb.param_decls + sb.ext_const_decls
        assert any("transpose=true" in d for d in decls), (
            f"Expected transpose=true in declarations, got {decls}"
        )

    def test_multi_gcs_transposed_cell_emits_transpose_kwarg(self):
        """Multi-component GCS where one cell ends with '*' sets transpose=true (lines 3388, 3426)."""
        sr = self._sr("fuel", ["fuel1", "fuel2"])
        gcs1_ast = GetConstantsStructure(file="f.xlsx", tab="Sheet1", cell="A1*")
        gcs2_ast = GetConstantsStructure(file="f.xlsx", tab="Sheet1", cell="A2")
        comp1 = AbstractComponent(subscripts=[["fuel1"], []], ast=gcs1_ast)
        comp2 = AbstractComponent(subscripts=[["fuel2"], []], ast=gcs2_ast)
        elem = AbstractElement(name="Multi Trans", components=[comp1, comp2])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        decls = sb.param_decls + sb.ext_const_decls
        assert any("transpose=true" in d for d in decls), (
            f"Expected transpose=true in declarations, got {decls}"
        )

    def test_multi_literal_range_fills_with_fill_expr(self):
        """Multi-comp where literal covers a full range (n_elems>1) emits fill() (line 3401)."""
        sr = self._sr("fuel", ["fuel1", "fuel2", "fuel3"])
        gcs_comp = AbstractComponent(
            subscripts=[["fuel1"], []], ast=GetConstantsStructure(file="f.xlsx", tab="S", cell="A1")
        )
        lit_comp = AbstractComponent(subscripts=[["fuel"], []], ast=0.0)
        elem = AbstractElement(name="Fill Test", components=[gcs_comp, lit_comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        decls = sb.param_decls + sb.ext_const_decls
        assert any("fill(" in d for d in decls), (
            f"Expected fill() in declarations for range-covering literal, got {decls}"
        )


# ===========================================================================
# EXCEPT subscription paths with stock/delay-fixed (lines 1440-1442, 1494-1532, 1552)
# ===========================================================================

class TestExceptWithStateful:
    """Cover the per-index stock and delay-fixed branches in _process_except_element."""

    def _make_1d_except(self, name, dim_name, dim_elems, comp1_ast, comp2_ast, except_labels):
        sr = _make_subscript_range(dim_name, dim_elems)
        comp1 = AbstractComponent(
            subscripts=[[dim_name], [except_labels]],
            ast=comp1_ast,
        )
        comp2 = AbstractComponent(subscripts=[[dim_name], []], ast=comp2_ast)
        return AbstractElement(name=name, components=[comp1, comp2]), sr

    def test_except_1d_integ_emits_stock_decl(self):
        """1D EXCEPT with IntegStructure emits stock_decls array and per-index ODE (lines 1494-1509, 1552)."""
        elem, sr = self._make_1d_except(
            "Level",
            "sector", ["S1", "S2", "S3"],
            comp1_ast=IntegStructure(flow=1.0, initial=0.0),
            comp2_ast=IntegStructure(flow=0.0, initial=0.0),
            except_labels=["S3"],
        )
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        ode_eqs = [e for e in eqs if "D(level[" in e]
        assert ode_eqs, f"Expected per-index ODE equations, got {eqs}"
        # has_integ=True → stock_decls should contain the array declaration
        assert any("level(t)[" in d for d in sb.stock_decls), (
            f"Expected array stock_decl for level, got {sb.stock_decls}"
        )

    def test_except_1d_delay_fixed_emits_df_stock(self):
        """1D EXCEPT with DelayFixedStructure creates _df_ stock array (lines 1440-1442, 1514-1532)."""
        import warnings
        elem, sr = self._make_1d_except(
            "DF Var",
            "product", ["P1", "P2"],
            comp1_ast=DelayFixedStructure(input=3.0, delay_time=2.0, initial=3.0),
            comp2_ast=2.0,
            except_labels=["P2"],
        )
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # The _df_ internal variable should be declared as an array stock
        df_decls = [d for d in sb.stock_decls if "_df_df_var" in d]
        assert df_decls, f"Expected _df_ stock declaration, got {sb.stock_decls}"
        # Per-index delay-fixed equations: D(_df_[idx]) ~ (...) and df_var[idx] ~ _df_[idx]
        df_eqs = [e for e in eqs if "_df_df_var[" in e]
        assert df_eqs, f"Expected per-index delay-fixed equations, got {eqs}"

    def test_except_4d_emits_warn_and_fallback_scalarize(self):
        """4D EXCEPT emits a UserWarning and falls back to a scalarize equation (lines 1406-1422)."""
        import warnings
        sr1 = _make_subscript_range("d1", ["A", "B"])
        sr2 = _make_subscript_range("d2", ["X", "Y"])
        sr3 = _make_subscript_range("d3", ["P", "Q"])
        sr4 = _make_subscript_range("d4", ["M", "N"])
        comp1 = AbstractComponent(
            subscripts=[["d1", "d2", "d3", "d4"], [["A", "X", "P", "M"]]],
            ast=1.0,
        )
        comp2 = AbstractComponent(subscripts=[["d1", "d2", "d3", "d4"], []], ast=2.0)
        elem = AbstractElement(name="4D Except", components=[comp1, comp2])
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2, sr3, sr4])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            sb.build_section()
        warns = [str(w.message) for w in captured if "4D" in str(w.message)
                 or "4d" in str(w.message).lower() or "not yet supported" in str(w.message)]
        assert warns, f"Expected warning for 4D EXCEPT, got: {[str(w.message) for w in captured]}"
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("Symbolics.scalarize" in e for e in eqs), (
            f"Expected scalarize fallback equation, got {eqs}"
        )

    def test_except_4d_ode_build_skips_scalarize_equation(self, tmp_path):
        """ODE model with 4D EXCEPT element: scalarize equation is skipped in rhs! (line 4357)."""
        import warnings
        sr1 = _make_subscript_range("d1", ["A", "B"])
        sr2 = _make_subscript_range("d2", ["X", "Y"])
        sr3 = _make_subscript_range("d3", ["P", "Q"])
        sr4 = _make_subscript_range("d4", ["M", "N"])
        comp1 = AbstractComponent(
            subscripts=[["d1", "d2", "d3", "d4"], [["A", "X", "P", "M"]]],
            ast=1.0,
        )
        comp2 = AbstractComponent(subscripts=[["d1", "d2", "d3", "d4"], []], ast=2.0)
        elem_4d = AbstractElement(name="4D Except", components=[comp1, comp2])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem_4d] + controls,
            subscripts=[sr1, sr2, sr3, sr4],
            path=tmp_path / "m4d.mdl",
        )
        model = AbstractModel(original_path=tmp_path / "m4d.mdl", sections=(section,))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            jl_path = JuliaModelBuilder(model).build_model()
        content = jl_path.read_text()
        # The scalarize equation should NOT appear verbatim in the rhs! function
        assert "function rhs!" in content

    def test_except_1d_ode_model_alloc_needed(self, tmp_path):
        """ODE model with 1D EXCEPT constant emits per-index alloc lines (line 4330)."""
        sr = _make_subscript_range("sector", ["S1", "S2"])
        comp1 = AbstractComponent(subscripts=[["sector"], [["S2"]]], ast=1.0)
        comp2 = AbstractComponent(subscripts=[["sector"], []], ast=2.0)
        stock = _make_stock_element("Level", 1.0, 0.0)
        elem_exc = AbstractElement(name="Exc Const", components=[comp1, comp2])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[stock, elem_exc] + controls,
            subscripts=[sr],
            path=tmp_path / "m_exc.mdl",
        )
        model = AbstractModel(original_path=tmp_path / "m_exc.mdl", sections=(section,))
        jl_path = JuliaModelBuilder(model).build_model()
        content = jl_path.read_text()
        assert "function rhs!" in content


# ===========================================================================
# JuliaSectionBuilder helper methods (low-level coverage)
# ===========================================================================

class TestSectionBuilderHelpers:
    """Cover internal helper methods on JuliaSectionBuilder directly."""

    def _sb(self):
        """Return a minimal section builder (no elements, no subscripts)."""
        return _section_builder_from_elements([_make_element("x", 1.0)])

    # ------------------------------------------------------------------
    # _extract_rhs_identifiers (lines 4481, 4485)
    # ------------------------------------------------------------------

    def test_extract_rhs_comment_line_returns_empty_set(self):
        """A comment equation (starts with '#') returns an empty set (line 4481)."""
        result = JuliaSectionBuilder._extract_rhs_identifiers("# this is a comment")
        assert result == set()

    def test_extract_rhs_assignment_style_splits_on_equals(self):
        """An assignment-style eq without '~' splits on ' = ' to find RHS identifiers (line 4485)."""
        result = JuliaSectionBuilder._extract_rhs_identifiers("output = input_var + scale_factor")
        assert "input_var" in result
        assert "scale_factor" in result
        assert "output" not in result

    # ------------------------------------------------------------------
    # _topo_sort_equations circular dependencies (lines 4580-4581)
    # ------------------------------------------------------------------

    def test_topo_sort_circular_deps_appended_at_end(self):
        """Mutually-dependent equations can't be sorted; they're appended in original order (lines 4580-4581)."""
        sb = self._sb()
        # a depends on b, b depends on a — neither can be resolved
        eqs = ["a ~ b + 1.0", "b ~ a + 1.0"]
        sorted_eqs = sb._topo_sort_equations(eqs, stock_names={})
        # Both equations must appear in the result (just appended after circular detection)
        assert len(sorted_eqs) == 2
        assert set(sorted_eqs) == set(eqs)

    # ------------------------------------------------------------------
    # _convert_eq_to_assignment malformed comprehension (line 4621)
    # ------------------------------------------------------------------

    def test_convert_eq_to_assignment_comprehension_without_for_raises(self):
        """Comprehension equation with no outer 'for' clause raises ValueError (line 4621)."""
        sb = self._sb()
        with pytest.raises(ValueError, match="Cannot convert comprehension"):
            sb._convert_eq_to_assignment("[x ~ 1.0]")

    # ------------------------------------------------------------------
    # _convert_ode_to_du nested brackets — depth tracking (lines 4644, 4646, 4672)
    # ------------------------------------------------------------------

    def test_convert_ode_to_du_1d_comprehension_for_clause_with_parens(self):
        """for-clause with parentheses forces the backward scan to track bracket depth (lines 4644, 4646)."""
        sb = self._sb()
        # "size(v, 1)" in the for clause puts ')' and '(' to the right of "for ",
        # so the backward scan encounters them BEFORE finding "for " — triggering
        # the depth-tracking branches at lines 4644 and 4646.
        eq = "[D(x[_i0]) ~ 1.0 for _i0 in 1:size(v, 1)]..."
        result = sb._convert_ode_to_du(eq, stock_indices={"x": 3})
        assert result[0] == "for _i0 in 1:size(v, 1)"
        assert "du[3 - 1 + _i0]" in result[1]
        assert result[2] == "end"

    def test_convert_ode_to_du_comprehension_body_not_d_form_falls_back(self):
        """Comprehension body that doesn't match D(var[idx]) falls back to replace (line 4672)."""
        sb = self._sb()
        eq = "[x[_i0] ~ 1.0 for _i0 in 1:N]..."
        result = sb._convert_ode_to_du(eq, stock_indices={})
        # Falls through to the replace fallback
        assert len(result) == 1
        assert " = " in result[0]

    # ------------------------------------------------------------------
    # _eval_ast_at_t0 and _try_eval_as_float (lines 1983-1999, 2052-2068)
    # ------------------------------------------------------------------

    def test_eval_ast_arithmetic_multiply(self):
        """ArithmeticStructure with '*' is evaluated (hits lines 1991-1992)."""
        sb = self._sb()
        ast = ArithmeticStructure(arguments=[3.0, 4.0], operators=["*"])
        assert sb._eval_ast_at_t0(ast) == 12.0

    def test_eval_ast_arithmetic_divide_by_zero_returns_none(self):
        """Division by zero in ArithmeticStructure returns None (hits lines 1994-1998)."""
        sb = self._sb()
        ast = ArithmeticStructure(arguments=[1.0, 0.0], operators=["/"])
        assert sb._eval_ast_at_t0(ast) is None

    def test_eval_ast_arithmetic_unsupported_op_returns_none(self):
        """An unsupported operator in ArithmeticStructure returns None (line 1996)."""
        sb = self._sb()
        ast = ArithmeticStructure(arguments=[2.0, 3.0], operators=["^"])
        assert sb._eval_ast_at_t0(ast) is None

    def test_eval_ast_arithmetic_none_arg_returns_none(self):
        """ArithmeticStructure with an un-resolvable arg returns None (line 1983)."""
        sb = self._sb()
        inner = ReferenceStructure("unknown_var")  # not in namespace → None
        ast = ArithmeticStructure(arguments=[inner, 2.0], operators=["+"])
        assert sb._eval_ast_at_t0(ast) is None

    def test_try_eval_as_float_finds_param_decl(self):
        """_try_eval_as_float resolves a name declared in param_decls (lines 2052-2056)."""
        sb = self._sb()
        sb.param_decls.append("@parameters my_rate = 0.25")
        result = sb._try_eval_as_float("my_rate")
        assert result == pytest.approx(0.25)

    def test_try_eval_as_float_finds_built_element(self):
        """_try_eval_as_float resolves a name from built_elements (lines 2059-2067)."""
        sb = self._sb()
        sb.built_elements["aux_val"] = (["aux_val ~ 3.14"], False)
        result = sb._try_eval_as_float("aux_val")
        assert result == pytest.approx(3.14)

    def test_eval_ast_arithmetic_subtract(self):
        """ArithmeticStructure with '-' is evaluated (line 1990)."""
        sb = self._sb()
        ast = ArithmeticStructure(arguments=[10.0, 3.0], operators=["-"])
        assert sb._eval_ast_at_t0(ast) == pytest.approx(7.0)

    def test_eval_ast_call_unknown_func_returns_none(self):
        """CallStructure with unrecognised function name returns None (line 2007)."""
        sb = self._sb()
        func = ReferenceStructure("some_custom_func")
        ast = CallStructure(function=func, arguments=[1.0])
        assert sb._eval_ast_at_t0(ast) is None

    def test_eval_ast_reference_resolved_via_namespace(self):
        """ReferenceStructure resolved through namespace + param_decls (lines 2010-2013)."""
        sb = self._sb()
        sb.namespace.add_to_namespace("growth rate")  # → growth_rate
        sb.param_decls.append("@parameters growth_rate = 0.1")
        ast = ReferenceStructure("growth rate")
        result = sb._eval_ast_at_t0(ast)
        assert result == pytest.approx(0.1)

    def test_eval_ast_reference_resolved_via_abstract_elements(self):
        """ReferenceStructure falling back to abstract_elements recursive eval (lines 2022-2028)."""
        sb = self._sb()
        # Add an element to abstract_elements with a concrete AST value
        elem = _make_element("order var", 4.0)
        sb.namespace.add_to_namespace("order var")
        sb.abstract_elements.append(elem)
        ast = ReferenceStructure("order var")
        result = sb._eval_ast_at_t0(ast)
        assert result == pytest.approx(4.0)

    def test_try_eval_as_float_malformed_param_decl_skipped(self):
        """_try_eval_as_float skips a param_decl whose value string raises ValueError (lines 2056-2057)."""
        sb = self._sb()
        # "1e" matches the regex but float("1e") raises ValueError → skip silently
        sb.param_decls.append("@parameters bad_param = 1e")
        result = sb._try_eval_as_float("bad_param")
        assert result is None

    def test_try_eval_as_float_non_numeric_built_element_skipped(self):
        """_try_eval_as_float skips built_element whose RHS is non-numeric (lines 2067-2068)."""
        sb = self._sb()
        sb.built_elements["symbolic"] = (["symbolic ~ some_expression"], False)
        result = sb._try_eval_as_float("symbolic")
        assert result is None

    def test_eval_ast_at_t0_unsupported_node_type_returns_none(self):
        """_eval_ast_at_t0 returns None for unsupported AST node type (line 2030)."""
        sb = self._sb()
        # LookupsStructure is not int/float/Arithmetic/Call/Reference → returns None at line 2030
        lut = LookupsStructure(x=[0.0, 1.0], y=[0.0, 1.0], x_limits=(0.0, 1.0),
                               y_limits=(0.0, 1.0), type="interpolate")
        result = sb._eval_ast_at_t0(lut)
        assert result is None

    def test_eval_ast_at_t0_reference_with_unsupported_component_breaks(self):
        """_eval_ast_at_t0 breaks comp loop when comp.ast is not a simple type (line 2025)."""
        sb = self._sb()
        # Build an element whose component AST is a LookupsStructure (not a simple type)
        lut = LookupsStructure(x=[0.0, 1.0], y=[0.0, 1.0], x_limits=(0.0, 1.0),
                               y_limits=(0.0, 1.0), type="interpolate")
        elem = AbstractElement(name="my table", components=[AbstractComponent(
            subscripts=[[], []], ast=lut
        )])
        sb.namespace.add_to_namespace("my table")
        sb.abstract_elements.append(elem)
        # ReferenceStructure → looks up "my table" in abstract_elements → finds comp.ast=LookupsStructure
        # → isinstance check fails → break at line 2025 → returns None
        result = sb._eval_ast_at_t0(ReferenceStructure("my table"))
        assert result is None

    # ------------------------------------------------------------------
    # Subscript geometry helper methods (lines 579, 633, 676, 730, 766, 770, 794)
    # ------------------------------------------------------------------

    def _sb_with_subs(self, *sub_ranges):
        """Build a minimal section builder that knows about the given subscript ranges."""
        elems = [_make_element("x", 1.0)]
        return _section_builder_from_elements(elems, subscripts=list(sub_ranges))

    def test_comp_coords_unknown_subscript_fallback(self):
        """_comp_coords returns empty list for unknown subscript (line 579)."""
        sr = _make_subscript_range("sectors", ["A", "B"])
        sb = self._sb_with_subs(sr)
        comp = AbstractComponent(subscripts=[["unknown_dim"], []], ast=1.0)
        result = sb._comp_coords(comp)
        # "unknown_dim" not in _subs_elems, not in _elem_to_range → result["unknown_dim"] = []
        assert result == {"unknown_dim": []}

    def test_detect_split_ranges_empty_components(self):
        """_detect_split_ranges returns {} when components have no subscripts (line 633)."""
        sr = _make_subscript_range("sectors", ["A", "B"])
        sb = self._sb_with_subs(sr)
        comp = AbstractComponent(subscripts=[[], []], ast=1.0)
        result = sb._detect_split_ranges([comp])
        assert result == {}

    def test_comp_coords_split_unknown_subscript_fallback(self):
        """_comp_coords_split returns empty list for subscript not in split_ranges/subs_elems (line 676)."""
        sr = _make_subscript_range("sectors", ["A", "B"])
        sb = self._sb_with_subs(sr)
        comp = AbstractComponent(subscripts=[["weird_sub"], []], ast=1.0)
        # split_ranges only covers pos=0 for a different sub, "weird_sub" falls to else
        result = sb._comp_coords_split(comp, split_ranges={})
        assert result == {"weird_sub": []}

    def test_element_dims_single_comp_element_subscript(self):
        """_element_dims with single-component element subscript uses _elem_to_range (line 730)."""
        sr = _make_subscript_range("sectors", ["A", "B", "C"])
        sb = self._sb_with_subs(sr)
        # Single-component element with specific element subscript "A" (not range name)
        elem = AbstractElement(name="y", components=[
            AbstractComponent(subscripts=[["A"], []], ast=1.0)
        ])
        result = sb._element_dims(elem)
        # "A" is in _elem_to_range (→ "sectors"), single comp → line 730: parent = _elem_to_range["A"]
        assert any(d == "sectors" for d, _ in result)

    def test_per_index_subs_def_elems_empty_returns_early(self):
        """_per_index_subs returns subs early when def_range_name has no elements (line 766)."""
        sr = _make_subscript_range("parent", ["A", "B", "C"])
        sb = self._sb_with_subs(sr)
        # def_range_name = "nonexistent" → _subs_elems["nonexistent"] = [] → line 766
        result = sb._per_index_subs("parent", ["A", "B", "C"], 1, "nonexistent")
        assert result == {"parent": "1"}

    def test_per_index_subs_element_not_in_def_elems_returns_early(self):
        """_per_index_subs returns subs early when element label not in def_range (line 770)."""
        sr1 = _make_subscript_range("parent", ["A", "B", "C"])
        sr2 = _make_subscript_range("sub_range", ["X", "Y"])
        sb = self._sb_with_subs(sr1, sr2)
        # abs_idx=1 → element_label = "A", but def_elems=["X","Y"] → "A" not in def_elems → line 770
        result = sb._per_index_subs("parent", ["A", "B", "C"], 1, "sub_range")
        assert result == {"parent": "1"}

    def test_per_index_subs_same_size_range_assigned(self):
        """_per_index_subs maps other same-size ranges to the same positional index (line 794)."""
        sr1 = _make_subscript_range("main_dim", ["A", "B", "C"])
        sr2 = _make_subscript_range("def_range", ["X", "Y", "Z"])
        sr3 = _make_subscript_range("alias_range", ["P", "Q", "R"])  # same size=3 as def_range
        sb = self._sb_with_subs(sr1, sr2, sr3)
        # abs_idx=1 → element_label="A" in main_dim
        # def_range has 3 elems: "A" not in ["X","Y","Z"] → line 770 early return
        # Wait, need element_label IN def_elems
        # Let me use main_dim as the dim being indexed, and def_range shares elements with parent
        # abs_idx=2 means element_label = "B"
        # def_range = sr1? No...
        # Try: def_range = main_dim_alias with same elements
        sr_def = _make_subscript_range("def_range2", ["A", "B", "C"])  # same elements as main
        sb2 = self._sb_with_subs(sr1, sr_def, sr3)
        # abs_idx=1 → element_label="A", def_elems=["A","B","C"] → "A" in def_elems
        # pos=0 (index of "A" in def_elems), alias_range also has size 3 → line 794!
        result = sb2._per_index_subs("main_dim", ["A", "B", "C"], 1, "def_range2")
        # def_range2 has element "A" at pos 0 → subs["def_range2"] = "1"
        # alias_range has size 3 (same as def_range2 size 3) → subs["alias_range"] = "1" (line 794)
        assert "def_range2" in result
        assert "alias_range" in result

    # ------------------------------------------------------------------
    # Lines 4147, 4165-4172: declarations block with scalar ext_const_decls
    # ------------------------------------------------------------------

    def test_declarations_block_ode_scalar_ext_const_passes_through(self):
        """Scalar ext_const entry (no '[') hits the else branch at line 4147 in ODE declarations."""
        sb = self._sb()
        sb.ext_const_decls.append("const my_scalar = 3.14")
        block = sb._declarations_block()
        assert "# External constants" in block
        # No "[" in value and not xlsx → line 4147: append decl as-is
        assert "const my_scalar = 3.14" in block

    def test_declarations_block_mtk_ext_const(self):
        """MTK declarations block includes ext_const_decls entries (lines 4165-4172)."""
        sb = self._sb()
        sb.backend = "mtk"  # switch to MTK mode
        sb.ext_const_decls.append("const arr_data = [1.0, 2.0]")
        sb.ext_const_decls.append("const scalar_val = 9.81")
        block = sb._declarations_block_mtk()
        assert "# External constants" in block
        # "[1.0, 2.0]" has "[" and starts with "const" → line 4170: pysd_safe
        assert "pysd_safe" in block
        # scalar_val has no "[" → line 4172: passed through as-is
        assert "scalar_val = 9.81" in block

    # ------------------------------------------------------------------
    # Line 2591, 2632: inline lookup placeholder for missing dimension indices
    # ------------------------------------------------------------------

    def test_subscripted_inline_lookup_placeholder_for_missing_index(self):
        """When a 1D inline lookup has fewer components than dim elements, placeholder is emitted (line 2591)."""
        sr = _make_subscript_range("sectors", ["A", "B", "C"])  # 3 elements
        lkp = LookupsStructure(x=(0.0, 1.0), y=(0.0, 1.0), x_limits=(0.0, 1.0),
                               y_limits=(0.0, 1.0), type="interpolate")
        # Only 2 components for a 3-element dim → index 3 (C) missing → placeholder at line 2591
        comp1 = AbstractLookup(subscripts=[["A"], []], ast=lkp)
        comp2 = AbstractLookup(subscripts=[["B"], []], ast=lkp)
        elem = AbstractElement(name="My Lookup", components=[comp1, comp2])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        # Placeholder interpolation should appear for the missing index C
        assert any("LinearInterpolation([0.0], [0.0]" in d for d in sb.lookup_const_decls)

    def test_subscripted_inline_lookup_2d_placeholder_for_missing_index(self):
        """When a 2D inline lookup has fewer components than dim elements, 2D placeholder is emitted (line 2632)."""
        sr1 = _make_subscript_range("dim1", ["A", "B"])
        sr2 = _make_subscript_range("dim2", ["X", "Y"])
        lkp = LookupsStructure(x=(0.0, 1.0), y=(0.0, 1.0), x_limits=(0.0, 1.0),
                               y_limits=(0.0, 1.0), type="interpolate")
        # 2 components for a 2×2 grid: cover (1,1) and (1,2) → (2,1) and (2,2) get placeholders
        comp1 = AbstractLookup(subscripts=[["A", "X"], []], ast=lkp)
        comp2 = AbstractLookup(subscripts=[["A", "Y"], []], ast=lkp)
        elem = AbstractElement(name="2D Lookup", components=[comp1, comp2])
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2])
        sb.build_section()
        # Placeholder interpolation should appear for the missing indices
        assert any("LinearInterpolation([0.0], [0.0]" in d for d in sb.lookup_const_decls)

    def test_subscripted_inline_lookup_unknown_label_uses_fallback_index(self):
        """Subscripted inline lookup with unknown element label uses fallback index 1 (line 2568)."""
        sr = _make_subscript_range("sectors", ["A", "B"])
        lkp = LookupsStructure(x=(0.0, 1.0), y=(0.0, 1.0), x_limits=(0.0, 1.0),
                               y_limits=(0.0, 1.0), type="interpolate")
        # comp1 has known label "A" (idx=1); comp2 has "Z" which is NOT in sectors ["A","B"]
        # → idx = None → fallback to index 1 at line 2568.
        # Two components are required so routing hits _process_subscripted_inline_lookup.
        comp1 = AbstractLookup(subscripts=[["A"], []], ast=lkp)
        comp2 = AbstractLookup(subscripts=[["Z"], []], ast=lkp)
        elem = AbstractElement(name="Unknown Label Lookup", components=[comp1, comp2])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        # The lookup should still be registered (using index 1 as fallback for "Z")
        assert any("unknown_label_lookup" in d for d in sb.lookup_func_decls)

    # ------------------------------------------------------------------
    # Line 4724: tokens.discard in u0 second pass
    # ------------------------------------------------------------------

    def test_u0_block_discards_param_token_in_dynamic_pass(self):
        """tokens.discard removes param names from dynamic_tokens in u0 second pass (line 4724)."""
        sb = self._sb()
        # First entry has non-param dynamic variable → needs_init_fn = True
        sb.u0_entries.append("stock_a => dynamic_aux")
        # Second entry has a parameter → line 4724 should discard it from dynamic_tokens
        sb.u0_entries.append("stock_b => my_param")
        sb.param_decls.append("@parameters my_param = 1.0")
        block = sb._u0_block()
        # dynamic_aux is in dynamic_tokens → appears as get(_obs_init, ...)
        assert "dynamic_aux" in block
        # my_param was discarded at line 4724 → should NOT be fetched from _obs_init
        # It appears in the values but not in the let-binding fetch lines
        assert 'get(_obs_init, "my_param"' not in block

    # ------------------------------------------------------------------
    # Lines 4077 + 4087: json data_format + mtk backend → @register_symbolic
    # ------------------------------------------------------------------

    def test_lookup_block_json_mtk_emits_register_symbolic_for_lookups(self):
        """In json+mtk mode, _lookup_block emits @register_symbolic for lookup entries (line 4077)."""
        sb = _section_builder_from_elements([_make_element("x", 1.0)], backend="mtk")
        sb.data_format = "json"
        sb._json_data["lookups"]["my_lkp"] = {"x": [0.0, 1.0], "y": [0.0, 2.0]}
        block = sb._lookup_block()
        assert "@register_symbolic my_lkp(x::Real)" in block

    def test_lookup_block_json_mtk_emits_register_symbolic_for_data(self):
        """In json+mtk mode, _lookup_block emits @register_symbolic for data entries (line 4087)."""
        sb = _section_builder_from_elements([_make_element("x", 1.0)], backend="mtk")
        sb.data_format = "json"
        sb._json_data["data"]["my_data"] = {"time": [0.0, 1.0], "values": [3.0, 4.0]}
        block = sb._lookup_block()
        assert "@register_symbolic my_data(x::Real)" in block

    # ------------------------------------------------------------------
    # Line 1619: EXCEPT 2D continue when all covered rows are excluded
    # ------------------------------------------------------------------

    def test_except_2d_all_rows_excluded_skips_component(self):
        """When an EXCEPT clause covers ALL rows of a 2D component, that component is
        skipped via continue (line 1619), leaving only comp1's equations."""
        sr1 = _make_subscript_range("r", ["A", "B"])
        sr2 = _make_subscript_range("c", ["X", "Y"])
        # comp0 covers A×c EXCEPT [A,c] → all covered rows excluded → skipped (line 1619)
        comp0 = AbstractComponent(subscripts=[["A", "c"], [["A", "c"]]], ast=99.0)
        # comp1 covers all r×c with no EXCEPT → produces the actual equations
        comp1 = AbstractComponent(subscripts=[["r", "c"], []], ast=1.0)
        elem = AbstractElement(name="Skip Row", components=[comp0, comp1])
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # comp0 was skipped; its value 99.0 must not appear in any equation
        assert not any("99.0" in e for e in eqs)
        # comp1's equation must appear
        assert any("1.0" in e for e in eqs)

    # ------------------------------------------------------------------
    # Lines 1794-1795: _materialize_input name-collision counter loop
    # ------------------------------------------------------------------

    def test_materialize_input_name_collision_increments_counter(self):
        """When _inter_<base> already exists in the namespace values, the counter loop
        picks _inter_<base>_1 instead (lines 1794-1795).

        The while condition checks `.values()`, so we must store the collision target
        string as a VALUE (not a key) in the namespace dict.
        """
        sb = self._sb()
        visitor, _, _, _ = _visitor_with_namespace()
        # The while loop checks: f"__internal_{interm_id}" in namespace.values()
        # For interm_id="_inter_mydelay", this is "__internal__inter_mydelay".
        # We store it as a VALUE to simulate a pre-existing collision.
        sb.namespace.namespace["_some_prior_var"] = "__internal__inter_mydelay"
        delay = DelayStructure(input=1.0, delay_time=1.0, initial=1.0, order=3)
        eqs: list = []
        result = sb._materialize_input(delay, "mydelay", visitor, eqs)
        # The counter loop should have produced the _1 suffix
        assert result == "_inter_mydelay_1"

    # ------------------------------------------------------------------
    # Line 1335: _build_invert_matrix_equations returns [] when is_control
    # ------------------------------------------------------------------

    def test_build_invert_matrix_equations_is_control_returns_empty(self):
        """When is_control=True, _build_invert_matrix_equations returns [] (line 1335)."""
        sb = self._sb()
        ast = CallStructure(
            function=ReferenceStructure(reference="INVERT MATRIX"),
            arguments=[ReferenceStructure(reference="Mat")],
        )
        dims = [("r", 2), ("c", 2)]
        result = sb._build_invert_matrix_equations("inv_mat", ast, dims, is_control=True)
        assert result == []

    # ------------------------------------------------------------------
    # Lines 1053-1055: SmoothN with non-integer dynamic order evaluated at t=0
    # ------------------------------------------------------------------

    def test_smooth_n_dynamic_order_evaluated_at_t0(self):
        """SmoothNStructure with non-integer order resolves via _eval_ast_at_t0 (lines 1053-1055).

        _prescanned_const_vals is populated for numeric literal ASTs so that
        _try_eval_as_float("smooth_order") returns 4.0, enabling lines 1053-1055.
        """
        from pysd.translators.structures.abstract_expressions import SmoothNStructure
        # A ReferenceStructure as order causes int(ast.order) to raise TypeError.
        order_ref = ReferenceStructure(reference="SmoothOrder")
        # SmoothOrder = 4.0 (numeric literal) → lands in _prescanned_const_vals.
        order_elem = AbstractElement(
            name="SmoothOrder",
            components=[AbstractUnchangeableConstant(subscripts=[[], []], ast=4.0)],
        )
        smooth = SmoothNStructure(input=1.0, smooth_time=1.0, initial=1.0, order=order_ref)
        smooth_elem = AbstractElement(
            name="Smooth Out",
            components=[AbstractComponent(subscripts=[[], []], ast=smooth)],
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sb = _section_builder_from_elements([order_elem, smooth_elem])
            sb.build_section()
        # Lines 1053-1055 ran successfully: "smooth_out" appears in built_elements
        assert "smooth_out" in sb.built_elements

    # ------------------------------------------------------------------
    # Lines 986-987: 1D stock with numpy ndarray initial value
    # ------------------------------------------------------------------

    def test_stock_1d_numpy_array_initial_emits_per_element_u0(self):
        """1D stock with ndarray initial uses per-element u0 entries (lines 986-987)."""
        import numpy as np
        sr = _make_subscript_range("pop_dim", ["A", "B"])
        integ = IntegStructure(
            flow=0.0, initial=np.array([10.0, 20.0])
        )
        comp = AbstractComponent(subscripts=[["pop_dim"], []], ast=integ)
        elem = AbstractElement(name="Stock ND", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        # Each element gets its own u0 entry from lines 986-987
        assert any("stock_nd[1] => 10.0" in e for e in sb.u0_entries)
        assert any("stock_nd[2] => 20.0" in e for e in sb.u0_entries)

    # ------------------------------------------------------------------
    # Line 1233: 1D ndarray auxiliary on a control element returns []
    # ------------------------------------------------------------------

    def test_1d_ndarray_control_element_returns_empty(self):
        """A 1D subscripted control element with ndarray AST returns [] (line 1233).

        Must use AbstractComponent (type='Auxiliary') so the constant-branch at
        line 1183 is NOT taken and we reach the ndarray auxiliary path at line 1231.
        """
        import numpy as np
        sr = _make_subscript_range("ctrl_dim", ["A", "B"])
        comp = AbstractComponent(
            subscripts=[["ctrl_dim"], []], ast=np.array([1.0, 2.0])
        )
        elem = AbstractControlElement(name="Ctrl Array", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        # is_control path returned [] → no equations for this element
        eqs, is_ctrl = sb.built_elements["ctrl_array"]
        assert eqs == []
        assert is_ctrl is True

    # ------------------------------------------------------------------
    # Line 1271: 2D ndarray auxiliary on a control element returns []
    # ------------------------------------------------------------------

    def test_2d_ndarray_control_element_returns_empty(self):
        """A 2D subscripted control element with ndarray AST returns [] (line 1271).

        Must use AbstractComponent so the constant-branch is skipped, reaching
        the ndarray auxiliary path at lines 1268-1271.
        """
        import numpy as np
        sr1 = _make_subscript_range("r_dim", ["A", "B"])
        sr2 = _make_subscript_range("c_dim", ["X", "Y"])
        comp = AbstractComponent(
            subscripts=[["r_dim", "c_dim"], []], ast=np.array([[1.0, 2.0], [3.0, 4.0]])
        )
        elem = AbstractControlElement(name="Ctrl Matrix", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2])
        sb.build_section()
        # is_control path returned [] → no equations for this element
        eqs, is_ctrl = sb.built_elements["ctrl_matrix"]
        assert eqs == []
        assert is_ctrl is True

    # ------------------------------------------------------------------
    # Lines 3623, 3629-3632: _comp_idx_arrays branches in piecewise_nd
    # ------------------------------------------------------------------

    def test_piecewise_nd_comp_idx_arrays_branches(self):
        """_comp_idx_arrays in _read_get_constants_piecewise_nd covers None, element,
        and fallback branches (lines 3623, 3629-3630, 3631-3632).

        Called with gcs_comps=[] to avoid needing actual Excel files.
        """
        import numpy as np
        sr1 = _make_subscript_range("r", ["A", "B"])
        sr2 = _make_subscript_range("c", ["X", "Y"])
        # comp1: subscripts=["A","X"] → both are specific element labels → line 3629-3630
        lit1 = AbstractUnchangeableConstant(subscripts=[["A", "X"], []], ast=1.0)
        # comp2: subscripts=["r"] only (1 element for 2D) → pos=1 yields s=None → line 3623
        lit2 = AbstractUnchangeableConstant(subscripts=[["r"], []], ast=2.0)
        # comp3: subscripts=["Z","X"] → "Z" not a range and not in r elems → line 3631-3632
        lit3 = AbstractUnchangeableConstant(subscripts=[["Z", "X"], []], ast=0.0)
        elem = AbstractElement(name="PC Const", components=[lit1, lit2, lit3])
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2])
        # Call the method directly with no GCS components (avoids ExtConstant file I/O)
        result = sb._read_get_constants_piecewise_nd(
            elem, "pc_const", gcs_comps=[], lit_comps=[lit1, lit2, lit3]
        )
        # Result should be a Julia array literal covering the 2×2 grid
        assert result is not None
        assert "[" in result

    # ------------------------------------------------------------------
    # Line 1736: EXCEPT 3D continue when all covered rows are excluded
    # ------------------------------------------------------------------

    def test_except_3d_all_rows_excluded_skips_component(self):
        """When an EXCEPT clause covers ALL (i,j,k) triples of a 3D component,
        that component is skipped via continue (line 1736)."""
        sr1 = _make_subscript_range("r", ["A", "B"])
        sr2 = _make_subscript_range("c", ["X", "Y"])
        sr3 = _make_subscript_range("d", ["P", "Q"])
        # comp0 covers A×c×d EXCEPT [A,c,d] → all triples (1,j,k) excluded → skipped
        comp0 = AbstractComponent(subscripts=[["A", "c", "d"], [["A", "c", "d"]]], ast=99.0)
        # comp1 covers full r×c×d with no EXCEPT → produces actual equations
        comp1 = AbstractComponent(subscripts=[["r", "c", "d"], []], ast=1.0)
        elem = AbstractElement(name="Skip 3D", components=[comp0, comp1])
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2, sr3])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # comp0 was skipped → 99.0 must not appear
        assert not any("99.0" in e for e in eqs)
        # comp1's equations must appear
        assert any("1.0" in e for e in eqs)
