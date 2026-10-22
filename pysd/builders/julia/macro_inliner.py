"""
Expand Vensim macros in place for the Julia builder.

Every call of a macro is replaced by copies of the macro's elements, renamed
for that call, with the call's arguments in place of the macro's parameters.
The call itself becomes a reference to the copy of the macro's output.  Stocks
inside a macro then become ordinary stocks of the model, so both backends
handle them like any other stock, and a macro may call other macros.
"""
import copy
import dataclasses
from typing import Callable, Dict, List

from pysd.translators.structures.abstract_expressions import (
    CallStructure,
    ReferenceStructure,
)
from pysd.translators.structures.abstract_model import (
    AbstractElement,
    AbstractModel,
    AbstractSection,
)


def _clean(name: str) -> str:
    """Name as the translator writes it in references."""
    return name.lower().replace(" ", "_")


def _map_ast(ast: object, fn: Callable[[object], object]) -> object:
    """Rebuild *ast* bottom-up, replacing every node by ``fn(node)``."""
    if dataclasses.is_dataclass(ast) and not isinstance(ast, type):
        new = copy.copy(ast)
        for field in dataclasses.fields(ast):
            setattr(new, field.name, _map_ast(getattr(ast, field.name), fn))
        ast = new
    elif isinstance(ast, tuple):
        ast = tuple(_map_ast(item, fn) for item in ast)
    elif isinstance(ast, list):
        ast = [_map_ast(item, fn) for item in ast]
    return fn(ast)


class _MacroExpander:
    def __init__(self, macros: Dict[str, AbstractSection]):
        self.macros = macros
        self.new_elements: List[AbstractElement] = []
        self._calls: Dict[tuple, int] = {}

    def expand(self, ast: object, owner: str) -> object:
        """*ast* of element *owner* with every macro call expanded."""
        return _map_ast(ast, lambda node: self._expand_call(node, owner))

    def _expand_call(self, node: object, owner: str) -> object:
        if not (isinstance(node, CallStructure)
                and isinstance(node.function, ReferenceStructure)
                and node.function.reference in self.macros):
            return node
        macro = self.macros[node.function.reference]
        key = (owner, macro.name)
        self._calls[key] = self._calls.get(key, 0) + 1
        prefix = f"{owner}_{macro.name}"
        if self._calls[key] > 1:
            prefix += f"_{self._calls[key]}"

        # the arguments are already expanded (bottom-up)
        params = dict(zip(macro.params, node.arguments))
        local = {
            _clean(element.name): f"{prefix}_{_clean(element.name)}"
            for element in macro.elements
        }

        def substitute(ref: object) -> object:
            if not isinstance(ref, ReferenceStructure):
                return ref
            if ref.reference in params:
                arg = params[ref.reference]
                if (ref.subscripts and isinstance(arg, ReferenceStructure)
                        and not arg.subscripts):
                    return dataclasses.replace(arg, subscripts=ref.subscripts)
                return arg
            if ref.reference in local:
                return dataclasses.replace(ref, reference=local[ref.reference])
            return ref

        for element in macro.elements:
            name = local[_clean(element.name)]
            new_element = copy.copy(element)
            new_element.name = name
            new_element.components = []
            for component in element.components:
                new_component = copy.copy(component)
                new_component.ast = self.expand(
                    _map_ast(component.ast, substitute), name)
                new_element.components.append(new_component)
            self.new_elements.append(new_element)

        return ReferenceStructure(local[_clean(macro.name)])


def inline_macros(model: AbstractModel) -> AbstractModel:
    """*model* with its macros expanded into the main section."""
    main, *macro_sections = model.sections
    if not macro_sections:
        return model
    expander = _MacroExpander({_clean(s.name): s for s in macro_sections})
    elements = []
    for element in main.elements:
        new_element = copy.copy(element)
        new_element.components = []
        for component in element.components:
            new_component = copy.copy(component)
            new_component.ast = expander.expand(
                component.ast, _clean(element.name))
            new_element.components.append(new_component)
        elements.append(new_element)
    new_main = copy.copy(main)
    new_main.elements = elements + expander.new_elements
    new_model = copy.copy(model)
    new_model.sections = (new_main,)
    return new_model
