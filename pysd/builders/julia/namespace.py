"""
Julia namespace manager: maps Vensim variable names to valid Julia identifiers.
"""
import re
from typing import Dict, Optional

# Julia reserved keywords (https://docs.julialang.org/en/v1/base/base/#Keywords)
JULIA_KEYWORDS = frozenset([
    "baremodule", "begin", "break", "catch", "const", "continue", "do",
    "else", "elseif", "end", "export", "false", "finally", "for",
    "function", "global", "if", "import", "importall", "in", "isa",
    "let", "local", "macro", "module", "mutable", "outer", "primitive",
    "quote", "return", "struct", "true", "try", "type", "using",
    "where", "while", "abstract",
])


class JuliaNamespaceManager:
    """Manages the mapping from Vensim variable names to Julia identifiers.

    Vensim names are case-insensitive and may contain spaces and special
    characters.  This manager produces valid, collision-free Julia identifiers
    and supports case-insensitive lookup via a secondary ``cleanspace`` dict.
    """

    def __init__(self) -> None:
        # original Vensim name -> Julia identifier
        self.namespace: Dict[str, str] = {"Time": "t"}
        # cleaned (lowercase + non-alnum → '_') version -> original Vensim name
        self.cleanspace: Dict[str, str] = {"time": "Time"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_to_namespace(self, name: str) -> str:
        """Register *name* and return its Julia identifier.

        If *name* was already registered, the existing identifier is returned.
        """
        if name in self.namespace:
            return self.namespace[name]

        identifier = self._make_identifier(name)
        self.namespace[name] = identifier
        self.cleanspace[_clean(name)] = name
        return identifier

    def get(self, name: str) -> Optional[str]:
        """Return the Julia identifier for *name* (case-insensitive).

        Returns ``None`` if the name has not been registered.
        """
        if name in self.namespace:
            return self.namespace[name]
        original = self.cleanspace.get(_clean(name))
        if original is not None:
            return self.namespace.get(original)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_identifier(self, name: str) -> str:
        """Convert a Vensim name to a unique, valid Julia identifier."""
        # Replace non-alphanumeric characters with underscores, use lowercase
        ident = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower())
        # Collapse runs of underscores and strip leading/trailing ones
        ident = re.sub(r"_+", "_", ident).strip("_")
        # Must start with a letter or underscore
        if not ident:
            ident = "_var"
        elif ident[0].isdigit():
            ident = "_" + ident
        # Avoid Julia reserved words
        if ident in JULIA_KEYWORDS:
            ident = ident + "_var"
        # Resolve collisions with already-registered identifiers
        existing = set(self.namespace.values())
        if ident in existing:
            base, i = ident, 1
            while f"{base}_{i}" in existing:
                i += 1
            ident = f"{base}_{i}"
        return ident


def _clean(name: str) -> str:
    """Normalise a name for case-insensitive comparison."""
    return re.sub(r"[^a-z0-9]", "_", name.lower())
