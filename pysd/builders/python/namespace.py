import re

from unicodedata import normalize
from typing import List

# used to create Python safe names with the variable reserved_words
from keyword import kwlist
from builtins import __dir__ as bidir
from pysd.py_backend.allocation import __dir__ as adir
from pysd.py_backend.cache import __dir__ as cadir
from pysd.py_backend.components import __dir__ as cdir
from pysd.py_backend.data import __dir__ as ddir
from pysd.py_backend.external import __dir__ as edir
from pysd.py_backend.functions import __dir__ as fdir
from pysd.py_backend.statefuls import __dir__ as sdir
from pysd.py_backend.utils import __dir__ as udir


class NamespaceManager:
    """
    NamespaceManager object allows includying new elements to the namespace
    and searching for elements in the namespace. When includying new
    elements a Python safe name is used to be able to write the equations.

    Parameters
    ----------
    parameters: list (optional)
        List of the parameters that are used as argument in the Macro.
        By defaukt it is an empty list.

    """
    _reserved_words = set(
        dir() + adir() + bidir() + cadir() + cdir() + ddir() + edir()
        + fdir() + sdir() + udir()).union(kwlist)

    def __init__(self, parameters: List[str] = []):
        self._used_words = self._reserved_words.copy()
        # inlcude time to the namespace
        self.namespace = {"Time": "time"}
        # include time to the cleanspace (case and whitespace/underscore
        # insensitive namespace)
        self.cleanspace = {"time": "time"}
        for parameter in parameters:
            self.add_to_namespace(parameter)

    def add_to_namespace(self, string: str) -> None:
        """
        Add a new string to the namespace.

        Parameters
        ----------
        string: str
            String to add to the namespace.

        Returns
        -------
        None

        """
        self.make_python_identifier(string, add_to_namespace=True)

    def make_python_identifier(self, string: str, prefix: str = None,
                               add_to_namespace: bool = False) -> str:
        """
        Takes an arbitrary string and creates a valid Python identifier.

        If the Python identifier created is already in the namespace,
        but the input string is not (ie, two similar strings resolve to
        the same Python identifier) or if the identifier is a reserved
        word in the reserved_words list, or is a Python default
        reserved word, adds _1, or if _1 is in the namespace, _2, etc.

        Parameters
        ----------
        string: str
            The text to be converted into a valid Python identifier.

        prefix: str or None (optional)
            If given it will be used as a prefix for the output string.
            Default is None.

        add_to_namespace: bool (optional)
            If True it will add the passed string to the namespace and
            to the cleanspace. Default is False.

        Returns
        -------
        identifier: str
            A vaild Python identifier based on the input string.

        Examples
        --------
        >>> make_python_identifier('Capital')
        'capital'

        >>> make_python_identifier('multiple words')
        'multiple_words'

        >>> make_python_identifier('multiple     spaces')
        'multiple_spaces'

        When the name is a Python keyword, add '_1' to differentiate it
        >>> make_python_identifier('for')
        'for_1'

        Remove leading and trailing whitespace
        >>> make_python_identifier('  whitespace  ')
        'whitespace'

        Remove most special characters outright:
        >>> make_python_identifier('H@t tr!ck')
        'ht_trck'

        add valid string to leading digits
        >>> make_python_identifier('123abc')
        'nvs_123abc'

        already in namespace
        >>> make_python_identifier('Var$')  # namespace={'Var$': 'var'}
        'var'

        namespace conflicts
        >>> make_python_identifier('Var@')  # namespace={'Var$': 'var'}
        'var_1'

        >>> make_python_identifier('Var$')  # namespace={'Var@': 'var',
        ...                                              'Var%':'var_1'}
        'var_2'

        References
        ----------
        Identifiers must follow the convention outlined here:
            https://docs.python.org/2/reference/lexical_analysis.html#identifiers

        """
        s = string.lower()
        clean_s = s.replace(" ", "_")

        # Make spaces into underscores
        s = re.sub(r"[\s\t\n_]+", "_", s)

        # remove accents, diaeresis and others ó -> o
        s = normalize("NFD", s).encode("ascii", "ignore").decode("utf-8")

        # Remove invalid characters
        s = re.sub(r"[^0-9a-zA-Z_]", "", s)

        # If leading character is not a letter add nvs_.
        # Only letters can be leading characters.
        if prefix is not None:
            s = prefix + "_" + s
        elif re.findall(r"^[0-9]", s) or not s:
            s = "nvs_" + s
        elif re.findall(r"^_", s):
            s = "nvs" + s

        # replace multiple _ after cleaning
        s = re.sub(r"[_]+", "_", s)

        # Check that the string is not a Python identifier
        identifier = s
        i = 1
        while identifier in self._used_words:
            identifier = s + '_' + str(i)
            i += 1

        # include the word in used words to avoid using it againg
        self._used_words.add(identifier)

        if add_to_namespace:
            # include word to the namespace
            self.namespace[string] = identifier
            self.cleanspace[clean_s] = identifier

        return identifier

    def get_original_name(self, identifier):
        """
        Search for the original name of a variable's Python identifier.

        Parameters
        ----------
        identifier: str
            It should be a value in the namespace.

        Rerturns
        --------
        original_name: str
            The original name of the variable.

        """
        for key, value in self.namespace.items():
            if value == identifier:
                return key

        raise ValueError(f"'{identifier}' not found in the namespace.")
