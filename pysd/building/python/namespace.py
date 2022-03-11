import re

from unicodedata import normalize

# used to create python safe names with the variable reserved_words
from keyword import kwlist
from builtins import __dir__ as bidir
from pysd.py_backend.components import __dir__ as cdir
from pysd.py_backend.data import __dir__ as ddir
from pysd.py_backend.decorators import __dir__ as dedir
from pysd.py_backend.external import __dir__ as edir
from pysd.py_backend.functions import __dir__ as fdir
from pysd.py_backend.statefuls import __dir__ as sdir
from pysd.py_backend.utils import __dir__ as udir


class NamespaceManager:
    reserved_words = set(
        dir() + bidir() + cdir() + ddir() + dedir() + edir() + fdir()
        + sdir() + udir()).union(kwlist)

    def __init__(self, parameters=[]):
        self.used_words = self.reserved_words.copy()
        self.namespace = {"Time": "time"}
        self.cleanspace = {"time": "time"}
        for parameter in parameters:
            self.add_to_namespace(parameter)

    def add_to_namespace(self, string):
        self.make_python_identifier(string, add_to_namespace=True)

    def make_python_identifier(self, string, prefix=None, add_to_namespace=False):
        """
        Takes an arbitrary string and creates a valid Python identifier.

        If the input string is in the namespace, return its value.

        If the python identifier created is already in the namespace,
        but the input string is not (ie, two similar strings resolve to
        the same python identifier)

        or if the identifier is a reserved word in the reserved_words
        list, or is a python default reserved word,
        adds _1, or if _1 is in the namespace, _2, etc.

        Parameters
        ----------
        string: str
            The text to be converted into a valid python identifier.

        namespace: dict
            Map of existing translations into python safe identifiers.
            This is to ensure that two strings are not translated into
            the same python identifier. If string is already in the namespace
            its value will be returned. Otherwise, namespace will be mutated
            adding string as a new key and its value.

        Returns
        -------
        identifier: str
            A vaild python identifier based on the input string.

        Examples
        --------
        >>> make_python_identifier('Capital')
        'capital'

        >>> make_python_identifier('multiple words')
        'multiple_words'

        >>> make_python_identifier('multiple     spaces')
        'multiple_spaces'

        When the name is a python keyword, add '_1' to differentiate it
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
        >>> make_python_identifier('Var$', namespace={'Var$': 'var'})
        'var'

        namespace conflicts
        >>> make_python_identifier('Var@', namespace={'Var$': 'var'})
        'var_1'

        >>> make_python_identifier('Var$', namespace={'Var@': 'var',
        ...                                           'Var%':'var_1'})
        'var_2'

        References
        ----------
        Identifiers must follow the convention outlined here:
            https://docs.python.org/2/reference/lexical_analysis.html#identifiers

        """
        s = string.lower()
        clean_s = s.replace(" ", "_")

        if prefix is None and clean_s in self.cleanspace:
            return self.cleanspace[clean_s]

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

        # Check that the string is not a python identifier
        identifier = s
        i = 1
        while identifier in self.used_words:
            identifier = s + '_' + str(i)
            i += 1

        self.used_words.add(identifier)

        if add_to_namespace:
            self.namespace[string] = identifier
            self.cleanspace[clean_s] = identifier

        return identifier
