Abstract Model Representation
=============================
The Abstract Model representation allows a separation of concern between
translation and the building. The translation will be called anything that
happens between the source code and the Abstract Model representation. While the
building will be everything that happens between the Abstract Model and the
final code.

This approach allows easily including new source codes or output codes,
without needing to make a lot of changes in the whole library. The
:py:class:`AbstractModel` object should keep as mutch information as
possible from the original model. Althought the information is not used
in the output code, it may be necessary for other future output languages
or for improvements in the currently supported outputs. For example, currently
the unchangeable constanst (== defined in Vensim) are treated as regular
components with Python, but in the future we may want to protect them
from user interaction.

The lowest level of this representation is the Abstract Syntax Tree (AST).
Which includes all the operations and calls in a given component expression.

Main abstract structures
------------------------
.. automodule:: pysd.translation.structures.abstract_model
   :members:

Abstrat structures for the AST
------------------------------
.. automodule:: pysd.translation.structures.abstract_expressions
   :members:
