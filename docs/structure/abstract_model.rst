Abstract Model Representation
=============================
The Abstract Model representation allows a separation of concern between
translation and building. Translation involves anything that
happens from the moment the source code of the original model is loaded
into memory up to the creation of the Abstract Model representation. Similarly,
the building will be everything that takes place between the Abstract Model and the
source code of the model written in a programming language different than that
of the original model.This approach allows to easily include new code to the translation or or building process,
without the the risk of affecting one another. 

The :py:class:`AbstractModel` object should retain as much information from the
original model as possible. Although the information is not used
in the output code, it may be necessary for other future output languages
or for improvements in the currently supported outputs. For example, currently
unchangeable constants (== defined in Vensim) are treated as regular
components with Python, but in the future we may want to protect them
from user interaction.

The lowest level of this representation is the :py:class:`AbstractSyntax` Tree (AST).
This includes all the operations and calls in a given component expression.

Main abstract structures
------------------------
.. automodule:: pysd.translators.structures.abstract_model
   :members:

Abstract structures for the AST
-------------------------------
.. automodule:: pysd.translators.structures.abstract_expressions
   :members:
