Python builder
==============

The Python builder allows to build models that can be run with the PySD Model class.

The use of a one-to-one dictionary in translation means that the breadth of functionality is inherently limited. In the case where no direct Python equivalent is available, PySD provides a library of functions such as `pulse`, `step`, etc. that are specific to dynamic model behavior.

In addition to translating individual commands between Vensim/XMILE and Python, PySD reworks component identifiers to be Python-safe by replacing spaces with underscores. The translator allows source identifiers to make use of alphanumeric characters, spaces, or the $ symbol.

Main builders
-------------

.. automodule:: pysd.building.python.python_model_builder
   :members:
   :undoc-members:

Expression builders
-------------------
.. automodule:: pysd.building.python.python_expressions_builder
   :members:
   :undoc-members:


Namespace manager
-----------------

.. automodule:: pysd.building.python.namespace
   :members:
   :undoc-members:


Subscript manager
-----------------
.. automodule:: pysd.building.python.subscripts
   :members:
   :undoc-members:


Imports manager
---------------
.. automodule:: pysd.building.python.imports
   :members:
   :undoc-members:
