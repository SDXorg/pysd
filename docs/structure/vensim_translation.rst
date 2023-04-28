Vensim Translation
==================

PySD allows parsing a Vensim `.mdl` file and translates the result to an :py:class:`AbstractModel` object that can later (building process) be used to build the model in another programming language.


Translation workflow
-------------------------
The following translation workflow allows splitting the Vensim file while parsing its contents in order to build an :py:class:`AbstractModel` object. The workflow may be summarized as follows:

1. **Vensim file**: splits the model equations from the sketch and allows splitting the model in sections (main section and macro sections).
2. **Vensim section**: is a full set of variables and definitions that is integrable. The Vensim section can then be split into model expressions.
3. **Vensim element**: a definition in the mdl file which could be a subscript (sub)range definition or a variable definition. It includes units and comments. Definitions for the same variable are grouped after in the same :py:class:`AbstractElement` object. Allows parsing its left hand side (LHS) to get the name of the subscript (sub)range or variable and it is returned as a specific type of component depending on the used assign operator (=, ==, :=, (), :)
4. **Vensim component**: the classified object for a variable definition, it depends on the operator used to define the variable. Its right hand side (RHS) can be parsed to get the Abstract Syntax Tree (AST) of the expression.

Once the model is parsed and broken following the previous steps, the :py:class:`AbstractModel` is returned.


.. image:: ../images/Vensim_file.svg
   :alt: Vensim file parts


Vensim file
^^^^^^^^^^^

.. automodule:: pysd.translators.vensim.vensim_file
   :members: VensimFile
   :undoc-members:

Vensim section
^^^^^^^^^^^^^^

.. automodule:: pysd.translators.vensim.vensim_section
   :members: Section
   :undoc-members:

Vensim element
^^^^^^^^^^^^^^

.. automodule:: pysd.translators.vensim.vensim_element
   :members: SubscriptRange, Element, Component, UnchangeableConstant, Data, Lookup
   :undoc-members:


.. _Vensim supported functions:

Supported Functions and Features
--------------------------------

Ongoing development of the translator will support the full subset of Vensim functionality. The current release supports the following operators, functions and features.

Operators
^^^^^^^^^
All the basic operators are supported, this includes the ones shown in the tables below.

.. csv-table:: Supported unary operators
   :file: ../tables/unary_vensim.csv
   :header-rows: 1

.. csv-table:: Supported binary operators
   :file: ../tables/binary_vensim.csv
   :header-rows: 1

Moreover, the Vensim :EXCEPT: operator is also supported to manage exceptions in the subscripts. See the :ref:`Subscripts section` section.


Functions
^^^^^^^^^
The list of currently supported Vensim functions are detailed below:

.. csv-table:: Supported basic functions
   :file: ../tables/functions_vensim.csv
   :header-rows: 1

.. csv-table:: Supported delay functions
   :file: ../tables/delay_functions_vensim.csv
   :header-rows: 1

.. csv-table:: Supported get functions
   :file: ../tables/get_functions_vensim.csv
   :header-rows: 1


Stocks
^^^^^^
Stocks defined in Vensim as `INTEG(flow, initial_value)` are supported and are translated to the AST as `IntegStructure(flow, initial_value)`.


.. _Subscripts section:

Subscripts
^^^^^^^^^^
Several subscript related features are also supported. These include:

- Basic subscript operations with different ranges.
- Subscript ranges and subranges definitions.
- Basic subscript mapping, where the subscript range is mapping to a full range (e.g. new_dim: A, B, C -> dim, dim_other). Mapping to a partial range is not yet supported (e.g. new_dim: A, B, C -> dim: E, F, G).
- Subscript copy (e.g. new_dim <-> dim).
- \:EXCEPT: operator with any number of arguments.
- Subscript usage as a variable (e.g. my_var[dim] = another var * dim).
- Subscript vectorial operations (e.g. SUM(my var[dim, dim!])).

Lookups
^^^^^^^
Vensim Lookups expressions are supported. They can be defined using hardcoded values, using `GET LOOKUPS` function or using `WITH LOOKUPS` function.

Data
^^^^
Data definitions with GET functions and empty data definitions (no expressions, Vensim uses a VDF file) are supported. These definitions may or may not include any of the possible interpolation keywords: :INTERPOLATE:, :LOOK FORWARD:, :HOLD BACKWARD:, :RAW:. These keywords will be stored in the 'keyword' argument of :py:class:`AbstractData` as 'interpolate', 'look_forward', 'hold_backward' and 'raw', respectively. The Abstract Structure for GET XLS/DATA is given in the supported GET functions table. The Abstract Structure for the empty Data declarations is a :py:class:`DataStructure`.

For the moment, any specific functions applying over data are supported (e.g. SHIFT IF TRUE, TIME SHIFT...), but new ones may be included in the future.

Macro
^^^^^
Vensim macros are supported. The macro content between the keywords \:MACRO: and \:END OF MACRO: is classified as a section of the model and is subsequently used to build an independent section from the rest of the model.

Planed New Functions and Features
---------------------------------


Discarded Functions and Features
--------------------------------

SHIFT IF TRUE
^^^^^^^^^^^^^
Due to the complexity of `SHIFT IF TRUE <https://www.vensim.com/documentation/fn_shift_if_true.html>`_ function and the need to mutate already computed values in another variable, the implementation of this function has been discarded for now.

However, it is possible to reproduce the same behaviour using IF THEN ELSE in the stock flow. A model with a workaround to avoid SHIFT IF TRUE and make the model work with PySD is given in :issue:`265`.
