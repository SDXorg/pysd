Vensim Translation
==================

PySD  allows parsing a Vensim '.mdl' file and translates the result to an :py:class:`AbstractModel` object that can be used to builde the model.


The translation workflow
-------------------------
The following translation workflow allows splitting the Vensim file while parsing each part of it in order to make it possible to build an :py:class:`AbstractModel` type object. The workflow could be summarized as follows:

1. Vensim file: Splits the file content from the sketch and allows splitting the model in sections (main section, macro section)
2. Vensim section: Full set of varibles and definitions that can be integrated. Allows splitting the model expressions.
3. Vensim element: A definition in the mdl file which could be a subscript (sub)range definition or a variable definition. It includes units and comments. Definitions for the same variable are grouped after in the same :py:class:`AbstractElement` object. Allows parsing its left hand side (LHS) to get the name of the subscript (sub)range or variable and it is returned as a specific type of component depending on the used assing operator (=, ==, :=, (), :)
4. Vensim component: The classified object for a variable definition, it depends on the opperator used to define the variable. Its right hand side (RHS) can be parsed to get the Abstract Syntax Tree (AST) of the expression.

Once the model is parsed and broken following the previous steps. The :py:class:`AbstractModel` can be returned.


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

Moreover, the Vensim :EXCEPT: operator is also supported to manage exceptions in the subscripts. See the :ref:`Subscripts section`.

Functions
^^^^^^^^^
Not all the Vensim functions all included yet, the list of supported functions are included below.

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
Several subscript related features all supported. This include:

- Basic subscript operations with different ranges.
- Subscript ranges and subranges definitions.
- Basic subscript mapping where the subscript range is mapping to a full range (e.g. new_dim: A, B, C -> dim, dim_other), mapping to a partial range is not supported yet (e.g. new_dim: A, B, C -> dim: E, F, G).
- Subscript copy (e.g. new_dim <-> dim).
- \:EXCEPT: operator with any number of arguments.
- Subscript usage as a variable (e.g. my_var[dim] = another var * dim).
- Subscript vectorial opperations (e.g. SUM(my var[dim, dim!])).

Lookups
^^^^^^^
Vensim Lookups expressions are supported they can be given hardcoded, using `GET LOOKUPS` function or using `WITH LOOKUPS` function.

Data
^^^^
Data definitions from GET functions and empty definitions (no expressions, Vensim uses a VDF file) are supported. This definitions can include or not any of the possible interpolation keywords: :INTERPOLATE:, :LOOK FORWARD:, :HOLD BACKWARD:, :RAW:. This keywords will be stored in the 'keyword' argument of :py:class:`AbstractData` as 'interpolate', 'look_forward', 'hold_backward' and 'raw', respectively. The Abstract Structure for GET XLS/DATA is given in the supported get functions table, the Abstract Structure for the empty Data declarations is a :py:class:`DataStructure`.

For the moment, any specific functions applying over data are supported (e.g. SHIFT IF TRUE, TIME SHIFT...), but they may be includded in the future.

Macro
^^^^^
Vensim macros are supported, The macro content between the keywords \:MACRO: and \:END OF MACRO: is classified as a section of the model and used for latelly build an independent section from the rest of the model.

Planed New Functions and Features
---------------------------------
- ALLOCATE BY PRIORITY
- GET TIME VALUE
- SHIFT IF TRUE
- VECTOR SELECT
