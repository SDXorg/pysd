Xmile Translation
=================

PySD allows parsing a Xmile file and translates the result to an :py:class:`AbstractModel` object that can be used to build the model.


.. warning::
    Currently no Xmile users are working on the development of PySD. This is causing a gap between the Xmile and Vensim developments. Stella users are encouraged to take part in the development of PySD by including new `test models <https://github.com/SDXorg/test-models>`_ and adding support for new functions and features.


The translation workflow
-------------------------
The following translation workflow allows splitting the Xmile file while parsing each part of it to build an :py:class:`AbstractModel` type object. The workflow may be summarized as follows:

1. **Xmile file**: Parses the file with etree library and creates a section for the model.
2. **Xmile section**: Full set of variables and definitions that can be integrated. Allows splitting the model elements.
3. **Xmile element**: A variable definition. It includes units and comments. Allows parsing the expressions it contains and saving them inside AbstractComponents, that are part of an AbstractElement.

Once the model is parsed and split following the previous steps. The :py:class:`AbstractModel` can be returned.


Xmile file
^^^^^^^^^^

.. automodule:: pysd.translators.xmile.xmile_file
   :members: XmileFile
   :undoc-members:

Xmile section
^^^^^^^^^^^^^

.. automodule:: pysd.translators.xmile.xmile_section
   :members: Section
   :undoc-members:

Xmile element
^^^^^^^^^^^^^

.. automodule:: pysd.translators.xmile.xmile_element
   :members: SubscriptRange, Element, Aux, Flow, Gf, Stock
   :undoc-members:


.. _Xmile supported functions:

Supported Functions and Features
--------------------------------

Ongoing development of the translator will support the full set of Xmile functionality. The current release supports the following operators, functions and features:

.. warning::
   Not all the supported functions and features are properly tested. Any new test model to cover the missing functions test will be welcome.

Operators
^^^^^^^^^
All the basic operators are supported, this includes the ones shown in the tables below.:

.. csv-table:: Supported unary operators
   :file: ../tables/unary_xmile.csv
   :header-rows: 1

.. csv-table:: Supported binary operators
   :file: ../tables/binary_xmile.csv
   :header-rows: 1


Functions
^^^^^^^^^
Not all the Xmile functions are included yet. The list of supported functions is shown below:

.. csv-table:: Supported basic functions
   :file: ../tables/functions_xmile.csv
   :header-rows: 1

.. csv-table:: Supported delay functions
   :file: ../tables/delay_functions_xmile.csv
   :header-rows: 1


Stocks
^^^^^^
Stocks are supported with any number of inflows and outflows. Stocks are translated to the AST as `IntegStructure(flows, initial_value, non_negative)`. Non-negative flag is parsed for both stocks and flows, this can be set element by element or using the `behavior section <http://docs.oasis-open.org/xmile/xmile/v1.0/errata01/csprd01/xmile-v1.0-errata01-csprd01-complete.html#_Toc442104128>`_. Flows with non-negative flags are read as flows with a maximum condition, while for stocks this information is saved in the :py:class:`pysd.translators.structures.abstract_expressions.IntegStructure` object.

Subscripts
^^^^^^^^^^
Several subscript related features are supported. These include:

- Basic subscript operations with different ranges.
- Subscript ranges and subranges definitions.

Graphical functions
^^^^^^^^^^^^^^^^^^^
Xmile graphical functions (gf), also known as lookups, are supported. They can be hardcoded or inlined.

.. warning::
   Interpolation methods 'extrapolate' and 'discrete' are implemented but not tested. Full integration models with these methods are required.

Supported in Vensim but not in Xmile
------------------------------------
Macro
^^^^^
Currently Xmile macros are not supported. In Vensim, macros are classified as an independent section of the model. If they are properly parsed in the :py:class:`XmileFile`, adding support for Xmile should be easy.

Planed New Functions and Features
---------------------------------
Nothing yet.
