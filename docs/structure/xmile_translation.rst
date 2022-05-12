Xmile Translation
=================

PySD allows parsing a Xmile file and translates the result to an :py:class:`AbstractModel` object that can be used to builde the model.


.. warning::
    Currently no Xmile users are working in the development of PySD, this is causing a gap between the Xmile and Vensim developments. Stella users are encouraged to take part in the development of PySD by invcludying new `test models <https://github.com/SDXorg/test-models>`_ and adding support for new functions and features.


The translation workflow
-------------------------
The following translation workflow allows splitting the Xmile file while parsing each part of it in order to make it possible to build an :py:class:`AbstractModel` type object. The workflow could be summarized as follows:

1. Xmile file: Parses the file with etree library and creates a section for the model.
2. Xmile section: Full set of varibles and definitions that can be integrated. Allows splitting the model elements.
3. Xmile element: A variable definition. It includes units and commnets. Allows parsing its the expressions that contains and saving them inside AbstractComponents that are part of an AbstractElement.

Once the model is parsed and broken following the previous steps. The :py:class:`AbstractModel` can be returned.


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
   :members: SubscriptRange, Element, Flaux, Gf, Stock
   :undoc-members:


Supported Functions and Features
--------------------------------

Ongoing development of the translator will support the full subset of Xmile functionality. The current release supports the following operators, functions and features.

.. warning::
   Not all the supported functions and features are properly tested. Any new test model to cover the missing functions test will be wellcome.

Operators
^^^^^^^^^
All the basic operators are supported, this includes the ones shown in the tables below.

.. csv-table:: Supported unary operators
   :file: ../tables/unary_xmile.csv
   :header-rows: 1

.. csv-table:: Supported binary operators
   :file: ../tables/binary_xmile.csv
   :header-rows: 1


Functions
^^^^^^^^^
Not all the Xmile functions all included yet, the list of supported functions are included below.

.. csv-table:: Supported basic functions
   :file: ../tables/functions_xmile.csv
   :header-rows: 1

.. csv-table:: Supported delay functions
   :file: ../tables/delay_functions_xmile.csv
   :header-rows: 1


Stocks
^^^^^^
Stocks are supported with any number of inflows and outflows. The stocks are translated to the AST as `IntegStructure(flows, initial_value)`.

Subscripts
^^^^^^^^^^
Several subscript related features all supported. This include:

- Basic subscript operations with different ranges.
- Subscript ranges and subranges definitions.

Graphical functions
^^^^^^^^^^^^^^^^^^^
Xmile graphical functions (gf), also known as lookups, are supported. They can be given hardcoded or inline.

.. warning::
   Interpolation methods 'extrapolate' and 'discrete' are implemented but not tested. Full integration models with this methods are required.

Supported in Vensim but not in Xmile
------------------------------------
Macro
^^^^^
Currently Xmile macros are not supported. In Vensim macros are classified as an independent section of the model. If they are properly parsed in the :py:class:`XmileFile` adding support for Xmile should be easy.

Planed New Functions and Features
---------------------------------
Nothing yet.
