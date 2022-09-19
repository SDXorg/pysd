Python API
==========

.. toctree::
   :hidden:

   model_loading
   model_class
   functions

This sections describes the main functions and functionalities to translate
models to Python and run them. If you need more detailed description about
the translation and building process, please see the :doc:`../structure/structure_index` section.

The model loading information can be found in :doc:`model_loading` and consists of the following functions:

.. list-table:: Translating and loading functions
   :widths: 25 75
   :header-rows: 0

   * - :py:func:`pysd.read_vensim`
     - Translates a Vensim file to Python and returns a :py:class:`Model` object.
   * - :py:func:`pysd.read_xmile`
     - Translates a Xmile file to Python and returns a :py:class:`Model` object.
   * - :py:func:`pysd.load`
     - Loads a translated Python file and returns a :py:class:`Model` object.

The Model and Macro classes information ad public methods and attributes can be found in :doc:`model_class`.

.. list-table:: Translating and loading functions
   :widths: 25 75
   :header-rows: 0

   * - :py:class:`pysd.py_backend.model.Model`
     - Implements functionalities to load a translated model and interact with it. The :py:class:`Model` class inherits from  :py:class:`Macro`, therefore, some public methods and properties are defined in the :py:class:`Macro` class.
   * - :py:class:`pysd.py_backend.model.Macro`
     - Implements functionalities to load a translated macro and interact with it. Most of its core methods are also use by :py:class:`Model` class.


Provided functions and stateful classes to integrate python models are described in :doc:`functions`.
