Internal Functions
==================

This section documents the functions that are going on behaind the scenes, for the benefit of developers.

Special functions needed for model execution
--------------------------------------------
These functions have no direct analog in the standard python data analytics stack, or require information about the internal state of the system beyond what is present in the function call. We provide them in a structure that makes it easy for the model elements to call.

.. autoclass:: pysd.functions.Functions
    :members:
    :undoc-members:
    

Building the python model file
------------------------------
These elements are used by the translator to construct the model from the interpreted results. It is technically possible to use these functions to build a model from scratch. But - it would be rather error prone.

.. autoclass:: pysd.builder.ComponentClass
    :members:
    :undoc-members:
    
.. autofunction:: pysd.builder.new_model

.. autofunction:: pysd.builder.add_stock

.. autofunction:: pysd.builder.add_flaux

.. autofunction:: pysd.builder.add_lookup

.. autofunction:: pysd.builder.add_n_delay