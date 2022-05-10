Structure of the PySD module
============================

PySD provides a set of translators that allow to build an original model into an abstract model representation (AMR), also called Abstract Model. This representation is based on a series of Python classes that allow to have a version of the model independent of the source language which classifies its elements depending on their type and expresses the mathematical formulations in an abstract syntax tree. This representation can be used by a builder, which allows to write the final functional language in another programming language. See the example of the complete process in the figure below.

.. image:: ../images/abstract_model.png
   :width: 700 px
   :align: center

Currently, PYSD can translate Vensim models (mdl format) or models in Xmile format (exported from Vensim, Stella or other software). The only builder available at the moment builds the models in Python.

For models translated into Python, all the necessary functions and classes are incorporated in this library so that they can be executed. The Model class is the main class that allows loading and running a model, as well as modifying the values of its parameters, among many other possibilities.

Translation
-----------

The internal functions of the model translation components and relevant objects can be seen in the following documents:

.. toctree::
   :maxdepth: 2

   vensim_translation
   xmile_translation
   abstract_model



The PySD module is capable of importing models from a Vensim model file (\*.mdl) or an XMILE format xml file. Translation makes use of a Parsing Expression Grammar parser, using the third party Python library Parsimonious to construct an abstract syntax tree based upon the full model file (in the case of Vensim) or individual expressions (in the case of XMILE). The translators then crawl the tree, using a set of classes to define a pseudo model representation called :doc:`Abstract Model <abstract_model>`.


Building the model
------------------

The builders allow you to build the final model in the desired language. To do so, they use a series of classes that subtract the information from the abstract model and convert it into the desired code. Currently there is only one builder to build the models in Python, any contribution to add new builders is welcome.

.. toctree::
   :maxdepth: 2

   python_builder

The Python model
----------------
For loading a translated model with Python see :doc:`basic usage <../basic_usage>` or:

.. toctree::
   :maxdepth: 2

   model_loading

The Python builder constructs a Python class that represents the system dynamics model. The class maintains a dictionary representing the current values of each of the system stocks, and the current simulation time, making it a `statefull` model in much the same way that the system itself has a specific state at any point in time.

The model class also contains a function for each of the model components, representing the essential model equations. The each function contains its units, subcscripts type infromation and documentation as translated from the original model file. A query to any of the model functions will calculate and return its value according to the stored state of the system.

The model class maintains only a single state of the system in memory, meaning that all functions must obey the Markov property  - that the future state of the system can be calculated entirely based upon its current state. In addition to simplifying integration, this requirement enables analyses that interact with the model at a step-by-step level.

Lastly, the model class provides a set of methods that are used to facilitate simulation. The .run() function returns to the user a Pandas dataframe representing the output of their simulation run. A variety of options allow the user to specify which components of the model they would like returned, and the timestamps at which they would like those measurements. Additional parameters make parameter changes to the model, modify its starting conditions, or specify how simulation results should be logged.

.. toctree::
   :maxdepth: 2

   model_class