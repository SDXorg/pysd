Structure of the PySD library
=============================

PySD provides translators that allow to build an original model into an Abstract Model Representation (AMR), or :doc:`Abstract Model <abstract_model>` for short. This representation allows to gather all the model equations and behavior into a number of Python data classes. Therefore, the AMR is Python code, hence independent of the source language used to write the model. The AMR is then passed to a builder, which converts it to source code of a programming language of our choice. See the example of the complete process in the figure below.

.. image:: ../images/abstract_model.png
   :width: 700 px
   :align: center

Currently, PySD can translate Vensim models (mdl format) or models in Xmile format (exported from Vensim, Stella or other software) into an AMR. The only builder available at the moment builds the models in Python.

For models translated into Python, all the necessary functions and classes to run it are incorporated in PySD. The :py:class:`Model` class is the main class that allows loading and running a model, as well as modifying the values of its parameters, among many other possibilities.

Translation
-----------

The internals of the translation process may be found in the following documents:

.. toctree::
   :maxdepth: 2

   vensim_translation
   xmile_translation
   abstract_model



PySD can import models in Vensim's \*.mdl file format and in XMILE format (\*xml file). Parsimonious is the Parsing Expression Grammar (PEG) parser library used in PySD to parse the original models and construct an abstract syntax tree. The translators then crawl the tree, using a set of classes to define a pseudo model representation called :doc:`Abstract Model <abstract_model>`.


Building the model
------------------

The builders allow you to build the final model in the desired programming language. To do so, they use a series of classes that obtain the information from the Abstract Model and convert it into the desired code. Currently PySD only includes a builder to build the models in Python. Any contribution to add new builders (and solvers) for other programming languages is welcome.

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

The Model class also contains a function for each of the model components, representing the essential model equations. Each function contains its units, subcscripts type infromation and documentation as translated from the original model file. A query to any of the model functions will calculate and return its value according to the stored state of the system.

The Model class maintains only a single state of the system in memory, meaning that all functions must obey the Markov property  - that the future state of the system can be calculated entirely based upon its current state. In addition to simplifying integration, this requirement enables analyses that interact with the model at a step-by-step level.

Lastly, the model class provides a set of methods that are used to facilitate simulation. The :py:func:`run()` method returns to the user a Pandas dataframe representing the output of their simulation run. A variety of options allow the user to specify which components of the model they would like returned, and the timestamps at which they would like those measurements. Additional parameters make parameter changes to the model, modify its starting conditions, or specify how simulation results should be logged.

.. toctree::
   :maxdepth: 2

   model_class