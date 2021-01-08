Installation
============

Installing via pip
------------------
To install the PySD package from the Python package index into an established
Python environment, use the pip command:

.. code-block:: bash

   pip install pysd


Installing from source
----------------------
To install from the source, clone the project with git:

.. code-block:: bash

   git clone https://github.com/JamesPHoughton/pysd.git

Or download the latest version from the project webpage: https://github.com/JamesPHoughton/pysd

In the source directory use the command

.. code-block:: bash

   python setup.py install



Required Dependencies
---------------------
PySD was originally built on python 2.7. Hoewever, the last version requires at least **python 3.7**.

PySD calls on the core Python data analytics stack, and a third party parsing library:

* Numpy
* Scipy
* Pandas
* Matplotlib
* Parsimonious
* black
* openpyxl

These modules should build automatically if you are installing via `pip`. If you are building from
the source code, or if pip fails to load them, they can be loaded with the same `pip` syntax as
above.


Optional Dependencies
---------------------
These Python libraries bring additional data analytics capabilities to the analysis of SD models:

* PyMC: a library for performing Markov chain Monte Carlo analysis
* Scikit-learn: a library for performing machine learning in Python
* NetworkX: a library for constructing networks
* GeoPandas: a library for manipulating geographic data

Additionally, the System Dynamics Translator utility developed by Robert Ward is useful for
translating models from other system dynamics formats into the XMILE standard, to be read by PySD.

These modules can be installed using pip with syntax similar to the above.


Additional Resources
--------------------
The `PySD Cookbook <https://github.com/JamesPHoughton/PySD-Cookbook>`_ contains recipes that can help you get set up with PySD.

