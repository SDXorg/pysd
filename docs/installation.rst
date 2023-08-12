Installation
============

Installing via pip
------------------
To install the PySD package from the Python package index use the pip command:

.. code-block:: bash

   pip install pysd

Installing with conda
---------------------
To install PySD with conda, using the conda-forge channel, use the following command:

.. code-block:: bash

   conda install -c conda-forge pysd

Installing from source
----------------------
To install from source, clone the project with git:

.. code-block:: bash

   git clone https://github.com/SDXorg/pysd.git

or download the latest version from the project repository: https://github.com/SDXorg/pysd

In the source directory use the command:

.. code-block:: bash

   python setup.py install



Required Dependencies
---------------------
PySD requires **Python 3.7** or above.

PySD builds on the core Python data analytics stack, and the following third party libraries:

* Numpy
* Scipy
* Pandas
* Parsimonious
* xarray
* lxml
* regex
* chardet
* black
* openpyxl
* progressbar2
* portion

These modules should build automatically if you are installing via `pip`. If you are building from
source, or if pip fails to load them, they can be loaded with the same `pip` syntax as
above.


Optional Dependencies
---------------------
In order to plot model outputs as shown in :doc:`Getting started <../getting_started>`:

* Matplotlib

To export data to netCDF (*.nc*) files:

* netCDF4
* importlib-metadata==2.0 (only for Python 3.7)

To export netCDF data to comma or tab separated files with parallel processing:

* dask[array]
* dask[diagnostics]
* dask[distributed]


These Python libraries bring additional data analytics capabilities to the analysis of SD models:

* PyMC: a library for performing Markov chain Monte Carlo analysis
* Scikit-learn: a library for performing machine learning in Python
* NetworkX: a library for constructing networks
* GeoPandas: a library for manipulating geographic data

Additionally, the System Dynamics Translator utility developed by Robert Ward is useful for
translating models from other system dynamics formats into the XMILE standard, to be read by PySD.

These modules can be installed using pip with a syntax similar to the above.


Additional Resources
--------------------
The `PySD Cookbook <https://github.com/SDXorg/PySD-Cookbook>`_ contains recipes that can help you get set up with PySD.

