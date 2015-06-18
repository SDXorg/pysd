Installation
============

To install the PySD package from the Python package index into an established
Python environment, use the pip command:

   pip install pysd

To install from the source, clone the project with git:

   git clone https://github.com/JamesPHoughton/pysd.git

Or download the latest version from the project webpage: https://github.com/JamesPHoughton/pysd

In the source directory use the command

   python setup.py install



Dependencies
------------
PySD builds on the core Python data analytics stack:
* Numpy
* Scipy
* Pandas
* Matplotlib.

In addition, it calls on Parsimonious to handle translation.

Additional Python libraries that integrate well with PySD and bring additional data analytics capabilities to the analysis of SD models include

* PyMC, a library for performing Markov chain Monte Carlo analysis
* Scikit-learn, a library for performing machine learning in Python
* NetworkX22, a library for constructing networks
* GeoPandas23, a library for manipulating geographic data

Additionally, the System Dynamics Translator utility developed by Robert Ward is useful for translating models from other system dynamics formats into the XMILE standard, to be read by PySD.

These modules can be installed using pip with syntax similar to the above.