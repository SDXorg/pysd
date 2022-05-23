PySD Development Pathway
========================

High priority features, bugs, and other elements of active effort are listed on the `github issue
tracker. <https://github.com/JamesPHoughton/pysd/issues>`_ To get involved see :doc:`guidelines`.


High Priority
-------------
* Improve running speed using numpy.arrays instead of xarray.DataArrays
* Adding unit and full tests for Xmile translation
* Subscripts/arrays support for Xmile models


Medium Priority
---------------
* Improve model execution speed using cython, theano, numba, or another package


Low Priority
------------
* Import model component documentation in a way that enables doctest, to enable writing unit tests within the modeling environment
* Handle simulating over timeseries
* Implement run memoization to improve speed of larger analyses
* Implement an interface for running the model over a range of conditions, build in intelligent parallelization.


Not Planned
-----------
* Model Construction
* Outputting models to XMILE or other formats


Ideas for Other Projects
------------------------
* SD-lint checker (units, modeling conventions, bounds/limits, etc)
* Contribution to external Data Science tools to make them more appropriate for dynamic assistant


Current Features
----------------

* Basic XMILE and Vensim parser
* Established library structure and data formats
* Simulation using existing Python integration tools
* Integration with basic Python Data Science functionality
* Run-at-a-time parameter modification
* Time-variant exogenous inputs
* Extended backends for storing parameters and output values
* Demonstration of integration with Machine Learning/Monte Carlo/Statistical Methods
* Python methods for programmatically manipulating SD model structure
* Turn off and on 'traces' or records of the values of variables
