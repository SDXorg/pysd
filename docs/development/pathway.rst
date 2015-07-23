PySD Development Pathway
========================

High priority features, bugs, and other elements of active effort are listed on the `github issue
tracker. <https://github.com/JamesPHoughton/pysd/issues>`_ To get involved see :doc:`contributing`.


High Priority
-------------
* Subscripts/arrays `Github Issue Track <https://github.com/JamesPHoughton/pysd/issues/21>`_
* Refactor delays to take advantage of array architecture
* Improve translation of model documentation strings and units into python function docstrings


Medium Priority
---------------
* Outsource model translation to `SDXchange <https://github.com/SDXchange>`_ model translation toolset
* Improve model exexution speed using cython, theano, numba, or another package
* Improve performance when returning non-stock model elements


Low Priority
------------
* Import model component documentation in a way that enables doctest, to enable writing unit tests
  within the modeling environment.
* Handle simulating over timeseries
* Implement run memoization to improve speed of larger analyses
* Implement an interface for running the model over a range of conditions, build in intelligent
  parallelization.


Not Planned
-----------
* Model Construction
* Display of Model Diagrams
* Outputting models to XMILE or other formats


Ideas for Other Projects
------------------------
* SD-lint checker (units, modeling conventions, bounds/limits, etc)
* Contribution to external Data Science tools to make them more appropriate for dynamic assistant


Current Features
----------------

* Basic XMILE and Vensim parser
* Established library structure and data formats
* Simulation using existing python integration tools
* Integration with basic python Data Science functionality
* Run-at-a-time parameter modification
* Time-variant exogenous inputs
* Extended backends for storing parameters and output values
* Demonstration of integration with Machine Learning/Monte Carlo/Statistical Methods
* Python methods for programmatically manipulating SD model structure
* Turn off and on 'traces' or records of the values of variables
