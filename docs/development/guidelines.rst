Development Guidelines
======================


If you are interested in helping to develop PySD, the :doc:`pathway` lists areas that are ripe
for contribution.

To get started, you can fork the repository and make contributions to your own version.
When you are happy with your edits, submit a pull request to the main branch.

.. note::
  In order to open a pull request, the new features and changes should be througly tested.
  To do so, unit tests of new features or translated functions should be added, please check the Development Tools section below. When opening a pull request all tests are run and the coverage and pep8 style are checked.

Development Tools
-----------------
There are a number of tools that you might find helpful in development:

Test Suite
^^^^^^^^^^
PySD uses the common model test suite found `on github <https://github.com/SDXorg/test-models>`_
which are run using `pytest_integration_test_vensim_pathway.py`  and `pytest_integration_test_xmile_pathway.py`.
PySD also has own tests for internal funtionality, `pytest_*.py` files
of the `/tests/` directory.

In order to run all the tests :py:mod:`pytest` should be used.
A `Makefile` is given to run easier the tests with :py:mod:`pytest`, check
`tests/README <https://github.com/JamesPHoughton/pysd/tree/master/tests/README.md>`_
for more information.

These tests run quickly and should be executed when any changes are made to ensure
that current functionality remains intact. If any new functionality is added or a
bug is corrected, the tests should be updated with new models in test suite or
complementary tests in the corresponding `pytest_*.py` file.

.. note::
  If your changes correct some existing bug related to the translation or running
  of a Vensim (or Xmile) model. You should add a new test in the `test suite repo <https://github.com/SDXorg/test-models>`_ reproducing the solved bug and addthe necessary lines in `pytest_integration/pytest_integration_test_vensim_pathway.py` (or `pytest_integration/pytest_integration_test_xmile_pathway.py`) to run the new test. Then, it is encoraged to add also unit test in `pytest_translators` reproducing the translation of the new function and test of the workflow in
  `pytest_types/functions/pytest_functions.py` (or `pytest_types/statefuls/pytest_statefuls.py`).

Speed Tests
^^^^^^^^^^^
Speed tests may be developed in the future. Any contribution is welcome.


Profiler
^^^^^^^^
Profiling the code can help to identify bottlenecks in operation. To understand how changes to the
code influence its speed, we should construct a profiling test that executes the PySD components in
question.

The profiler depends on :py:mod:`cProfile` and `cprofilev <https://github.com/ymichael/cprofilev>`_


Python Linter
^^^^^^^^^^^^^
`Pylint <http://docs.pylint.org/>`_ is a module that checks that your code meets proper Python
coding practices. It is helpful for making sure that the code will be easy for other people to read,
and also is good fast feedback for improving your coding practice. The lint checker can be run for
the entire packages, and for individual Python modules or classes. It should be run at a local level
(ie, on specific files) whenever changes are made, and globally before the package is committed.
It doesn't need to be perfect, but we should aspire always to move in a positive direction.'


PySD Design Philosophy
----------------------
Understanding that a focussed project is both more robust and maintainable, PySD adheres to the
following philosophy:


* Do as little as possible.

 * Anything that is not endemic to System Dynamics (such as plotting, integration, fitting, etc)
   should either be implemented using external tools, or omitted.
 * Stick to SD. Let other disciplines (ABM, Discrete Event Simulation, etc) create their own tools.
 * Use external model creation tools.

* Use the core language of System Dynamics.

 * Limit implementation to the basic XMILE standard.
 * Resist the urge to include everything that shows up in all vendors' tools.

* Emphasize ease of use. Let SD practitioners who haven't used Python before understand the basics.
* Take advantage of general Python constructions and best practices.
* Develop and use strong testing and profiling components. Share your work early. Find bugs early.
* Avoid firefighting or rushing to add features quickly. SD knows enough about short term thinking
  in software development to know where that path leads.
