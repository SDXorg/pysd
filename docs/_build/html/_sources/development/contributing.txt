Contributing to PySD
====================


If you are interested in helping to develop PySD, there are a number of tools that you might find
helpful. The :doc:`pathway` lists areas that are ripe for contribution.

Test Suite
----------
The test suite is found in the main repository in the `test_pysd.py` module. These tests run
quickly and should be executed when any changes are made to ensure that current functionality
remains intact.

The tests outlined in this file depend on a set of models present in the `tests` directory of the
main repository that are categorized by model source.

The test suite depends on the standard python :py:mod:`unittest` library.


Speed Tests
-----------
A set of speed tests are found in the `speed_test.py` module. These speed tests help understand how
changes to the PySD module influence the speed of execution. These tests take a little longer to run
than the basic test suite, but are not onerous. They should be run before any submission to the
repository.

The speed test results are appended to 'speedtest_results.json', along with version and date
information, so that before and after comparisons can be made.

The speed tests depend on the standard python :py:mod:`timeit` library.

Profiler
--------
Profiling the code can help to identify bottlenecks in operation. To understand how changes to the
code influence its speed, we should construct a profiling test that executes the PySD components in
question. The file 'profile_pysd.py' gives an example for how this profiling can be conducted, and
the file 'run_profiler.sh' executes the profiler and launches a view of the results that can be
explored in the browser.

The profiler depends on :py:mod:`cProfile` and `cprofilev<https://github.com/ymichael/cprofilev>`_

