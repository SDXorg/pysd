Adding new functions
====================
In this section you may found some helpful examples for adding a new function to the PySD Python builder. Before starting adding any new feature or fuction, please, make sure that no one is working on it. Search if any open issue exists with the feature you want to work on or open a new one if it does not exist. Then, claim that you are working on it.

Adding a hardcoded function
---------------------------
The most simple cases are when the existing Abstract Structure :py:class:`pysd.translators.structures.abstract_expressions.CallStructure` can be used. This structure holds a reference to the function name and the passed arguments in a :py:class:`tuple`. Sometimes the function can be directly added to the model file without the needing of defining any specific function. This can be done when the function is already implemented in Python or a Python library, and the behaviour is the same. For example, `Vensim's ABS <https://www.vensim.com/documentation/fn_abs.html>`_ and `XMILE's ABS <http://docs.oasis-open.org/xmile/xmile/v1.0/xmile-v1.0.pdf#page=30>`_ functions can be replaced by :py:func:`numpy.abs`.

In this case we only need to include the translation in the :py:data:`functionspace` dictionary from :py:mod:`pysd.builders.python.python_functions.py`::

    "abs": ("np.abs(%(0)s)", ("numpy",)),

They key (:py:data:`"abs"`) is the name of the Vensim/XMILE function in lowercase. The first argument in the value (:py:data:`"np.abs(%(0)s)"`) is the python repressentation, the :py:data:`%(0)s` standd for the first argument of the original function. The last arguments stands for the dependencies of that function, in this case the used functions is included in :py:mod:`numpy` module. Hence, we need to import `numpy` in our model file, which is done by adding the dependency :py:data:`("numpy",)`, note that the dependency is a tuple.

The next step is to test the new function, in order to do that we need to include integration tests in the `test-models repo <https://github.com/SDXorg/test-models>`_, please follow the instructions to add a new test in the `README of that repo <https://github.com/SDXorg/test-models/blob/master/README.md>`_. In this case, we would need to add test models for Vensim's `mdl`file and a XMILE file, as we are adding support for both. The tests should cover all the possible cases, in this case we should test the absolute of positive and negative floats and positive, negative and mixed arrays. In this case, we included the tests `test-models/tests/abs/test_abs.mdl` and `test-models/tests/abs/test_abs.xmile`, with their corresponding outputs file. Now we include the test in the testing script. We need to add the following entry in the :py:data:`vensim_test` dictionary of :py:mod:`tests/pytest_integration/pytest_integration_test_vensim_pathway.py`::

    "abs": {
        "folder": "abs",
        "file": "test_abs.mdl"
    },

and the following one in the :py:data:`xmile_test` dictionary of :py:mod:`tests/pytest_integration/pytest_integration_test_xmile_pathway.py`::

    "abs": {
        "folder": "abs",
        "file": "test_abs.xmile"
    },

At this point we should be able to run the test and, if the implementation was correctly done, pass it. We need to make sure that we did not break any other feature by running all the tests.

In order to finish the contribution, we should update the documentation. The tables of :ref:`supported Vensim functions <Vensim supported functions>`, :ref:`supported Xmile functions <Xmile supported functions>`, and :ref:`supported Python functions <Python supported functions>` are automatically generated from `docs/tables/*.tab`, which are tab separated files. In this case, we should add the following line to `docs/tables/functions.tab`:

.. list-table:: ABS
   :header-rows: 1

   * - Vensim
     - Vensim example
     - Xmile
     - Xmile example
     - Abstract Syntax
     - Python Translation
   * - ABS
     - ABS(A)
     - abs
     - abs(A)
     - CallStructure('abs', (A,))
     - numpy.abs(A)

To finish, we create a new release notes block at the top of `docs/whats_new.rst` file and update the software version. Commit all the changes, includying the test-models repo, and open a new PR.


Adding a simple function
------------------------
The second most simple case is we still are able to use the Abstract Structure :py:class:`pysd.translators.structures.abstract_expressions.CallStructure`, but we need to define a new function as the complexity of the source function would mess up the model code.

Let's suppose we want to add support for `Vensim's VECTOR SORT ORDER function <https://www.vensim.com/documentation/fn_vector_sort_order.html>`_. First of all, we may need to check Vensim's documentation to see how this function works and try to think what is the fatest way to solve it. VECTOR SORT ORDER functions takes two arguments, `vector` and `direction`. The function returns the order of the elements of the `vector`` based on the `direction`. Therefore, we do not need to save previous states information or to pass other information as arguments, we should have enought with a basic Python function that takes the same arguments.

Then, we define the Python function base on the Vensim's documentation. We also include the docstring with the same style of other functions and add this function to the file :py:mod:`pysd.py_backend.functions`::


    def vector_sort_order(vector, direction):
        """
        Implements Vensim's VECTOR SORT ORDER function. Sorting is done on
        the complete vector relative to the last subscript.
        https://www.vensim.com/documentation/fn_vector_sort_order.html

        Parameters
        -----------
        vector: xarray.DataArray
            The vector to sort.
        direction: float
            The direction to sort the vector. If direction > 1 it will sort
            the vector entries are from smallest to biggest, otherwise from
            biggest to smallest.

        Returns
        -------
        vector_sorted: xarray.DataArray
            The sorted vector.

        """
        if direction <= 0:
            flip = np.flip(vector.argsort(), axis=-1)
            return xr.DataArray(flip.values, vector.coords, vector.dims)
        return vector.argsort()

Now, we need to link the defined function with its corresponent abstract repressentation. So we include the following entry in the :py:data:`functionspace` dictionary from :py:mod:`pysd.builders.python.python_functions.py`::

    "vector_sort_order": (
        "vector_sort_order(%(0)s, %(1)s)",
        ("functions", "vector_sort_order"))

They key (:py:data:`"vector_sort_order"`) is the name of the Vensim function in lowercase and replacing the whitespaces by underscores. The first argument in the value (:py:data:`"vector_sort_order(%(0)s, %(1)s)"`) is the python repressentation, the :py:data:`%(0)s` and :py:data:`%(1)s` stand for the first and second argument of the original function, respectively. In this case, the repressentation is quite similar to the one in Vensim, we will move from `VECTOR SORT ORDER(vec, direction)` to `vector_sort_order(vec, direction)`. The last arguments stands for the dependencies of that function, in this case the function has been included in the functions submodule. Hence, we need to import `vector_sort_order` from `functions`, which is done by adding the dependency :py:data:`("functions", "vector_sort_order")`.

The next step is to add a test model for Vensim's `mdl`file. The test should cover all the possible cases, in this case, we should test the results for one and more dimensions arrays with different values along dimensions to genrate different order combinations, we also include cases for the both possible directions. We included the test `test-models/tests/vector_order/test_vector_order.mdl`, with its corresponding outputs file. Now we include the test in the testing script. We need to add the following entry in the :py:data:`vensim_test` dictionary of :py:mod:`tests/pytest_integration/pytest_integration_test_vensim_pathway.py`::

    "vector_order": {
        "folder": "vector_order",
        "file": "test_vector_order.mdl"
    },

At this point we should be able to run the test and, if the implementation was correctly done, pass it. We need to make sure that we did not break any other feature by running all the tests.

In order to finish the contribution, we should update the documentation by adding the following line to `docs/tables/functions.tab`:

.. list-table:: VECTOR SORT ORDER
   :header-rows: 1

   * - Vensim
     - Vensim example
     - Xmile
     - Xmile example
     - Abstract Syntax
     - Python Translation
   * - VECTOR SORT ORDER
     - VECTOR SORT ORDER(vec, direction)
     -
     -
     - CallStructure('vector_sort_order', (vec, direction))
     - vector_sort_order(vec, direction)

To finish, we create a new release notes block at the top of `docs/whats_new.rst` file and update the software version. Commit all the changes, includying the test-models repo, and open a new PR.
