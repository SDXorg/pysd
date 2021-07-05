Command Line Usage
==================

Basic command line usage
------------------------

Most of the features available in :doc:`basic usage <../basic_usage>` are also available using command line. Running:

.. code-block:: text

    python -m pysd Teacup.mdl


will translate *Teacup.mdl* to *Teacup.py* and run it with the default values. The output will be saved in *Teacup_output_%Y_%m_%d-%H_%M_%S_%f.tab*. The command line accepts several arguments, this can be checked using the *-h/--help* argument:

.. code-block:: text

    python -m pysd --help

Set output file
^^^^^^^^^^^^^^^
In order to set the output file *-o/--output-file* argument can be used:

.. code-block:: text

    python -m pysd -o my_output_file.csv Teacup.mdl

.. note::
    The output file can be a *.csv* or *.tab*.

.. note::
    If *-o/--output-file* is not given the output will be saved in a file
    that starts with the model file name and has a time stamp to avoid
    overwritting files.

Activate progress bar
^^^^^^^^^^^^^^^^^^^^^
The progress bar can be activated using *-p/--progress* command:

.. code-block:: text

    python -m pysd --progress Teacup.mdl

Translation options
-------------------

Only translate model file
^^^^^^^^^^^^^^^^^^^^^^^^^
To only translate the model file, it does not run the model, *-t/--trasnlate* command is provided:

.. code-block:: text

    python -m pysd --translate Teacup.mdl

Splitting Vensim views in different files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to split the Vensim model views in different files as explained in :doc:`advanced usage <../advanced_usage>`:

.. code-block:: text

    python -m pysd --split-modules many_views_model.mdl

Outputting various run information
----------------------------------
The output number of variables can be modified bu passing them as arguments separated by commas, using *-r/return_columns* argument:

.. code-block:: text

    python -m pysd -r 'Teacup Temperature, Room Temperature' Teacup.mdl

Note that the argument passed after *-r/return_columns* should be inside '' to be properly read. Moreover each variable name must be split with commas.

Sometimes, the variable names have special characteres, such as commas, which can happen when trying to return a variable with subscripts.
In this case whe can save a *.txt* file with one variable name per row and use it as an argument:

.. code-block:: text

    python -m pysd -r output_selected_vars.txt Teacup.mdl


*-R/--return-timestamps* command can be used to set the *return timestamps*:

.. code-block:: text

    python -m pysd -R '0, 1, 3, 7, 9.5, 13.178, 21, 25, 30' Teacup.mdl

.. note::
    Each time stamp should be able to be computed as *initial_time + N x time_step*,
    where *N* is an integer.

.. note::
    The time outputs can be also modified using the model control variables, explained in next section.

Modify model variables
----------------------

Modify model control variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The model control variables such as the *initial time*. *final time*, *time step* and *saving step* can be easily
modified using the *-I/--initial_time*, *-F/--final-time*, *-T/--time-step* and *-S/--saveper* commands respectively. For example:

.. code-block:: text

    python -m pysd -I 2005 --final-time=2010 --time-step=1 Teacup.mdl

will set the initial time to 2005, the final time to 2010 and the time step to 1.

.. note::
    If *-R/--return-timestamps* argument is used the *final time* and *saving step* will be ignored.




Modify model variables
^^^^^^^^^^^^^^^^^^^^^^
In order to modify the values of model variables they can be passed after the model file:

.. code-block:: text

    python -m pysd Teacup.mdl 'Room Temperature'=5

this will set *Room Temperature* variable to the constant value 5. A series can be also passed
to change a value of a value to a time dependent series or the interpolation values
of a lookup variable two lists of the same length must be given:

.. code-block:: text

    python -m pysd Teacup.mdl 'Temperature Lookup=[[1, 2, 3, 4], [10, 15, 17, 18]]'

The first list will be used for the *time* or *x* values and the second for the data. See setting parameter values in :doc:`basic usage <../basic_usage>` for more information.

.. note::

    If a variable name or the right hand side are defined with whitespaces
    it is needed to add '' define it, as has been done in the last example.

Several variables can be changed at the same time, e.g.:

.. code-block:: text

    python -m pysd Teacup.mdl 'Room Temperature'=5 temperature_lookup='[[1, 2, 3, 4], [10, 15, 17, 18]]' 'Initial Temperature'=5

Modify initial conditions of model variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sometimes we do not want to change a variable value to a constant but change its initial value, for example change initial value of a stock object, this can be similarly done to the previos case but using ':' instead of '=':

.. code-block:: text

    python -m pysd Teacup.mdl 'Teacup Temperature':30

this will set initial *Teacup Temperature* to 30.

Putting It All Together
-----------------------
Several commands can be used together, first need to add optional arguments, those starting with '-', next the model file, and last the variable or variables to change, for example:

.. code-block:: text

    python -m pysd -o my_output_file.csv --progress --final-time=2010 --time-step=1 Teacup.mdl 'Room Temperature'=5 temperature_lookup='[[1, 2, 3, 4], [10, 15, 17, 18]]' 'Teacup Temperature':30

will save step 1 outputs until 2010 in *my_output_file.csv*, showing a progressbar during integration and settung foo to *5* and *temperature_lookup* to ((1, 10), (2, 15), (3, 17), (4, 18)) and  initial *Teacup Temperature* to 30.