Command Line Usage
==================

Basic command line usage
------------------------

Most of the features available in :doc:`Getting started <../getting_started>` are also available using the command line. Running:

.. code-block:: text

    python -m pysd Teacup.mdl


will translate *Teacup.mdl* to *Teacup.py* and run it with the default values. The output will be saved in *Teacup_output_%Y_%m_%d-%H_%M_%S_%f.tab*. The command line interface accepts several arguments, this can be checked using the *-h/--help* argument:

.. code-block:: text

    python -m pysd --help

Set output file
^^^^^^^^^^^^^^^
In order to set the output file path, the *-o/--output-file* argument can be used:

.. code-block:: text

    python -m pysd -o my_output_file.csv Teacup.mdl

.. note::
    The output file format may be *.csv*, *.tab* or *.nc*.

.. note::
    If *-o/--output-file* is not given, the output will be saved in a *.tab*
    file that starts with the model file name followed by a time stamp to avoid
    overwriting files.

Activate progress bar
^^^^^^^^^^^^^^^^^^^^^
The progress bar can be activated using the *-p/--progress* argument:

.. code-block:: text

    python -m pysd --progress Teacup.mdl

Translation options
-------------------

Only translate the model file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To translate the model file and not run the model, the *-t/--translate* command is provided:

.. code-block:: text

    python -m pysd --translate Teacup.mdl

Splitting Vensim views in different files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to split the Vensim model views in different files, as explained in :doc:`advanced usage <../advanced_usage>`, use the *--split-views* argument:

.. code-block:: text

    python -m pysd many_views_model.mdl --split-views

The previous code will put each model view in a separate Python module. Additionally, if the names of the views include the concepts of subsubmodules (e.g., ENERGY-transformation.efficiency_improvement), the *--subview-sep* (subview separators) argument may be used to further classify the model equations:

.. code-block:: text

    python -m pysd many_views_and_subviews_model.mdl --split-views --subview-sep - .

Note that passing any positional argument right after the *--subview-sep* argument will raise an error, so it is recommended to pass this argument as the last one.


Outputting various run information
----------------------------------
The number of output variables can be modified by passing them as arguments separated by commas, using the *-r/return_columns* argument:

.. code-block:: text

    python -m pysd -r 'Teacup Temperature, Room Temperature' Teacup.mdl

Note that the a single string must be passed after the *-r/return_columns* argument, containing the names of the variables separated by commas.

Sometimes, variable names have special characters, such as commas, which can happen when trying to return a variable with subscripts.
In this case we can save a *.txt* file with one variable name per row and use it as an argument:

.. code-block:: text

    python -m pysd -r output_selected_vars.txt Teacup.mdl


*-R/--return-timestamps* command can be used to set the *return timestamps*:

.. code-block:: text

    python -m pysd -R '0, 1, 3, 7, 9.5, 13.178, 21, 25, 30' Teacup.mdl

.. note::
    Each time stamp should be able to be computed as *initial_time + N x time_step*,
    where *N* is an integer.

.. note::
    The time outputs can also be modified using the model control variables, explained in next section.

Modify model variables
----------------------

Modify model control variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The values of the model control variables (i.e. *initial time*. *final time*, *time step* and *saving step*) can be
modified using the *-I/--initial_time*, *-F/--final-time*, *-T/--time-step* and *-S/--saveper* arguments, respectively. For example:

.. code-block:: text

    python -m pysd -I=2005 --final-time=2010 --time-step=1 Teacup.mdl

will set the initial time to 2005, the final time to 2010 and the time step to 1.

.. note::
    If the *-R/--return-timestamps* argument is used, the *final time* and *saving step* will be ignored.



Modify model variables
^^^^^^^^^^^^^^^^^^^^^^
To modify the values of model variables, their new values may be passed after the model file:

.. code-block:: text

    python -m pysd Teacup.mdl 'Room Temperature'=5

this will set *Room Temperature* variable to 5. A time series or a lookup can also be passed
as the new value of a variable as two lists of the same length:

.. code-block:: text

    python -m pysd Teacup.mdl 'Temperature Lookup=[[1, 2, 3, 4], [10, 15, 17, 18]]'

The first list will be used for the *time* or *x* values, and the second for the data values. See setting parameter values in :doc:`Getting started <../getting_started>` for further details.

.. note::

    If a variable name or the right hand side are defined with white spaces, they must be enclosed in quotes, as in the previous example.

Several variables can be changed at the same time, e.g.:

.. code-block:: text

    python -m pysd Teacup.mdl 'Room Temperature'=5 temperature_lookup='[[1, 2, 3, 4], [10, 15, 17, 18]]' 'Initial Temperature'=5

Modify initial conditions of model variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sometimes we do not want to change the actual value of a variable but we want to change its initial value instead. An example of this would be changing the initial value of a stock object. This can be done similarly to what was shown in the previous case, but using ':' instead of '=':

.. code-block:: text

    python -m pysd Teacup.mdl 'Teacup Temperature':30

this will set initial *Teacup Temperature* to 30.

Putting it all together
-----------------------
Several commands may be used together. The optional arguments and model arguments go first (those starting with '-' or '--'), then the model file path, and finally the variable or variables to change:

.. code-block:: text

    python -m pysd -o my_output_file.csv --progress --final-time=2010 --time-step=1 Teacup.mdl 'Room Temperature'=5 temperature_lookup='[[1, 2, 3, 4], [10, 15, 17, 18]]' 'Teacup Temperature':30

will save step 1 outputs until 2010 in *my_output_file.csv*, showing a progressbar during integration and setting foo to *5*, *temperature_lookup* to ((1, 10), (2, 15), (3, 17), (4, 18)) and initial *Teacup Temperature* to 30.