Advanced Usage
==============

The power of PySD, and its motivation for existence, is its ability to tie in to other models and analysis packages in the Python environment. In this section we discuss how those connections happen.

Running models one (or more) step(s) at a time
----------------------------------------------

Coupling different models requires the exchange of certain variables between them. This is only possible if a stepping mechanism is available, which allows for running any number of steps with certain boundary conditions that evolve along time.
In this section, we show how the teacup model may be run in a for loop using the :py:meth:`pysd.py_backend.model.Model.step` function (rather than the :py:meth:`pysd.py_backend.model.Model.run` function) and update the value of a model variable before each step::

   from pysd.py_backend.output import ModelOutput

   # instantiate ModelOutput object
   output = ModelOutput()

   # configure the stepper behavior, passing a list of the variables that
   # will be updated before running each model step in the step_vars argument
   model.set_stepper(output,
                     step_vars=["room_temperature"],
                     final_time=5)

   # run 40 steps increasing the room_temperature by one degree at each step
   for _ in range(40):
      model.step(1, {"room_temperature": model["room_temperature"] + 1})

   # store model results in a pandas DataFrame (default unless a file name
   # is passed when instantiating the ModelOutput object)
   result_df = output.collect(model)


Replacing model components with more complex objects
----------------------------------------------------
In the last section we saw that a parameter could take on a single value, or a series of values over time, with PySD linearly interpolating between the supplied time-series values. Behind the scenes, PySD is translating that constant or time-series into a function that then goes on to replace the original component in the model. For instance, in the teacup example, the room temperature was originally a function defined through parsing the model file as something similar to::

   def room_temperature():
      return 75

However, when we made the room temperature something that varied with time, PySD replaced this function with something like::

   def room_temperature():
      return np.interp(t, series.index, series.values)

This drew on the internal state of the system, namely the time t, and the time-series data series that we wanted the variable to represent. This process of substitution is available to the user, and we can replace functions ourselves, if we are careful.

Because PySD assumes that all components in a model are represented as functions taking no arguments, any component that we wish to modify must be replaced with a function taking no arguments. As the state of the system and all auxiliary or flow methods are public, our replacement function can call these methods as part of its internal structure.

In our teacup example, suppose we did not know the functional form for calculating the heat lost to the room, but instead had a lot of data of teacup temperatures and heat flow rates. We could use a regression model (here a support vector regression from Scikit-Learn) in place of the analytic function::

   from sklearn.svm import SVR
   regression = SVR()
   regression.fit(X_training, Y_training)

Once the regression model is fit, we write a wrapper function for its predict method that accesses the input components of the model and formats the prediction for PySD::

   def new_heatflow_function():
      """ Replaces the original flowrate equation
          with a regression model"""
      tea_temp = model.components.teacup_temperature()
      room_temp = model.components.room_temperature()
      return regression.predict([room_temp, tea_temp])[0]

To substitute this function directly for the heat_loss_to_room model component using the :py:meth:`.set_components` method::

   model.set_components({'heat_loss_to_room': new_heatflow_function})

If you want to replace a subscripted variable, you need to ensure that the output from the new function is the same as the previous one. You can check the current coordinates and dimensions of a component by using :py:meth:`.get_coords` as it is explained in :doc:`Getting started <../getting_started>`.

.. note::
   Alternatively, you can also set a model component directly::

      model.components.heat_loss_to_room = new_heatflow_function

   However, this will only accept the python name of the model component. While for the :py:meth:`.set_components` method, the original name can be also used.

Splitting Vensim views in separate Python files (modules)
---------------------------------------------------------
In order to replicate the Vensim views in the translated models, the user can set the `split_views` argument to True in the :py:func:`pysd.read_vensim` function::

   read_vensim("many_views_model.mdl", split_views=True)


The option to split the model in views is particularly interesting for large models with tens of views. Translating those models into a single file may make the resulting Python model difficult to read and maintain.

In a Vensim model with three separate views (e.g. `view_1`, `view_2` and `view_3`), setting `split_views` to True would create the following tree inside the directory where the `.mdl` model is located:

| main-folder
| ├── modules_many_views_model
| │   ├── _modules.json
| │   ├── view_1.py
| │   ├── view_2.py
| │   └── view_3.py
| ├── _subscripts_many_views_model.json
| ├── many_views_model.py

The variables in each file will be sorted alphabetically, using their Python name.

.. note ::
    Often, modelers wish to organise views further. To that end, a common practice is to include a particular character in the View name to indicate that what comes after it is the name of the subview. For instance, we could name one view as `ENERGY.Supply` and another one as `ENERGY.Demand`.
    In that particular case, setting the `subview_sep` kwarg equal to `["."]`, as in the code below, would name the translated views as `demand.py` and `supply.py` and place them inside the `ENERGY` folder::

      read_vensim("many_views_model.mdl", split_views=True, subview_sep=["."])

.. note ::
    If a variable appears as a `workbench variable` in more than one view, it will be added only to the module corresponding to the first view and a warning message will be printed. If a variable does not appear as a workbench variable in any view, it will be added to the main model file printing a warning message.

If macros are present, they will be self-contained in files named after the macro itself. The macro inner variables will be placed inside the module that corresponds with the view in which they were defined.


Starting simulations from an end-state of another simulation
------------------------------------------------------------
The current state of a model can be saved in a pickle file using the :py:meth:`.export` method::

   import pysd
   model1 = pysd.read_vensim("my_model.mdl")
   model1.run(final_time=50)
   model1.export("final_state.pic")

then the exported data can be used in another session::

   import pysd
   model2 = pysd.load("my_model.py")
   model2 = run(initial_condition="final_state.pic", return_timestamps=[55, 60])

the new simulation will have initial time equal to 50 with the saved values from the previous one.

.. note::
   You can set the exact final time of the simulation using the *final_time* argument.
   If you want to avoid returning the dataframe of the stocks you can use *return_timestamps=[]*::

     model1.run(final_time=50, return_timestamps=[])

.. note::
   The changes made with the *params* arguments are not ported to the new model (*model2*) object that you initialize with *final_state.pic*. If you want to keep them, you need to call run with the same *params* values as in the original model (*model1*).

.. warning::
  Exported data is saved and loaded using `pickle <https://docs.python.org/3/library/pickle.html>`_. The data stored in the pickles may be incompatible with future versions of
  *PySD* or *xarray*. In order to prevent data losses, always save the source code.


Selecting and running a submodel
--------------------------------
A submodel of a translated model can be run as a standalone model. This can be done through the :py:meth:`.select_submodel` method:

.. automethod:: pysd.py_backend.model.Model.select_submodel
   :noindex:


In order to preview the needed exogenous variables, the :py:meth:`.get_dependencies` method can be used:

.. automethod:: pysd.py_backend.model.Model.get_dependencies
   :noindex:


Initializing external data from netCDF file
-------------------------------------------

IO operations are expensive, especially when reading non-binary files. This makes the model initialization process slow when lots of datasets need to be read from spreadsheet files.

From PySD 3.8, users can export a subset or all model external data to a netCDF file, and use this file for subsequent model initializations.

Suppose we have a model (*my_model.mdl*) that loads *param_1* from *parameters_1.xls*, *param_2* from *parameters_2.xls*, and *policy_1* and *policy_2* from *scenario.xls*. Imagine we want to test different policy configurations, by changing the values of *policy_1* and *policy_2*, while keeping all other parameters unchanged.
In this case, we might want to export the external objects that we do not intend to modify (*param_1* and *param_2*) to a netCDF file, so that they are initialized instantaneously:

The code below shows how this can be achieved::

    import pysd
    model = pysd.read_vensim("my_model.mdl", initialize=False)
    model.serialize_externals(export_path="parameters.nc",
                              include_externals="all",
                              exclude_externals=["scenario.xls"])


This will export only the external data defined in *parameters_1.xls* and *parameters_2.xls* to the *parameters.xls* file. A good thing of this is that the resulting netCDF file will include all the metadata defined in the model for such variable (description, units, etc.)

Note that the exact same result could have been achieved with::

    import pysd

    model = pysd.read_vensim("my_model.mdl", initialize=False)
    model.serialize_externals(export_path="parameters.nc",
                              include_externals="all",
                              exclude_externals=["policy_1", "policy_2"])

Or even::

    import pysd

    model = pysd.read_vensim("my_model.mdl", initialize=False)
    model.serialize_externals(export_path="parameters.nc",
                              include_externals=["parameters_1.xls",
                                                 "parameters_2.xls"],
                              exclude_externals=None)

Or we could have also combined variable names and spreadsheet files in the **include_externals** argument, the **exclude_externals** argument or both::

    import pysd

    model = pysd.read_vensim("my_model.mdl", initialize=False)
    model.serialize_externals(export_path="parameters.nc",
                              include_externals=["param_1", "parameters_2.xls"],
                              exclude_externals=None)


Then, to run a simulation loading the external objects stored in *parameters.nc*, we write::

    import pysd

    model = pysd.read_vensim("my_model.mdl", initialize=False)
    model.initialize_external_data(externals="parameters.nc")

And from here we could run the model normally with :py:meth:`pysd.py_backend.Model.run`.