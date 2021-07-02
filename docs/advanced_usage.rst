Advanced Usage
==============

The power of PySD, and its motivation for existence, is its ability to tie in to other models and analysis packages in the Python environment. In this section we’ll discuss how those connections happen.


Replacing model components with more complex objects
----------------------------------------------------
In the last section we saw that a parameter could take on a single value, or a series of values over time, with PySD linearly interpolating between the supplied time-series values. Behind the scenes, PySD is translating that constant or time-series into a function that then goes on to replace the original component in the model. For instance, in the teacup example, the room temperature was originally a function defined through parsing the model file as something similar to::

   def room_temperature():
      return 75

However, when we made the room temperature something that varied with time, PySD replaced this function with something like::

   def room_temperature():
      return np.interp(t, series.index, series.values)

This drew on the internal state of the system, namely the time t, and the time-series data series that that we wanted to variable to represent. This process of substitution is available to the user, and we can replace functions ourselves, if we are careful.

Because PySD assumes that all components in a model are represented as functions taking no arguments, any component that we wish to modify must be replaced with a function taking no arguments. As the state of the system and all auxiliary or flow methods are public, our replacement function can call these methods as part of its internal structure.

In our teacup example, suppose we didn’t know the functional form for calculating the heat lost to the room, but instead had a lot of data of teacup temperatures and heat flow rates. We could use a regression model (here a support vector regression from Scikit-Learn) in place of the analytic function::

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

We can substitute this function directly for the heat_loss_to_room model component::

   model.components.heat_loss_to_room = new_heatflow_function

If you want to replace a subscripted variable, you need to ensure that the output from the new function is the same as the previous one. You can check the current coordinates and dimensions of a component by using :py:data:`.get_coords(variable_name)` as it is explained in :doc:`basic usage <../basic_usage>`.

Starting simulations from an end-state of another simulation
------------------------------------------------------------
The current state of a model can be saved in a pickle file using the :py:data:`.export()`method::

   import pysd
   model1 = pysd.read_vensim("my_model.mdl")
   model1.run(final_time=50)
   model1.export("final_state.pic")

Then the exported data can be used in another session::

   import pysd
   model2 = pysd.load("my_model.py")
   model2 = run(initial_condition="final_state.pic", return_timestamps=[55, 60])

the new simulation will have initial time equal to 50 with the saved values from the previous one.

.. note::
   You can set the exact final time of the simulation using the *final_time* argument.
   If you want to avoid returning the dataframe of the stocks you can use *return_timestamps=[]*::

     model1.run(final_time=50, return_timestamps=[])

.. note::
   The changes done with *params* arguments are not ported to the new model (*model2*) object that you initialize with *final_state.pic*. If you want to keep them, you need to call run with the same *params* values as in the original model (*model1*).

.. warning::
  Exported data is saved and loaded using `pickle <https://docs.python.org/3/library/pickle.html>`_, this data can be not compatible with future versions of
  *PySD* or *xarray*. In order to prevent data losses save always the source code.
