Tools
=====

Some additional tools are provided with the library.

Benchmarking
------------
.. automodule:: pysd.tools.benchmarking
   :members:

Exporting netCDF data_vars to csv or tab
----------------------------------------

Simulation results can be stored as netCDF (*.nc*) files (see :doc:`Storing simulation results on a file <./getting_started>`).

The :py:class:`pysd.tools.ncfiles.NCFile` allows loading netCDF files generated with PySD as an :py:class:`xarray.Dataset`. When passing the argument `parallel=True` to the constructor, :py:class:`xarray.DataArray` inside the Dataset will be loded as `dask arrays <https://docs.dask.org/en/stable/array.html>`_, with `chunks=-1 <https://docs.dask.org/en/stable/array-chunks.html>`_.

Once the Dataset is loaded, a subset (or all) of the data_vars can be exported into:

* A :py:class:`pandas.DataFrame`, using the :py:meth:`pysd.tools.ncfiles.NCFile.to_df` method
* A `*.csv` or `*.tab` files, using the the :py:meth:`pysd.tools.ncfiles.NCFile.to_text_file` method

Alternatively, to get further control of the chunking, users can load the :py:class:`xarray.Dataset` using :py:meth:`xarray.open_dataset` and then use the :py:meth:`pysd.tools.ncfiles.NCFile.ds_to_df` or :py:meth:`pysd.tools.ncfiles.NCFile.df_to_text_file` static methods.

.. automodule:: pysd.tools.ncfiles
   :members: