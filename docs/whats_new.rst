What's New
==========
v3.10.0 (2023/04/28)
--------------------
New Features
~~~~~~~~~~~~
- Parse TABBED ARRAYS Vensim function. (`@rogersamso <https://github.com/rogersamso>`_)
- Add support for Vensim's `POWER <https://www.vensim.com/documentation/fn_power.html>`_ function. (`@rogersamso <https://github.com/rogersamso>`_)
- Add possibility to pass data_files in netCDF format. (`@rogersamso <https://github.com/rogersamso>`_)
- Add support for XMILE's non-negative flows and stocks. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Add support for XMILE's MIN and MAX functions with one argument. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Set the final_subscripts to an empty dictionary for ELMCOUNT function in :py:meth:`pysd.builders.python_exressions_builder.CallBuilder.build_function_call`. (`@rogersamso <https://github.com/rogersamso>`_)
- Define comp_subtype of Unchangeable tabbed arrays as Unchangeable. This is done in :py:meth:`pysd.builders.python.python_expressions_builder.ArrayBuilder.build`. (`@rogersamso <https://github.com/rogersamso>`_)

Documentation
~~~~~~~~~~~~~
- Add information about slack channel https://slofile.com/slack/sdtoolsandmet-slj3251. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Update XMILE stocks section. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
- Add a weekly scheduled run to all CI workflows, which run each Monday at 06:00 UTC. (`@EwoutH <https://github.com/EwoutH>`_)
- Fix CI pipeline for Python 3.11 and remove Python 3.10 pipeline in favour of 3.11. (`@kinow <https://github.com/kinow>`_)
- Add non_negative argument in :py:class:`pysd.translators.structures.abstract_expressions.IntegStructure`. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

v3.9.1 (2023/03/11)
-------------------

New Features
~~~~~~~~~~~~
- Add :py:const:`numpy.py` as translation for the call to the function `PI()`. (`@lionel42 <https://github.com/lionel42>`_)

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Set :py:mod:`numpy` <1.24 to avoid errors with least squares equation in :py:func:`pysd.py_backend.allocation.allocate_available`. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Keep the attributes of a component when using :py:meth:`pysd.py_backend.model.Macro.set_components` to avoid losing coords or arguments information. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Set :py:mod:`openpyxl` <3.1 to avoid errors due to non-backwards compatible changes. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Include time dependency in random functions to avoid them using constant cache. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Documentation
~~~~~~~~~~~~~

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
- Run test for Python 3.11 with ubuntu-latest (hdf5-headers need to be installed using apt manager). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)


v3.9.0 (2022/12/15)
-------------------

New Features
~~~~~~~~~~~~
- Parses and ignores reality check functions during translation of Vensim models. (`@rogersamso <https://github.com/rogersamso>`_)

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Fix issue with the classification of variables in modules and submodules (:issue:`388`). When a model had a view with 3 sublevels (e.g. energy-transformation.losses) but another view was defined with only two of them (e.g. energy-transformation), the variables in the second view were placed in the main model file. Now, if this happens, the variables in the second view will be placed in a main.py file (i.e. energy/transformation/main.py). (`@rogersamso <https://github.com/rogersamso>`_)
- Fix bug on the CLI when passing a hyphen as first value to the *--subview-sep* argument (:issue:`388`). (`@rogersamso <https://github.com/rogersamso>`_)
- Fix bug on the CLI when parsing initial conditions (:issue:`395`). (`@rogersamso <https://github.com/rogersamso>`_)

Documentation
~~~~~~~~~~~~~
- The `Splitting Vensim views in different files` section in :doc:`command_line_usage` has been updated to include an example of the usage of the *--subview-sep* CLI argument. (`@rogersamso <https://github.com/rogersamso>`_)

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
- The :py:meth:`_merge_nested_dicts` method from the :py:class:`pysd.translators.vensim.vensim_file.VensimFile` class has been made a static method, as it does not need to access any attribute of the instance, and it does facilitate unit testing. (`@rogersamso <https://github.com/rogersamso>`_)
- The `pysd/translators/vensim/parsing_grammars/element_object.peg` grammar has been modified to be able to parse reality check elements. (`@rogersamso <https://github.com/rogersamso>`_)
- :py:class:`pysd.translators.vensim.vensim_element.Constraint`  and :py:class:`pysd.translators.vensim.vensim_element.TestInputs` classes have been added, which inherit from the also newly created :py:class:`pysd.translators.vensim.vensim_element.GenericComponent`, which include the :py:meth:`parse` and :py:meth:`get_abstract_component` methods. (`@rogersamso <https://github.com/rogersamso>`_ and `@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- The :py:class:`pysd.translators.structures.abstract_model.AbstractSection` class now has two extra attributes (:py:data:`constraints` and :py:data:`input_tests`), which hold the :py:class:`pysd.translators.structures.abstract_model.AbstractConstraint` and :py:class:`pysd.translators.structures.abstract_model.AbstractTestInputs` objects. (`@rogersamso <https://github.com/rogersamso>`_)

v3.8.0 (2022/11/03)
-------------------

New Features
~~~~~~~~~~~~
- Adds ncfile.py module with helper functions to export a subset or all of the data_vars in netCDF files generated with PySD to :py:class:`pandas.DataFrame`, csv or tab files. (`@rogersamso <https://github.com/rogersamso>`_)
- Adds possibility to initialize and export a subset or all external objects to netCDF, and then initialize the external objects from the file. (`@rogersamso <https://github.com/rogersamso>`_)

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~
- Deprecate :py:meth:`pysd.py_backend.model.Model._get_dependencies` replacing it with :py:meth:`pysd.py_backend.model.Model.get_dependencies`. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Bug fixes
~~~~~~~~~
- Include new class :py:class:`pysd.py_backend.utils.Dependencies` to return by :py:meth:`pysd.py_backend.model.Model.get_dependencies` (:issue:`379`). (`@lionel42 <https://github.com/lionel42>`_)

Documentation
~~~~~~~~~~~~~
- Updates the :doc:`getting_started` page with instructions on how to use the new helper functions for netCDF files. (`@rogersamso <https://github.com/rogersamso>`_)
- Updates the :doc:`advanced_usage` page with instructions on how to export externals to netCDF and initialize a model from it. (`@rogersamso <https://github.com/rogersamso>`_)
- Update citation information to include the new paper published in JOSS. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Performance
~~~~~~~~~~~
- Initializing external objects from netCDF file is much faster than reading from spreadsheet files.(`@rogersamso <https://github.com/rogersamso>`_)

Internal Changes
~~~~~~~~~~~~~~~~
- Adds the :py:meth:`pysd.py_backend.model.Macro.serialize_externals` and :py:meth:`pysd.py_backend.model.Macro.initialize_external_data` methods, and a few other private methods.(`@rogersamso <https://github.com/rogersamso>`_)
- Adds the :py:class:`pysd.py_backend.utils.UniqueDims` class for renaming model dimensions with unique names.(`@rogersamso <https://github.com/rogersamso>`_)
- Force :py:class:`pysd.py_backend.external.External` objects to always have the full element dimensions, missing dimensions are filled with `numpy.nan`. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Add `dependabot <https://docs.github.com/en/code-security/dependabot/working-with-dependabot/keeping-your-actions-up-to-date-with-dependabot>`_ configuration for GitHub Actions updates. (`@EwoutH <https://github.com/EwoutH>`_)
- Include new error messages for initialization of :py:class:`pysd.py_backend.lookups.HardcodedLookups` (:issue:`376`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Include new warning message when a translated variable has several types or subtypes. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Set CI test to run in parallel in 2 cores. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

v3.7.1 (2022/09/19)
-------------------

New Features
~~~~~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Fix bugs with :py:class:`pandas.DataFrame` 1.5.0 (:issue:`366`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Documentation
~~~~~~~~~~~~~

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~

v3.7.0 (2022/09/19)
-------------------

New Features
~~~~~~~~~~~~
- Simulation results can now be stored as netCDF4 files. (`@rogersamso <https://github.com/rogersamso>`_)
- The CLI also accepts netCDF4 file paths after the -o argument. (`@rogersamso <https://github.com/rogersamso>`_)

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Fix bug when a WITH LOOKUPS argument has subscripts. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Fix bug of exportig csv files with multiple subscripts variables. (`@rogersamso <https://github.com/rogersamso>`_)
- Fix bug of missing dimensions in variables defined with not all the subscripts of a range (:issue:`364`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Fix bug when running a model with variable final time or time step and progressbar (:issue:`361`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Documentation
~~~~~~~~~~~~~
- Add `Storing simulation results on a file` section in the :doc:`getting_started` page. (`@rogersamso <https://github.com/rogersamso>`_)
- Include cookbook information in the :doc:`getting_started` page. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Include an introduction of main historical changes in the :doc:`about` page. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Performance
~~~~~~~~~~~
- Exporting outputs as netCDF4 is much faster than exporting a pandas DataFrame, especially for large models. (`@rogersamso <https://github.com/rogersamso>`_)

Internal Changes
~~~~~~~~~~~~~~~~
- Make PySD work with :py:mod:`parsimonius` 0.10.0. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Add netCDF4 dependency for tests. (`@rogersamso <https://github.com/rogersamso>`_)
- Improve warning message when replacing a stock with a parameter.  (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Include more pytest parametrizations in some test and make them translate the models in temporary directories.  (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Include lychee-action in the GHA workflow to check the links. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Update License. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Include `Maintained? Yes` and `Contributions welcome` badges. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Update links to the new repository location. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Reduce relative precision from 1e-10 to 1e-5 to compute the saving times and final time. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Add convergence tests for euler integration method. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Include build docs check in the GHA workflow to avoid warnings with sphinx. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

v3.6.1 (2022/09/05)
-------------------

New Features
~~~~~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

Documentation
~~~~~~~~~~~~~

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
- Set :py:mod:`parsimonius` requirement to 0.9.0 to avoid a breaking-change in the newest version. Pending to update PySD to run it with :py:mod:`parsimonious` 0.10.0. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

v3.6.0 (2022/08/31)
-------------------

New Features
~~~~~~~~~~~~
- Include warning messages when a variable is defined in more than one view, when a control variable appears in a view or when a variable doesn't appear in any view as a `workbench variable` (:issue:`357`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Force variables in a module to be saved alphabetically for being able to compare differences between versions (only for the models that are split by views). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Classify control variables in the main file always (:issue:`357`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Documentation
~~~~~~~~~~~~~

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
- Include :py:class:`pysd.translators.structures.abstract_model.AbstractControlElement` child of :py:class:`pysd.translators.structures.abstract_model.AbstractElement` to differentiate the control variables. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)


v3.5.2 (2022/08/15)
-------------------

New Features
~~~~~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Make sketch's `font_size` optional. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Documentation
~~~~~~~~~~~~~
- Correct typos.

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~

v3.5.1 (2022/08/11)
-------------------

New Features
~~~~~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Fix bug generated when :EXCEPT: keyword is used with subscript subranges (:issue:`352`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Fix bug of precision error for :py:func:`pysd.py_backend.allocation.allocate_by_priority` (:issue:`353`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Fix bug of constant cache assignment. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Documentation
~~~~~~~~~~~~~

Performance
~~~~~~~~~~~
- Improve the performance of reading :py:class:`pysd.py_backend.external.External` data with cellrange names by loading the data in memory with :py:mod:`pandas`. As recommended by :py:mod:`openpyxl` developers, this is a possible way of improving performance to avoid parsing all rows up each time for getting the data (`issue 1867 in openpyxl <https://foss.heptapod.net/openpyxl/openpyxl/-/issues/1867>`_). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Internal Changes
~~~~~~~~~~~~~~~~

v3.5.0 (2022/07/25)
-------------------

New Features
~~~~~~~~~~~~
- Add support for subscripted arguments in :py:func:`pysd.py_backend.functions.ramp` and :py:func:`pysd.py_backend.functions.step` (:issue:`344`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Fix bug related to the order of elements in 1D GET expressions (:issue:`343`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Fix bug in request 0 values in allocate by priority (:issue:`345`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Fix a numerical error in starting time of step and ramp. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Documentation
~~~~~~~~~~~~~
- Include new PySD logo. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
- Ignore 'distutils Version classes are deprecated. Use packaging.version instead' error in tests as it is an internal error of `xarray`. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Add a warning message when a subscript range is duplicated in a variable reference. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)


v3.4.0 (2022/06/29)
-------------------

New Features
~~~~~~~~~~~~
- Add support for Vensim's `ALLOCATE AVAILABLE <https://www.vensim.com/documentation/fn_allocate_available.html>`_ (:py:func:`pysd.py_backend.allocation.allocate_available`) function (:issue:`339`). Integer allocation cases have not been implemented neither the fixed quantity and constant elasticity curve priority functions. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

Documentation
~~~~~~~~~~~~~
- Improve the documentation of the :py:mod:`pysd.py_backend.allocation` module. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
- Add a class to manage priority profiles so it can be also used by the `many-to-many allocation <https://www.vensim.com/documentation/24340.html>`_. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)


v3.3.0 (2022/06/22)
-------------------

New Features
~~~~~~~~~~~~
- Add support for Vensim's `ALLOCATE BY PRIORITY <https://www.vensim.com/documentation/fn_allocate_by_priority.html>`_ (:py:func:`pysd.py_backend.allocation.allocate_by_priority`) function (:issue:`263`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Fix bug of using subranges to define a bigger range (:issue:`335`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Documentation
~~~~~~~~~~~~~

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
- Improve error messages for :class:`pysd.py_backend.External` objects. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

v3.2.0 (2022/06/10)
-------------------

New Features
~~~~~~~~~~~~
- Add support for Vensim's `GET TIME VALUE <https://www.vensim.com/documentation/fn_get_time_value.html>`_ (:py:func:`pysd.py_backend.functions.get_time_value`) function (:issue:`332`). Not all cases have been implemented. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Add support for Vensim's `VECTOR SELECT <http://vensim.com/documentation/fn_vector_select.html>`_ (:py:func:`pysd.py_backend.functions.vector_select`) function (:issue:`266`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

Documentation
~~~~~~~~~~~~~

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~



v3.1.0 (2022/06/02)
-------------------

New Features
~~~~~~~~~~~~
- Add support for Vensim's `VECTOR SORT ORDER <https://www.vensim.com/documentation/fn_vector_sort_order.html>`_ (:py:func:`pysd.py_backend.functions.vector_sort_order`) function (:issue:`326`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Add support for Vensim's `VECTOR RANK <https://www.vensim.com/documentation/fn_vector_rank.html>`_ (:py:func:`pysd.py_backend.functions.vector_rank`) function (:issue:`326`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Add support for Vensim's `VECTOR REORDER <https://www.vensim.com/documentation/fn_vector_reorder.html>`_ (:py:func:`pysd.py_backend.functions.vector_reorder`) function (:issue:`326`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

Documentation
~~~~~~~~~~~~~
- Add the section :doc:`/development/adding_functions` with examples for developers. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~

- Include a template for PR.


v3.0.1 (2022/05/26)
-------------------

New Features
~~~~~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

- Simplify subscripts dictionaries for :py:class:`pysd.py_backend.data.TabData` objects. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Documentation
~~~~~~~~~~~~~
- Improve tests/README.md.
- Minor improvements in the documentation.

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
- Add Python 3.10 to CI pipeline and include it in the supported versions list. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Correct LICENSE file extension in the `setup.py`. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Move from `importlib`'s :py:func:`load_module` to :py:func:`exec_module`. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Remove warnings related to :py:data:`set` usage. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Move all the missing test to :py:mod:`pytest`. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Remove warning messages from test and make test fail if there is any warning. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)


v3.0.0 (2022/05/23)
-------------------

New Features
~~~~~~~~~~~~

- The new :doc:`Abstract Model Representation <structure/structure_index>` translation and building workflow will allow to add new output languages in the future. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Added new properties to the :py:class:`pysd.py_backend.model.Macro` to make more accessible some information: :py:attr:`.namespace`, :py:attr:`.subscripts`, :py:attr:`.dependencies`, :py:attr:`.modules`, :py:attr:`.doc`. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Cleaner Python models: (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
    - :py:data:`_namespace` and :py:data:`_dependencies` dictionaries have been removed from the file.
    - Variables original names, dependencies metadata now are given through :py:meth:`pysd.py_backend.components.Component.add` decorator, instead of having them in the docstring.
    - Merging of variable equations is now done using the coordinates to a pre-allocated array, instead of using the `magic` function :py:data:`pysd.py_backend.utils.xrmerge()`.
    - Arranging and subseting arrays are now done inplace instead of using the magic function :py:data:`pysd.py_backend.utils.rearrange()`.

Breaking changes
~~~~~~~~~~~~~~~~

- Set the argument :py:data:`flatten_output` from :py:meth:`.run` to :py:data:`True` by default. Previously it was set to :py:data:`False` by default. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Move the docstring of the model to a property, :py:attr:`.doc`. Thus, it is not callable anymore. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Allow the function :py:func:`pysd.py_backend.functions.pulse` to also perform the operations performed by :py:data:`pysd.py_backend.functions.pulse_train()` and :py:data:`pysd.py_backend.functions.pulse_magnitude()`. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Change first argument of :py:func:`pysd.py_backend.functions.active_initial`, now it is the `stage of the model` and not the `time`. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Simplify the function :py:data:`pysd.py_backend.utils.rearrange()` orienting it to perform simple rearrange cases for user interaction. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Move :py:data:`pysd.py_backend.statefuls.Model` and  :py:data:`pysd.py_backend.statefuls.Macro` to  :py:class:`pysd.py_backend.model.Model` and :py:class:`pysd.py_backend.model.Macro`, respectively. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Manage all kinds of lookups with the :py:class:`pysd.py_backend.lookups.Lookups` class. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Include a second optional argument to lookups functions to set the final coordinates when a subscripted variable is passed as an argument. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Deprecations
~~~~~~~~~~~~

- Remove :py:data:`pysd.py_backend.utils.xrmerge()`, :py:data:`pysd.py_backend.functions.pulse_train()`, :py:data:`pysd.py_backend.functions.pulse_magnitude()`, :py:data:`pysd.py_backend.functions.lookup()`, :py:data:`pysd.py_backend.functions.lookup_discrete()`, :py:data:`pysd.py_backend.functions.lookup_extrapolation()`, :py:data:`pysd.py_backend.functions.logical_and()`, :py:data:`pysd.py_backend.functions.logical_or()`, :py:data:`pysd.py_backend.functions.bounded_normal()`, :py:data:`pysd.py_backend.functions.log()`. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Remove old translation and building files (:py:data:`pysd.translation`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)


Bug fixes
~~~~~~~~~

- Generate the documentation of the model when loading it to avoid lossing information when replacing a variable value (:issue:`310`, :pull:`312`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Make random functions return arrays of the same shape as the variable, to avoid repeating values over a dimension (:issue:`309`, :pull:`312`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Fix bug when Vensim's :MACRO: definition is not at the top of the model file (:issue:`306`, :pull:`312`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Make builder identify the subscripts using a main range and subrange to allow using subscripts as numeric values as Vensim does (:issue:`296`, :issue:`301`, :pull:`312`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Fix bug of missmatching of functions and lookups names (:issue:`116`, :pull:`312`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Parse Xmile models case insensitively and ignoring the new lines characters (:issue:`203`, :issue:`253`, :pull:`312`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Add support for Vensim's `\:EXCEPT\: keyword <https://www.vensim.com/documentation/exceptionequations.html>`_ (:issue:`168`, :issue:`253`, :pull:`312`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Add spport for Xmile's FORCST and SAFEDIV functions (:issue:`154`, :pull:`312`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Add subscripts support for Xmile (:issue:`289`, :pull:`312`). (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Fix numeric error bug when using :py:data:`return_timestamps` and time step with non-integer values. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Documentation
~~~~~~~~~~~~~

- Review the whole documentation, refract it, and describe the new features. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Performance
~~~~~~~~~~~

- The variables defined in several equations are now assigned to a pre-allocated array instead of using :py:data:`pysd.py_backend.utils.xrmerge()`. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- The arranging and subseting of arrays is now done inplace instead of using the magic function :py:data:`pysd.py_backend.utils.rearrange()`. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- The grammars for Parsimonious are only compiled once per translation. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Internal Changes
~~~~~~~~~~~~~~~~
- The translation and the building of models has been totally modified to use the :doc:`Abstract Model Representation <structure/structure_index>`. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
