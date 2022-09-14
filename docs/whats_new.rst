What's New
==========
v3.7.0 (to be released)
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
- Add netCDF4 as a dependency for tests. (`@rogersamso <https://github.com/rogersamso>`_)
- Include lychee-action in the GHA workflow to check the links. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Update License. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Include `Maintained? Yes` and `Contributions welcome` badges. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Update links to the new repository location. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

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
