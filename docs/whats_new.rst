
What's New
==========

v3.0.0 (unreleased)
-----------------------

New Features
~~~~~~~~~~~~

- The new :doc:`Abstract Model Representation <structure/structure_index>` translation and building workflow will allow to add new output languages in the future.
- Properties added to the :py:class:`pysd.py_backend.model.Macro` to make more accessible some information: :py:attr:`.namespace`, :py:attr:`.subscripts`, :py:attr:`.dependencies`, :py:attr:`.modules`, :py:attr:`.doc`.
- The Python models now look cleaner:
    - :py:data:`_namespace` and :py:data:`_dependencies` dictionaries are removed from the file.
    - Variables original names, dependencies metadata are given through :py:meth:`pysd.py_backend.components.Component.add` decorator, instead of having them in the docstring.
    - The merging of variable equations is done using the coordinates to a pre-allocated array, instead of using the `magic` function :py:data:`pysd.py_backend.utils.xrmerge()`.
    - The arranging and subseting arrays are now done inplace instead of using the magic function :py:data:`pysd.py_backend.utils.rearrange()`.

Breaking changes
~~~~~~~~~~~~~~~~

- The argument :py:data:`flatten_output` from :py:meth:`.run` is now set to :py:data:`True` by default.
- The docstring of the model is now a property and thus is it not callable,:py:attr:`.doc`.
- Allow the function :py:func:`pysd.py_backend.functions.pulse` to also perform the operations performed by :py:data:`pysd.py_backend.functions.pulse_train()` and :py:data:`pysd.py_backend.functions.pulse_magnitude()`.
- The first argument of :py:func:`pysd.py_backend.functions.active_initial` now is the stage and not the time.
- The function :py:data:`pysd.py_backend.utils.rearrange()` now its mutch simpler oriented to perform simple rearrange cases for user interaction.
- The translation and the building of models has been totally modified to use the :doc:`Abstract Model Representation <structure/structure_index>`.
- Move :py:data:`pysd.py_backend.statefuls.Model` and  :py:data:`pysd.py_backend.statefuls.Macro` to  :py:class:`pysd.py_backend.model.Model` and :py:class:`pysd.py_backend.model.Macro`, respectively.
- All kinds of lookups are now managed with the :py:class:`pysd.py_backend.lookups.Lookups` class.
- The lookups functions may now take a second argument to set the final coordinates when a subscripted variable is passed as an argument.

Deprecations
~~~~~~~~~~~~

- Remove :py:data:`pysd.py_backend.utils.xrmerge()`, :py:data:`pysd.py_backend.functions.pulse_train()`, :py:data:`pysd.py_backend.functions.pulse_magnitude()`, :py:data:`pysd.py_backend.functions.lookup()`, :py:data:`pysd.py_backend.functions.lookup_discrete()`, :py:data:`pysd.py_backend.functions.lookup_extrapolation()`, :py:data:`pysd.py_backend.functions.logical_and()`, :py:data:`pysd.py_backend.functions.logical_or()`, :py:data:`pysd.py_backend.functions.bounded_normal()`, :py:data:`pysd.py_backend.functions.log()`.
- Remove old translation and building files.


Bug fixes
~~~~~~~~~

- Generate the documentation of the model when loading it to avoid lossing information when replacing a variable value (:issue:`310`, :pull:`312`).
- Make random functions return arrays of the same shape as the variable, to avoid repeating values over a dimension (:issue:`309`, :pull:`312`).
- Fix bug when Vensim's :MACRO: definition is not at the top of the model file (:issue:`306`, :pull:`312`).
- Make builder identify the subscripts using a main range and subrange to allow using subscripts as numeric values as Vensim does (:issue:`296`, :issue:`301`, :pull:`312`).
- Fix bug of missmatching of functions and lookups names (:issue:`116`, :pull:`312`).
- Parse Xmile models case insensitively and ignoring the new lines characters (:issue:`203`, :issue:`253`, :pull:`312`).
- Add support for Vensim's `\:EXCEPT\: keyword <https://www.vensim.com/documentation/exceptionequations.html>`_ (:issue:`168`, :issue:`253`, :pull:`312`).
- Add spport for Xmile's FORCST and SAFEDIV functions (:issue:`154`, :pull:`312`).
- Add subscripts support for Xmile (:issue:`289`, :pull:`312`).
- Fix numeric error bug when using :py:data:`return_timestamps` and time step with non-integer values.

Documentation
~~~~~~~~~~~~~

- Review the whole documentation, refract it, and describe the new features.

Performance
~~~~~~~~~~~

- The variables defined in several equations are now assigned to a pre-allocated array instead of using :py:data:`pysd.py_backend.utils.xrmerge()`.
- The arranging and subseting of arrays is now done inplace instead of using the magic function :py:data:`pysd.py_backend.utils.rearrange()`.
- The grammars for Parsimonious are only compiled once per translation.

Internal Changes
~~~~~~~~~~~~~~~~
- The translation and the building of models has been totally modified to use the :doc:`Abstract Model Representation <structure/structure_index>`.
