
What's New
==========

v3.0.0 (unreleased)
-----------------------

New Features
~~~~~~~~~~~~

- Properties added to the :py:class:`Model` to make more accessible some information: :py:attr:`.namespace`, :py:attr:`.subscripts`, :py:attr:`.dependencies`, :py:attr:`.modules`, :py:attr:`.doc`.

Breaking changes
~~~~~~~~~~~~~~~~

- The argument :py:data:`flatten_output` from :py:meth:`.run` is now set to :py:data:`True` by default.
- The docstring of the model is now a property and thus is it not callable,:py:attr:`.doc`.
- Allow the function :py:func:`py_backend.functions.pulse` to also perform the operations performed by :py:data:`py_backend.functions.pulse_train()` and :py:data:`py_backend.functions.pulse_magnitude()`.
- The first argument of :py:func:`py_backend.functions.active_initial` now is the stage and not the time.
- The function :py:data:`py_backend.utils.rearrange` now its mutch simpler oriented to perform simple rearrange cases for user interaction.
- The translation and the building of models has been totally modified to use the :doc:`Abstract Model Representation <structure/structure_index>`.
- Move :py:data:`py_backend.statefuls.Model` and  :py:data:`py_backend.statefuls.Macro` to  :py:class:`py_backend.model.Model` and :py:class:`py_backend.model.Macro`, respectively.
- All kinds of lookups are now managed with the :py:class:`py_backend.lookups.Lookups` class.

Deprecations
~~~~~~~~~~~~

- Remove :py:data:`py_backend.utils.xrmerge()`, :py:data:`py_backend.functions.pulse_train()`, :py:data:`py_backend.functions.pulse_magnitude()`, :py:data:`py_backend.functions.lookup()`, :py:data:`py_backend.functions.lookup_discrete()`, :py:data:`py_backend.functions.lookup_extrapolation()`, :py:data:`py_backend.functions.logical_and()`, :py:data:`py_backend.functions.logical_or()`, :py:data:`py_backend.functions.bounded_normal()`, :py:data:`py_backend.functions.log()`.
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

- The variables defined in several equations are now assigned to a pre-allocated array instead of using :py:data:`py_backend.utils.xrmerge`. This improves the speed of subscripted models.
- The grammars for Parsimonious are only compiled once per translation.

Internal Changes
~~~~~~~~~~~~~~~~
- The translation and the building of models has been totally modified to use the :doc:`Abstract Model Representation <structure/structure_index>`.
