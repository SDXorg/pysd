PySD Design Philosophy
======================

* Do as little as possible.
 * Anything that is not endemic to System Dynamics (such as plotting, integration, fitting, etc) should either be implemented using external tools, or omitted.
 * Stick to SD. Let other disciplines (ABM, Discrete Event Simulation, etc) create their own tools.
 * Use external model creation tools
* Use the language of system dynamics.
* Be simple to use. Let SD practitioners who haven't used python before understand the basics.
* Take advantage of general python constructions and best practices.
* Be simple to maintain.