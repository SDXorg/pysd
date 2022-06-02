"""
The ModelBuilder class allows converting the AbstractModel into a
PySD model writing the Python code in files that can be loaded later
with PySD Model class. Each Abstract level has its own Builder. However,
the user is only required to create a ModelBuilder object using the
AbstractModel and call the `build_model` method.
"""
import textwrap
import black
import json
from pathlib import Path
from typing import Union

from pysd.translators.structures.abstract_model import\
    AbstractComponent, AbstractElement, AbstractModel, AbstractSection

from . import python_expressions_builder as vs
from .namespace import NamespaceManager
from .subscripts import SubscriptManager
from .imports import ImportsManager
from pysd._version import __version__


class ModelBuilder:
    """
    ModelBuilder allows building a PySD Python model from the
    Abstract Model.

    Parameters
    ----------
    abstract_model: AbstractModel
        The abstract model to build.

    """

    def __init__(self, abstract_model: AbstractModel):
        self.__dict__ = abstract_model.__dict__.copy()
        # load sections
        self.sections = [
            SectionBuilder(section)
            for section in abstract_model.sections
        ]
        # create the macrospace (namespace of macros)
        self.macrospace = {
            section.name: section for section in self.sections[1:]}

    def build_model(self) -> Path:
        """
        Build the Python model in a file callled as the orginal model
        but with '.py' suffix.

        Returns
        -------
        path: pathlib.Path
            The path to the new PySD model.

        """
        for section in self.sections:
            # add macrospace information to each section and build it
            section.macrospace = self.macrospace
            section.build_section()

        # return the path to the main file
        return self.sections[0].path


class SectionBuilder:
    """
    SectionBuilder allows building a section of the PySD model. Each
    section will be a file unless the model has been set to be split
    in modules.

    Parameters
    ----------
    abstract_section: AbstractSection
        The abstract section to build.

    """
    def __init__(self, abstract_section: AbstractSection):
        self.__dict__ = abstract_section.__dict__.copy()
        self.root = self.path.parent  # the folder where the model is
        self.model_name = self.path.with_suffix("").name  # name of the model
        # Create subscript manager object with subscripts_dict
        self.subscripts = SubscriptManager(
            abstract_section.subscripts, self.root)
        # Load the elements in the section
        self.elements = [
            ElementBuilder(element, self)
            for element in abstract_section.elements
        ]
        # Create the namespace of the section
        self.namespace = NamespaceManager(self.params)
        # Create an imports manager
        self.imports = ImportsManager()
        # Create macrospace (namespace of macros)
        self.macrospace = {}
        # Create parameters dict necessary in macros
        self.params = {
            key: self.namespace.namespace[key]
            for key in self.params
        }

    def build_section(self) -> None:
        """
        Build the Python section in a file callled as the orginal model
        if the section is main or in a file called as the macro name
        if the section is a macro.
        """
        # Firts iteration over elements to recover their information
        for element in self.elements:
            # Add element to namespace
            self.namespace.add_to_namespace(element.name)
            identifier = self.namespace.namespace[element.name]
            element.identifier = identifier
            # Add element subscripts information to the subscript manager
            self.subscripts.elements[identifier] = element.subscripts

        # Build elements
        for element in self.elements:
            element.build_element()

        if self.split:
            # Build modular section
            self._build_modular(self.views_dict)
        else:
            # Build one-file section
            self._build()

    def _process_views_tree(self, view_name: str,
                            view_content: Union[dict, set],
                            wdir: Path) -> dict:
        """
        Creates a directory tree based on the elements_per_view dictionary.
        If it's the final view, it creates a file, if not, it creates a folder.
        """
        if isinstance(view_content, set):
            # Will become a module
            # Convert subview elements names to Python names
            view_content = {
                self.namespace.cleanspace[var] for var in view_content
            }
            # Get subview elements
            subview_elems = [
                element for element in self.elements_remaining
                if element.identifier in view_content
            ]
            # Remove elements from remaining ones
            [
                self.elements_remaining.remove(element)
                for element in subview_elems
            ]
            # Build the module
            self._build_separate_module(subview_elems, view_name, wdir)
            return sorted(view_content)
        else:
            # The current view has subviews
            wdir = wdir.joinpath(view_name)
            wdir.mkdir(exist_ok=True)
            return {
                subview_name:
                self._process_views_tree(subview_name, subview_content, wdir)
                for subview_name, subview_content in view_content.items()
            }

    def _build_modular(self, elements_per_view: dict) -> None:
        """ Build modular section """
        self.elements_remaining = self.elements.copy()
        elements_per_view = self._process_views_tree(
            "modules_" + self.model_name, elements_per_view, self.root)
        # Building main file using the build function
        self._build_main_module(self.elements_remaining)

        # Build subscripts dir and moduler .json files
        for file, values in {
          "modules_%s/_modules": elements_per_view,
          "_subscripts_%s": self.subscripts.subscripts}.items():

            with self.root.joinpath(
                file % self.model_name).with_suffix(
                    ".json").open("w") as outfile:
                json.dump(values, outfile, indent=4, sort_keys=True)

    def _build_separate_module(self, elements: list, module_name: str,
                               module_dir: str) -> None:
        """
        Constructs and writes the Python representation of a specific model
        module, when the split_views=True in the read_vensim function.

        Parameters
        ----------
        elements: list
            Elements belonging to the module module_name.

        module_name: str
            Name of the module

        module_dir: str
            Path of the directory where module files will be stored.

        Returns
        -------
        None

        """
        text = textwrap.dedent('''
        """
        Module %(module_name)s
        Translated using PySD version %(version)s
        """
        ''' % {
            "module_name": module_name,
            "version": __version__,
        })
        funcs = self._generate_functions(elements)
        text += funcs
        text = black.format_file_contents(
            text, fast=True, mode=black.FileMode())

        outfile_name = module_dir.joinpath(module_name + ".py")

        with outfile_name.open("w", encoding="UTF-8") as out:
            out.write(text)

    def _build_main_module(self, elements: list) -> None:
        """
        Constructs and writes the Python representation of the main model
        module, when the split_views=True in the read_vensim function.

        Parameters
        ----------
        elements: list
            Elements belonging to the main module. Ideally, there should
            only be the initial_time, final_time, saveper and time_step,
            functions, though there might be others in some situations.
            Each element is a dictionary, with the various components
            needed to assemble a model component in Python syntax. This
            will contain multiple entries for elements that have multiple
            definitions in the original file, and which need to be combined.

        Returns
        -------
        None

        """
        # separating between control variables and rest of variables
        control_vars, funcs = self._build_variables(elements)

        self.imports.add("utils", "load_model_data")
        self.imports.add("utils", "load_modules")

        # import of needed functions and packages
        text = self.imports.get_header(self.path.name)

        # import subscript dict from json file
        text += textwrap.dedent("""
        __pysd_version__ = '%(version)s'

        __data = {
            'scope': None,
            'time': lambda: 0
        }

        _root = Path(__file__).parent
        %(params)s
        _subscript_dict, _modules = load_model_data(
            _root, "%(model_name)s")

        component = Component()
        """ % {
            "params": f"\n        _params = {self.params}\n"
                      if self.params else "",
            "model_name": self.model_name,
            "version": __version__
        })

        text += self._get_control_vars(control_vars)

        text += textwrap.dedent("""
            # load modules from modules_%(model_name)s directory
            exec(load_modules("modules_%(model_name)s", _modules, _root, []))

            """ % {
                "model_name": self.model_name,
            })

        text += funcs
        text = black.format_file_contents(
            text, fast=True, mode=black.FileMode())

        with self.path.open("w", encoding="UTF-8") as out:
            out.write(text)

    def _build(self) -> None:
        """
        Constructs and writes the Python representation of a section.

        Returns
        -------
        None

        """
        control_vars, funcs = self._build_variables(self.elements)

        text = self.imports.get_header(self.path.name)
        indent = "\n        "
        # Generate params dict for macro parameters
        params = f"{indent}_params = {self.params}\n"\
            if self.params else ""
        # Generate subscripts dir
        subs = f"{indent}_subscript_dict = {self.subscripts.subscripts}"\
            if self.subscripts.subscripts else ""

        text += textwrap.dedent("""
        __pysd_version__ = '%(version)s'

        __data = {
            'scope': None,
            'time': lambda: 0
        }

        _root = Path(__file__).parent
        %(params)s
        %(subscript_dict)s

        component = Component()
        """ % {
            "subscript_dict": subs,
            "params": params,
            "version": __version__,
        })

        text += self._get_control_vars(control_vars) + funcs

        text = black.format_file_contents(
            text, fast=True, mode=black.FileMode())

        with self.path.open("w", encoding="UTF-8") as out:
            out.write(text)

    def _build_variables(self, elements: dict) -> tuple:
        """
        Build model variables (functions) and separate then in control
        variables and regular variables.

        Returns
        -------
        control_vars, regular_vars: tuple, str
            control_vars is a tuple of length 2. First element is the
            dictionary of original control vars. Second is the string to
            add the control variables' functions. regular_vars is the
            string to add the regular variables' functions.

        """
        # returns of the control variables
        control_vars_dict = {
            "initial_time": "__data['time'].initial_time()",
            "final_time": "__data['time'].final_time()",
            "time_step": "__data['time'].time_step()",
            "saveper": "__data['time'].saveper()"
        }
        regular_vars = []
        control_vars = []

        for element in elements:
            if element.identifier in control_vars_dict:
                # change the return expression in the element and update
                # the dict with the original expression
                control_vars_dict[element.identifier], element.expression =\
                    element.expression, control_vars_dict[element.identifier]
                control_vars.append(element)
            else:
                regular_vars.append(element)

        if len(control_vars) == 0:
            # macro objects, no control variables
            control_vars_dict = ""
        else:
            control_vars_dict = """
        _control_vars = {
            "initial_time": lambda: %(initial_time)s,
            "final_time": lambda: %(final_time)s,
            "time_step": lambda: %(time_step)s,
            "saveper": lambda: %(saveper)s
        }
        """ % control_vars_dict

        return (control_vars_dict,
                self._generate_functions(control_vars)),\
            self._generate_functions(regular_vars)

    def _generate_functions(self, elements: dict) -> str:
        """
        Builds all model elements as functions in string format.
        NOTE: this function calls the build_element function, which
        updates the import_modules.
        Therefore, it needs to be executed before the method
        _generate_automatic_imports.

        Parameters
        ----------
        elements: dict
            Each element is a dictionary, with the various components
            needed to assemble a model component in Python syntax. This
            will contain multiple entries for elements that have multiple
            definitions in the original file, and which need to be combined.

        Returns
        -------
        funcs: str
            String containing all formated model functions

        """
        return "\n".join(
            [element._build_element_out() for element in elements]
        )

    def _get_control_vars(self, control_vars: str) -> str:
        """
        Create the section of control variables

        Parameters
        ----------
        control_vars: str
            Functions to define control variables.

        Returns
        -------
        text: str
            Control variables section and header of model variables section.

        """
        text = textwrap.dedent("""
        #######################################################################
        #                          CONTROL VARIABLES                          #
        #######################################################################
        %(control_vars_dict)s
        def _init_outer_references(data):
            for key in data:
                __data[key] = data[key]


        @component.add(name="Time")
        def time():
            '''
            Current time of the model.
            '''
            return __data['time']()

        """ % {"control_vars_dict": control_vars[0]})

        text += control_vars[1]

        text += textwrap.dedent("""
        #######################################################################
        #                           MODEL VARIABLES                           #
        #######################################################################
        """)

        return text


class ElementBuilder:
    """
    ElementBuilder allows building an element of the PySD model.

    Parameters
    ----------
    abstract_element: AbstractElement
        The abstract element to build.
    section: SectionBuilder
        The section where the element is defined. Necessary to give the
        acces to the subscripts and namespace.

    """
    def __init__(self, abstract_element: AbstractElement,
                 section: SectionBuilder):
        self.__dict__ = abstract_element.__dict__.copy()
        # Set element type and subtype to None
        self.type = None
        self.subtype = None
        # Get the arguments of the element
        self.arguments = getattr(self.components[0], "arguments", "")
        # Load the components of the element
        self.components = [
            ComponentBuilder(component, self, section)
            for component in abstract_element.components
        ]
        self.section = section
        # Get the subscripts of the element after merging all the components
        self.subscripts = section.subscripts.make_merge_list(
            [component.subscripts[0] for component in self.components])
        # Get the subscript dictionary of the element
        self.subs_dict = section.subscripts.make_coord_dict(self.subscripts)
        # Dictionaries to save dependencies and objects related to the element
        self.dependencies = {}
        self.other_dependencies = {}
        self.objects = {}

    def build_element(self) -> None:
        """
        Build the element. Returns the string to include in the section which
        will be a decorated function definition and possible objects.
        """
        # TODO think better how to build the components at once to build
        # in one declaration the external objects
        # TODO include some kind of magic vectorization to identify patterns
        # that can be easily vecorized (GET, expressions, Stocks...)

        # Build the components of the element
        [component.build_component() for component in self.components]
        expressions = []
        for component in self.components:
            expr, subs, except_subscripts = component.get()
            if expr is None:
                # The expr is None when the component has been "added"
                # to an existing object using the add method
                continue
            if isinstance(subs, list):
                # Get the list of locs for the component
                # Subscripts dict will be a list when the component is
                # translated to an object that groups may components
                # via 'add' method.
                loc = [vs.visit_loc(subsi, self.subs_dict, True)
                       for subsi in subs]
            else:
                # Get the loc of the component
                loc = vs.visit_loc(subs, self.subs_dict, True)

            # Get the locs of the :EXCLUDE: parameters if any
            exc_loc = [
                vs.visit_loc(subs_e, self.subs_dict, True)
                for subs_e in except_subscripts
            ]
            expressions.append({
                "expr": expr,
                "subs": subs,
                "loc": loc,
                "loc_except": exc_loc
            })

        if len(expressions) > 1:
            # NUMPY: xrmerge would be sustitute by a multiple line definition
            # e.g.:
            # value = np.empty((len(dim1), len(dim2)))
            # value[:, 0] = expression1
            # value[:, 1] = expression2
            # return value
            # This allows reference to the same variable
            # from: VAR[A] = 5; VAR[B] = 2*VAR[A]
            # to: value[0] = 5; value[1] = 2*value[0]
            self.section.imports.add("numpy")
            self.pre_expression =\
                "value = xr.DataArray(np.nan, {%s}, %s)\n" % (
                    ", ".join("'%(dim)s': _subscript_dict['%(dim)s']" %
                              {"dim": subs} for subs in self.subscripts),
                    self.subscripts)
            for expression in expressions:
                # Generate the pre_expression, operations to compute in
                # the body of the function
                if expression["expr"].subscripts:
                    # Get the values
                    # NUMPY not necessary
                    expression["expr"].lower_order(-1)
                    expression["expr"].expression += ".values"
                if expression["loc_except"]:
                    # There is an excep in the definition of the component
                    self.pre_expression += self._manage_except(expression)
                elif isinstance(expression["subs"], list):
                    # There are mixed definitions which include multicomponent
                    # object
                    self.pre_expression += self._manage_multi_def(expression)
                else:
                    # Regular loc for a component
                    self.pre_expression +=\
                        "value.loc[%(loc)s] = %(expr)s\n" % expression

            # Return value
            self.expression = "value"
        else:
            self.pre_expression = ""
            # NUMPY: reshape to the final shape if needed
            # expressions[0]["expr"].reshape(self.section.subscripts, {})
            if not expressions[0]["expr"].subscripts and self.subscripts:
                # Updimension the return value to an array
                self.expression = "xr.DataArray(%s, %s, %s)\n" % (
                     expressions[0]["expr"],
                     self.section.subscripts.simplify_subscript_input(
                         self.subs_dict)[1],
                     list(self.subs_dict)
                )
            else:
                # Return the expression
                self.expression = expressions[0]["expr"]

        # Merge the types of the components (well defined element should
        # have only one type and subtype)
        self.type = ", ".join(
            set(component.type for component in self.components)
        )
        self.subtype = ", ".join(
            set(component.subtype for component in self.components)
        )

    def _manage_multi_def(self, expression: dict) -> str:
        """
        Manage multiline definitions when some of them (not all) are
        merged to one object.
        """
        final_expr = "def_subs = xr.zeros_like(value, dtype=bool)\n"
        for loc in expression["loc"]:
            # coordinates of the object
            final_expr += f"def_subs.loc[{loc}] = True\n"

        # replace the values matching the coordinates
        return final_expr + "value.values[def_subs.values] = "\
            "%(expr)s[def_subs.values]\n" % expression

    def _manage_except(self, expression: dict) -> str:
        """
        Manage except declarations by not asigning its values.
        """
        if expression["subs"] == self.subs_dict:
            # Final subscripts are the same as the main subscripts
            # of the component. Generate a True array like value
            final_expr = "except_subs = xr.ones_like(value, dtype=bool)\n"
        else:
            # Final subscripts are greater than the main subscripts
            # of the component. Generate a False array like value and
            # set to True the subarray of the component coordinates
            final_expr = "except_subs = xr.zeros_like(value, dtype=bool)\n"\
                         "except_subs.loc[%(loc)s] = True\n" % expression

        for except_subs in expression["loc_except"]:
            # We set to False the dimensions in the EXCEPT
            final_expr += "except_subs.loc[%s] = False\n" % except_subs

        if expression["expr"].subscripts:
            # Assign the values of an array
            return final_expr + "value.values[except_subs.values] = "\
                "%(expr)s[except_subs.values]\n" % expression
        else:
            # Assign the values of a float
            return final_expr + "value.values[except_subs.values] = "\
                "%(expr)s\n" % expression

    def _build_element_out(self) -> str:
        """
        Returns a string that has processed a single element dictionary.

        Returns
        -------
        func: str
            The function to write in the model file.

        """
        # Contents of the function (body + return)
        contents = self.pre_expression + "return %s" % self.expression

        # Get the objects to create as string
        objects = "\n\n".join([
            value["expression"] % {
                "final_subs":
                self.section.subscripts.simplify_subscript_input(
                    value.get("final_subs", {}))[1]
            }   # Replace the final subs in the objects that merge
                # several components
            for value in self.objects.values()
            if value["expression"] is not None
        ])

        # Format the limits to get them as a string
        self.limits = self._format_limits(self.limits)

        # Update arguments with final subs to alllow passing arguments
        # with subscripts to the lookups
        if self.arguments == 'x':
            self.arguments = 'x, final_subs=None'

        # Define variable metadata for the @component decorator
        self.name = repr(self.name)
        meta_data = ["name=%(name)s"]

        # Include basic metadata (units, limits, dimensions)
        if self.units:
            meta_data.append("units=%(units)s")
            self.units = repr(self.units)
        if self.limits:
            meta_data.append("limits=%(limits)s")
        if self.subscripts:
            self.section.imports.add("subs")
            meta_data.append("subscripts=%(subscripts)s")

        # Include component type and subtype
        meta_data.append("comp_type='%(type)s'")
        meta_data.append("comp_subtype='%(subtype)s'")

        # Include dependencies
        if self.dependencies:
            meta_data.append("depends_on=%(dependencies)s")
        if self.other_dependencies:
            meta_data.append("other_deps=%(other_dependencies)s")

        # Get metadata decorator
        self.meta_data = f"@component.add({', '.join(meta_data)})"\
            % self.__dict__

        # Clean the documentation and add it to the beggining of contents
        if self.documentation:
            doc = self.documentation.replace("\\", "\n")
            contents = f'"""\n{doc}\n"""\n'\
                + contents

        indent = 12

        # Convert newline indicator and add expected level of indentation
        self.contents = contents.replace("\n", "\n" + " " * (indent+4))
        self.objects = objects.replace("\n", "\n" + " " * indent)

        # Return the decorated function definition with the object declarations
        return textwrap.dedent('''
            %(meta_data)s
            def %(identifier)s(%(arguments)s):
                %(contents)s


            %(objects)s
            ''' % self.__dict__)

    def _format_limits(self, limits: tuple) -> str:
        """Format the limits of an element to print them properly"""
        if limits == (None, None):
            return None

        new_limits = []
        for value in limits:
            value = repr(value)
            if value == "nan" or value == "None":
                # add numpy.nan to the values
                self.section.imports.add("numpy")
                new_limits.append("np.nan")
            elif value.endswith("inf"):
                # add numpy.inf to the values
                self.section.imports.add("numpy")
                new_limits.append(value.strip("inf") + "np.inf")
            else:
                # add numeric value
                new_limits.append(value)

        if new_limits[0] == "np.nan" and new_limits[1] == "np.nan":
            # if both are numpy.nan do not include limits
            return None

        return "(" + ", ".join(new_limits) + ")"


class ComponentBuilder:
    """
    ComponentBuilder allows building a component of the PySD model.

    Parameters
    ----------
    abstract_component: AbstracComponent
        The abstract component to build.
    element: ElementBuilder
        The element where the component is defined. Necessary to give the
        acces to the merging subscripts and other components.
    section: SectionBuilder
        The section where the element is defined. Necessary to give the
        acces to the subscripts and namespace.

    """
    def __init__(self, abstract_component: AbstractComponent,
                 element: ElementBuilder, section: SectionBuilder):
        self.__dict__ = abstract_component.__dict__.copy()
        self.element = element
        self.section = section
        if not hasattr(self, "keyword"):
            self.keyword = None

    def build_component(self) -> None:
        """
        Build model component parsing the Abstract Syntax Tree.
        """
        self.subscripts_dict = self.section.subscripts.make_coord_dict(
            self.subscripts[0])
        self.except_subscripts = [self.section.subscripts.make_coord_dict(
            except_list) for except_list in self.subscripts[1]]
        self.ast_build = vs.ASTVisitor(self).visit()

    def get(self) -> tuple:
        """
        Get build component to build the element.

        Returns
        -------
        ast_build: BuildAST
            Parsed AbstractSyntaxTree.
        subscript_dict: dict or list of dicts
            The subscripts of the component.
        except_subscripts: list of dicts
            The subscripts to avoid.

        """
        return self.ast_build, self.subscripts_dict, self.except_subscripts
