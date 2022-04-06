import textwrap
import black
import json

from pysd.translation.structures.abstract_model import\
    AbstractComponent, AbstractElement, AbstractModel, AbstractSection

from . import python_expressions_builder as vs
from .namespace import NamespaceManager
from .subscripts import SubscriptManager
from .imports import ImportsManager
from pysd._version import __version__


class ModelBuilder:

    def __init__(self, abstract_model: AbstractModel):
        self.__dict__ = abstract_model.__dict__.copy()
        self.sections = [
            SectionBuilder(section)
            for section in abstract_model.sections
        ]
        self.macrospace = {
            section.name: section for section in self.sections[1:]}

    def build_model(self):
        # TODO: add special building for main
        for section in self.sections:
            section.macrospace = self.macrospace
            section.build_section()

        return self.sections[0].path


class SectionBuilder:

    def __init__(self, abstract_section: AbstractSection):
        self.__dict__ = abstract_section.__dict__.copy()
        self.root = self.path.parent
        self.model_name = self.path.with_suffix("").name
        self.subscripts = SubscriptManager(
            abstract_section.subscripts, self.root)
        self.elements = [
            ElementBuilder(element, self)
            for element in abstract_section.elements
        ]
        self.namespace = NamespaceManager(self.params)
        self.imports = ImportsManager()
        self.macrospace = {}
        self.dependencies = {}

    def build_section(self):
        # Create namespace
        for element in self.elements:
            self.namespace.add_to_namespace(element.name)
            identifier = self.namespace.namespace[element.name]
            element.identifier = identifier
            self.subscripts.elements[identifier] = element.subscripts

        for element in self.elements:
            element.build_element()
            self.dependencies[element.identifier] = element.dependencies
            for subelement in element.objects.values():
                if "calls" in subelement:
                    self.dependencies[subelement["name"]] = subelement["calls"]

        if self.split:
            self._build_modular(self.views_dict)
        else:
            self._build()

    def process_views_tree(self, view_name, view_content, wdir):
        """
        Creates a directory tree based on the elements_per_view dictionary.
        If it's the final view, it creates a file, if not, it creates a folder.
        """
        if isinstance(view_content, set):
            # will become a module

            # convert subview elements names to python names
            view_content = {
                self.namespace.cleanspace[var] for var in view_content
            }

            # get subview elements
            subview_elems = [
                element for element in self.elements_remaining
                if element.identifier in view_content
            ]

            # remove elements from remaining ones
            [
                self.elements_remaining.remove(element)
                for element in subview_elems
            ]

            self._build_separate_module(subview_elems, view_name, wdir)

            return sorted(view_content)

        else:
            # the current view has subviews
            wdir = wdir.joinpath(view_name)
            wdir.mkdir(exist_ok=True)
            return {
                subview_name:
                self.process_views_tree(subview_name, subview_content, wdir)
                for subview_name, subview_content in view_content.items()
            }

    def _build_modular(self, elements_per_view):
        self.elements_remaining = self.elements.copy()
        elements_per_view = self.process_views_tree(
            "modules_" + self.model_name, elements_per_view, self.root)
        # building main file using the build function
        self._build_main_module(self.elements_remaining)

        for file, values in {
          "modules_%s/_modules": elements_per_view,
          "_namespace_%s": self.namespace.namespace,
          "_subscripts_%s": self.subscripts.subscripts,
          "_dependencies_%s": self.dependencies}.items():

            with self.root.joinpath(
                file % self.model_name).with_suffix(
                    ".json").open("w") as outfile:
                json.dump(values, outfile, indent=4, sort_keys=True)

    def _build_separate_module(self, elements, module_name, module_dir):
        """
        Constructs and writes the python representation of a specific model
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

    def _build_main_module(self, elements):
        """
        Constructs and writes the python representation of the main model
        module, when the split_views=True in the read_vensim function.

        Parameters
        ----------
        elements: list
            Elements belonging to the main module. Ideally, there should
            only be the initial_time, final_time, saveper and time_step,
            functions, though there might be others in some situations.
            Each element is a dictionary, with the various components
            needed to assemble a model component in python syntax. This
            will contain multiple entries for elements that have multiple
            definitions in the original file, and which need to be combined.

        Returns
        -------
        None or text: None or str
            If file_name="return" it will return the content of the output file
            instead of saving it. It is used for testing.

        """
        # separating between control variables and rest of variables
        control_vars, funcs = self._build_variables(elements)

        self.imports.add("utils", "load_model_data")
        self.imports.add("utils", "load_modules")

        # import of needed functions and packages
        text = self.imports.get_header(self.path.name)

        # import namespace from json file
        text += textwrap.dedent("""
        __pysd_version__ = '%(version)s'

        __data = {
            'scope': None,
            'time': lambda: 0
        }

        _root = Path(__file__).parent

        _namespace, _subscript_dict, _dependencies, _modules = load_model_data(
            _root, "%(model_name)s")
        """ % {
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

    def _build(self):
        control_vars, funcs = self._build_variables(self.elements)

        text = self.imports.get_header(self.path.name)
        text += textwrap.dedent("""
        __pysd_version__ = '%(version)s'

        __data = {
            'scope': None,
            'time': lambda: 0
        }

        _root = Path(__file__).parent

        _subscript_dict = %(subscript_dict)s

        _namespace = %(namespace)s

        _dependencies = %(dependencies)s
        """ % {
            "subscript_dict": repr(self.subscripts.subscripts),
            "namespace": repr(self.namespace.namespace),
            "dependencies": repr(self.dependencies),
            "version": __version__,
        })

        text += self._get_control_vars(control_vars) + funcs

        text = black.format_file_contents(
            text, fast=True, mode=black.FileMode())

        with self.path.open("w", encoding="UTF-8") as out:
            out.write(text)

    def _build_variables(self, elements):
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

    def _generate_functions(self, elements):
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
            needed to assemble a model component in python syntax. This
            will contain multiple entries for elements that have multiple
            definitions in the original file, and which need to be combined.

        Returns
        -------
        funcs: str
            String containing all formated model functions

        """
        return "\n".join([element.build_element_out() for element in elements])

    def _get_control_vars(self, control_vars):
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


        def time():
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

    def __init__(self, abstract_element: AbstractElement,
                 section: SectionBuilder):
        self.__dict__ = abstract_element.__dict__.copy()
        self.type = None
        self.subtype = None
        self.arguments = getattr(self.components[0], "arguments", "")
        self.components = [
            ComponentBuilder(component, self, section)
            for component in abstract_element.components
        ]
        self.section = section
        self.subscripts = section.subscripts.make_merge_list(
            [component.subscripts[0] for component in self.components])
        self.subs_dict = section.subscripts.make_coord_dict(self.subscripts)
        self.dependencies = {}
        self.objects = {}

    def build_element(self):
        # TODO think better how to build the components at once to build
        # in one declaration the external objects
        # TODO include some kind of magic vectorization to identify patterns
        # that can be easily vecorized (GET, expressions, Stocks...)
        expressions = []
        [component.build_component() for component in self.components]
        for component in self.components:
            expr, subs, except_subscripts = component.get()
            if expr is None:
                continue
            if isinstance(subs, list):
                subs = [{
                    esubs: subsi[csubs]
                    for csubs, esubs in zip(subsi, self.subscripts)
                } for subsi in subs]
            else:
                subs = {
                    esubs: subs[csubs]
                    for csubs, esubs in zip(subs, self.subscripts)
                }
            exc_subs = [
                {
                    esubs: subs_e[csubs]
                    for csubs, esubs in zip(subs_e, self.subscripts)
                }
                for subs_e in except_subscripts
            ]
            expressions.append(
                {"expr": expr, "subs": subs, "subs_except": exc_subs})

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
                if expression["expr"].subscripts:
                    # get the values
                    # NUMPY not necessary
                    expression["expr"].lower_order(0, force_0=True)
                    expression["expr"].expression += ".values"
                if expression["subs_except"]:
                    # there is an excep in the definition of the component
                    self.pre_expression += self.manage_except(expression)
                elif isinstance(expression["subs"], list):
                    self.pre_expression += self.manage_multi_def(expression)
                else:
                    self.pre_expression +=\
                        "value.loc[%(subs)s] = %(expr)s\n" % expression
            self.expression = "value"
        else:
            self.pre_expression = ""
            # NUMPY: reshape to the final shape if meeded
            # expressions[0]["expr"].reshape(self.section.subscripts, {})
            self.expression = expressions[0]["expr"]

        self.type = ", ".join(
            set(component.type for component in self.components)
        )
        self.subtype = ", ".join(
            set(component.subtype for component in self.components)
        )

    def manage_multi_def(self, expression):
        final_expr = "def_subs = xr.zeros_like(value, dtype=bool)\n"
        for subs in expression["subs"]:
            final_expr += "def_subs.loc[%s] = True\n" % subs

        return final_expr + "value.values[def_subs.values] = "\
            "%(expr)s[def_subs.values]\n" % expression

    def manage_except(self, expression):
        if expression["subs"] == self.subs_dict:
            # Final subscripts are the same as the main subscripts
            # of the component. Generate a True array like value
            final_expr = "except_subs = xr.ones_like(value, dtype=bool)\n"
        else:
            # Final subscripts are greater than the main subscripts
            # of the component. Generate a False array like value and
            # set to True the subarray of the component coordinates
            final_expr = "except_subs = xr.zeros_like(value, dtype=bool)\n"\
                         "except_subs.loc[%(subs)s] = True\n" % expression

        for except_subs in expression["subs_except"]:
            # We set to False the dimensions in the EXCEPT
            final_expr += "except_subs.loc[%s] = False\n" % except_subs

        if expression["expr"].subscripts:
            # assign the values of an array
            return final_expr + "value.values[except_subs.values] = "\
                "%(expr)s[except_subs.values]\n" % expression
        else:
            # assign the values of a float
            return final_expr + "value.values[except_subs.values] = "\
                "%(expr)s\n" % expression

    def build_element_out(self):
        """
        Returns a string that has processed a single element dictionary.

        Returns
        -------
        func: str
            The function to write in the model file.

        """
        # TODO: merge with the previous build to do all at once
        contents = self.pre_expression + "return %s" % self.expression

        self.subs_dec = ""
        self.subs_doc = "None"

        if self.subscripts:
            # We add the list of the subs to the __doc__ of the function
            # this will give more information to the user and make possible
            # to rewrite subscripted values with model.run(params=X) or
            # model.run(initial_condition=(n,x))
            self.subs_doc = "%s" % self.subscripts
            self.subs_dec =\
                "@subs(%s, _subscript_dict)" % self.subscripts
            self.section.imports.add("subs")

        objects = "\n\n".join([
            value["expression"] % {
                "final_subs": value.get("final_subs", "")}
            for value in self.objects.values()
            if value["expression"] is not None
        ])

        indent = 12

        # convert newline indicator and add expected level of indentation
        self.contents = contents.replace("\n", "\n" + " " * (indent+4))
        self.objects = objects.replace("\n", "\n" + " " * indent)
        self.documentation = self.documentation.replace(
            "\\", "\n").replace("\n", "\n" + " " * indent)

        return textwrap.dedent('''
            %(subs_dec)s
            def %(identifier)s(%(arguments)s):
                """
                Real Name: %(name)s
                Units: %(units)s
                Limits: %(range)s
                Type: %(type)s
                Subtype: %(subtype)s
                Subs: %(subscripts)s

                %(documentation)s
                """
                %(contents)s


            %(objects)s
            ''' % self.__dict__)


class ComponentBuilder:

    def __init__(self, abstract_component: AbstractComponent,
                 element: ElementBuilder, section: SectionBuilder):
        self.__dict__ = abstract_component.__dict__.copy()
        self.element = element
        self.section = section
        if not hasattr(self, "keyword"):
            self.keyword = None

    def build_component(self):
        self.subscripts_dict = self.section.subscripts.make_coord_dict(
            self.subscripts[0])
        # NUMPY: use vs.ExceptVisitor
        self.except_subscripts = [self.section.subscripts.make_coord_dict(
            except_list) for except_list in self.subscripts[1]]
        self.ast_build = vs.ASTVisitor(self).visit()

    def get(self):
        return self.ast_build, self.subscripts_dict, self.except_subscripts
