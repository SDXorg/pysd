import warnings
from dataclasses import dataclass

import numpy as np
from pysd.py_backend.utils import compute_shape

from pysd.translation.structures import abstract_expressions as ae
from .python_functions import functionspace


@dataclass
class BuildAST:
    expression: str
    calls: dict
    subscripts: dict
    order: int

    def __str__(self):
        # makes easier building
        return self.expression

    def reshape(self, subscripts, final_subscripts):
        subscripts_out = subscripts.simplify_subscript_input(
                final_subscripts, list(final_subscripts))[1]
        if not final_subscripts or (
          self.subscripts == final_subscripts
          and list(self.subscripts) == list(final_subscripts)):
            # same dictionary in the same orde, do nothing
            pass
        elif not self.subscripts:
            # original expression is not an array
            # NUMPY: object.expression = np.full(%s, %(shape)s)
            self.expression = "xr.DataArray(%s, %s, %s)" % (
                self.expression, subscripts_out, list(final_subscripts)
            )
            self.order = 0
        else:
            # original expression is an array
            # NUMPY: reorder dims if neccessary with np.moveaxis or similar
            # NUMPY: add new axis with [:, None, :] or np.tile,
            #        depending on an input argument
            # NUMPY: if order is not 0 need to lower the order to 0
            # using force!
            self.expression = "(xr.DataArray(0, %s, %s) + %s)" % (
                subscripts_out, list(final_subscripts), self.expression
                )
            self.order = 0
        self.subscripts = final_subscripts

    def lower_order(self, new_order, force_0=False):
        if self.order >= new_order and self.order != 0\
          and (new_order != 0 or force_0):
            # if current operator order is 0 do not need to do anything
            # if the order of operations conflicts add parenthesis
            # if new order is 0 do not need to do anything, as it may be
            # an argument to a function, unless force_0 is True which
            # will force the parenthesis (necessary to reshape some
            # numpy arrays)
            self.expression = "(%s)" % self.expression
            self.order = 0


class StructureBuilder:
    def __init__(self, value, component):
        self.value = value
        self.arguments = {}
        self.component = component
        self.element = component.element
        self.section = component.section
        self.def_subs = component.subscripts_dict

    def build(self, arguments):
        return BuildAST(
            expression=repr(self.value),
            calls={},
            subscripts={},
            order=0)

    def join_calls(self, arguments):
        if len(arguments) == 0:
            return {}
        elif len(arguments) == 1:
            return arguments["0"].calls
        else:
            return merge_dependencies(
                *[val.calls for val in arguments.values()])

    def reorder(self, arguments, def_subs=None, force=None):

        if force == "component":
            final_subscripts = def_subs or {}
        else:
            final_subscripts = self.get_final_subscripts(
                arguments, def_subs)

        [arguments[key].reshape(self.section.subscripts, final_subscripts)
         for key in arguments
         if arguments[key].subscripts or force == "equal"]

        return final_subscripts

    def get_final_subscripts(self, arguments, def_subs):
        if len(arguments) == 0:
            return {}
        elif len(arguments) == 1:
            return arguments["0"].subscripts
        else:
            return self._compute_final_subscripts(
                [arg.subscripts for arg in arguments.values()],
                def_subs)

    def _compute_final_subscripts(self, subscripts_list, def_subs):
        expression = {}
        [expression.update(subscript)
         for subscript in subscripts_list if subscript]
        # TODO reorder final_subscripts taking into account def_subs
        return expression

    def update_object_subscripts(self, name, component_final_subs):
        origin_comp = self.element.objects[name]["component"]
        if isinstance(origin_comp.subscripts_dict, dict):
            if len(list(origin_comp.subscripts_dict)) == 1:
                key = list(origin_comp.subscripts_dict.keys())[0]
                value = list(component_final_subs.values())[0]
                origin_comp.subscripts_dict[key] += value
                self.element.objects[name]["final_subs"] =\
                    origin_comp.subscripts_dict
            else:
                origin_comp.subscripts_dict = [origin_comp.subscripts_dict]
                self.element.objects[name]["final_subs"] =\
                    self.element.subs_dict
        if isinstance(origin_comp.subscripts_dict, list):
            origin_comp.subscripts_dict.append(component_final_subs)


class OperationBuilder(StructureBuilder):
    operators_build = {
        "^": ("%(left)s**%(right)s", None, 1),
        "*": ("%(left)s*%(right)s", None, 2),
        "/": ("%(left)s/%(right)s", None, 2),
        "+": ("%(left)s + %(right)s", None, 3),
        "-": ("%(left)s - %(right)s", None, 3),
        "=": ("%(left)s == %(right)s", None, 4),
        "<>": ("%(left)s != %(right)s", None, 4),
        ">=": ("%(left)s >= %(right)s", None, 4),
        ">": ("%(left)s > %(right)s", None, 4),
        "<=": ("%(left)s <= %(right)s", None, 4),
        "<": ("%(left)s < %(right)s", None, 4),
        ":NOT:": ("np.logical_not(%s)", ("numpy",), 0),
        ":AND:": ("np.logical_and(%(left)s, %(right)s)", ("numpy",), 0),
        ":OR:": ("np.logical_or(%(left)s, %(right)s)", ("numpy",), 0),
        "negative": ("-%s", None, 3),
    }

    def __init__(self, operation, component):
        super().__init__(None, component)
        self.operators = operation.operators.copy()
        self.arguments = {
            str(i): arg for i, arg in enumerate(operation.arguments)}

    def build(self, arguments):
        operands = {}
        calls = self.join_calls(arguments)
        final_subscripts = self.reorder(arguments, def_subs=self.def_subs)
        arguments = [arguments[str(i)] for i in range(len(arguments))]
        dependencies, order = self.operators_build[self.operators[-1]][1:]

        if dependencies:
            self.section.imports.add(*dependencies)

        if self.operators[-1] == "^":
            # right side of the exponential can be from higher order
            arguments[-1].lower_order(2)
        else:
            arguments[-1].lower_order(order)

        if len(arguments) == 1:
            # not and negative operations (only 1 element)
            if self.operators[0] == "negative":
                order = 1
            expression = self.operators_build[self.operators[0]][0]
            return BuildAST(
                expression=expression % arguments[0],
                calls=calls,
                subscripts=final_subscripts,
                order=order)

        operands["right"] = arguments.pop()
        while arguments or self.operators:
            expression = self.operators_build[self.operators.pop()][0]
            operands["left"] = arguments.pop()
            operands["left"].lower_order(order)
            operands["right"] = expression % operands

        return BuildAST(
            expression=operands["right"],
            calls=calls,
            subscripts=final_subscripts,
            order=order)


class GameBuilder(StructureBuilder):
    def __init__(self, game_str, component):
        super().__init__(None, component)
        self.arguments = {"expr": game_str.expression}

    def build(self, arguments):
        return arguments["expr"]


class CallBuilder(StructureBuilder):
    def __init__(self, call_str, component):
        super().__init__(None, component)
        function_name = call_str.function.reference
        self.arguments = {
            str(i): arg for i, arg in enumerate(call_str.arguments)}
        # move this to a setter
        if function_name in self.section.macrospace:
            # build macro
            self.macro_name = function_name
            self.build = self.build_macro_call
        elif function_name in self.section.namespace.cleanspace:
            # build lookupcall
            self.arguments["function"] = call_str.function
            self.build = self.build_lookups_call
        elif function_name in functionspace:
            # build direct function
            self.function = function_name
            self.build = self.build_function_call
        elif function_name == "a_function_of":
            self.build = self.build_incomplete_call
        else:
            self.function = function_name
            self.build = self.build_not_implemented

    def build_not_implemented(self, arguments):
        final_subscripts = self.reorder(arguments, def_subs=self.def_subs)
        warnings.warn(
            "\n\nTrying to translate "
            + self.function
            + " which it is not implemented on PySD. The translated "
            + "model will crash... "
        )
        self.section.imports.add("functions", "not_implemented_function")

        return BuildAST(
            expression="not_implemented_function('%s', %s)" % (
                self.function,
                ", ".join(arg.expression for arg in arguments.values())),
            calls=self.join_calls(arguments),
            subscripts=final_subscripts,
            order=0)

    def build_macro_call(self, arguments):
        self.section.imports.add("statefuls", "Macro")
        macro = self.section.macrospace[self.macro_name]

        calls = self.join_calls(arguments)
        final_subscripts = self.reorder(arguments, def_subs=self.def_subs)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.macro_name + "_" + self.element.identifier, prefix="_macro")
        arguments["file"] = macro.path.name
        arguments["macro_name"] = macro.name
        arguments["args"] = "{%s}" % ", ".join([
            "'%s': lambda: %s" % (key, val)
            for key, val in zip(macro.params, arguments.values())
        ])

        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = Macro(_root.joinpath('%(file)s'), "
                          "%(args)s, '%(macro_name)s', "
                          "time_initialization=lambda: __data['time'], "
                          "py_name='%(name)s')" % arguments,
            "calls": {
                "initial": calls,
                "step": calls
            }
        }
        return BuildAST(
            expression="%s()" % arguments["name"],
            calls={arguments["name"]: 1},
            subscripts=final_subscripts,
            order=0)

    def build_incomplete_call(self, arguments):
        warnings.warn(
            "%s has no equation specified" % self.element.name,
            SyntaxWarning, stacklevel=2
        )
        self.section.imports.add("functions", "incomplete")
        return BuildAST(
            expression="incomplete(%s)" % ", ".join(
                arg.expression for arg in arguments.values()),
            calls=self.join_calls(arguments),
            subscripts=self.def_subs,
            order=0)

    def build_lookups_call(self, arguments):
        if arguments["0"].subscripts:
            final_subscripts =\
                self.get_final_subscripts(arguments, self.def_subs)
            expression = arguments["function"].expression.replace(
                "()", f"(%(0)s, {final_subscripts})")
        else:
            final_subscripts = arguments["function"].subscripts
            expression = arguments["function"].expression.replace(
                "()", "(%(0)s)")

        # NUMPY: we need to manage inside lookup with subscript and later
        #        return the values in a correct ndarray
        return BuildAST(
            expression=expression % arguments,
            calls=self.join_calls(arguments),
            subscripts=final_subscripts,
            order=0)

    def build_function_call(self, arguments):
        expression, modules = functionspace[self.function]
        if modules:
            self.section.imports.add(*modules)

        calls = self.join_calls(arguments)

        if "__data['time']" in expression:
            merge_dependencies(calls, {"time": 1}, inplace=True)

        # TODO modify dimensions of BuildAST
        if "%(axis)s" in expression:
            final_subscripts, arguments["axis"] = self.compute_axis(arguments)

        elif "%(size)s" in expression:
            final_subscripts = self.reorder(
                arguments,
                def_subs=self.def_subs,
                force="component"
            )
            arguments["size"] = compute_shape(final_subscripts)

        elif self.function == "active_initial":
            # we need to ensure that active initial outputs are always the
            # same and update dependencies as stateful object
            name = self.section.namespace.make_python_identifier(
                self.element.identifier, prefix="_active_initial")
            final_subscripts = self.reorder(
                arguments,
                def_subs=self.def_subs,
                force="equal"
            )
            self.element.objects[name] = {
                "name": name,
                "expression": None,
                "calls": {
                    "initial": arguments["1"].calls,
                    "step": arguments["0"].calls
                }

            }
            calls = {name: 1}
        else:
            final_subscripts = self.reorder(
                arguments,
                def_subs=self.def_subs
            )
            if self.function == "xidz" and final_subscripts:
                if not arguments["1"].subscripts:
                    new_args = {"0": arguments["0"], "2": arguments["2"]}
                    self.reorder(
                        new_args,
                        def_subs=self.def_subs,
                        force="equal"
                    )
                    arguments.update(new_args)
            if self.function == "if_then_else" and final_subscripts:
                if not arguments["0"].subscripts:
                    # NUMPY: we need to ensure that if_then_else always returs
                    # the same shape object
                    new_args = {"1": arguments["1"], "2": arguments["2"]}
                    self.reorder(
                        new_args,
                        def_subs=self.def_subs,
                        force="equal"
                    )
                    arguments.update(new_args)
                else:
                    self.reorder(
                        arguments,
                        def_subs=self.def_subs,
                        force="equal"
                    )

        return BuildAST(
            expression=expression % arguments,
            calls=calls,
            subscripts=final_subscripts,
            order=0)

    def compute_axis(self, arguments):
        subscripts = arguments["0"].subscripts
        axis = []
        coords = {}
        for subs in subscripts:
            if subs.endswith("!"):
                # dimensions to apply along
                axis.append(subs)
            else:
                # dimensions remaining
                coords[subs] = subscripts[subs]
        return coords, axis


class ExtLookupBuilder(StructureBuilder):
    def __init__(self, getlookup_str,  component):
        super().__init__(None, component)
        self.file = getlookup_str.file
        self.tab = getlookup_str.tab
        self.x_row_or_col = getlookup_str.x_row_or_col
        self.cell = getlookup_str.cell
        self.arguments = {}

    def build(self, arguments):
        self.component.type = "Lookup"
        self.component.subtype = "External"
        arguments["params"] = "'%s', '%s', '%s', '%s'" % (
            self.file, self.tab, self.x_row_or_col, self.cell
        )
        final_subs, arguments["subscripts"] =\
            self.section.subscripts.simplify_subscript_input(
                self.def_subs, self.element.subscripts)

        if "ext_lookups" in self.element.objects:
            # object already exists
            self.element.objects["ext_lookups"]["expression"] += "\n\n"\
                + self.element.objects["ext_lookups"]["name"]\
                + ".add(%(params)s, %(subscripts)s)" % arguments

            self.update_object_subscripts("ext_lookups", final_subs)

            return None
        else:
            # create a new object
            self.section.imports.add("external", "ExtLookup")

            arguments["name"] = self.section.namespace.make_python_identifier(
                self.element.identifier, prefix="_ext_lookup")
            arguments["final_subs"] = "%(final_subs)s"
            self.component.subscripts_dict = final_subs

            self.element.objects["ext_lookups"] = {
                "name": arguments["name"],
                "expression": "%(name)s = ExtLookup(%(params)s, "
                              "%(subscripts)s, _root, "
                              "%(final_subs)s , '%(name)s')" % arguments,
                "component": self.component,
                "final_subs": final_subs
            }

            return BuildAST(
                expression=arguments["name"] + "(x, final_subs)",
                calls={
                    "__external__": arguments["name"],
                    "__lookup__": arguments["name"]
                },
                subscripts=final_subs,
                order=0)


class ExtDataBuilder(StructureBuilder):
    def __init__(self, getdata_str,  component):
        super().__init__(None, component)
        self.file = getdata_str.file
        self.tab = getdata_str.tab
        self.time_row_or_col = getdata_str.time_row_or_col
        self.cell = getdata_str.cell
        self.keyword = component.keyword
        self.arguments = {}

    def build(self, arguments):
        self.component.type = "Data"
        self.component.subtype = "External"
        arguments["params"] = "'%s', '%s', '%s', '%s'" % (
            self.file, self.tab, self.time_row_or_col, self.cell
        )
        final_subs, arguments["subscripts"] =\
            self.section.subscripts.simplify_subscript_input(
                self.def_subs, self.element.subscripts)
        arguments["method"] = "'%s'" % self.keyword if self.keyword else None

        if "ext_data" in self.element.objects:
            # object already exists
            self.element.objects["ext_data"]["expression"] += "\n\n"\
                + self.element.objects["ext_data"]["name"]\
                + ".add(%(params)s, %(method)s, %(subscripts)s)" % arguments

            self.update_object_subscripts("ext_data", final_subs)

            return None
        else:
            # create a new object
            self.section.imports.add("external", "ExtData")

            arguments["name"] = self.section.namespace.make_python_identifier(
                self.element.identifier, prefix="_ext_data")
            arguments["final_subs"] = "%(final_subs)s"
            self.component.subscripts_dict = final_subs

            self.element.objects["ext_data"] = {
                "name": arguments["name"],
                "expression": "%(name)s = ExtData(%(params)s, "
                              " %(method)s, %(subscripts)s, "
                              "_root, %(final_subs)s ,'%(name)s')" % arguments,
                "component": self.component,
                "final_subs": final_subs
            }

            return BuildAST(
                expression=arguments["name"] + "(time())",
                calls={
                    "__external__": arguments["name"],
                    "__data__": arguments["name"],
                    "time": 1},
                subscripts=final_subs,
                order=0)


class ExtConstantBuilder(StructureBuilder):
    def __init__(self, getconstant_str,  component):
        super().__init__(None, component)
        self.file = getconstant_str.file
        self.tab = getconstant_str.tab
        self.cell = getconstant_str.cell
        self.arguments = {}

    def build(self, arguments):
        self.component.type = "Constant"
        self.component.subtype = "External"
        arguments["params"] = "'%s', '%s', '%s'" % (
            self.file, self.tab, self.cell
        )
        final_subs, arguments["subscripts"] =\
            self.section.subscripts.simplify_subscript_input(
                self.def_subs, self.element.subscripts)

        if "constants" in self.element.objects:
            # object already exists
            self.element.objects["constants"]["expression"] += "\n\n"\
                + self.element.objects["constants"]["name"]\
                + ".add(%(params)s, %(subscripts)s)" % arguments

            self.update_object_subscripts("constants", final_subs)

            return None
        else:
            # create a new object
            self.section.imports.add("external", "ExtConstant")

            arguments["name"] = self.section.namespace.make_python_identifier(
                self.element.identifier, prefix="_ext_constant")
            arguments["final_subs"] = "%(final_subs)s"
            self.component.subscripts_dict = final_subs

            self.element.objects["constants"] = {
                "name": arguments["name"],
                "expression": "%(name)s = ExtConstant(%(params)s, "
                              "%(subscripts)s, _root, %(final_subs)s, "
                              "'%(name)s')" % arguments,
                "component": self.component,
                "final_subs": final_subs
            }

            return BuildAST(
                expression=arguments["name"] + "()",
                calls={"__external__": arguments["name"]},
                subscripts=final_subs,
                order=0)


class TabDataBuilder(StructureBuilder):
    def __init__(self, data_str,  component):
        super().__init__(None, component)
        self.keyword = component.keyword
        self.arguments = {}

    def build(self, arguments):
        self.section.imports.add("data", "TabData")

        final_subs, arguments["subscripts"] =\
            self.section.subscripts.simplify_subscript_input(
                self.def_subs, self.element.subscripts)

        arguments["real_name"] = self.element.name
        arguments["py_name"] =\
            self.section.namespace.namespace[self.element.name]
        arguments["subscripts"] = self.def_subs
        arguments["method"] = "'%s'" % self.keyword if self.keyword else None

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_data")

        self.element.objects["tab_data"] = {
            "name": arguments["name"],
            "expression": "%(name)s = TabData('%(real_name)s', '%(py_name)s', "
                          "%(subscripts)s,  %(method)s)" % arguments
        }

        return BuildAST(
            expression=arguments["name"] + "(time())",
            calls={"time": 1, "__data__": arguments["name"]},
            subscripts=final_subs,
            order=0)


class InitialBuilder(StructureBuilder):
    def __init__(self, initial_str, component):
        super().__init__(None, component)
        self.arguments = {
            "initial": initial_str.initial
        }

    def build(self, arguments):
        self.component.type = "Stateful"
        self.component.subtype = "Initial"
        self.section.imports.add("statefuls", "Initial")

        arguments["initial"].reshape(self.section.subscripts, self.def_subs)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_initial")

        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = Initial(lambda: %(initial)s, "
                          "'%(name)s')" % arguments,
            "calls": {
                "initial": arguments["initial"].calls,
                "step": {}
            }

        }
        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class IntegBuilder(StructureBuilder):
    def __init__(self, integ_str, component):
        super().__init__(None, component)
        self.arguments = {
            "flow": integ_str.flow,
            "initial": integ_str.initial
        }

    def build(self, arguments):
        self.component.type = "Stateful"
        self.component.subtype = "Integ"
        self.section.imports.add("statefuls", "Integ")

        arguments["initial"].reshape(self.section.subscripts, self.def_subs)
        arguments["flow"].reshape(self.section.subscripts, self.def_subs)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_integ")

        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = Integ(lambda: %(flow)s, "
                          "lambda: %(initial)s, '%(name)s')" % arguments,
            "calls": {
                "initial": arguments["initial"].calls,
                "step": arguments["flow"].calls
            }

        }
        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class DelayBuilder(StructureBuilder):
    def __init__(self, dtype, delay_str,  component):
        super().__init__(None, component)
        self.arguments = {
            "input": delay_str.input,
            "delay_time": delay_str.delay_time,
            "initial": delay_str.initial,
            "order": delay_str.order
        }
        self.dtype = dtype

    def build(self, arguments):
        self.component.type = "Stateful"
        self.component.subtype = "Delay"
        self.section.imports.add("statefuls", self.dtype)

        arguments["input"].reshape(self.section.subscripts, self.def_subs)
        arguments["delay_time"].reshape(self.section.subscripts, self.def_subs)
        arguments["initial"].reshape(self.section.subscripts, self.def_subs)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix=f"_{self.dtype.lower()}")
        arguments["dtype"] = self.dtype

        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = %(dtype)s(lambda: %(input)s, "
                          "lambda: %(delay_time)s, lambda: %(initial)s, "
                          "lambda: %(order)s, "
                          "time_step, '%(name)s')" % arguments,
            "calls": {
                "initial": merge_dependencies(
                    arguments["initial"].calls,
                    arguments["delay_time"].calls,
                    arguments["order"].calls),
                "step": merge_dependencies(
                    arguments["input"].calls,
                    arguments["delay_time"].calls)

            }
        }
        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class DelayFixedBuilder(StructureBuilder):
    def __init__(self, delay_str,  component):
        super().__init__(None, component)
        self.arguments = {
            "input": delay_str.input,
            "delay_time": delay_str.delay_time,
            "initial": delay_str.initial,
        }

    def build(self, arguments):
        self.component.type = "Stateful"
        self.component.subtype = "DelayFixed"
        self.section.imports.add("statefuls", "DelayFixed")

        arguments["input"].reshape(self.section.subscripts, self.def_subs)
        arguments["initial"].reshape(self.section.subscripts, self.def_subs)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_delayfixed")

        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = DelayFixed(lambda: %(input)s, "
                          "lambda: %(delay_time)s, lambda: %(initial)s, "
                          "time_step, '%(name)s')" % arguments,
            "calls": {
                "initial": merge_dependencies(
                    arguments["initial"].calls,
                    arguments["delay_time"].calls),
                "step": arguments["input"].calls
            }
        }
        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class SmoothBuilder(StructureBuilder):
    def __init__(self, smooth_str,  component):
        super().__init__(None, component)
        self.arguments = {
            "input": smooth_str.input,
            "smooth_time": smooth_str.smooth_time,
            "initial": smooth_str.initial,
            "order": smooth_str.order
        }

    def build(self, arguments):
        self.component.type = "Stateful"
        self.component.subtype = "Smooth"
        self.section.imports.add("statefuls", "Smooth")

        arguments["input"].reshape(
            self.section.subscripts, self.def_subs)
        arguments["smooth_time"].reshape(
            self.section.subscripts, self.def_subs)
        arguments["initial"].reshape(
            self.section.subscripts, self.def_subs)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_smooth")

        # TODO in the future we need to ad timestep to show warnings about
        # the smooth time as its done with delays (see vensim help for smooth)
        # TODO in the future we may want to have 2 py_backend classes for
        # smooth as the behaviour is different for SMOOTH and SMOOTH N when
        # using RingeKutta scheme
        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = Smooth(lambda: %(input)s, "
                          "lambda: %(smooth_time)s, lambda: %(initial)s, "
                          "lambda: %(order)s, '%(name)s')" % arguments,
            "calls": {
                "initial": merge_dependencies(
                    arguments["initial"].calls,
                    arguments["smooth_time"].calls,
                    arguments["order"].calls),
                "step": merge_dependencies(
                    arguments["input"].calls,
                    arguments["smooth_time"].calls)
            }

        }
        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class TrendBuilder(StructureBuilder):
    def __init__(self, trend_str,  component):
        super().__init__(None, component)
        self.arguments = {
            "input": trend_str.input,
            "average_time": trend_str.average_time,
            "initial_trend": trend_str.initial_trend,
        }

    def build(self, arguments):
        self.component.type = "Stateful"
        self.component.subtype = "Trend"
        self.section.imports.add("statefuls", "Trend")

        arguments["input"].reshape(
            self.section.subscripts, self.def_subs)
        arguments["average_time"].reshape(
            self.section.subscripts, self.def_subs)
        arguments["initial_trend"].reshape(
            self.section.subscripts, self.def_subs)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_trend")

        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = Trend(lambda: %(input)s, "
                          "lambda: %(average_time)s, "
                          "lambda: %(initial_trend)s, "
                          "'%(name)s')" % arguments,
            "calls": {
                "initial": merge_dependencies(
                    arguments["initial_trend"].calls,
                    arguments["input"].calls,
                    arguments["average_time"].calls),
                "step": merge_dependencies(
                    arguments["input"].calls,
                    arguments["average_time"].calls)
            }

        }
        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class ForecastBuilder(StructureBuilder):
    def __init__(self, forecast_str,  component):
        super().__init__(None, component)
        self.arguments = {
            "input": forecast_str.input,
            "average_time": forecast_str.average_time,
            "horizon": forecast_str.horizon,
            "initial_trend": forecast_str.initial_trend
        }

    def build(self, arguments):
        self.component.type = "Stateful"
        self.component.subtype = "Forecast"
        self.section.imports.add("statefuls", "Forecast")

        arguments["input"].reshape(
            self.section.subscripts, self.def_subs)
        arguments["average_time"].reshape(
            self.section.subscripts, self.def_subs)
        arguments["horizon"].reshape(
            self.section.subscripts, self.def_subs)
        arguments["initial_trend"].reshape(
            self.section.subscripts, self.def_subs)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_forecast")

        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = Forecast(lambda: %(input)s, "
                          "lambda: %(average_time)s, lambda: %(horizon)s, "
                          "lambda: %(initial_trend)s, '%(name)s')" % arguments,
            "calls": {
                "initial": merge_dependencies(
                    arguments["input"].calls,
                    arguments["initial_trend"].calls),
                "step": merge_dependencies(
                    arguments["input"].calls,
                    arguments["average_time"].calls,
                    arguments["horizon"].calls)
            }

        }
        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class SampleIfTrueBuilder(StructureBuilder):
    def __init__(self, sampleiftrue_str,  component):
        super().__init__(None, component)
        self.arguments = {
            "condition": sampleiftrue_str.condition,
            "input": sampleiftrue_str.input,
            "initial": sampleiftrue_str.initial,
        }

    def build(self, arguments):
        self.component.type = "Stateful"
        self.component.subtype = "SampleIfTrue"
        self.section.imports.add("statefuls", "SampleIfTrue")

        arguments["condition"].reshape(self.section.subscripts, self.def_subs)
        arguments["input"].reshape(self.section.subscripts, self.def_subs)
        arguments["initial"].reshape(self.section.subscripts, self.def_subs)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_sampleiftrue")

        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = SampleIfTrue(lambda: %(condition)s, "
                          "lambda: %(input)s, lambda: %(initial)s, "
                          "'%(name)s')" % arguments,
            "calls": {
                "initial":
                    arguments["initial"].calls,
                "step": merge_dependencies(
                    arguments["condition"].calls,
                    arguments["input"].calls)
            }

        }
        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class LookupsBuilder(StructureBuilder):
    def __init__(self, lookups_str,  component):
        super().__init__(None, component)
        self.arguments = {}
        self.x = lookups_str.x
        self.y = lookups_str.y
        self.keyword = lookups_str.type

    def build(self, arguments):
        self.component.type = "Lookup"
        self.component.subtype = "Normal"
        arguments["x"] = np.array2string(
            np.array(self.x),
            separator=",",
            threshold=len(self.x)
        )
        arguments["y"] = np.array2string(
            np.array(self.y),
            separator=",",
            threshold=len(self.y)
        )
        arguments["subscripts"] = self.def_subs
        arguments["interp"] = self.keyword

        if "hardcoded_lookups" in self.element.objects:
            # object already exists
            self.element.objects["hardcoded_lookups"]["expression"] += "\n\n"\
                + self.element.objects["hardcoded_lookups"]["name"]\
                + ".add(%(x)s, %(y)s, %(subscripts)s)" % arguments

            return None
        else:
            # create a new object
            self.section.imports.add("lookups", "HardcodedLookups")

            arguments["name"] = self.section.namespace.make_python_identifier(
                self.element.identifier, prefix="_hardcodedlookup")

            arguments["final_subs"] = self.element.subs_dict

            self.element.objects["hardcoded_lookups"] = {
                "name": arguments["name"],
                "expression": "%(name)s = HardcodedLookups(%(x)s, %(y)s, "
                              "%(subscripts)s, '%(interp)s', "
                              "%(final_subs)s, '%(name)s')"
                              % arguments
            }

            return BuildAST(
                expression=arguments["name"] + "(x, final_subs)",
                calls={"__lookup__": arguments["name"]},
                subscripts=self.def_subs,
                order=0)


class InlineLookupsBuilder(StructureBuilder):
    def __init__(self, inlinelookups_str,  component):
        super().__init__(None, component)
        self.arguments = {
            "value": inlinelookups_str.argument
        }
        self.lookups = inlinelookups_str.lookups

    def build(self, arguments):
        self.component.type = "Auxiliary"
        self.component.subtype = "with Lookup"
        self.section.imports.add("numpy")
        arguments["x"] = np.array2string(
            np.array(self.lookups.x),
            separator=",",
            threshold=len(self.lookups.x)
        )
        arguments["y"] = np.array2string(
            np.array(self.lookups.y),
            separator=",",
            threshold=len(self.lookups.y)
        )
        return BuildAST(
            expression="np.interp(%(value)s, %(x)s, %(y)s)" % arguments,
            calls=arguments["value"].calls,
            subscripts=arguments["value"].subscripts,
            order=0)


class ReferenceBuilder(StructureBuilder):
    def __init__(self, reference_str,  component):
        super().__init__(None, component)
        self.mapping_subscripts = {}
        self.reference = reference_str.reference
        self.subscripts = reference_str.subscripts
        self.arguments = {}
        self.section.imports.add("xarray")

    @property
    def subscripts(self):
        return self._subscripts

    @subscripts.setter
    def subscripts(self, subscripts):
        """Get subscript dictionary from reference"""
        self._subscripts = self.section.subscripts.make_coord_dict(
            getattr(subscripts, "subscripts", {}))

        # get the subscripts after applying the mapping if necessary
        for dim, coordinates in self._subscripts.items():
            if len(coordinates) > 1:
                # we create the mapping only with those subscripts that are
                # ranges as we need to ignore singular subscripts because
                # that dimension is removed from final element
                if dim not in self.def_subs and not dim.endswith("!"):
                    # the reference has a subscripts which is it not
                    # applied (!) and does not appear in the definition
                    # of the variable
                    for mapped in self.section.subscripts.mapping[dim]:
                        # check the mapped subscripts
                        # TODO update this and the parser to make it
                        # compatible with more complex mappings
                        if mapped in self.def_subs\
                          and mapped not in self._subscripts:
                            # the mapped subscript appears in the definition
                            # and it is not already in the variable
                            self.mapping_subscripts[mapped] =\
                                self.section.subscripts.subscripts[mapped]
                            break
                else:
                    # the subscript is in the variable definition,
                    # do not change it
                    self.mapping_subscripts[dim] = coordinates

    def build(self, arguments):
        if self.reference not in self.section.namespace.cleanspace:
            # Manage references to subscripts (subscripts used as variables)
            expression, subscripts =\
                self.section.subscripts.subscript2num[self.reference]
            subscripts_out = self.section.subscripts.simplify_subscript_input(
                subscripts, list(subscripts))[1]
            if subscripts:
                self.section.imports.add("numpy")
                # NUMPY: not need this if
                expression = "xr.DataArray(%s, %s, %s)" % (
                    expression, subscripts_out, list(subscripts))
            return BuildAST(
                expression=expression,
                calls={},
                subscripts=subscripts,
                order=0)

        reference = self.section.namespace.cleanspace[self.reference]

        expression = reference + "()"

        if not self.subscripts:
            return BuildAST(
                expression=expression,
                calls={reference: 1},
                subscripts={},
                order=0)

        original_subs = self.section.subscripts.make_coord_dict(
                    self.section.subscripts.elements[reference])

        expression, final_subs = self.visit_subscripts(
            expression, original_subs)

        return BuildAST(
            expression=expression,
            calls={reference: 1},
            subscripts=final_subs,
            order=0)

    def visit_subscripts(self, expression, original_subs):
        final_subs, rename, loc, reset_coords, float = {}, {}, [], False, True
        for (dim, coord), (orig_dim, orig_coord)\
          in zip(self.subscripts.items(), original_subs.items()):
            if len(coord) == 1:
                # subset a 1 dimension value
                # NUMPY: subset value [:, N, :, :]
                loc.append(repr(coord[0]))
                reset_coords = True
            elif len(coord) < len(orig_coord):
                # subset a subrange
                # NUMPY: subset value [:, :, np.array([1, 0]), :]
                # NUMPY: as order may change we need to check if
                #        dim != orig_dim
                # NUMPY: use also ranges [:, :, 2:5, :] when possible
                if dim.endswith("!"):
                    loc.append("_subscript_dict['%s']" % dim[:-1])
                else:
                    loc.append("_subscript_dict['%s']" % dim)
                final_subs[dim] = coord
                float = False
            else:
                # do nothing
                # NUMPY: same, we can remove float = False
                loc.append(":")
                final_subs[dim] = coord
                float = False

            if dim != orig_dim and len(coord) != 1:
                # NUMPY: check order of dimensions, make all subranges work
                #        with the same dimensions?
                # NUMPY: this could be solved in the previous if/then/else
                rename[orig_dim] = dim

        if any(dim != ":" for dim in loc):
            # NUMPY: expression += "[%s]" % ", ".join(loc)
            expression += ".loc[%s]" % ", ".join(loc)
        if reset_coords and float:
            # NUMPY: Not neccessary
            expression = "float(" + expression + ")"
        elif reset_coords:
            # NUMPY: Not neccessary
            expression += ".reset_coords(drop=True)"
        if rename:
            # NUMPY: Not neccessary
            expression += ".rename(%s)" % rename

        # NUMPY: This will not be necessary, we only need to return
        #        self.mapping_subscripts
        if self.mapping_subscripts != final_subs:
            subscripts_out = self.section.subscripts.simplify_subscript_input(
                self.mapping_subscripts, list(self.mapping_subscripts))[1]
            expression = "xr.DataArray(%s.values, %s, %s)" % (
                expression, subscripts_out, list(self.mapping_subscripts)
            )

        return expression, self.mapping_subscripts


class NumericBuilder(StructureBuilder):
    def build(self, arguments):
        if np.isnan(self.value):
            self.section.imports.add("numpy")

            return BuildAST(
                expression="np.nan",
                calls={},
                subscripts={},
                order=0)
        else:
            return BuildAST(
                expression=repr(self.value),
                calls={},
                subscripts={},
                order=0)


class ArrayBuilder(StructureBuilder):
    def build(self, arguments):
        self.value = np.array2string(
            self.value.reshape(compute_shape(self.def_subs)),
            separator=",",
            threshold=np.prod(self.value.shape)
        )
        self.component.type = "Constant"
        self.component.subtype = "Normal"

        final_subs, subscripts_out =\
            self.section.subscripts.simplify_subscript_input(
                self.def_subs, self.element.subscripts)

        return BuildAST(
            expression="xr.DataArray(%s, %s, %s)" % (
                self.value, subscripts_out, list(final_subs)),
            calls={},
            subscripts=final_subs,
            order=0)


def merge_dependencies(*dependencies, inplace=False):
    # TODO improve dependencies in the next major release, include info
    # about external objects and simplify the stateful objects, think about
    # how to include data/lookups objects
    current = dependencies[0]
    if inplace:
        current = dependencies[0]
    else:
        current = dependencies[0].copy()
    for new in dependencies[1:]:
        if not current:
            current.update(new)
        elif new:
            # regular element
            _merge_dependencies(current, new)

    return current


def _merge_dependencies(current, new):
    """
    Merge two dependencies dicts of an element.

    Parameters
    ----------
    current: dict
        Current dependencies of the element. It will be mutated.

    new: dict
        New dependencies to add.

    Returns
    -------
    None

    """
    current_set, new_set = set(current), set(new)
    for dep in current_set.intersection(new_set):
        # if dependency is in both sum the number of calls
        if dep.startswith("__"):
            # if it is special (__lookup__, __external__) continue
            continue
        else:
            current[dep] += new[dep]
    for dep in new_set.difference(current_set):
        # if dependency is only in new copy it
        current[dep] = new[dep]


class ASTVisitor:
    builders = {
        ae.InitialStructure: InitialBuilder,
        ae.IntegStructure: IntegBuilder,
        ae.DelayStructure: lambda x, y: DelayBuilder("Delay", x, y),
        ae.DelayNStructure: lambda x, y: DelayBuilder("DelayN", x, y),
        ae.DelayFixedStructure: DelayFixedBuilder,
        ae.SmoothStructure: SmoothBuilder,
        ae.SmoothNStructure: SmoothBuilder,
        ae.TrendStructure: TrendBuilder,
        ae.ForecastStructure: ForecastBuilder,
        ae.SampleIfTrueStructure: SampleIfTrueBuilder,
        ae.GetConstantsStructure: ExtConstantBuilder,
        ae.GetDataStructure: ExtDataBuilder,
        ae.GetLookupsStructure: ExtLookupBuilder,
        ae.LookupsStructure: LookupsBuilder,
        ae.InlineLookupsStructure: InlineLookupsBuilder,
        ae.DataStructure: TabDataBuilder,
        ae.ReferenceStructure: ReferenceBuilder,
        ae.CallStructure: CallBuilder,
        ae.GameStructure: GameBuilder,
        ae.LogicStructure: OperationBuilder,
        ae.ArithmeticStructure: OperationBuilder,
        int: NumericBuilder,
        float: NumericBuilder,
        np.ndarray: ArrayBuilder,
    }

    def __init__(self, component):
        self.ast = component.ast
        self.subscripts = component.subscripts_dict
        self.component = component

    def visit(self):
        visit_out = self._visit(self.ast)

        if not visit_out:
            # external objects that are declared with other expression
            return None

        if not visit_out.calls and self.component.type == "Auxiliary":
            self.component.type = "Constant"
            self.component.subtype = "Normal"

        # include dependencies of the current component in the element
        merge_dependencies(
            self.component.element.dependencies,
            visit_out.calls,
            inplace=True)

        if not visit_out.subscripts:
            # expression is a float
            return visit_out

        # NUMPY not needed
        # get subscript in elements as name of the ranges may change
        subscripts_in_element = {
            dim: coords
            for dim, coords
            in zip(self.component.element.subscripts, self.subscripts.values())
        }

        reshape = (
            (visit_out.subscripts != self.subscripts
             or list(visit_out.subscripts) != list(self.subscripts))
            and
            (visit_out.subscripts != subscripts_in_element
             or list(visit_out.subscripts) != list(subscripts_in_element))
        )

        if reshape:
            # We are only comparing the dictionaries (set of dimensions)
            # and not the list (order).
            # With xarray we don't need to compare the order because the
            # decorator @subs will reorder the objects
            # NUMPY: in this case we need to tile along dims if neccessary
            #        or reorder the dimensions
            # NUMPY: if the output is a float or int and they are several
            #        definitions we can return float or int as we can
            #        safely do "var[:, 1, :] = 3"
            visit_out.reshape(
                self.component.section.subscripts, self.subscripts)

        return visit_out

    def _visit(self, ast_object):
        builder = self.builders[type(ast_object)](ast_object, self.component)
        arguments = {
            name: self._visit(value)
            for name, value in builder.arguments.items()
        }
        return builder.build(arguments)


class ExceptVisitor:  # pragma: no cover
    # this class will be used in the numpy array backend
    def __init__(self, component):
        self.except_definitions = component.subscripts[1]
        self.subscripts = component.section.subscripts
        self.subscripts_dict = component.subscripts_dict

    def visit(self):
        excepts = [
            BuildAST("", self.subscripts_dict, {}, 0)
            for _ in self.except_definitions
        ]
        [
            except_def.reshape(
                self.subscripts,
                self.subscripts.make_coord_dict(except_list))
            for except_def, except_list
            in zip(excepts, self.except_definitions)
        ]
        return excepts
