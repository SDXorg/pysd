"""
The translation from Abstract Syntax Tree to Python happens in both ways.
The outer expression is visited with its builder, which will split its
arguments and visit them with their respective builders. Once the lowest
level is reached, it will be translated into Python returning a BuildAST
object, this object will include the python expression, its subscripts,
its calls to other and its arithmetic order (see Build AST for more info).
BuildAST will be returned for each visited argument from the lower
lever to the top level, giving the final expression.
"""
import warnings
from dataclasses import dataclass
from typing import Union

import numpy as np
from pysd.py_backend.utils import compute_shape

from pysd.translators.structures.abstract_expressions import\
    AbstractSyntax, AllocateAvailableStructure, AllocateByPriorityStructure,\
    ArithmeticStructure, CallStructure, DataStructure, DelayFixedStructure,\
    DelayStructure, DelayNStructure, ForecastStructure, GameStructure,\
    GetConstantsStructure, GetDataStructure, GetLookupsStructure,\
    InitialStructure, InlineLookupsStructure, IntegStructure,\
    LogicStructure, LookupsStructure, ReferenceStructure,\
    SampleIfTrueStructure, SmoothNStructure, SmoothStructure,\
    SubscriptsReferenceStructure, TrendStructure

from .python_functions import functionspace
from .subscripts import SubscriptManager


@dataclass
class BuildAST:
    """
    Python expression holder.

    Parameters
    ----------
    expression: str
        The Python expression.
    calls: dict
        The calls to other variables for the dependencies dictionary.
    subscripts: dict
        The subscripts dict of the expression.
    order: int
        Arithmetic order of the expression. The arithmetic order depends
        on the last arithmetic operation. If the expression is a number,
        a call to a function, or is between parenthesis; its order will
        be 0. If the expression its an exponential of two terms its order
        will be 1. If the expression is a product or division its order
        will be 2. If the expression is a sum or substraction its order
        will be 3. If the expression is a logical comparison its order
        will be 4.

    """
    expression: str
    calls: dict
    subscripts: dict
    order: int

    def __str__(self) -> str:
        # makes easier building
        return self.expression

    def reshape(self, subscripts: SubscriptManager,
                final_subscripts: dict,
                final_element: bool = False) -> None:
        """
        Reshape the object to the desired subscripts. It will modify the
        expression and lower the order if it is not 0.

        Parameters
        ----------
        subscripts: SubscriptManager
            The subscripts of the section.
        final_subscripts: dict
            The desired final subscripts.
        final_element: bool (optional)
            If True the array will be reshaped with the final subscripts
            to have the shame shape. Otherwise, a length 1 dimension
            will be included in the position to allow arithmetic
            operations with other arrays. Default is False.

        """
        if not final_subscripts or (
          self.subscripts == final_subscripts
          and list(self.subscripts) == list(final_subscripts)):
            # Same dictionary in the same order, do nothing
            pass
        elif not self.subscripts:
            # Original expression is not an array
            # NUMPY: object.expression = np.full(%s, %(shape)s)
            subscripts_out = subscripts.simplify_subscript_input(
                final_subscripts)[1]
            self.expression = "xr.DataArray(%s, %s, %s)" % (
                self.expression, subscripts_out, list(final_subscripts)
            )
            self.order = 0
            self.subscripts = final_subscripts
        else:
            # Original expression is an array
            self.lower_order(-1)

            # Reorder subscrips
            final_order = {
                sub: self.subscripts[sub]
                for sub in final_subscripts
                if sub in self.subscripts
            }
            if list(final_order) != list(self.subscripts):
                # NUMPY: reorder dims if neccessary with np.moveaxis or similar
                self.expression +=\
                    f".transpose({', '.join(map(repr, final_order))})"
                self.subscripts = final_order

            # add new dimensions
            if final_element and final_subscripts != self.subscripts:
                # NUMPY: remove final_element condition from top
                # NUMPY: add new axis with [:, None, :]
                # NUMPY: move final_element condition here and use np.tile
                for i, dim in enumerate(final_subscripts):
                    if dim not in self.subscripts:
                        subscripts_out = subscripts.simplify_subscript_input(
                            {dim: final_subscripts[dim]})[1]
                        self.expression +=\
                            f".expand_dims({subscripts_out}, {i})"

                self.subscripts = final_subscripts

    def lower_order(self, new_order: int) -> None:
        """
        Lower the order to maintain the correct order in arithmetic
        operations. If the requested order is smaller than the current
        order parenthesis will be added to the expression to lower its
        order to 0.

        Parameters
        ----------
        new_order: int
            The required new order of the expression. If 0 it will be
            assumed that the expression will be passed as an argument
            of a function and therefore no operations will be done. If
            order 0 is required, a negative value can be used for
            new_order.

        """
        if self.order >= new_order and self.order != 0 and new_order != 0:
            # if current operator order is 0 do not need to do anything
            # if the order of operations conflicts add parenthesis
            # if new order is 0 do not need to do anything, as it may be
            # an argument to a function. To force the 0 order a negative
            # value can be used, which will force the parenthesis
            # (necessary to reshape some arrays)
            self.expression = "(%s)" % self.expression
            self.order = 0


class StructureBuilder:
    """
    Main builder for Abstract Syntax Tree structures. All the builders
    are children of this class, which allows them inheriting the methods.
    """
    def __init__(self, value: object, component: object):
        # component typing should be ComponentBuilder, but importing it
        # for typing would create a circular dependency :S
        self.value = value
        self.arguments = {}
        self.component = component
        self.element = component.element
        self.section = component.section
        self.def_subs = component.subscripts_dict

    @staticmethod
    def join_calls(arguments: dict) -> dict:
        """
        Merge the calls of the arguments.

        Parameters
        ----------
        arguments: dict
            The dictionary of arguments. The keys should br strings of
            ordered integer numbers starting from 0.

        Returns
        -------
        calls: dict
            The merged dictionary of calls.

        """
        if len(arguments) == 0:
            # No arguments
            return {}
        elif len(arguments) == 1:
            # Only one argument
            return list(arguments.values())[0].calls
        else:
            # Several arguments
            return merge_dependencies(
                *[val.calls for val in arguments.values()])

    def reorder(self, arguments: dict, force: bool = None) -> dict:
        """
        Reorder the subscripts of the arguments to make them match.

        Parameters
        ----------
        arguments: dict
            The dictionary of arguments. The keys should br strings of
            ordered integer numbers starting from 0.
        force: 'component', 'equal', or None (optional)
            If force is 'component' it will force the arguments to have
            the subscripts of the component definition. If force is
            'equal' it will force all the arguments to have the same
            subscripts, includying the floats. If force is None, it
            will only modify the shape of the arrays adding length 1
            dimensions to allow operation between different shape arrays.
            Default is None.

        Returns
        -------
        final_subscripts: dict
            The final_subscripts after reordering all the elements.

        """
        if force == "component":
            final_subscripts = self.def_subs or {}
        else:
            final_subscripts = self.get_final_subscripts(arguments)

        [arguments[key].reshape(
            self.section.subscripts, final_subscripts, bool(force))
         for key in arguments
         if arguments[key].subscripts or force == "equal"]

        return final_subscripts

    def get_final_subscripts(self, arguments: dict) -> dict:
        """
        Get the final subscripts of a combination of arguments.

        Parameters
        ----------
        arguments: dict
            The dictionary of arguments. The keys should br strings of
            ordered integer numbers starting from 0.

        Returns
        -------
        final_subscripts: dict
            The final_subscripts of combining all the elements.

        """
        if len(arguments) == 0:
            return {}
        elif len(arguments) == 1:
            return arguments["0"].subscripts
        else:
            return self._compute_final_subscripts(
                [arg.subscripts for arg in arguments.values()])

    def _compute_final_subscripts(self, subscripts_list: list) -> dict:
        """
        Compute final subscripts from a list of subscript dictionaries.

        Parameters
        ----------
        subscript_list: list of dicts
            List of subscript dictionaries.

        """
        expression = {}
        [expression.update(subscript)
         for subscript in subscripts_list if subscript]
        # TODO reorder final_subscripts taking into account def_subs
        # this way try to minimize the reordering operations
        return expression

    def update_object_subscripts(self, name: str,
                                 component_final_subs: dict) -> None:
        """
        Update the object subscripts. Needed for those objects that
        use 'add' method to load several components at once and mixed
        definitions are used.

        Parameters
        ----------
        name: str
            The name of the object in the objects dictionary from the
            element.
        component_final_subs: dict
            The subscripts of the component but with the element
            subscript ranges as keys. This can differ from the component
            subscripts when the component is defined with subranges of
            the final subscript ranges.

        """
        # Get the component used to define the object first time
        origin_comp = self.element.objects[name]["component"]
        # The original component subscript dictionary is a list
        origin_comp.subscripts_dict.append(component_final_subs)


class OperationBuilder(StructureBuilder):
    """Builder for arithmetic and logical operations."""
    _operators_build = {
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

    def __init__(self, operation: Union[ArithmeticStructure, LogicStructure],
                 component: object):
        super().__init__(None, component)
        self.operators = operation.operators.copy()
        self.arguments = {
            str(i): arg for i, arg in enumerate(operation.arguments)}

    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        operands = {}
        calls = self.join_calls(arguments)
        final_subscripts = self.reorder(arguments)
        arguments = [arguments[str(i)] for i in range(len(arguments))]
        dependencies, order = self._operators_build[self.operators[-1]][1:]

        if dependencies:
            # Add necessary dependencies to the imports
            self.section.imports.add(*dependencies)

        if self.operators[-1] == "^":
            # Right side of the exponential can be from higher order
            arguments[-1].lower_order(2)
        else:
            arguments[-1].lower_order(order)

        if len(arguments) == 1:
            # not and negative operations (only 1 element)
            if self.operators[0] == "negative":
                order = 1
            expression = self._operators_build[self.operators[0]][0]
            return BuildAST(
                expression=expression % arguments[0],
                calls=calls,
                subscripts=final_subscripts,
                order=order)

        # Add the arguments to the expression with the operator,
        # they are built from right to left
        # Get the last argument as the RHS of the first operation
        operands["right"] = arguments.pop()
        while arguments or self.operators:
            # Get the operator and the LHS of the operation
            expression = self._operators_build[self.operators.pop()][0]
            operands["left"] = arguments.pop()
            # Lower the order of the LHS if neccessary
            operands["left"].lower_order(order)
            # Include the operation in the RHS for next iteration
            operands["right"] = expression % operands

        return BuildAST(
            expression=operands["right"],
            calls=calls,
            subscripts=final_subscripts,
            order=order)


class GameBuilder(StructureBuilder):
    """Builder for GAME expressions."""
    def __init__(self, game_str: GameStructure, component: object):
        super().__init__(None, component)
        self.arguments = {"expr": game_str.expression}

    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        # Game calls are ignored as we have no support for a similar
        # feature, we simply return the content inside the GAME call
        return arguments["expr"]


class CallBuilder(StructureBuilder):
    """Builder for calls to functions, macros and lookups."""
    def __init__(self, call_str: CallStructure, component: object):
        super().__init__(None, component)
        function_name = call_str.function.reference
        self.arguments = {
            str(i): arg for i, arg in enumerate(call_str.arguments)}

        if function_name in self.section.macrospace:
            # Build macro
            self.macro_name = function_name
            self.build = self.build_macro_call
        elif function_name in self.section.namespace.cleanspace:
            # Build lookupcall
            self.arguments["function"] = call_str.function
            self.build = self.build_lookups_call
        elif function_name in functionspace:
            # Build direct function
            self.function = function_name
            self.build = self.build_function_call
        elif function_name == "a_function_of":
            # Build incomplete function
            self.build = self.build_incomplete_call
        else:
            # Build missing function
            self.function = function_name
            self.build = self.build_not_implemented

    def build_not_implemented(self, arguments: dict) -> BuildAST:
        """
        Build method for not implemented function calls.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        final_subscripts = self.reorder(arguments)
        warnings.warn(
            "Trying to translate '" + self.function.upper().replace("_", " ")
            + "' which it is not implemented on PySD. The translated "
            + "model will crash..."
        )
        self.section.imports.add("functions", "not_implemented_function")

        return BuildAST(
            expression="not_implemented_function('%s', %s)" % (
                self.function,
                ", ".join(arg.expression for arg in arguments.values())),
            calls=self.join_calls(arguments),
            subscripts=final_subscripts,
            order=0)

    def build_incomplete_call(self, arguments: dict) -> BuildAST:
        """
        Build method for incomplete function calls.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        warnings.warn(
            "'%s' has no equation specified" % self.element.name,
            SyntaxWarning, stacklevel=2
        )
        self.section.imports.add("functions", "incomplete")
        return BuildAST(
            expression="incomplete(%s)" % ", ".join(
                arg.expression for arg in arguments.values()),
            calls=self.join_calls(arguments),
            subscripts=self.def_subs,
            order=0)

    def build_macro_call(self, arguments: dict) -> BuildAST:
        """
        Build method for macro calls.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        self.section.imports.add("model", "Macro")
        # Get macro from macrospace
        macro = self.section.macrospace[self.macro_name]

        calls = self.join_calls(arguments)
        final_subscripts = self.reorder(arguments)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.macro_name + "_" + self.element.identifier, prefix="_macro")
        arguments["file"] = macro.path.name
        arguments["macro_name"] = macro.name
        arguments["args"] = "{%s}" % ", ".join([
            "'%s': lambda: %s" % (key, val)
            for key, val in zip(macro.params, arguments.values())
        ])

        # Create Macro object
        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = Macro(_root.joinpath('%(file)s'), "
                          "%(args)s, '%(macro_name)s', "
                          "time_initialization=lambda: __data['time'], "
                          "py_name='%(name)s')" % arguments,
        }
        # Add other_dependencies
        self.element.other_dependencies[arguments["name"]] = {
            "initial": calls,
            "step": calls
        }

        return BuildAST(
            expression="%s()" % arguments["name"],
            calls={arguments["name"]: 1},
            subscripts=final_subscripts,
            order=0)

    def build_lookups_call(self, arguments: dict) -> BuildAST:
        """
        Build method for loookups calls.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        if arguments["0"].subscripts:
            # Build lookups with subcripted arguments
            # it is neccessary to give the final subscripts information
            # in the call to rearrange it correctly
            final_subscripts =\
                self.get_final_subscripts(arguments)
            expression = arguments["function"].expression.replace(
                "()", f"(%(0)s, {final_subscripts})")
        else:
            # Build lookups with float arguments
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

    def build_function_call(self, arguments: dict) -> BuildAST:
        """
        Build method for function calls.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        # Get the function expression from the functionspace
        expression, modules = functionspace[self.function]
        for module in modules:
            # Update module dependencies in imports
            self.section.imports.add(*module)

        calls = self.join_calls(arguments)

        if "__data['time']" in expression:
            # If the expression depens on time add to the dependencies
            merge_dependencies(calls, {"time": 1}, inplace=True)

        if "%(axis)s" in expression:
            # Vectorial expressions, compute the axis using dimensions
            # with ! operator
            if "%(1)s" in expression:
                subs = self.reorder(arguments)
                # NUMPY: following line may be avoided
                [arguments[i].reshape(self.section.subscripts, subs, True)
                 for i in ["0", "1"]]
            else:
                subs = arguments["0"].subscripts
            final_subscripts, arguments["axis"] = self._compute_axis(subs)

        elif "%(size)s" in expression:
            # Random expressions, need to give the final size of the
            # component to create one value per final coordinate
            final_subscripts = self.reorder(arguments, force="component")
            arguments["size"] = tuple(compute_shape(final_subscripts))
            if arguments["size"]:
                # Create an xarray from the random function output
                # NUMPY: not necessary
                # generate an xarray from the output
                subs = self.section.subscripts.simplify_subscript_input(
                    self.def_subs)[1]
                expression = f"xr.DataArray({expression}, {subs}, "\
                    f"{list(self.def_subs)})"
            calls["time"] = 1

        elif self.function == "active_initial":
            # Ee need to ensure that active initial outputs are always the
            # same and update dependencies as stateful object
            name = self.section.namespace.make_python_identifier(
                self.element.identifier, prefix="_active_initial")
            final_subscripts = self.reorder(arguments, force="equal")
            self.element.other_dependencies[name] = {
                "initial": arguments["1"].calls,
                "step": arguments["0"].calls
            }

            calls = {name: 1}

        elif self.function == "elmcount":
            final_subscripts = {}

        else:
            final_subscripts = self.reorder(arguments)
            if self.function == "xidz" and final_subscripts:
                # xidz must always return the same shape object
                if not arguments["1"].subscripts:
                    [arguments[i].reshape(
                        self.section.subscripts, final_subscripts, True)
                     for i in ["0", "1"]]
                elif arguments["0"].subscripts or arguments["2"].subscripts:
                    # NUMPY: not need this statement
                    [arguments[i].reshape(
                        self.section.subscripts, final_subscripts, True)
                     for i in ["0", "1", "2"]
                     if arguments[i].subscripts]
            elif self.function == "zidz" and final_subscripts:
                # zidz must always return the same shape object
                arguments["0"].reshape(
                    self.section.subscripts, final_subscripts, True)
                if arguments["1"].subscripts:
                    # NUMPY: not need this statement
                    arguments["1"].reshape(
                        self.section.subscripts, final_subscripts, True)
            elif self.function == "if_then_else" and final_subscripts:
                # if_then_else must always return the same shape object
                if not arguments["0"].subscripts:
                    # condition is a float
                    [arguments[i].reshape(
                        self.section.subscripts, final_subscripts, True)
                     for i in ["1", "2"]]
                else:
                    # condition has dimensions
                    [arguments[i].reshape(
                        self.section.subscripts, final_subscripts, True)
                     for i in ["0", "1", "2"]]

        return BuildAST(
            expression=expression % arguments,
            calls=calls,
            subscripts=final_subscripts,
            order=0)

    def _compute_axis(self, subscripts: dict) -> tuple:
        """
        Compute the axis to apply a vectorial function.

        Parameters
        ----------
        subscripts: dict
            The final_subscripts after reordering all the elements.

        Returns
        -------
        coords: dict
            The final coordinates after executing the vectorial function
        axis: list
            The list of dimensions to apply the function. Uses the
            dimensions with "!" at the end.

        """
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


class AllocateAvailableBuilder(StructureBuilder):
    """Builder for allocate_available function."""

    def __init__(self, allocate_str: AllocateAvailableStructure,
                 component: object):
        super().__init__(None, component)

        pp = allocate_str.pp
        pp_sub = self.section.subscripts.elements[pp.reference][-1:]
        pp.subscripts.subscripts = pp.subscripts.subscripts[:-1] + pp_sub
        self.arguments = {
            "request": allocate_str.request,
            "pp": pp,
            "avail": allocate_str.avail
        }

    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        self.section.imports.add("allocation", "allocate_available")

        calls = self.join_calls(arguments)

        # the last sub of the request must be keep last sub of request and
        # priority
        last_sub = list(arguments["request"].subscripts)[-1]
        pp_sub = list(arguments["pp"].subscripts)[-1]
        # compute the merged subscripts
        final_subscripts = self.get_final_subscripts(arguments)
        # remove last sub from request
        last_sub_value = final_subscripts[last_sub]
        pp_sub_value = final_subscripts[pp_sub]
        del final_subscripts[last_sub], final_subscripts[pp_sub]

        # Update the susbcripts of avail
        arguments["avail"].reshape(
            self.section.subscripts, final_subscripts, True)

        # Include last sub of request in the last position and update
        # the subscripts of request
        final_subscripts[last_sub] = last_sub_value
        arguments["request"].reshape(
            self.section.subscripts, final_subscripts, True)

        # Include priority subscripts and update the subscripts of pp
        final_subscripts[pp_sub] = pp_sub_value
        arguments["pp"].reshape(
            self.section.subscripts, final_subscripts, True)

        expression = "allocate_available(%(request)s, %(pp)s, %(avail)s)"
        return BuildAST(
            expression=expression % arguments,
            calls=calls,
            subscripts=arguments["request"].subscripts,
            order=0)


class AllocateByPriorityBuilder(StructureBuilder):
    """Builder for allocate_by_priority function."""

    def __init__(self, allocate_str: AllocateByPriorityStructure,
                 component: object):
        super().__init__(None, component)
        self.arguments = {
            "request": allocate_str.request,
            "priority": allocate_str.priority,
            "width": allocate_str.width,
            "supply": allocate_str.supply
        }

    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        self.section.imports.add("allocation", "allocate_by_priority")

        calls = self.join_calls(arguments)

        # the last sub of the request must be keep last sub of request and
        # priority
        last_sub = list(arguments["request"].subscripts)[-1]
        # compute the merged subscripts
        final_subscripts = self.get_final_subscripts(arguments)
        # remove last sub from request
        last_sub_value = final_subscripts[last_sub]
        del final_subscripts[last_sub]

        # Update the susbcripts of width and supply
        arguments["width"].reshape(
            self.section.subscripts, final_subscripts, True)
        arguments["supply"].reshape(
            self.section.subscripts, final_subscripts, True)

        # Include last sub of request in the last position and update
        # the subscripts of request and priority
        final_subscripts[last_sub] = last_sub_value
        arguments["request"].reshape(
            self.section.subscripts, final_subscripts, True)
        arguments["priority"].reshape(
            self.section.subscripts, final_subscripts, True)

        expression = "allocate_by_priority(%(request)s, %(priority)s, "\
                     "%(width)s, %(supply)s)"
        return BuildAST(
            expression=expression % arguments,
            calls=calls,
            subscripts=arguments["request"].subscripts,
            order=0)


class ExtLookupBuilder(StructureBuilder):
    """Builder for External Lookups."""
    def __init__(self, getlookup_str: GetLookupsStructure,  component: object):
        super().__init__(None, component)
        self.file = getlookup_str.file
        self.tab = getlookup_str.tab
        self.x_row_or_col = getlookup_str.x_row_or_col
        self.cell = getlookup_str.cell
        self.arguments = {}

    def build(self, arguments: dict) -> Union[BuildAST, None]:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST or None
            The built object, unless the component has been added to an
            existing object using the 'add' method.

        """
        self.component.type = "Lookup"
        self.component.subtype = "External"
        arguments["params"] = "r'%s', '%s', '%s', '%s'" % (
            self.file, self.tab, self.x_row_or_col, self.cell
        )
        final_subs, arguments["subscripts"] =\
            self.section.subscripts.simplify_subscript_input(
                self.def_subs, self.element.subscripts)

        if "ext_lookups" in self.element.objects:
            # Object already exists, use 'add' method
            self.element.objects["ext_lookups"]["expression"] += "\n\n"\
                + self.element.objects["ext_lookups"]["name"]\
                + ".add(%(params)s, %(subscripts)s)" % arguments

            self.update_object_subscripts("ext_lookups", final_subs)

            return None
        else:
            # Create a new object
            self.section.imports.add("external", "ExtLookup")

            arguments["name"] = self.section.namespace.make_python_identifier(
                self.element.identifier, prefix="_ext_lookup")
            arguments["final_subs"] =\
                self.section.subscripts.simplify_subscript_input(
                    self.element.subs_dict)[1]
            self.component.subscripts_dict = [final_subs]

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
    """Builder for External Data."""
    def __init__(self, getdata_str: GetDataStructure,  component: object):
        super().__init__(None, component)
        self.file = getdata_str.file
        self.tab = getdata_str.tab
        self.time_row_or_col = getdata_str.time_row_or_col
        self.cell = getdata_str.cell
        self.keyword = component.keyword
        self.arguments = {}

    def build(self, arguments: dict) -> Union[BuildAST, None]:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST or None
            The built object, unless the component has been added to an
            existing object using the 'add' method.

        """
        self.component.type = "Data"
        self.component.subtype = "External"
        arguments["params"] = "r'%s', '%s', '%s', '%s'" % (
            self.file, self.tab, self.time_row_or_col, self.cell
        )
        final_subs, arguments["subscripts"] =\
            self.section.subscripts.simplify_subscript_input(
                self.def_subs, self.element.subscripts)
        arguments["method"] = "'%s'" % self.keyword if self.keyword else None

        if "ext_data" in self.element.objects:
            # Object already exists, use add method
            self.element.objects["ext_data"]["expression"] += "\n\n"\
                + self.element.objects["ext_data"]["name"]\
                + ".add(%(params)s, %(method)s, %(subscripts)s)" % arguments

            self.update_object_subscripts("ext_data", final_subs)

            return None
        else:
            # Create a new object
            self.section.imports.add("external", "ExtData")

            arguments["name"] = self.section.namespace.make_python_identifier(
                self.element.identifier, prefix="_ext_data")
            arguments["final_subs"] =\
                self.section.subscripts.simplify_subscript_input(
                    self.element.subs_dict)[1]
            self.component.subscripts_dict = [final_subs]

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
    """Builder for External Constants."""
    def __init__(self, getconstant_str: GetConstantsStructure,
                 component: object):
        super().__init__(None, component)
        self.file = getconstant_str.file
        self.tab = getconstant_str.tab
        self.cell = getconstant_str.cell
        self.arguments = {}

    def build(self, arguments: dict) -> Union[BuildAST, None]:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST or None
            The built object, unless the component has been added to an
            existing object using the 'add' method.

        """
        self.component.type = "Constant"
        self.component.subtype = "External"
        arguments["params"] = "r'%s', '%s', '%s'" % (
            self.file, self.tab, self.cell
        )
        final_subs, arguments["subscripts"] =\
            self.section.subscripts.simplify_subscript_input(
                self.def_subs, self.element.subscripts)

        if "constants" in self.element.objects:
            # Object already exists, use 'add' method
            self.element.objects["constants"]["expression"] += "\n\n"\
                + self.element.objects["constants"]["name"]\
                + ".add(%(params)s, %(subscripts)s)" % arguments

            self.update_object_subscripts("constants", final_subs)

            return None
        else:
            # Create a new object
            self.section.imports.add("external", "ExtConstant")

            arguments["name"] = self.section.namespace.make_python_identifier(
                self.element.identifier, prefix="_ext_constant")
            arguments["final_subs"] =\
                self.section.subscripts.simplify_subscript_input(
                    self.element.subs_dict)[1]
            self.component.subscripts_dict = [final_subs]

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
    """Builder for empty DATA expressions."""
    def __init__(self, data_str: DataStructure,  component: object):
        super().__init__(None, component)
        self.keyword = component.keyword
        self.arguments = {}

    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        self.section.imports.add("data", "TabData")

        final_subs, arguments["subscripts"] =\
            self.section.subscripts.simplify_subscript_input(
                self.def_subs, self.element.subscripts)

        arguments["real_name"] = self.element.name
        arguments["py_name"] =\
            self.section.namespace.namespace[self.element.name]
        arguments["method"] = "'%s'" % self.keyword if self.keyword else None

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_data")

        # Create TabData object
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
    """Builder for Initials."""
    def __init__(self, initial_str: InitialStructure, component: object):
        super().__init__(None, component)
        self.arguments = {
            "initial": initial_str.initial
        }

    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        self.component.type = "Stateful"
        self.component.subtype = "Initial"
        self.section.imports.add("statefuls", "Initial")

        arguments["initial"].reshape(
            self.section.subscripts, self.def_subs, True)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_initial")

        # Create the object
        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = Initial(lambda: %(initial)s, "
                          "'%(name)s')" % arguments,
        }
        # Add other-dependencies
        self.element.other_dependencies[arguments["name"]] = {
            "initial": arguments["initial"].calls,
            "step": {}
        }

        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class IntegBuilder(StructureBuilder):
    """Builder for Integs/Stocks."""
    def __init__(self, integ_str: IntegStructure, component: object):
        super().__init__(None, component)
        self.arguments = {
            "flow": integ_str.flow,
            "initial": integ_str.initial
        }
        self.non_negative = integ_str.non_negative

    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        self.component.type = "Stateful"
        self.component.subtype = "Integ"

        arguments["initial"].reshape(
            self.section.subscripts, self.def_subs, True)
        arguments["flow"].reshape(
            self.section.subscripts, self.def_subs, True)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_integ")

        # Create the object
        if self.non_negative:
            # Non-negative stocks
            self.section.imports.add("statefuls", "NonNegativeInteg")
            self.element.objects[arguments["name"]] = {
                "name": arguments["name"],
                "expression": "%(name)s = NonNegativeInteg("
                              "lambda: %(flow)s, "
                              "lambda: %(initial)s, '%(name)s')" % arguments
            }
        else:
            # Regular stocks
            self.section.imports.add("statefuls", "Integ")
            self.element.objects[arguments["name"]] = {
                "name": arguments["name"],
                "expression": "%(name)s = Integ(lambda: %(flow)s, "
                              "lambda: %(initial)s, '%(name)s')" % arguments
            }

        # Add other dependencies
        self.element.other_dependencies[arguments["name"]] = {
            "initial": arguments["initial"].calls,
            "step": arguments["flow"].calls
        }

        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class DelayBuilder(StructureBuilder):
    """Builder for regular Delays."""
    def __init__(self, dtype: str,
                 delay_str: Union[DelayStructure, DelayNStructure],
                 component: object):
        super().__init__(None, component)
        self.arguments = {
            "input": delay_str.input,
            "delay_time": delay_str.delay_time,
            "initial": delay_str.initial,
            "order": delay_str.order
        }
        self.dtype = dtype

    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        self.component.type = "Stateful"
        self.component.subtype = "Delay"
        self.section.imports.add("statefuls", self.dtype)

        arguments["input"].reshape(
            self.section.subscripts, self.def_subs, True)
        arguments["delay_time"].reshape(
            self.section.subscripts, self.def_subs, True)
        arguments["initial"].reshape(
            self.section.subscripts, self.def_subs, True)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix=f"_{self.dtype.lower()}")
        arguments["dtype"] = self.dtype

        # Add the object
        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = %(dtype)s(lambda: %(input)s, "
                          "lambda: %(delay_time)s, lambda: %(initial)s, "
                          "lambda: %(order)s, "
                          "time_step, '%(name)s')" % arguments,
        }
        # Add other dependencies
        self.element.other_dependencies[arguments["name"]] = {
            "initial": merge_dependencies(
                arguments["initial"].calls,
                arguments["delay_time"].calls,
                arguments["order"].calls),
            "step": merge_dependencies(
                arguments["input"].calls,
                arguments["delay_time"].calls)

        }

        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class DelayFixedBuilder(StructureBuilder):
    """Builder for Delay Fixed."""
    def __init__(self, delay_str: DelayFixedStructure, component: object):
        super().__init__(None, component)
        self.arguments = {
            "input": delay_str.input,
            "delay_time": delay_str.delay_time,
            "initial": delay_str.initial,
        }

    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        self.component.type = "Stateful"
        self.component.subtype = "DelayFixed"
        self.section.imports.add("statefuls", "DelayFixed")

        arguments["input"].reshape(
            self.section.subscripts, self.def_subs, True)
        arguments["initial"].reshape(
            self.section.subscripts, self.def_subs, True)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_delayfixed")

        # Create object
        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = DelayFixed(lambda: %(input)s, "
                          "lambda: %(delay_time)s, lambda: %(initial)s, "
                          "time_step, '%(name)s')" % arguments,
        }
        # Add other dependencies
        self.element.other_dependencies[arguments["name"]] = {
            "initial": merge_dependencies(
                arguments["initial"].calls,
                arguments["delay_time"].calls),
            "step": arguments["input"].calls
        }

        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class SmoothBuilder(StructureBuilder):
    """Builder for Smooths."""
    def __init__(self, smooth_str: Union[SmoothStructure, SmoothNStructure],
                 component: object):
        super().__init__(None, component)
        self.arguments = {
            "input": smooth_str.input,
            "smooth_time": smooth_str.smooth_time,
            "initial": smooth_str.initial,
            "order": smooth_str.order
        }

    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        self.component.type = "Stateful"
        self.component.subtype = "Smooth"
        self.section.imports.add("statefuls", "Smooth")

        arguments["input"].reshape(
            self.section.subscripts, self.def_subs, True)
        arguments["smooth_time"].reshape(
            self.section.subscripts, self.def_subs, True)
        arguments["initial"].reshape(
            self.section.subscripts, self.def_subs, True)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_smooth")

        # TODO in the future we need to ad timestep to show warnings about
        # the smooth time as its done with delays (see vensim help for smooth)
        # TODO in the future we may want to have 2 py_backend classes for
        # smooth as the behaviour is different for SMOOTH and SMOOTH N when
        # using RingeKutta scheme

        # Create object
        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = Smooth(lambda: %(input)s, "
                          "lambda: %(smooth_time)s, lambda: %(initial)s, "
                          "lambda: %(order)s, '%(name)s')" % arguments,
        }
        # Add other dependencies
        self.element.other_dependencies[arguments["name"]] = {
            "initial": merge_dependencies(
                arguments["initial"].calls,
                arguments["order"].calls),
            "step": merge_dependencies(
                arguments["input"].calls,
                arguments["smooth_time"].calls)
        }

        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class TrendBuilder(StructureBuilder):
    """Builder for Trends."""
    def __init__(self, trend_str: TrendStructure, component: object):
        super().__init__(None, component)
        self.arguments = {
            "input": trend_str.input,
            "average_time": trend_str.average_time,
            "initial_trend": trend_str.initial_trend,
        }

    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        self.component.type = "Stateful"
        self.component.subtype = "Trend"
        self.section.imports.add("statefuls", "Trend")

        arguments["input"].reshape(
            self.section.subscripts, self.def_subs, True)
        arguments["average_time"].reshape(
            self.section.subscripts, self.def_subs, True)
        arguments["initial_trend"].reshape(
            self.section.subscripts, self.def_subs, True)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_trend")

        # Create object
        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = Trend(lambda: %(input)s, "
                          "lambda: %(average_time)s, "
                          "lambda: %(initial_trend)s, "
                          "'%(name)s')" % arguments,
        }
        # Add other dependencies
        self.element.other_dependencies[arguments["name"]] = {
            "initial": merge_dependencies(
                arguments["initial_trend"].calls,
                arguments["input"].calls,
                arguments["average_time"].calls),
            "step": merge_dependencies(
                arguments["input"].calls,
                arguments["average_time"].calls)
        }

        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class ForecastBuilder(StructureBuilder):
    """Builder for Forecasts."""
    def __init__(self, forecast_str: ForecastStructure, component: object):
        super().__init__(None, component)
        self.arguments = {
            "input": forecast_str.input,
            "average_time": forecast_str.average_time,
            "horizon": forecast_str.horizon,
            "initial_trend": forecast_str.initial_trend
        }

    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        self.component.type = "Stateful"
        self.component.subtype = "Forecast"
        self.section.imports.add("statefuls", "Forecast")

        arguments["input"].reshape(
            self.section.subscripts, self.def_subs, True)
        arguments["average_time"].reshape(
            self.section.subscripts, self.def_subs, True)
        arguments["horizon"].reshape(
            self.section.subscripts, self.def_subs, True)
        arguments["initial_trend"].reshape(
            self.section.subscripts, self.def_subs, True)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_forecast")

        # Create object
        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = Forecast(lambda: %(input)s, "
                          "lambda: %(average_time)s, lambda: %(horizon)s, "
                          "lambda: %(initial_trend)s, '%(name)s')" % arguments,
        }
        # Add other dependencies
        self.element.other_dependencies[arguments["name"]] = {
            "initial": merge_dependencies(
                arguments["input"].calls,
                arguments["initial_trend"].calls),
            "step": merge_dependencies(
                arguments["input"].calls,
                arguments["average_time"].calls,
                arguments["horizon"].calls)
        }

        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class SampleIfTrueBuilder(StructureBuilder):
    """Builder for Sample If True."""
    def __init__(self, sampleiftrue_str: SampleIfTrueStructure,
                 component: object):
        super().__init__(None, component)
        self.arguments = {
            "condition": sampleiftrue_str.condition,
            "input": sampleiftrue_str.input,
            "initial": sampleiftrue_str.initial,
        }

    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        self.component.type = "Stateful"
        self.component.subtype = "SampleIfTrue"
        self.section.imports.add("statefuls", "SampleIfTrue")

        arguments["condition"].reshape(
            self.section.subscripts, self.def_subs, True)
        arguments["input"].reshape(
            self.section.subscripts, self.def_subs, True)
        arguments["initial"].reshape(
            self.section.subscripts, self.def_subs, True)

        arguments["name"] = self.section.namespace.make_python_identifier(
            self.element.identifier, prefix="_sampleiftrue")

        # Create object
        self.element.objects[arguments["name"]] = {
            "name": arguments["name"],
            "expression": "%(name)s = SampleIfTrue(lambda: %(condition)s, "
                          "lambda: %(input)s, lambda: %(initial)s, "
                          "'%(name)s')" % arguments,
        }
        # Add other dependencies
        self.element.other_dependencies[arguments["name"]] = {
            "initial":
                arguments["initial"].calls,
            "step": merge_dependencies(
                arguments["condition"].calls,
                arguments["input"].calls)
        }

        return BuildAST(
            expression=arguments["name"] + "()",
            calls={arguments["name"]: 1},
            subscripts=self.def_subs,
            order=0)


class LookupsBuilder(StructureBuilder):
    """Builder for regular Lookups."""
    def __init__(self, lookups_str: LookupsStructure, component: object):
        super().__init__(None, component)
        self.arguments = {}
        self.x = lookups_str.x
        self.y = lookups_str.y
        self.keyword = lookups_str.type

    def build(self, arguments: dict) -> Union[BuildAST, None]:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST or None
            The built object, unless the component has been added to an
            existing object using the 'add' method.

        """
        self.component.type = "Lookup"
        self.component.subtype = "Normal"
        # Get the numeric values as numpy arrays
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
            # Object already exists, use 'add' method
            self.element.objects["hardcoded_lookups"]["expression"] += "\n\n"\
                + self.element.objects["hardcoded_lookups"]["name"]\
                + ".add(%(x)s, %(y)s, %(subscripts)s)" % arguments

            return None
        else:
            # Create a new object
            self.section.imports.add("lookups", "HardcodedLookups")

            arguments["name"] = self.section.namespace.make_python_identifier(
                self.element.identifier, prefix="_hardcodedlookup")
            arguments["final_subs"] =\
                self.section.subscripts.simplify_subscript_input(
                    self.element.subs_dict)[1]

            self.element.objects["hardcoded_lookups"] = {
                "name": arguments["name"],
                "expression": "%(name)s = HardcodedLookups(%(x)s, %(y)s, "
                              "%(subscripts)s, '%(interp)s', "
                              "%(final_subs)s, '%(name)s')"
                              % arguments,
                "final_subs": self.element.subs_dict
            }

            return BuildAST(
                expression=arguments["name"] + "(x, final_subs)",
                calls={"__lookup__": arguments["name"]},
                subscripts=self.def_subs,
                order=0)


class InlineLookupsBuilder(StructureBuilder):
    """Builder for inline Lookups."""
    def __init__(self, inlinelookups_str: InlineLookupsStructure,
                 component: object):
        super().__init__(None, component)
        self.arguments = {
            "value": inlinelookups_str.argument
        }
        self.lookups = inlinelookups_str.lookups

    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        self.component.type = "Auxiliary"
        self.component.subtype = "with Lookup"
        self.section.imports.add("numpy")
        # Get the numeric values as numpy arrays
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
        if arguments["value"].subscripts:
            subs = arguments["value"].subscripts
            expression = "np.interp(%(value)s, %(x)s, %(y)s)" % arguments
            return BuildAST(
                expression="xr.DataArray(%s, %s, %s)" % (
                    expression, subs, list(subs)),
                calls=arguments["value"].calls,
                subscripts=subs,
                order=0)
        else:
            return BuildAST(
                expression="np.interp(%(value)s, %(x)s, %(y)s)" % arguments,
                calls=arguments["value"].calls,
                subscripts={},
                order=0)


class ReferenceBuilder(StructureBuilder):
    """Builder for references to other variables."""
    def __init__(self, reference_str: ReferenceStructure, component: object):
        super().__init__(None, component)
        self.mapping_subscripts = {}
        self.reference = reference_str.reference
        self.subscripts = reference_str.subscripts
        self.arguments = {}

    @property
    def subscripts(self):
        return self._subscripts

    @subscripts.setter
    def subscripts(self, subscripts: SubscriptsReferenceStructure):
        """Get subscript dictionary from reference"""
        ref_subs = getattr(subscripts, "subscripts", [])

        self._subscripts = self.section.subscripts.make_coord_dict(ref_subs)

        if len(ref_subs) != len(self._subscripts):
            # The reference has repeated subscript ranges, this is
            # not compatible with Python as we use dictionaries to save
            # the subscript ranges information (duplicates a key)

            # Get the original name of the reference
            origin_ref = self.reference
            for origin_ref2 in self.section.namespace.namespace.keys():
                if origin_ref == origin_ref2.lower().replace(" ", "_"):
                    origin_ref = origin_ref2
                    break

            warnings.warn(
                "The reference to '%s' in variable '%s' has duplicated "
                "subscript ranges. If mapping is used in one of them, "
                "please, rewrite reference subscripts to avoid "
                "duplicates. Otherwise, the final model may crash..."
                % (origin_ref, self.element.name)
            )

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
                    not_mapped = True
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
                            not_mapped = False
                            break
                    if not_mapped:
                        # manage other not mapped subscripts
                        # this is necessary for Allocate Available
                        # where we must force the expression in the
                        # right to have more subscripts thant the
                        # expression in the left
                        self.mapping_subscripts[dim] = coordinates
                else:
                    # the subscript is in the variable definition,
                    # do not change it
                    self.mapping_subscripts[dim] = coordinates

    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        if self.reference not in self.section.namespace.cleanspace:
            # Manage references to subscripts (subscripts used as variables)
            expression, subscripts =\
                self.section.subscripts.subscript2num[self.reference]
            subscripts_out = self.section.subscripts.simplify_subscript_input(
                subscripts)[1]
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

        expression, final_subs = self._visit_subscripts(
            expression, original_subs)

        return BuildAST(
            expression=expression,
            calls={reference: 1},
            subscripts=final_subs,
            order=0)

    def _visit_subscripts(self, expression: str, original_subs: dict) -> tuple:
        """
        Visit the subcripts of a reference to subset a subarray if neccessary
        or apply mapping.

        Parameters
        ----------
        expression: str
            The expression of visiting the variable.
        original_subs: dict
            The original subscript dict of the variable.

        Returns
        -------
        expression: str
            The expression with the necessary operations.
        mapping_subscirpts: dict
            The final subscripts of the reference after applying mapping.

        """
        loc, rename, final_subs, reset_coords, to_float =\
            visit_loc(self.subscripts, original_subs)

        if loc is not None:
            # NUMPY: expression += "[%s]" % ", ".join(loc)
            expression += f".loc[{loc}]"
        if to_float:
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
                self.mapping_subscripts)[1]
            expression = "xr.DataArray(%s.values, %s, %s)" % (
                expression, subscripts_out, list(self.mapping_subscripts)
            )

        return expression, self.mapping_subscripts


class NumericBuilder(StructureBuilder):
    """Builder for numeric and nan values."""
    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
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
    """Builder for arrays."""
    def build(self, arguments: dict) -> BuildAST:
        """
        Build method.

        Parameters
        ----------
        arguments: dict
            The dictionary of builded arguments.

        Returns
        -------
        built_ast: BuildAST
            The built object.

        """
        self.value = np.array2string(
            self.value.reshape(compute_shape(self.def_subs)),
            separator=",",
            threshold=np.prod(self.value.shape)
        )
        self.component.type = "Constant"
        self.component.subtype = self.component.subtype or "Normal"

        final_subs, subscripts_out =\
            self.section.subscripts.simplify_subscript_input(
                self.def_subs, self.element.subscripts)

        return BuildAST(
            expression="xr.DataArray(%s, %s, %s)" % (
                self.value, subscripts_out, list(final_subs)),
            calls={},
            subscripts=final_subs,
            order=0)


def merge_dependencies(*dependencies: dict, inplace: bool = False) -> dict:
    """
    Merge two dependencies dicts of an element.

    Parameters
    ----------
    dependencies: dict
        The dictionaries of dependencies to merge.

    inplace: bool (optional)
        If True the final dependencies dict will be updated in the first
        dependencies argument, mutating it. Default is False.

    Returns
    -------
    current: dict
        The final dependencies dict.

    """
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

    return current


def visit_loc(current_subs: dict, original_subs: dict,
              keep_shape: bool = False) -> tuple:
    """
    Compares the original subscripts and the current subscripts and
    returns subindexing information if needed.

    Parameters
    ----------
    current_subs: dict
        The dictionary of the subscripts that are used in the variable.

    original_subs: dict
        The dictionary of the original subscripts of the variable.

    keep_shape: bool (optional)
        If True will keep the number of dimensions of the original element
        and return only loc. Default is False.

    Returns
    -------
    loc: list of str or None
        List of the subscripting in each dimensions. If all are full (":"),
        None is rerned wich means that array indexing is not needed.

    rename: dict
        Dictionary of the dimensions to rename.

    final_subs: dict
        Dictionary of the final subscripts of the variable.

    reset_coords: bool
        Boolean indicating if the coords need to be reseted.

    to_float: bool
        Boolean indicating if the variable should be converted to a float.

    """
    final_subs, rename, loc, reset_coords, to_float = {}, {}, [], False, True
    subscripts_zipped = zip(current_subs.items(), original_subs.items())
    for (dim, coord), (orig_dim, orig_coord) in subscripts_zipped:
        if len(coord) == 1:
            # subset a 1 dimension value
            # NUMPY: subset value [:, N, :, :]
            if keep_shape:
                # NUMPY: not necessary
                loc.append(f"[{repr(coord[0])}]")
            else:
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
                if dim != orig_dim:
                    loc.append("_subscript_dict['%s']" % dim)
                else:
                    # workaround for locs from external objects merge
                    loc.append(repr(coord))
            final_subs[dim] = coord
            to_float = False
        else:
            # do nothing
            # NUMPY: same, we can remove float = False
            loc.append(":")
            final_subs[dim] = coord
            to_float = False

        if dim != orig_dim and len(coord) != 1:
            # NUMPY: check order of dimensions, make all subranges work
            #        with the same dimensions?
            # NUMPY: this could be solved in the previous if/then/else
            rename[orig_dim] = dim

    if all(dim == ":" for dim in loc):
        # if all are ":" then no need to loc
        loc = None
    else:
        loc = ", ".join(loc)

    if keep_shape:
        return loc

    # convert to float if also coords are reseted (input is an array)
    to_float = to_float and reset_coords

    # NUMPY: save and return only loc, the other are not needed
    return loc, rename, final_subs, reset_coords, to_float


class ASTVisitor:
    """
    ASTVisitor allows visiting the Abstract Synatx Tree of a component
    returning the Python object and generating the neccessary objects.

    Parameters
    ----------
    component: ComponentBuilder
        The component builder to build.

    """
    _builders = {
        InitialStructure: InitialBuilder,
        IntegStructure: IntegBuilder,
        DelayStructure: lambda x, y: DelayBuilder("Delay", x, y),
        DelayNStructure: lambda x, y: DelayBuilder("DelayN", x, y),
        DelayFixedStructure: DelayFixedBuilder,
        SmoothStructure: SmoothBuilder,
        SmoothNStructure: SmoothBuilder,
        TrendStructure: TrendBuilder,
        ForecastStructure: ForecastBuilder,
        SampleIfTrueStructure: SampleIfTrueBuilder,
        GetConstantsStructure: ExtConstantBuilder,
        GetDataStructure: ExtDataBuilder,
        GetLookupsStructure: ExtLookupBuilder,
        LookupsStructure: LookupsBuilder,
        InlineLookupsStructure: InlineLookupsBuilder,
        DataStructure: TabDataBuilder,
        ReferenceStructure: ReferenceBuilder,
        CallStructure: CallBuilder,
        GameStructure: GameBuilder,
        AllocateAvailableStructure: AllocateAvailableBuilder,
        AllocateByPriorityStructure: AllocateByPriorityBuilder,
        LogicStructure: OperationBuilder,
        ArithmeticStructure: OperationBuilder,
        int: NumericBuilder,
        float: NumericBuilder,
        np.ndarray: ArrayBuilder,
    }

    def __init__(self, component: object):
        # component typing should be ComponentBuilder, but importing it
        # for typing would create a circular dependency :S
        self.ast = component.ast
        self.subscripts = component.subscripts_dict
        self.component = component

    def visit(self) -> Union[None, BuildAST]:
        """
        Visit the Abstract Syntax Tree of the component.

        Returns
        -------
        visit_out: BuildAST or None
            The BuildAST object resulting from visiting the AST. If the
            component content has been added to an existing object
            using the 'add' method it will return None.

        """
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
            # NUMPY: in this case we need to tile along dims if neccessary
            #        or reorder the dimensions
            visit_out.reshape(
                self.component.section.subscripts, self.subscripts, True)

        return visit_out

    def _visit(self, ast_object: AbstractSyntax) -> AbstractSyntax:
        """
        Visit one Builder and its arguments.
        """
        builder = self._builders[type(ast_object)](ast_object, self.component)
        arguments = {
            name: self._visit(value)
            for name, value in builder.arguments.items()
        }
        return builder.build(arguments)
