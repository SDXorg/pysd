from collections import defaultdict

class IndentedString:
    def __init__(self, indent_level=0):
        self.indent_level = indent_level
        self.string = ""

    def __iadd__(self, other: str):
        prefix = " " * 4 * self.indent_level
        if other != "\n":
            self.string += prefix
        self.string += other
        return self

    def add_raw(self, string, ignore_indent=False):
        if ignore_indent:
            self.string += string
        else:
            self.__iadd__(string)

    def __str__(self):
        return self.string


class StatementTopoSort:
    def __init__(self, ignored_variables=tuple()):
        self.dependency_graph = dict()
        self.sorted_order = []
        self.ignored_variables = ignored_variables

    def add_stmt(self, lhs_var, rhs_vars):
        if lhs_var not in self.dependency_graph:
            self.dependency_graph[lhs_var] = set()
        self.dependency_graph[lhs_var].update(rhs_vars)
        for var in rhs_vars:
            if var not in self.dependency_graph:
                self.dependency_graph[var] = set()


    def recursive_order_search(self, current, visited):
        if current in self.ignored_variables:
            return
        visited.add(current)
        for child in self.dependency_graph[current]:
            if child == current:
                continue
            if child in self.ignored_variables:
                continue
            if child not in visited:
                self.recursive_order_search(child, visited)
        if current not in self.sorted_order:
            self.sorted_order.append(current)

    def sort(self, reversed=False):
        """
        reversed=False(default) means it will sort according to LHS given RHS
        so a = b + c would mean b, c will come before a
        """
        for key in self.dependency_graph.keys():
            self.recursive_order_search(key, set())

        return self.sorted_order if not reversed else self.sorted_order[::-1]


def vensim_name_to_identifier(name: str):
    return name.lower().replace(" ", "_")
