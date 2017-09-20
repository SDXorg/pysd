"""
Created August 14 2014
James Houghton <james.p.houghton@gmail.com>

Changed May 03 2017
Alexey Prey Mulyukin <alexprey@yandex.ru>
    Changes:
    
    [May 03 2017] Alexey Prey Mulyukin: Integrate support to
        logical operators like 'AND', 'OR' and 'NOT'.
        Fix support the whitespaces in expressions between 
        operators and operands.
        Add support to modulo operator - 'MOD'.
        Fix support for case insensitive in function names.
    
This module converts a string of SMILE syntax into Python
    
"""
import parsimonious
from parsimonious.nodes import NodeVisitor
import pkg_resources
import re

# Here we define which python function each XMILE keyword corresponds to
functions = {
    "abs": "abs", "int": "int", "exp": "np.exp", "inf": "np.inf", "log10": "np.log10",
    "pi": "np.pi", "sin": "np.sin", "cos": "np.cos", "sqrt": "np.sqrt", "tan": "np.tan",
    "lognormal": "np.random.lognormal", "normal": "np.random.normal",
    "poisson": "np.random.poisson", "ln": "np.log", "exprnd": "np.random.exponential",
    "random": "np.random.rand", "min": "min", "max": "max", "arccos": "np.arccos",
    "arcsin": "np.arcsin", "arctan": "np.arctan",
    "if_then_else": "functions.if_then_else",
    "step": "functions.step", "pulse": "functions.pulse"
}

prefix_operators = {
    "not": " not ",
    "-": "-", "+": " ",
}

infix_operators = {
    "and": " and ", "or": " or ",
    "=": "==", "<=": "<=", "<": "<", ">=": ">=", ">": ">", "<>": "!=",
    "^": "**", "+": "+", "-": "-",
    "*": "*", "/": "/", "mod": "%",
}

builders = {}


def format_word_list(word_list):
    return '|'.join(
        [re.escape(k) for k in reversed(sorted(word_list, key=len))])


class SMILEParser(NodeVisitor):
    def __init__(self, model_namespace):
        self.model_namespace = model_namespace

        self.extended_model_namespace = {
            key.replace(' ', '_'): value for key, value in self.model_namespace.items()}
        self.extended_model_namespace.update(self.model_namespace)
        self.extended_model_namespace.update({'dt': 'time_step'})

        grammar = pkg_resources.resource_string("pysd", "py_backend/xmile/smile.grammar")
        grammar = grammar.decode('ascii').format(
            funcs=format_word_list(functions.keys()),
            in_ops=format_word_list(infix_operators.keys()),
            pre_ops=format_word_list(prefix_operators.keys()),
            identifiers=format_word_list(self.extended_model_namespace.keys()),
            build_keywords=format_word_list(builders.keys())
        )

        self.grammar = parsimonious.Grammar(grammar)

    def parse(self, text, context='eqn'):
        """
           context : <string> 'eqn', 'defn'
                If context is set to equation, lone identifiers will be parsed as calls to elements
                If context is set to definition, lone identifiers will be cleaned and returned.
        """
        self.ast = self.grammar.parse(text)
        self.context = context
        return self.visit(self.ast)

    def visit_identifier(self, n, vc):
        return self.extended_model_namespace[n.text] + '()'

    def visit_func(self, n, vc):
        return functions[n.text.lower()]

    def visit_pre_oper(self, n, vc):
        return prefix_operators[n.text.lower()]

    def visit_in_oper(self, n, vc):
        return infix_operators[n.text.lower()]

    def generic_visit(self, n, vc):
        """
        Replace childbearing nodes with a list of their children;
        for leaves, return the node text;
        for empty nodes, return an empty string.

        Handles:
        - call
        - parens
        -
        """
        return ''.join(filter(None, vc)) or n.text or ''
