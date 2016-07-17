## Begun on 20 June 2016 by Simon De Stercke
## File to parse docstring information for all the elements in a model

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

class SDVarDoc(NodeVisitor):

    def __init__(self, grammar, text):
        self.sdVar = {}
        ast = Grammar(grammar).parse(text)
        self.visit(ast)
    def visit_name(self, n, vc):
        self.sdVar['name'] = n.text
    def visit_modelName(self, n, vc):
        self.sdVar['modelName'] = n.text
    def visit_unit(self, n, vc):
        self.sdVar['unit'] = n.text
    def visit_comment(self, n, vc):
        self.sdVar['comment'] = n.text
    def generic_visit(self, n, vc):
        pass


## Below is the grammar and strings on which I tested.
#
# grammar = """\
#     sdVar = (sep? name sep "-"* sep modelNameWrap sep unit sep+ comment? " "*)?
#     sep = ws "\\n" ws
#     ws = " "*
#     name = ~"[A-z ]+"
#     modelNameWrap = '(' modelName ')'
#     modelName = ~"[A-z_]+"
#     unit = ~"[A-z\\, \\/\\*\\[\\]\\?0-9]*"
#     comment = ~"[A-z _+-/*\\n]+"
#     """
#
# text= """
#     Teacup Temperature
#     ------------------
#     (teacup_temperature)
#     Degrees
#
#
#     """
#
# text2= """
#     Implicit
#     --------
#     (_init_teacup_temperature)
#     See docs for teacup_temperature
#
#     Provides initial conditions for teacup_temperature function
#     """
#
# text3 = """"""
#
# text4 = """
#     TIME STEP
#     ---------
#     (time_step)
#     Minute [0,?]
#
#     The time step for the simulation.
#     """
#
# print SDVarDoc(grammar,text).sdVar
# print SDVarDoc(grammar,text2).sdVar
# print SDVarDoc(grammar,text3).sdVar
# print SDVarDoc(grammar,text4).sdVar