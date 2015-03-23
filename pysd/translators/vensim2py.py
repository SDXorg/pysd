'''
    created: August 13, 2014
    last update: February, 19 2015
    James Houghton <james.p.houghton@gmail.com>
'''
import parsimonious
from pysd import functions
import inspect
from helpers import *
import component_class_template



###############################################################
#Updates:
#
# - Feb 16: brought parsing of full sections of the model into the PEG parser.
###############################################################



###############################################################
#Todo:
#
# - what we should probably be doing here is extending the components class
# - parse geometry information
# - parser can't handle multi-line equations
# - check that the identifiers discovered are not the same as python keywords
# - we don't necessarily need to force everything to lowercase before parsing it. It would be nice, at least, to preserve the case of the docstring
# - parse excessive spaces out of docstrings
# - add functions that let us make changes to variables, override things, etc.
# - It would be good to include the original vensim/xmile non-python-safed identifiers for model components in the docstring, along with the cleaned version
# - also, include the  original equation, and the python clean version
# - maybe we add elements to the functions which represent the pre and post translation components (one each?)
# - model documentation shouldn't include the 'self' parts of a descriptor
###############################################################


###############################################################
# Notes on PEG parser:
#
#  - backslashes in the .mdl file get removed by python  before they go into parsimonious, as python interprets them as
# a line continuation character. This is probably ok, as that is how vensim is using them as well, but
# if the user includes backslashes anywhere, will give them trouble.
#
# - for some reason, 'Factor' is consuming newlines, which it shouldn't. This might give issues
# if an equation is long enough to go to multiple lines
#
# - The docstring parser also consumes trailing newlines. This is probably ok, but
# something to keep in mind if we have problems later.
#
# - The docstring parser contains a lookahead to make sure we don't consume starlines
#
# - separated out single space characters from the _ entry, so that it can have any number or combination of space characters.
#
# - unfortunately the \s in the identifier can match newline characters - do we want this?
#
# - copair means 'coordinate pair'
#
# - # If you put a *+? quantifier on a node listing (ie: Docstring?) then it creates an anonymous
# node, which makes it hard to match up later in the tree crawler
#
# - 'Flaux' represents either a flow or an auxiliary, as the syntax is the same
#
# - we may need to have seperate identifiers for the 'Condition' elements in the Stock element.
# the first represents the expression, the second represents the initial condition
# but I'm not sure if the order is guaranteed.
#
# -

################################################################



grammar = """
    Model = _ NL* _ "{utf-8}" Content ~"...---///" Rubbish*
    
    Content = Entry+
    
    Entry = MetaEntry / ModelEntry
    Rubbish = ~"."* NL*
    
    MetaEntry  = _ NL* _ starline _ NL* _ Docstring _ NL* _ starline _ "~" NL* Docstring NL* "|" NL*
    ModelEntry = _ NL* _ Component _ NL* _ "~" _ Unit _ NL* _ "~" _ Docstring _ NL* _ "|" NL*
    
    Unit      = ~"[^~|]"*
    Docstring = (~"[^~|]" !~"(\*{3,})")*
    Component = Stock / Flaux / Lookup
    
    Lookup     = Identifier _ "(" _ NL _ Range _ AddCopair* _ NL _ ")"
    Range      = "[" _ Copair _ "-" _ Copair _ "]"
    AddCopair  = "," _ Copair
    Copair     = "(" _ Primary _ "," _ Primary _ ")"
    
    Stock     = Identifier _ "=" _ "integ" _ "(" _ NL _ Condition _ "," _ NL _ Condition _ ")"
    Flaux     = Identifier _ "=" _ NL* _ Condition
    
    Condition   = Term _ Conditional*
    Conditional = ("<=" / "<" / ">=" / ">" / "=") _ Term
    
    Term     = Factor _ Additive*
    Additive = ("+"/"-") _ Factor
    
    Factor   = ExpBase _ Multiplicative*
    Multiplicative = ("*" / "/") _ ExpBase
    
    ExpBase  = Primary _ Exponentive*
    Exponentive = "^" _ Primary
    
    Primary  = Call / Parens / Signed / Number / Identifier
    Parens   = "(" _ Condition _ ")"
    Call     = Keyword _ "(" _ Condition _ ("," _ Condition)* ")"
    Signed   = ("-"/"+") Primary
    Number   = ((~"[0-9]"+ "."? ~"[0-9]"*) / ("." ~"[0-9]"+)) (("e"/"E") ("-"/"+") ~"[0-9]"+)?
    Identifier =  ~"[a-z]" ~"[a-z0-9_\$\s]"*
    
    Keyword = "exprnd" / "exp" / "sin" / "cos" / "abs" / "integer" / "inf" / "log10" / "pi" /
    "sqrt" / "tan" / "lognormal" / "random normal" / "poisson" / "ln" / "min" / "max" /
    "random uniform" / "arccos" / "arcsin" / "arctan" / "if then else" / "step" / "modulo" /
    "pulse train" / "pulse" / "ramp"
    
    NL = _ ~"[\\r\\n]" _
    _ = spacechar*
    spacechar = " "* ~"\t"*
    starline = ~"\*{3,}"
    """
g = parsimonious.Grammar(grammar)


######################################################
#Dictionary:
#
# This is the vensim to python translation dictionary. Many expressions (+-/*, etc)
# translate directly, but some need to have their names changed, etc. This is
# how we decide that. If it isn't in the dictionary, it probably won't work
#
######################################################

dictionary = {"abs":"abs", "integer":"int", "exp":"np.exp", "inf":"np.inf", "log10":"np.log10",
    "pi":"np.pi", "sin":"np.sin", "cos":"np.cos", "sqrt":"np.sqrt", "tan":"np.tan",
    "lognormal":"np.random.lognormal", "random normal":"functions.bounded_normal",
    "poisson":"np.random.poisson", "ln":"np.log", "exprnd":"np.random.exponential",
    "random uniform":"np.random.rand", "min":"min", "max":"max", "arccos":"np.arccos",
    "arcsin":"np.arcsin", "arctan":"np.arctan", "if then else":"functions.if_then_else",
    "step":"functions.step", "modulo":"np.mod", "pulse":"functions.pulse",
    "pulse train":"functions.pulse_train", "ramp":"functions.ramp",
    "=":"==", "<=":"<=", "<":"<", ">=":">=", ">":">", "^":"**"}

add_t_param_list = ["step", "ramp", "pulse train", "pulse"]




def translate_expr(node):
    """
        This translates the AST below the structural level, at the expression level
        """
    if node.expr_name in ['Exponentive', 'Conditional']: #non-terminal lookup
        return dictionary[node.children[0].text] + ''.join([translate_expr(child) for child in node.children[1:]])
    
    elif node.expr_name == "Call":
        if node.children[0].text in add_t_param_list:
            return ''.join([translate_expr(child) for child in node if child.text != ")"]+[', self.t)']) #add a t parameter
        else:
            return ''.join([translate_expr(child) for child in node])  #just normal

    elif node.expr_name == "Keyword": #terminal lookup
        return dictionary[node.text]

    elif node.expr_name == "Identifier":
        return 'self.'+clean_identifier(node.text)+'()'

    else:
        if node.children:
            return ''.join([translate_expr(child) for child in node])
        else:
            return node.text



def translate_modelentry(node):
    """
        Takes a node of the AST that corresponds to a ModelEntry
        and returns a dictionary containing either just the function,
        in case of flow/aux variables or lookups, or a derivative
        function and an initial condition in the case of a stock
        """
    docnode = getChildren(node, 'Docstring')
    docs = docnode[0].text.strip()
    
    unitnode = getChildren(node, 'Unit')
    unit = unitnode[0].text.strip()
    
    # This method of dealing with different types of elements is rather inelegant,
    # and I would like to change it in the future
    component = getChildren(node, 'Component')[0]
    stocknode = getChildren(component, 'Stock')
    flauxnode = getChildren(component, 'Flaux')
    lookupnode = getChildren(component, 'Lookup')
    
    ret_dict = {}
    
    if len(stocknode):
        
        identifier = clean_identifier(getChildren(stocknode[0], 'Identifier')[0].text)
        
        exprnode, initnode = getChildren(stocknode[0], 'Condition')
        expr = translate_expr(exprnode).strip()
        init = translate_expr(initnode).strip()
        
        docstring = docs + "\n Units: " + unit + "\n Equation: " + expr + "\n Init: " + init
        
        ret_dict['d'+identifier+'_dt'] = lambda self: eval(expr) # the fact that this is a lambda function makes debugging harder.
        ret_dict['d'+identifier+'_dt'].func_name = 'd'+identifier+'_dt'
        ret_dict['d'+identifier+'_dt'].func_doc = docstring
        ret_dict['d'+identifier+'_dt'].units = unit
        ret_dict['d'+identifier+'_dt'].__str__ = 'd'+identifier+'_dt'+"()\n"+expr
        
        ret_dict['state'] = {identifier:0}
        ret_dict['_'+identifier+'_init'] = lambda self: eval(init)
        ret_dict[identifier] = lambda self: self.state[identifier]
    
    
    elif len(flauxnode):
        identifier = clean_identifier(getChildren(flauxnode[0], 'Identifier')[0].text)
        exprnode = getChildren(flauxnode[0], 'Condition')[0]
        expr = translate_expr(exprnode).strip()
        
        docstring = docs + "\n Units: " + unit + "\n Equation: " + expr + "\n"
        
        ret_dict[identifier] = lambda self: eval(expr)
        ret_dict[identifier].func_name = identifier
        ret_dict[identifier].func_doc = docstring
        ret_dict[identifier].units = unit
        ret_dict[identifier].__str__ = identifier+"()\n"+expr
    
    elif len(lookupnode):
        identifier = clean_identifier(getChildren(lookupnode[0], 'Identifier'))
        
    # do this later
    
    return ret_dict




class component_class(component_class_template.component_class_template):
    pass



def import_vensim(mdl_file):
    """
        This takes the filename of a vensim model and builds a
        class representing that model to be a subclass in the pysd
        main class.
        """

    #at some point in the future, this should be expanded to include the model-level description.
    component_class.__str__ = 'Import of '+mdl_file
    
    with open(mdl_file) as file:
        filetext = file.read().lower()

    ast = g.parse(filetext) #abstract syntax tree for the whole file! may or may not be a good strategy, but its what we got.

    content = getChildren(ast, 'Content')[0]
    entries = getChildren(content, 'Entry')
    for entry in entries:
        child = entry.children[0] # this is brittle, assumes that the entry will have exactly one child. Syntactically it should, but...
        if child.expr_name == 'MetaEntry':
            pass
            # this is where we should get some of the information about model
        elif child.expr_name == 'ModelEntry':
            elements = translate_modelentry(child)
            update(component_class.__dict__, elements) #use our own (nested recursive) dictionary update
    
    return component_class

