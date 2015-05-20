'''
    created: August 13, 2014
    last update: March 28 2015
    James Houghton <james.p.houghton@gmail.com>
'''
import parsimonious
from parsimonious.nodes import NodeVisitor
from pysd import functions
from helpers import *
import component_class_template



###############################################################
#Updates:
#
# - Feb 16, 2015: brought parsing of full sections of the model into the PEG parser.
# - March 28, 2015: incorporated parsimonious's node walker to take care of class construction.
###############################################################


###############################################################
#Todo:
#
# - parse geometry information
# - parser can't handle multi-line equations
# - check that the identifiers discovered are not the same as python keywords
# - parse excessive spaces out of docstrings

# - It would be good to include the original vensim/xmile non-python-safed identifiers for model components in the docstring, along with the cleaned version
# - also, include the  original equation, and the python clean version
# - maybe we add elements to the functions which represent the pre and post translation components (one each?)
# - model documentation shouldn't include the 'self' parts of a descriptor

# should build in a mechanism to fail gracefully
# maybe want to make the node visitor intelligently add spaces around things
# should check that sin/cos are using the same units in vensim and numpy
# the 'pi' keyword is broken

# are we parsing model control components properly? We may need to change the grammar so that identifiers that if we find an
# identifier that is a control

###############################################################


######################################################
#Dictionary:
#
# This is the vensim to python translation dictionary. Many expressions (+-/*, etc)
# translate directly, but some need to have their names changed, etc. This is
# how we decide that. If it isn't in the dictionary, it probably won't work
#
######################################################

dictionary = {"ABS":"abs", "INTEGER":"int", "EXP":"np.exp", "INF":"np.inf", "LOG10":"np.log10",
    "PI":"np.pi", "SIN":"np.sin", "COS":"np.cos", "SQRT":"np.sqrt", "TAN":"np.tan",
    "LOGNORMAL":"np.random.lognormal", "RANDOM NORMAL":"functions.bounded_normal",
    "POISSON":"np.random.poisson", "LN":"np.log", "EXPRND":"np.random.exponential",
    "RANDOM UNIFORM":"np.random.rand", "MIN":"min", "MAX":"max", "ARCCOS":"np.arccos",
    "ARCSIN":"np.arcsin", "ARCTAN":"np.arctan", "IF THEN ELSE":"functions.if_then_else",
    "STEP":"functions.step", "MODULO":"np.mod", "PULSE":"functions.pulse",
    "PULSE TRAIN":"functions.pulse_train", "RAMP":"functions.ramp",
    "=":"==", "<=":"<=", "<":"<", ">=":">=", ">":">", "^":"**"}



###############################################################
# General Notes on PEG parser:
#
# - we separate the grammar out into chunks to simplify testing.
#     This way, in the unit tests we can construct a parser based upon a subset of the grammar
#     and test that subset on its own.

#  - backslashes in the .mdl file get removed by python  before they go into parsimonious, as python interprets them as
#     a line continuation character. This is probably ok, as that is how vensim is using them as well, but
#     if the user includes backslashes anywhere, will give them trouble.
#
# - unfortunately the \s in the identifier can match newline characters - do we want this?
#
# - # If you put a *+? quantifier on a node listing (ie: Docstring?) then it creates an anonymous
#     node, which makes it hard to match up later in the tree crawler
#
# - we could think about putting the dictionary and the grammars within the class, because the class uses them exclusively
#
################################################################




#################################################################
# Notes on Expression Grammar
#
# - we have to break 'reference' out from identifier, because in reference situations we'll want to add self.<name>(), but
#      we also need to be able to clean and access the identifier independently. To force parsimonious to handle the reference as
#      a seperate object from the Identifier, we add a trailing optional space character
#
# - we separated out single space characters from the _ entry, so that it can have any number or combination of space characters.
#
# - Vensim uses a 'backslash, newline' syntax as a way to split an equation onto multiple lines. We address this by including it as
#      an allowed space character.
#
# - calls to lookup functions don't use keywords, so we have to use identifiers. They take only one parameter.
#
# - for some reason, 'Factor' is consuming newlines, which it shouldn't. This gives issues
#      if an equation is long enough to go to multiple lines, which happens sometimes
#
# - we have to sort keywords in decreasing order of length so that the peg parser doesnt quit early when finding a partial keyword
#################################################################

keywords = ' / '.join(['"%s"'%key for key in reversed(sorted(dictionary.keys(), key=len))])

expression_grammar = """
    Condition   = Term _ Conditional*
    Conditional = ("<=" / "<" / ">=" / ">" / "=") _ Term
    
    Term     = Factor _ Additive*
    Additive = ("+"/"-") _ Factor
    
    Factor   = ExpBase _ Multiplicative*
    Multiplicative = ("*" / "/") _ ExpBase
    
    ExpBase  = Primary _ Exponentive*
    Exponentive = "^" _ Primary
    
    Primary  = Call / LUCall / Parens / Signed / Number / Reference
    Parens   = "(" _ Condition _ ")"
    Call     = Keyword _ "(" _ Condition _ ("," _ Condition)* _ ")"
    LUCall   = Identifier _ "(" _ Condition _ ")"
    Signed   = ("-"/"+") Primary
    Reference = Identifier _

    Number   = ((~"[0-9]"+ "."? ~"[0-9]"*) / ("." ~"[0-9]"+)) (("e"/"E") ("-"/"+") ~"[0-9]"+)?
    Identifier =  ~"[a-zA-Z]" ~"[a-zA-Z0-9_\$\s]"*
    
    _ = spacechar*
    spacechar = " "* ~"\t"* (~r"\\\\" NL)*
    
    Keyword = """ + keywords

#################################################################
# Notes on Entry Gramamr
#
# - meta entry is the first entry, if there is model documentation, or the partition separating model control parameters.
#
# - SNL means 'spaced newline'
#
# - 'Flaux' represents either a flow or an auxiliary, as the syntax is the same
#
# - 'copair' represents a coordinate pair
#
# - The docstring parser also consumes trailing newlines. This is probably ok, but
#      something to keep in mind if we have problems later.
#
# - The docstring parser contains a lookahead to make sure we don't consume starlines
#
#################################################################

entry_grammar = """
    Entry = MetaEntry / ModelEntry
    
    MetaEntry  = SNL starline SNL Docstring SNL starline _ "~" SNL Docstring SNL "|" SNL*
    ModelEntry = SNL Component SNL "~" _ Unit SNL "~" _ Docstring SNL "|" SNL*
    
    Unit      = ~"[^~|]"*
    Docstring = (~"[^~|]" !~"(\*{3,})")*
    Component = Stock / Flaux / Lookup
    
    Lookup     = Identifier _ "(" _ NL _ Range _ CopairList _ ")"
    Range      = "[" _ Copair _ "-" _ Copair _ "]"
    CopairList = AddCopair*
    AddCopair  = "," _ Copair
    Copair     = "(" _ Primary _ "," _ Primary _ ")"
    
    Stock     = Identifier _ "=" _ "INTEG" _ "(" _ NL _ Condition _ "," _ NL _ Condition _ ")"
    Flaux     = Identifier _ "=" SNL Condition
    
    SNL = _ NL* _
    NL = _ ~"[\\r\\n]" _
    starline = ~"\*{3,}"
    """ + expression_grammar

#################################################################
# Notes on File Grammar
#
# - In the 'Model' definiton, we use arbitrary characters (.) to represent the backslashes, because escaping in here is a nightmare
#
#################################################################

file_grammar = """
    Model = SNL "{UTF-8}" Content ~"...---///" Rubbish*
    
    Content = Entry+
    
    Rubbish = ~"."* NL*
    """ + entry_grammar


#################################################################
# Notes on node visitor
#
# - each function in the parser takes an argument which is the result of having visited each of the children
#
# - we separate the parse function from the class initialization to facilitate unit testing,
#      although in practice, they will most certainly always be called sequentially
#################################################################


class TextParser(NodeVisitor):
    def __init__(self, grammar):
        class component_class(component_class_template.component_class_template):
            self.__str__ = 'Undefined'

        self.component_class = component_class
        self.grammar = parsimonious.Grammar(grammar)

    def parse(self, text):
        self.ast = self.grammar.parse(text)
        return self.visit(self.ast)

    ############# 'entry' level translators ############################
    visit_Entry = visit_Component = NodeVisitor.lift_child

    def visit_ModelEntry(self, n, (_1, Component_Identifier, _2, tld1, _3, Unit, _4, tld2, _5, Docstring, _6, pipe, _7)):
        """
            All we have to do here is add to the docstring of the relevant function the things that arent 
            available at lower levels. The stock/flaux/lookup visitors will take care of adding methods to the class
        """
        ds = 'Units: %s \n'%Unit + Docstring
        entry = getattr(self.component_class, Component_Identifier)
        
        if hasattr(entry, 'im_func'): #most functions
            entry.im_func.func_doc += ds
        else: #the lookups - which are represented as callable classes, instead of functions
            entry.__doc__ += ds

                         
    def visit_Stock(self, n, (Identifier, _1, eq, _2, integ, _3,
                              lparen, _4, NL1, _5, expression, _6,
                              comma, _7, NL2, _8, initial_condition, _9, rparen)):

        #create a place to store the stock's current value
        self.component_class.state[Identifier] = None
        
        #create a 'derivative function' that can be
        # called by the d_dt boilerplate function and passed to the integrator
        funcstr = ('def d%s_dt(self):\n'%Identifier +
                   '    return %s'%expression   )
        exec funcstr in self.component_class.__dict__
             
        #create an 'intialization function' of the form '<stock>_init()' that 
        # can be called when the model is reset to initialize the state variable
        funcstr = ('def %s_init(self):\n'%Identifier +
                   '    return %s'%initial_condition)
        exec funcstr in self.component_class.__dict__
        
        #create a function that points to the state dictionary, to let other
        # components reference the state without explicitly having to know that
        # it is a stock. This is the function that gets an elaborated docstring
        funcstr = ('def %s(self):\n'%Identifier +
                   '    """    %s = %s \n'%(Identifier, expression) + #include the docstring
                   '        Initial Value: %s \n'%initial_condition +
                   '        Type: Stock \n' +
                   '        Do not overwrite this function\n' +
                   '    """\n' +
                   '    return self.state["%s"]'%Identifier)
        exec funcstr in self.component_class.__dict__
        return Identifier
    
    
    def visit_Flaux(self, n, (Identifier, _1, eq, SNL, expression)):
        funcstr = ('def    %s(self):\n'%Identifier +
                   '    """%s = %s \n'%(Identifier, expression) +
                   '       Type: Flow or Auxiliary \n ' +
                   '    """\n' +
                   '    return %s'%expression)
        exec funcstr in self.component_class.__dict__

        return Identifier
    

    def visit_Lookup(self, n, (Identifier, _1, lparen, _2, NL1, _3, Range, _4, CopairList, _5, rparen)):
        """
            This is pretty complex, as we need to create a function that will be called elsewhere. We've
            created a structure in which function calls can only come from a very limited set of keywords,
            but the syntax that we expect to see with lookups is such that the keyword will fail, and the syntax
            be ambiguous.
        """
        # if we want to include range checking, we need to parse the range. for now, lazy...
        
        xs, ys = zip(*CopairList)
        self.component_class.__dict__.update({Identifier:functions.lookup(xs, ys)})
        
        #This docstring approach could be improved
        getattr(self.component_class, Identifier).__doc__ = ('%s is lookup with coordinates:\n%s'%(Identifier, CopairList) +
                                                             'Type: Flow or Auxiliary \n')
        
        return Identifier
    
    
    
    def visit_Macro(self, n, vc):
        """
            When we take on the task of implementing delay functions, this will be where
            we choose to deal with them. Delay functions require programmatically creating additional stocks.
        """
        pass
    
    
    def visit_Unit(self, n, vc):
        return n.text.strip()
    
    visit_Docstring = visit_Unit

    ######### 'expression' level visitors ###############################
    def visit_Keyword(self, n, vc):
        return dictionary[n.text]
    
    def visit_Reference(self, n, (Identifier, _)):
        return 'self.'+Identifier+'()'
    
    def visit_Identifier(self, n, vc):
        #todo: should check here that identifiers are not python keywords...
        string = n.text
        string = string.lower()
        string = string.strip()
        string = string.replace(' ', '_')
        return string

    def visit_LUCall(self, n, (Identifier, _1, lparen, _2, Condition, _3,  rparen)):
        return 'self.'+Identifier+'('+Condition+')'
        
    def visit_Copair(self, n, (lparen, _1, xcoord, _2, comma, _3, ycoord, _, rparen)):
        return (float(xcoord), float(ycoord))
        
    def visit_AddCopair(self, n, (comma, _1, Copair)):
        return Copair

    def visit_Range(self, n, copairs):
        pass
    
    def visit_CopairList(self, n, copairs):
        return copairs

    def visit_Conditional(self, n, (condition, _, term)):
        return dictionary[condition] + term

    visit_Exponentive = visit_Conditional
    
    def generic_visit(self, n, vc):
        """
        Replace childbearing nodes with a list of their children;
        for leaves, return the node text;
        for empty nodes, return an empty string.
        """
        return ''.join(filter(None, vc)) or n.text or ''




def import_vensim(mdl_file):
    """
        This takes the filename of a vensim model and builds a
        class representing that model to be a subclass in the pysd
        main class.
        """
    parser = TextParser(file_grammar)
    
    with open(mdl_file, 'rU') as file:
        text = file.read().decode('utf-8')

    parser.parse(text)

    component_class = parser.component_class
    component_class.__str__ = 'Import of '+mdl_file

    return component_class
