
# coding: utf-8

# In[1]:

import parsimonious
import re
import networkx as nx


# Need to add:
#

# In[2]:

# parsing components
#this is challenging as we have to account for spaces...

#I've added a '+' to the 'Neg' primary (making it really a 'signed value') because some students include that in their models...

_grammar = """
    Condition = Term " "* Conditional*
    Conditional = ("<=" / "<" / ">=" / ">" / "=") " "* Term
    
    Term     = Factor " "* Additive*
    Additive = ("+"/"-") " "* Factor
    
    Factor   = ExpBase " "* Multiplicative*
    Multiplicative = ("*" / "/") " "* ExpBase
    
    ExpBase  = Primary " "* Exponentive*
    Exponentive = "^" " "* Primary
    
    Primary  = Call / Parens / Neg / Number / Identifier
    Parens   = "(" " "* Condition " "* ")"
    Call     = Keyword " "* "(" " "* Condition " "* ("," " "* Condition)* ")"
    Neg      = ("-"/"+") Primary
    Number   = ((~"[0-9]"+ "."? ~"[0-9]"*) / ("." ~"[0-9]"+)) (("e"/"E") ("-"/"+") ~"[0-9]"+)?
    Identifier =  ~"[a-z]" ~"[a-z0-9_\$\s]"*
    
    Keyword = "exprnd" / "exp" / "sin" / "cos" / "abs" / "integer" / "inf" / "log10" / "pi" /
    "sqrt" / "tan" / "lognormal" / "random normal" / "poisson" / "ln" / "min" / "max" /
    "random uniform" / "arccos" / "arcsin" / "arctan" / "if then else" / "step" / "modulo" /
    "pulse train" / "pulse" / "ramp"
    """
_g = parsimonious.Grammar(_grammar)


_dictionary = {"abs":"abs", "integer":"int", "exp":"np.exp", "inf":"np.inf", "log10":"np.log10",
    "pi":"np.pi", "sin":"np.sin", "cos":"np.cos", "sqrt":"np.sqrt", "tan":"np.tan",
    "lognormal":"np.random.lognormal", "random normal":"functions.bounded_normal",
    "poisson":"np.random.poisson", "ln":"np.log", "exprnd":"np.random.exponential",
    "random uniform":"np.random.rand", "min":"min", "max":"max", "arccos":"np.arccos",
    "arcsin":"np.arcsin", "arctan":"np.arctan", "if then else":"functions.if_then_else",
    "step":"functions.step", "modulo":"np.mod", "pulse":"functions.pulse",
    "pulse train":"functions.pulse_train", "ramp":"functions.ramp",
    "=":"==", "<=":"<=", "<":"<", ">=":">=", ">":">", "^":"**"}

_add_t_param_list = ["step", "ramp", "pulse train", "pulse"]

def _clean_identifier(string):
    #at the moment, we may have training spaces on an identifier that need to be dealt with
    #in the future, it would be better to improve the parser so that it handles that whitespace properly
    string = string.strip()
    string = string.replace(' ', '_')
    return string



def _get_identifiers(node):
    """
        Its helpful to know what the identifiers are in a string of XMILE syntax,
        as we can then build up a dependancy graph based upon references to other
        elements.
        
        In this function we crawl through the Abstract Syntax Tree recursively
        and when we find a node that is an identifier, we add it to our list.
        """
    identifiers = []
    for child in node:
        identifiers += _get_identifiers(child) #merge all into one list
    identifiers += [_clean_identifier(node.text)] if node.expr_name in ['Identifier'] else []
    return identifiers


def _translate_tree(node):
    """
        At some point in the future, it may make sense to consolidate code that is
        repeated between the various parsers - XMILE, vensim, etc.
        """
    if node.expr_name in ['Exponentive', 'Conditional']: #non-terminal lookup
        return _dictionary[node.children[0].text] + ''.join([_translate_tree(child) for child in node.children[1:]])
    elif node.expr_name == "Call":
        if node.children[0].text in _add_t_param_list:
            return ''.join([_translate_tree(child) for child in node if child.text != ")"]+[', t)']) #add a t parameter
        else:
            return ''.join([_translate_tree(child) for child in node])  #just normal
    elif node.expr_name == "Keyword": #terminal lookup
        return _dictionary[node.text]
    elif node.expr_name == "Identifier":
        return _clean_identifier(node.text)
    else:
        if node.children:
            return ''.join([_translate_tree(child) for child in node])
        else:
            return node.text


def _translate(s=''):
    """
        given a string, clean it, parse it, translate it into python,
        and return a line of executable python code, and a list of
        variables which the code depends upon.
        """
    
    #need to clean the string, deal with spaces, etc.
    try:
        tree = _g.parse(s.strip())
    except:
        print 'failed to parse: "'+s+'"'
    return _translate_tree(tree), _get_identifiers(tree)

_translate('step(0, 40)')



#tree building section

#we need to add the ability to deal with initial conditions on stocks that are not floats.
#one way to do this would be to just take the string from the init, and then once the
#network is built up, parse and execute it. This should return some sort of value, if the model is well formed
#but we need to figure out how to give it the dependencies that it needs.
#we don't have access to the build-a-function function here...
#may need to push that execution off onto the pysd engine, instead of trying to do it in the model translator.
#that becomes a bit of a bigger project, and I'm not sure how it meshes with the xmile ethos.


def _build_execution_tree(string):
    
    initial_stocks = {}
    components = nx.DiGraph()
    
    def add_node(name, eqn):
        eqn = eqn.replace('\t', '').replace('\n','').replace('\r','').replace("\\",'')
        pyeqn, dependencies = _translate(eqn)
        components.add_node(name, attr_dict={'eqn':pyeqn})
        for dependency in dependencies:
            components.add_edge(name, dependency)
    
    for definition in re.findall("\{.*\}(?P<capture>(.*\r?\n)+?)\*{3,}", string)[0][0].split('|'):
        
        name = re.findall('\r?\n(?P<name>.*)=', definition)
        
        if name: #this is a really ugly way to pass on strings that don't even have a name (probably whitespace at the end of the section)
            name, _ = _translate(name[0].strip()) # if the name exists, we clean it up...
        else:
            continue
        
        stock_match = re.findall('=\s*integ\s*\(\r?\n\s*(?P<eqn>.*),\r?\n\s*(?P<init>.*)\)\r?\n\s*~', definition)
        if stock_match:
            eqn, init = stock_match[0]
            add_node('d'+name+'_dt', eqn) #add a node for the derivative
            add_node(name+'_init', init) #add a node for the initial condition
            initial_stocks[name] = None
        
        else:
            flowaux_match = re.findall('=\s*\r?\n\s*(?P<eqn>[^~]+)\r?\n\s*~', definition)
            if flowaux_match:
                eqn = flowaux_match[0]
                add_node(name, eqn)
    
    
    assert nx.is_directed_acyclic_graph(components)
    
    return components, initial_stocks




def _get_vensim_params(string):
    
    params = {}
    for control in re.findall("\*{3,}~\r?\n(?P<capture>(.*\r?\n)+?)\\\\", string)[0][0].split('|'):
        
        #this is a really ugly way to pass on strings that don't even have a name (probably whitespace at the end of the section)
        name = re.findall('\r?\n(?P<name>.*)=', control)
        if name:
            name, _ = _translate(name[0].strip())
        else:
            continue
        
        eqn = re.findall('=\s*(?P<eqn>[^~]+)\r?\n\s*~', control)
        if eqn:
            eqn = eqn[0].replace('\t', '').replace('\n','').replace('\r','').replace("\\",'')
            pyeqn, dependencies = _translate(eqn)
            params[name] = pyeqn
    
    return params





def import_vensim(mdl_file=''):
    with open(mdl_file) as file:
        string = file.read().lower()
    
    model, initial_stocks = _build_execution_tree(string)
    params = _get_vensim_params(string)
    
    return model, {'tstart':float(params['initial_time']), 'tstop':float(params['final_time']),
        'dt':float(params['time_step']),
        'stocknames':sorted(initial_stocks.keys()),
        'initial_values':[initial_stocks[key] for key in sorted(initial_stocks.keys())]}



