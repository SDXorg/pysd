'''
August 15 2014
James Houghton <james.p.houghton@gmail.com>
    
'''
import SMILE2Py
import networkx as nx
from lxml import etree

def import_XMILE(xmile_file):

    parser = etree.XMLParser(encoding="utf-8", recover=True)
    root = etree.parse(xmile_file, parser=parser).getroot()
    NS = root.nsmap.values()[0]


    # Read the Stock and Aux elemenst
    # Build directed acyclic graph with equations on nodes
    # At some point, it may be interesting to have nodes represent objects, and actually execute the tree itself.
    # For now, we execute the tree by unpacking it to a function
    components = nx.DiGraph()
    for element in root.xpath('//ns:model/ns:variables/ns:aux|//ns:model/ns:variables/ns:flow',namespaces={'ns':NS}):
        name = SMILE2Py.clean(element.attrib['name'])
        pyeqn, dependencies = SMILE2Py.translate(element.xpath('ns:eqn', namespaces={'ns':NS})[0].text)
        components.add_node(name, attr_dict={'eqn':pyeqn})
        for dependency in dependencies:
            components.add_edge(name, dependency)

    #check the model
    _validate_execution_graph(components)

    #if we topo sort the graph, we can build up equations making sure we get the dependancies right
    lines = []
    for node_name in nx.algorithms.topological_sort(components, reverse=True):
        if components.node[node_name]: #the leaves will return empty dicts - leaves are stocks
            lines.append(node_name + ' = ' + components.node[node_name]['eqn'])


    #now process the stock elements to initialize the model and work out the return elements
    initial_stocks = {}
    derivative_lines = []
    return_elements = {}
    for element in root.xpath('//ns:model/ns:variables/ns:stock',namespaces={'ns':NS}):
        name = SMILE2Py.clean(element.attrib['name'])
        val = element.xpath('ns:eqn', namespaces={'ns':NS})[0].text
        initial_stocks[name] = float(val)
        
        inflows = [e.text for e in element.xpath('ns:inflow', namespaces={'ns':NS})]
        outflows = [e.text for e in element.xpath('ns:outflow', namespaces={'ns':NS})]
        derivative_line = "d"+name+"_dt = "
        derivative_line += ' + '.join(inflows) if inflows else ''
        derivative_line += ' - '+' - '.join(outflows) if outflows else ''
        derivative_lines.append(derivative_line)
        
        return_elements[name] = "d"+name+"_dt"


    #Build up a string representing the function we intend to return, containing the model.
    #scipy.integrate.odeint expects the function it operates to have a list representing the state
    #vector. Assume stocks are in the array in alphabetical order.
    func_string = "def dstocks_dt(stocks, t): \n\n"
    func_string += "    "+', '.join(sorted(initial_stocks.keys()))+" = stocks\n\n" #unpack the input array

    for line in lines+derivative_lines:
        func_string += "    " + line + "\n" #make sure to indent properly!

    func_string += "\n    return ["+ ', '.join(sorted(return_elements.values()))+"]" #caution here- this only works because the derivative names are derived from the stock names

    #Executing the string we've created builds the function
    exec(func_string)


    #Get timeseries information from the XMILE file
    tstart = float(root.xpath('//ns:sim_specs/ns:start',namespaces={'ns':NS})[0].text)
    tstop = float(root.xpath('//ns:sim_specs/ns:stop',namespaces={'ns':NS})[0].text)
    dt = float(root.xpath('//ns:sim_specs/ns:dt',namespaces={'ns':NS})[0].text)

    return dstocks_dt, {'tstart':tstart, 'tstop':tstop, 'dt':dt,
                        'stocknames':sorted(initial_stocks.keys()),
                        'initial_values':[initial_stocks[key] for key in sorted(initial_stocks.keys())]}



def _validate_execution_graph(components):
    #if its a well formed set of equations, it should be a directed acyclic graph
    assert nx.algorithms.is_directed_acyclic_graph(components)


