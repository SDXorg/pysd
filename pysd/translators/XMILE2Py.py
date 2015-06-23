'''
August 15 2014
James Houghton <james.p.houghton@gmail.com>
Major edits June 22 2015
'''
import SMILE2Py
from lxml import etree

def import_XMILE(xmile_file):
    """
    Load the xml file and parse it to an element tree
    """
    parser = etree.XMLParser(encoding="utf-8", recover=True)
    root = etree.parse(xmile_file, parser=parser).getroot()
    return _build_execution_tree(root), _get_xmile_params(root)
#
#
#def _build_execution_tree(root):
#    """
#    Read the Stock and Aux elemenst
#    Build directed acyclic graph with equations on nodes
#    At some point, it may be interesting to have nodes represent objects, and actually execute the tree itself.
#    For now, we execute the tree by unpacking it to a function
#    """
#    NS = root.nsmap.values()[0]
#    components = nx.DiGraph()
#    # add aux and flow nodes
#    for element in root.xpath('//ns:model/ns:variables/ns:aux|//ns:model/ns:variables/ns:flow',namespaces={'ns':NS}):
#        name = SMILE2Py.clean(element.attrib['name'])
#        pyeqn, dependencies = SMILE2Py.translate(element.xpath('ns:eqn', namespaces={'ns':NS})[0].text)
#        components.add_node(name, attr_dict={'eqn':pyeqn, 'return':False})
#        for dependency in dependencies:
#            components.add_edge(name, dependency)
#    
#    # add nodes for the derivatives of stocks
#    for element in root.xpath('//ns:model/ns:variables/ns:stock',namespaces={'ns':NS}):
#        name = "d"+SMILE2Py.clean(element.attrib['name'])+"_dt"
#        inflows = [SMILE2Py.clean(e.text) for e in element.xpath('ns:inflow', namespaces={'ns':NS})]
#        outflows = [SMILE2Py.clean(e.text) for e in element.xpath('ns:outflow', namespaces={'ns':NS})]
#        pyeqn = ' + '.join(inflows) if inflows else ''
#        pyeqn += ' - '+' - '.join(outflows) if outflows else ''
#        components.add_node(name, attr_dict={'eqn':pyeqn, 'return':True})
#        for dependency in inflows+outflows:
#            components.add_edge(name, dependency)
#    
#    return components
#
#def _get_xmile_params(root):
#    """
#        Read the xmile file and get the components which don't go into the execution tree
#    """
#    
#    NS = root.nsmap.values()[0]
#    #Get timeseries information from the XMILE file
#    tstart = float(root.xpath('//ns:sim_specs/ns:start',namespaces={'ns':NS})[0].text)
#    tstop = float(root.xpath('//ns:sim_specs/ns:stop',namespaces={'ns':NS})[0].text)
#    dt = float(root.xpath('//ns:sim_specs/ns:dt',namespaces={'ns':NS})[0].text)
#
#    initial_stocks = {}
#    for element in root.xpath('//ns:model/ns:variables/ns:stock',namespaces={'ns':NS}):
#        name = SMILE2Py.clean(element.attrib['name'])
#        value = float(element.xpath('ns:eqn', namespaces={'ns':NS})[0].text)
#        initial_stocks[name] = value
#
#    return {'tstart':tstart, 'tstop':tstop, 'dt':dt,
#            'stocknames':sorted(initial_stocks.keys()),
#            'initial_values':[initial_stocks[key] for key in sorted(initial_stocks.keys())]}
#
