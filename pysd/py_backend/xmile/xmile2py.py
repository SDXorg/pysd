"""
Deals with accessing the components of the xmile file, and
formatting them for the builder

James Houghton <james.p.houghton@gmail.com>
Alexey Prey Mulyukin <alexprey@yandex.ru> from sdCloud.io development team.

"""
from __future__ import absolute_import
from .SMILE2Py import SMILEParser
from lxml import etree
from ...py_backend import builder, utils


def translate_xmile(xmile_file):
    """ Translate an xmile model file into a python class.
    Functionality is currently limited.

    """

    # process xml file
    xml_parser = etree.XMLParser(encoding="utf-8", recover=True)
    root = etree.parse(xmile_file, parser=xml_parser).getroot()
    NS = root.nsmap[None]  # namespace of the xmile document

    def get_xpath_text(node, path, ns={'ns': NS}, default=''):
        """ Safe access of occassionally missing elements """
        # defined here to take advantage of NS in default
        try:
            return node.xpath(path, namespaces=ns)[0].text
        except:
            return default
            
    def is_constant_expression(py_expr):
        try:
            val = float(py_expr)
            return True
        except ValueError:
            return False

    # build model namespace
    namespace = {
        'TIME': 'time', 
        'Time': 'time', 
        'time': 'time'
    }  # namespace of the python model
    names_xpath = '//ns:model/ns:variables/ns:aux|' \
                  '//ns:model/ns:variables/ns:flow|' \
                  '//ns:model/ns:variables/ns:stock'

    for node in root.xpath(names_xpath, namespaces={'ns': NS}):
        name = node.attrib['name']
        _, namespace = utils.make_python_identifier(name, namespace)

    model_elements = []
    smile_parser = SMILEParser(namespace)
    
    # add aux and flow elements
    flaux_xpath = '//ns:model/ns:variables/ns:aux|//ns:model/ns:variables/ns:flow'
    for node in root.xpath(flaux_xpath, namespaces={'ns': NS}):
        name = node.attrib['name']
        units = get_xpath_text(node, 'ns:units')
        doc = get_xpath_text(node, 'ns:doc')
        py_name = namespace[name]
        eqn = get_xpath_text(node, 'ns:eqn')
        
        element = {
            'kind': 'component',
            'real_name': name,
            'unit': units,
            'doc': doc,
            'eqn': eqn,
            'py_name': py_name,
            'subs': [],  # Todo later
            'arguments': '',
        }
                
        tranlation, new_structure = smile_parser.parse(eqn, element)
        element.update(tranlation)
        if is_constant_expression(element['py_expr']):
            element['kind'] = 'constant'
        
        model_elements += new_structure
        model_elements.append(element)

    # add stock elements
    stock_xpath = '//ns:model/ns:variables/ns:stock'
    for node in root.xpath(stock_xpath, namespaces={'ns': NS}):
        name = node.attrib['name']
        units = get_xpath_text(node, 'ns:units')
        doc = get_xpath_text(node, 'ns:doc')
        py_name = namespace[name]
        
        # Extract input and output flows equations
        inflows = [n.text for n in node.xpath('ns:inflow', namespaces={'ns': NS})]
        outflows = [n.text for n in node.xpath('ns:outflow', namespaces={'ns': NS})]
        
        eqn = ' + '.join(inflows) if inflows else ''
        eqn += (' - ' + ' - '.join(outflows)) if outflows else ''
        
        element = {
            'kind': 'component' if inflows or outflows else 'constant',
            'real_name': name,
            'unit': units,
            'doc': doc,
            'eqn': eqn,
            'py_name': py_name,
            'subs': [],  # Todo later
            'arguments': '' 
        }

        # Parse each flow equations
        py_inflows = []
        for inputFlow in inflows:
            translation, new_structure = smile_parser.parse(inputFlow, element)
            py_inflows.append(translation['py_expr'])
            model_elements += new_structure
        
        # Parse each flow equations
        py_outflows = []
        for outputFlow in outflows:
            translation, new_structure = smile_parser.parse(outputFlow, element)
            py_outflows.append(translation['py_expr'])
            model_elements += new_structure
        
        py_ddt = ' + '.join(py_inflows) if py_inflows else ''
        py_ddt += (' - ' + ' - '.join(py_outflows)) if py_outflows else ''

        # Read the initial value equation for stock element
        initial_value_eqn = get_xpath_text(node, 'ns:eqn')
        translation, new_structure = smile_parser.parse(initial_value_eqn, element)
        py_initial_value = translation['py_expr']
        model_elements += new_structure

        py_expr, new_structure = builder.add_stock(identifier=py_name,
                                                  subs=[],  # Todo later
                                                  expression=py_ddt,
                                                  initial_condition=py_initial_value,
                                                  subscript_dict={},  # Todo later
                                                  )
        element['py_expr'] = py_expr
        model_elements.append(element)
        model_elements += new_structure

    # remove timestamp pieces so as not to double-count
    model_elements_parsed = []
    for element in model_elements:
        if element['real_name'].lower() not in ['initial time', 'final time', 'time step', 'saveper']:
            model_elements_parsed.append(element)
    model_elements = model_elements_parsed

    # Add timeseries information
    
    # Read the start time of simulation
    sim_spec_node = root.xpath('//ns:sim_specs', namespaces={'ns': NS});
    time_units = sim_spec_node[0].attrib['time_units'] if (len(sim_spec_node) > 0 and sim_spec_node[0].attrib.has_key('time_units')) else ""
    
    tstart = root.xpath('//ns:sim_specs/ns:start', namespaces={'ns': NS})[0].text
    element = {
        'kind': 'constant',
        'real_name': 'INITIAL TIME',
        'unit': time_units,
        'doc': 'The initial time for the simulation.',
        'eqn': tstart,
        'py_name': 'initial_time',
        'subs': None,
        'arguments': '',
    }
    translation, new_structure = smile_parser.parse(tstart, element)
    element.update(translation)
    model_elements.append(element)
    model_elements += new_structure

    # Read the final time of simulation
    tstop = root.xpath('//ns:sim_specs/ns:stop', namespaces={'ns': NS})[0].text
    element = {
        'kind': 'constant',
        'real_name': 'FINAL TIME',
        'unit': time_units,
        'doc': 'The final time for the simulation.',
        'eqn': tstart,
        'py_name': 'final_time',
        'subs': None,
        'arguments': '',
    }
    
    translation, new_structure = smile_parser.parse(tstop, element)
    element.update(translation)
    model_elements.append(element)
    model_elements += new_structure

    # Read the time step of simulation
    dt = root.xpath('//ns:sim_specs/ns:dt', namespaces={'ns': NS})[0].text
    element = {
        'kind': 'constant',
        'real_name': 'TIME STEP',
        'unit': time_units,
        'doc': 'The time step for the simulation.',
        'eqn': dt,
        'py_name': 'time_step',
        'subs': None,
        'arguments': '',
    }
    translation, new_structure = smile_parser.parse(dt, element)
    element.update(translation)
    model_elements.append(element)
    model_elements += new_structure
    
    # Add the SAVEPER attribute to the model
    model_elements.append({
        'kind': 'constant',
        'real_name': 'SAVEPER',
        'unit': time_units,
        'doc': 'The time step for the simulation.',
        'eqn': dt,
        'py_name': 'saveper',
        'py_expr': 'time_step()',
        'subs': None,
        'arguments': '',
    })

    outfile_name = xmile_file.replace('.xmile', '.py')

    builder.build(elements=model_elements,
                  subscript_dict={},
                  namespace=namespace,
                  outfile_name=outfile_name)

    return outfile_name
