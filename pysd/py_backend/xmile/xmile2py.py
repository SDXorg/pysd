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

    def get_xpath_text(element, path, ns={'ns': NS}, default=''):
        """ Safe access of occassionally missing elements """
        # defined here to take advantage of NS in default
        try:
            return element.xpath(path, namespaces=ns)[0].text
        except:
            return default

    # build model namespace
    namespace = {'TIME': 'time', 'Time': 'time', 'time': 'time'}  # namespace of the python model
    names_xpath = '//ns:model/ns:variables/ns:aux|' \
                  '//ns:model/ns:variables/ns:flow|' \
                  '//ns:model/ns:variables/ns:stock'

    for element in root.xpath(names_xpath, namespaces={'ns': NS}):
        name = element.attrib['name']
        _, namespace = utils.make_python_identifier(name, namespace)

    model_elements = []
    smile_parser = SMILEParser(namespace)

    # add aux and flow elements
    flaux_xpath = '//ns:model/ns:variables/ns:aux|//ns:model/ns:variables/ns:flow'
    for element in root.xpath(flaux_xpath, namespaces={'ns': NS}):
        name = element.attrib['name']
        units = get_xpath_text(element, 'ns:units')
        doc = get_xpath_text(element, 'ns:doc')
        py_name = namespace[name]
        eqn = get_xpath_text(element, 'ns:eqn')
        py_expr = smile_parser.parse(eqn)

        model_elements.append({'kind': 'component',  # Not always the case - could be constant!
                               'real_name': name,
                               'unit': units,
                               'doc': doc,
                               'eqn': eqn,
                               'py_name': py_name,
                               'subs': [],  # Todo later
                               'py_expr': py_expr,
                               'arguments': '',})

    # add stock elements
    stock_xpath = '//ns:model/ns:variables/ns:stock'
    for element in root.xpath(stock_xpath, namespaces={'ns': NS}):
        name = element.attrib['name']
        units = get_xpath_text(element, 'ns:units')
        doc = get_xpath_text(element, 'ns:doc')
        py_name = namespace[name]

        inflows = [e.text for e in
                   element.xpath('ns:inflow', namespaces={'ns': NS})]
        outflows = [e.text for e in
                    element.xpath('ns:outflow', namespaces={'ns': NS})]

        eqn = ' + '.join(inflows)
        eqn += ' - ' + ' - '.join(outflows) if outflows else ''

        py_inflows = [smile_parser.parse(i) for i in inflows]
        py_outflows = [smile_parser.parse(o) for o in outflows]

        py_ddt = ' + '.join(py_inflows) if py_inflows else ''
        py_ddt += ' - ' + ' - '.join(py_outflows) if py_outflows else ''

        initial_value = get_xpath_text(element, 'ns:eqn')
        py_initial_value = smile_parser.parse(initial_value)

        py_expr, new_structure = builder.add_stock(identifier=py_name,
                                                  subs=[],  # Todo later
                                                  expression=py_ddt,
                                                  initial_condition=py_initial_value,
                                                  subscript_dict={},  # Todo later
                                                  )

        model_elements.append({'kind': 'component',  # Not always the case - could be constant!
                               'real_name': name,
                               'unit': units,
                               'doc': doc,
                               'eqn': eqn,
                               'py_name': py_name,
                               'subs': [],  # Todo later
                               'py_expr': py_expr,
                               'arguments': '', })

        model_elements += new_structure

    # remove timestamp pieces so as not to double-count
    model_elements_parsed = []
    for element in model_elements:
        if element['real_name'].lower() not in ['initial time', 'final time', 'time step']:
            model_elements_parsed.append(element)
    model_elements = model_elements_parsed

    # Add timeseries information
    time_units = root.xpath('//ns:sim_specs', namespaces={'ns': NS})[0].attrib['time_units']
    tstart = root.xpath('//ns:sim_specs/ns:start', namespaces={'ns': NS})[0].text
    py_tstart = smile_parser.parse(tstart)

    model_elements.append({
        'kind': 'constant',
        'real_name': 'INITIAL TIME',
        'unit': time_units,
        'doc': 'The initial time for the simulation.',
        'eqn': tstart,
        'py_name': 'initial_time',
        'subs': None,
        'py_expr': py_tstart,
        'arguments': '',
    })

    tstop = root.xpath('//ns:sim_specs/ns:stop', namespaces={'ns': NS})[0].text
    py_tstop = smile_parser.parse(tstop)

    model_elements.append({
        'kind': 'constant',
        'real_name': 'FINAL TIME',
        'unit': time_units,
        'doc': 'The final time for the simulation.',
        'eqn': tstart,
        'py_name': 'final_time',
        'subs': None,
        'py_expr': py_tstop,
        'arguments': '',
    })

    dt = root.xpath('//ns:sim_specs/ns:dt', namespaces={'ns': NS})[0].text
    py_dt = smile_parser.parse(dt)

    model_elements.append({
        'kind': 'constant',
        'real_name': 'TIME STEP',
        'unit': time_units,
        'doc': 'The time step for the simulation.',
        'eqn': dt,
        'py_name': 'time_step',
        'subs': None,
        'py_expr': py_dt,
        'arguments': '',
    })

    model_elements.append({
        'kind': 'constant',
        'real_name': 'SAVE PER',
        'unit': time_units,
        'doc': 'The time step for the simulation results.',
        'eqn': dt,
        'py_name': 'saveper',
        'subs': None,
        'py_expr': py_dt,
        'arguments': ''
    })

    outfile_name = xmile_file.replace('.xmile', '.py')

    builder.build(elements=model_elements,
                  subscript_dict={},
                  namespace=namespace,
                  outfile_name=outfile_name)

    return outfile_name
