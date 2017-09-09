"""
Deals with accessing the components of the xmile file, and
formatting them for the builder

James Houghton <james.p.houghton@gmail.com>
Alexey Prey Mulyukin

"""

from pysd.py_backend.xmile.SMILE2Py import SMILEParser
from lxml import etree
from pysd.py_backend import builder, utils


def translate_xmile(xmile_file):
    """ Translate an xmile model file into a python class.
    Functionality is currently limited.

    """

    # process xml file
    xml_parser = etree.XMLParser(encoding="utf-8", recover=True)
    root = etree.parse(xmile_file, parser=xml_parser).getroot()
    NS = root.nsmap.values()[0]  # namespace of the xmile document

    model_elements = []
    smile_parser = SMILEParser()
    namespace = {'TIME': 'time', 'Time': 'time'}  # namespace of the python model

    # add aux and flow elements
    flaux_xpath = '//ns:model/ns:variables/ns:aux|//ns:model/ns:variables/ns:flow'
    for element in root.xpath(flaux_xpath, namespaces={'ns': NS}):
        name = element.attrib['name']
        units = element.xpath('ns:units', namespaces={'ns': NS})[0].text
        doc = element.xpath('ns:doc', namespaces={'ns': NS})[0].text
        py_name, namespace = utils.make_python_identifier(name, namespace)
        eqn = element.xpath('ns:eqn', namespaces={'ns': NS})[0].text
        py_eqn = smile_parser.parse(eqn)

        model_elements.append({'kind': 'component',  # Not always the case - could be constant!
                               'real_name': name,
                               'unit': units,
                               'doc': doc,
                               'eqn': eqn,
                               'py_name': py_name,
                               'subs': None,  # Todo later
                               'py_eqn': py_eqn})

    # add stock elements
    stock_xpath = '//ns:model/ns:variables/ns:stock'
    for element in root.xpath(stock_xpath, namespaces={'ns': NS}):
        name = element.attrib['name']
        units = element.xpath('ns:units', namespaces={'ns': NS})[0].text
        doc = element.xpath('ns:doc', namespaces={'ns': NS})[0].text
        py_name, namespace = utils.make_python_identifier(name, namespace)

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

        initial_value = element.xpath('ns:eqn', namespaces={'ns': NS})[0].text
        py_initial_value = smile_parser.parse(initial_value)

        py_eqn, new_structure = builder.add_stock(identifier=py_name,
                                                  subs=None,  # Todo later
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
                               'subs': None,  # Todo later
                               'py_eqn': py_eqn})

        model_elements += new_structure

    # Add timeseries information
    time_units = root.xpath('//ns:sim_specs', namespaces={'ns': NS})[0].attrib['time_units']
    tstart = root.xpath('//ns:sim_specs/ns:start', namespaces={'ns': NS})[0].text
    py_tstart = smile_parser.parse(tstart)

    model_elements.append({
        'kind': 'component',  # Not always the case - could be constant!
        'real_name': 'INITIAL TIME',
        'unit': time_units,
        'doc': 'The initial time for the simulation.',
        'eqn': tstart,
        'py_name': 'initial_time',
        'subs': None,
        'py_eqn': py_tstart,
    })

    tstop = root.xpath('//ns:sim_specs/ns:stop', namespaces={'ns': NS})[0].text
    py_tstop = smile_parser.parse(tstop)

    model_elements.append({
        'kind': 'component',  # Not always the case - could be constant!
        'real_name': 'FINAL TIME',
        'unit': time_units,
        'doc': 'The final time for the simulation.',
        'eqn': tstart,
        'py_name': 'final_time',
        'subs': None,
        'py_eqn': py_tstop,
    })

    dt = root.xpath('//ns:sim_specs/ns:dt', namespaces={'ns': NS})[0].text
    py_dt = smile_parser.parse(dt)

    model_elements.append({
        'kind': 'component',  # Not always the case - could be constant!
        'real_name': 'TIME STEP',
        'unit': time_units,
        'doc': 'The time step for the simulation.',
        'eqn': dt,
        'py_name': 'time_step',
        'subs': None,
        'py_eqn': py_dt,
    })

    # Todo: Saveper

    outfile_name = xmile_file.replace('.xmile', '.py')

    builder.build(elements=model_elements,
                  subscript_dict={},
                  namespace=namespace,
                  outfile_name=outfile_name)

    return outfile_name
