'''
August 15 2014
James Houghton <james.p.houghton@gmail.com>
Major edits June 22 2015
'''
from pysd.translators.SMILE2Py import SMILEParser
from lxml import etree
from pysd import builder


def translate_xmile(xmile_file):
    """ Translate an xmile model file into a python class.
    Functionality is currently limited.
    
    """
    py_model_file = build_python(xmile_file)
    return py_model_file


def build_python(xmile_file):
    """ Load the xml file and pass the relevant elements to the builder class """

    smile_parser = SMILEParser()

    xml_parser = etree.XMLParser(encoding="utf-8", recover=True)
    root = etree.parse(xmile_file, parser=xml_parser).getroot()
    NS = root.nsmap.values()[0]

    filename = '.'.join(xmile_file.split('.')[:-1])+'.py'
    builder.new_model(filename)

    # add aux and flow nodes
    flaux_xpath = '//ns:model/ns:variables/ns:aux|//ns:model/ns:variables/ns:flow'
    for element in root.xpath(flaux_xpath, namespaces={'ns':NS}):
        identifier = smile_parser.parse(element.attrib['name'], context='defn')
        pyeqn = smile_parser.parse(element.xpath('ns:eqn', namespaces={'ns':NS})[0].text)
        builder.add_flaux(filename, identifier, pyeqn)

    
    # add nodes for the derivatives of stocks
    stock_xpath = '//ns:model/ns:variables/ns:stock'
    for element in root.xpath(stock_xpath,namespaces={'ns':NS}):
        identifier = smile_parser.parse(element.attrib['name'], context='defn')

        inflows = [smile_parser.parse(e.text) for e in element.xpath('ns:inflow', namespaces={'ns':NS})]
        outflows = [smile_parser.parse(e.text) for e in element.xpath('ns:outflow', namespaces={'ns':NS})]
        pyeqn = ' + '.join(inflows) if inflows else ''
        pyeqn += ' - '+' - '.join(outflows) if outflows else ''

        initial_value = smile_parser.parse(element.xpath('ns:eqn', namespaces={'ns':NS})[0].text)

        builder.add_stock(filename, identifier, pyeqn, initial_value)


    #Get timeseries information from the XMILE file
    tstart = smile_parser.parse(root.xpath('//ns:sim_specs/ns:start',namespaces={'ns':NS})[0].text)
    builder.add_flaux(filename, 'initial_time', tstart)

    tstop = smile_parser.parse(root.xpath('//ns:sim_specs/ns:stop',namespaces={'ns':NS})[0].text)
    builder.add_flaux(filename, 'final_time', tstop)

    dt = smile_parser.parse(root.xpath('//ns:sim_specs/ns:dt',namespaces={'ns':NS})[0].text)
    builder.add_flaux(filename, 'time_step', dt)

    return filename


