from pysd.translators.vensim.vensim_file import VensimFile
from pysd.translators.xmile.xmile_file import XmileFile

vf = VensimFile("vensim_models/ds_white_sterman.mdl")
#vf = VensimFile("vensim_models/arithmetic.mdl")
#vf = XmileFile("vensim_models/repair.xmile")
vf.parse()

am = vf.get_abstract_model()
for section in am.sections:
    for element in section.elements:
        print("*" * 10)
        print(f"name: {element.name}")
        print(f"length: {len(element.components)}")
        for component in element.components:
            print(f"type: {component.type}")
            print(f"subtype: {component.subtype}")
            print(f"subscript: {component.subscripts}")
            print(component.ast)