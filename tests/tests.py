from pysd.translators import vensim2py

import unittest

import inspect

class TestVensimImport(unittest.TestCase):
    
    def test_expression_grammar(self):
    
        # debug grammar
        #print vensim2py.expression_grammar
    
        # set up
        parser = vensim2py.TextParser(vensim2py.expression_grammar)

        # test identifiers and references
        self.assertEqual(parser.parse('Identifier OF DOOM 27'), 'self.identifier_of_doom_27()')
        
        # test keywords
        self.assertEqual(parser.parse('ABS(-41)'), 'abs(-41)')
        self.assertEqual(parser.parse('SIN(45)'), 'np.sin(45)')
        #self.assertEqual(parser.parse('PI'), 'np.pi') #because pi doesn't follow the call pattern with the parentheses, it doesnt work properly
        self.assertEqual(parser.parse('Fred >= 21'), 'self.fred()>=21')
        
        # test parenthesization (sp?)
        self.assertEqual(parser.parse('5*(3+9)'), '5*(3+9)')

        # test signed components
        self.assertEqual(parser.parse('-5*(+3+-9)'), '-5*(+3+-9)')


    def test_entry_grammar(self):

        parser = vensim2py.TextParser(vensim2py.entry_grammar)
        
        #test a constant, auxiliary variable with no comment
        text ="""
                Characteristic Time=
                    10
                    ~	Minutes
                    ~		|
              """
        parser.parse(text)
        instance = parser.component_class()
        #print [a for (a,b) in inspect.getmembers(instance)]
        self.assertEqual(instance.characteristic_time(), 10)
        
        # add another
        text="""
                Room Temperature=
                    70
                    ~
                    ~		|
             """
        parser.parse(text)
        instance = parser.component_class()

        # test a flow with a comment
        text ="""
                Heat Loss to Room=
                    (Teacup Temperature - Room Temperature) / Characteristic Time
                    ~	Degrees/Minute
                    ~	This is the rate at which heat flows from the cup into the room. We can \
                        ignore it at this point.
                    |
            """
        parser.parse(text)
        instance = parser.component_class()
        #print instance.heat_loss_to_room.__doc__

        # test a stock
        text ="""
                Teacup Temperature= INTEG (
                    -Heat Loss to Room,
                        180)
                    ~	Degrees
                    ~		|
              """
        parser.parse(text)
        instance = parser.component_class()
        self.assertEqual(instance.teacup_temperature(), 180)
        self.assertEqual(instance.dteacup_temperature_dt(), -11)
        self.assertEqual(instance.teacup_temperature_init(), 180)


    def test_file_grammar(self):
    
        # debug grammar
        #print vensim2py.file_grammar
        
        with open('tests/teacup.mdl') as file:
            text = file.read()
    
        parser = vensim2py.TextParser(vensim2py.file_grammar)
        parser.parse(text)



if __name__ == '__main__':
    unittest.main()