'''
created: August 15, 2014
last update: June 6 2015
version 0.2.5
James Houghton <james.p.houghton@gmail.com>
'''

#pysd specific imports
import translators as _translators

#third party imports
from scipy.integrate import odeint as _odeint
import pandas as _pd
import numpy as np
import imp

######################################################
# Todo:
# - passing optional arguments in the run command through to the integrator,
#       to give a finer level of control to those who know what to do with them. (such as `tcrit`)
# - add a logical way to run two or more models together, using the same integrator.
# - import translators within read_XMILE and read_Vensim, so we don't load both if we dont need them
######################################################


######################################################
# Issues:
#
# If two model components (A, and B) depend on the same third model component (C)
# then C will get computed twice. If C itself is dependant on many upstream components (D, E, F...)
# then these will also be computed multiple times.
#
# As the model class depends on its internal state for calculating
# the values of each element, instead of passing that state
# through the function execution network, we can't use caching
# to prevent multiple execution, as we don't know when to update the cache
# (maybe in the calling function?)
#
# Also, this multi-calculation bears the risk that if the state is
# changed during an execution step, the resulting calculation will be
# corrupted, and we won't have any indication of this corruption.
######################################################

def read_xmile(xmile_file):
    """ Construct a model object from `.xmile` file. """
    py_model_file = _translators.translate_xmile(xmile_file)
    model = load(py_model_file)
    model.__str__ = 'Import of ' + xmile_file
    return model
read_xmile.__doc__ += _translators.translate_xmile.__doc__

def read_vensim(mdl_file):
    """ Construct a model from Vensim `.mdl` file. """
    py_model_file = _translators.translate_vensim(mdl_file)
    model = load(py_model_file)
    model.__str__ = 'Import of ' + mdl_file
    return model
read_vensim.__doc__ += _translators.translate_vensim.__doc__

def load(py_model_file):
    """ Load a python-converted model file. """
    module = imp.load_source('modulename', py_model_file)
    model = module.Components
    model.__str__ = 'Import of ' + py_model_file
    return model

def run(model):
    components=getcomponents(model)
    set_initial_condition(model)
    
    final_time=getattr(model,'final_time')
    initial_time=getattr(model,'initial_time')
    time_step=getattr(model,'time_step')
    num_of_steps=int(math.ceil(((final_time())-initial_time())/time_step()))+1
    stocks=[]
    ddt=[]
    variables=[]
    numbersonly=[]
    dictofsubs={}
    dictofvars={}
    runtime=range(len(components))
    auxiliaries=[]
    
    for k in runtime:
        dictofvars[components[k]]=k
        
    for k in runtime:
        if components[k].startswith('subscript_'):
            stocks.append(k)
        elif (components[k].startswith('d') and components[k].endswith('_dt')):
            stocks.append(k)
            ddt.append(k)
        elif components[k].startswith('variable_numbersonly_'):
            stocks.append(k)
            numbersonly.append(k)
        elif components[k].startswith('variable_'):
            stocks.append(k)
            variables.append(k)
        elif components[k].endswith('_init'):
            stocks.append(k)
            for l in range(len(components)):
                if components[l]==components[k][:-5]:
                    stocks.append(l)
        elif len(checkforlookup.getargspec(getattr(model,components[k]))[0])>1:
            stocks.append(k)
    
    for k in runtime:
        if k not in stocks:
            auxiliaries.append(k)
            
    for k in runtime:            
        #if component is a subscript definition
        if components[k].startswith('subscript_'):
            value=getattr(model,components[k])
            array=(value().replace(" ","").split(','))
            getelempos.arrayofsubs.update(dict(zip(array,range(len(array)))))
            getelempos.arrayofsubs[components[k][10:]]=slice(len(array))
            
    #Initializing the stocks to their initial values
    for num in runtime:
        if components[num].endswith('_init'):
            stocknum=model.Dictionary.get(components[num][0:-5])
            value=getattr(model,components[num])
            model.arrayofresults[stocknum][0]=value()
            model.arrayofresults[stocknum][-1]=value()
    
    
    #Setting the values in each variable by calculating at each time step
    #First loop is for the total number of steps (end time - beginning time)/ step size           
    for i in range(1,num_of_steps):
    
        model.CurrentTime=i
    
        for j in variables:
            setattr(model,components[j],"")    
    
        for k in ddt:
            StockValue=components[k][1:-3]
            StockPos=dictofvars[StockValue]    
            model.arrayofresults[StockPos][i]=model.arrayofresults[StockPos][i-1]+getattr(model,components[k])()*time_step()
    
        for k in auxiliaries:
            model.arrayofresults[k][i]=getattr(model,components[k])()

def getcomponents(model):
    components=list(dir(model))
    components=[x for x in components if not x.startswith('__') and not x.startswith('_') 
                and not x in ['getsubs','getnumofelements','getelements','getsubname','getelempos','getnp','time','state','functions','t','saveper','CurrentTime','final_time','initial_time','time_step','doc','reset_state','returnvalues','state_vector','d_dt']]
    return components
    
def set_initial_condition(model):
    getnumofelements.getnumofelements_variable={}
    getelempos.arrayofsubs={}
    model.getnumofelements=getnumofelements
    model.getelempos=getelempos
    #Creating a repertoire to know components location
    model.Dictionary=dict(zip(components,range(len(components))))
    #Variable defining current run time
    model.CurrentTime=0
    model.arrayofresults=np.ndarray((len(components),num_of_steps),object)

    
def getnumofelements(self, subname):
    numofelements=[]
    try:
        numofelements=getnumofelements.getnumofelements_variable[subname]
    except:
        tempsubname=subname.split(",")
        for i in tempsubname:
            elements=getattr(self,"subscript_"+i)
            numofelements.append(len(elements().split(",")))
            getnumofelements.getnumofelements_variable[subname]=numofelements
    return numofelements
getnumofelements.getnumofelements_variable={}

def getelempos(self, element=""):
    return getelempos.arrayofsubs[element]
getelempos.arrayofsubs={}



