"""
functions.py

These are supports for functions that are included in modeling software but have no
straightforward equivalent in python.

"""

import numpy as np
import scipy.stats as stats  # Todo: can we get away from the scipy dependency?
import re
# Todo: Pull this out of a class and make it flat for inclusion in the model file


class Functions(object):
    """Provides SD-specific calculations that are not available in python standard libraries,
    or have a different functional form to those specified in the libraries, and therefore
    are in need of translation.
    
    This is implemented as a class, with a local pointer to the model components class, 
    because some of these functions need to reference the internal state of the model.
    Mostly they are just referencing the simulation time.
    """

    def __init__(self, component_class):
        self.components = component_class


    def if_then_else(self, condition, val_if_true, val_if_false):
        """Replicates vensim's IF THEN ELSE function. """
        if condition:
            return val_if_true
        else:
            return val_if_false


    def step(self, value, tstep):
        """" Impliments vensim's STEP function

        In range [-inf, tstep) returns 0
        In range [tstep, +inf] returns `value`
        """
        t = self.components.t
        return value if t >= tstep else 0


    def pulse(self, start, duration):
        """ Implements vensim's PULSE function

        In range [-inf, start) returns 0
        In range [start, start+duration) returns 1
        In range [start+duration, +inf] returns 0
        """
        t = self.components.t
        return 1 if t >= start and t < start+duration else 0

    # Warning: I'm not totally sure if this is correct
    def pulse_train(self, start, duration, repeattime, end):
        """ Implements vensim's PULSE TRAIN function

        """

        t = self.components.t
        return 1 if t >= start and (t-start)%repeattime < duration else 0

    def ramp(self, slope, start, finish):
        """ Implements vensim's RAMP function """
        t = self.components.t
        if t < start:
            return 0
        elif t > finish:
            return slope * (start-finish)
        else:
            return slope * (t-start)

    def bounded_normal(self, minimum, maximum, mean, std, seed):
        """ Implements vensim's BOUNDED NORMAL function """
        np.random.seed(seed)
        return stats.truncnorm.rvs(minimum, maximum, loc=mean, scale=std)


def lookup(x, xs, ys):
    """ Provides the working mechanism for lookup functions the builder builds """
    return np.interp(x, xs, ys)


def if_then_else(condition,val_if_true,val_if_false):
    return np.where(condition,val_if_true,val_if_false)


def pos(number):  # dont divide by 0
    return np.maximum(number, 0.000001)


def tuner(number, factor):
    if factor>1:
        if number == 0:
            return 0
        else:
            return max(number,0.000001)**factor
    else:
        return (factor*number)+(1-factor)
        
def shorthander(orig,dct,refdct,dictionary):
    if refdct == 0:
        return orig
    elif len(refdct) == 1:
        return orig

    def getnumofelements(element,dictionary):
        if element=="":
            return 0
        position=[]
        elements=element.replace('!','').replace('','').split(',')
        for element in elements:
            if element in dictionary.keys():
                if isinstance(dictionary[element],list):
                    position.append((getnumofelements(dictionary[element][-1],dictionary))[0])
                else:
                    position.append(len(dictionary[element]))
            else:
                for d in dictionary.itervalues():
                    try:
                        (d[element])
                    except: pass
                    else:
                        position.append(len(d))
        return position
    def getshape(refdct):
        return tuple(getnumofelements(','.join([names for names,keys in refdct.iteritems()]),dictionary))
    def tuplepopper(tup,pop):
        tuparray=list(tup)
        for i in pop:
            tuparray.remove(i)
        return tuple(tuparray)
    def swapfunction(dct,refdct,counter=0):
        if len(dct)<len(refdct):
            tempdct = {}
            sortcount=0
            for i in sorted(refdct.values()):
                if refdct.keys()[refdct.values().index(i)] not in dct:
                    tempdct[refdct.keys()[refdct.values().index(i)]]=sortcount
                    sortcount+=1
            finalval=len(tempdct)
            for i in sorted(dct.values()):
                tempdct[dct.keys()[dct.values().index(i)]]=finalval+i
        else:
            tempdct=dct.copy()
        if tempdct==refdct:
            return '(0,0)'
        else:
            for sub,pos in tempdct.iteritems():
                if refdct.keys()[refdct.values().index(counter)]==sub:
                    tempdct[tempdct.keys()[tempdct.values().index(counter)]]=pos
                    tempdct[sub]=counter
                    return '(%i,%i);'%(pos,counter)+swapfunction(tempdct,refdct,counter+1) 
#############################################################################    
    if len(dct)<len(refdct):
        dest=getshape(refdct)
        copyoforig=np.ones(tuplepopper(dest,np.shape(orig))+np.shape(orig))*orig
    else:
        copyoforig=orig
    process=swapfunction(dct,refdct).split(';')
    for i in process:
        j=re.sub(r'[\(\)]','',i).split(',')
        copyoforig=copyoforig.swapaxes(int(j[0]),int(j[1]))
    return copyoforig
#     add the variable to the dct so that it's similar to refdct, then do the swap axes
#
# def ramp(self, slope, start, finish):
#     """ Implements vensim's RAMP function """
#     t = self.components._t
#     try:
#         len(start)
#     except:
#         if t<start:
#             return 0
#         elif t>finish:
#             return slope * (start-finish)
#         else:
#             return slope * (t-start)
#     else:
#         returnarray=np.ndarray(len(start))
#         for i in range(len(start)):
#             if np.less(t,start)[i]:
#                 returnarray[i]=0
#             elif np.greater(t,finish)[i]:
#                 try:
#                     len(slope)
#                 except:
#                     returnarray[i]=slope*(start[i]-finish[i])
#                 else:
#                     returnarray[i]=slope[i]*(start[i]-finish[i])
#             else:
#                 try:
#                     len(slope)
#                 except:
#                     returnarray[i]=slope*(t-start[i])
#                 else:
#                     returnarray[i]=slope[i]*(t-start[i])
#         return returnarray