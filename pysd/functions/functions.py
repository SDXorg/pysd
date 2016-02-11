"""
functions.py

These are supports for functions that are included in modeling software but have no
straightforward equivalent in python.

"""

import numpy as np
import scipy.stats as stats  # Todo: can we get away from the scipy dependency?
import re


def ramp(slope, start, finish):
    """ Implements vensim's RAMP function """
    t = time()
    if t < start:
        return 0
    elif t > finish:
        return slope * (finish-start)
    else:
        return slope * (t-start)


def bounded_normal(minimum, maximum, mean, std, seed):
    """ Implements vensim's BOUNDED NORMAL function """
    np.random.seed(seed)
    return stats.truncnorm.rvs(minimum, maximum, loc=mean, scale=std)


def step(value, tstep):
    """" Impliments vensim's STEP function

    In range [-inf, tstep) returns 0
    In range [tstep, +inf] returns `value`
    """
    return value if time() >= tstep else 0


def pulse(start, duration):
    """ Implements vensim's PULSE function

    In range [-inf, start) returns 0
    In range [start, start+duration) returns 1
    In range [start+duration, +inf] returns 0
    """
    t = time()
    return 1 if t >= start and t < start+duration else 0


def pulse_train(start, duration, repeattime, end):
    """ Implements vensim's PULSE TRAIN function

    """
    t = time()
    return 1 if t >= start and (t-start)%repeattime < duration else 0


def lookup(x, xs, ys):
    """ Provides the working mechanism for lookup functions the builder builds """
    if not isinstance(xs, np.ndarray):
        return np.interp(x, xs, ys)
    resultarray=np.ndarray(np.shape(x))
    for i,j in np.ndenumerate(x):
        resultarray[i]=np.interp(j,np.array(xs)[i],np.array(ys)[i])
    return resultarray


def if_then_else(condition,val_if_true,val_if_false):
    return np.where(condition,val_if_true,val_if_false)


def pos(number):  # dont divide by 0
    return np.maximum(number, 0.000001)


def active_initial(expr, initval):
    if _t == initial_time():
        return initval
    else:
        return expr


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

def sums(expression,count=0):
    operations = ['+','-','*','/']
    merge=[]
    if count == len(operations):
        return 'np.sum(%s)'%expression
    for sides in expression.split(operations[count]):
        merge.append(sum(sides,count+1))
    return operations[count].join(merge)



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