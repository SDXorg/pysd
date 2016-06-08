"""
functions.py

These are supports for functions that are included in modeling software but have no
straightforward equivalent in python.

"""

import numpy as np
import re
from functools import wraps
import scipy.stats as stats


def cache(horizon):
    """
    Put a wrapper around a model function

    Decorators with parameters are tricky, you have to
    essentially create a decorator that returns a decorator,
    which itself then returns the function wrapper.

    Parameters
    ----------
    horizon: string
        - 'step' means cache just until the next timestep
        - 'run' means cache until the next initialization of the model

    Returns
    -------
    new_func: decorated function
        function wrapping the original function, handling caching

    """
    def cache_step(func):
        """ Decorator for caching at a step level"""
        @wraps(func)
        def cached(*args):
            """Step wise cache function"""
            try:  # fails if cache is out of date or not instantiated
                assert cached.t == func.func_globals['_t']
            except (AssertionError, AttributeError):
                cached.cache = func(*args)
                cached.t = func.func_globals['_t']
            return cached.cache
        return cached

    def cache_run(func):
        """ Decorator for caching at  the run level"""
        @wraps(func)
        def cached(*args):
            """Run wise cache function"""
            try:  # fails if cache is not instantiated
                return cached.cache
            except AttributeError:
                cached.cache = func(*args)
                return cached.cache
        return cached

    if horizon == 'step':
        return cache_step

    elif horizon == 'run':
        return cache_run

    else:
        raise(AttributeError('Bad horizon for cache decorator'))


def ramp(slope, start, finish):
    """
    Implements vensim's RAMP function

    Parameters
    ----------
    slope: float
        The slope of the ramp starting at zero at time start
    start: float
        Time at which the ramp begins
    finish: float
        Time at which the ramo ends

    Returns
    -------
    response: float
        If prior to ramp start, returns zero
        If after ramp ends, returns top of ramp
    Examples
    --------


    """

    t = time()
    if t < start:
        return 0
    elif t > finish:
        return slope * (finish-start)
    else:
        return slope * (t-start)


def step(value, tstep):
    """"
    Implements vensim's STEP function

    Parameters
    ----------
    value: float
        The height of the step
    tstep: float
        The time at and after which `result` equals `value`

    Returns
    -------
    - In range [-inf, tstep) returns 0
    - In range [tstep, +inf] returns `value`
    """
    return value if time() >= tstep else 0


def pulse(start, duration):
    """ Implements vensim's PULSE function

    In range [-inf, start) returns 0
    In range [start, start + duration) returns 1
    In range [start + duration, +inf] returns 0
    """
    t = time()
    return 1 if start <= t < start + duration else 0


def pulse_train(start, duration, repeat_time, end):
    """ Implements vensim's PULSE TRAIN function

    In range [-inf, start) returns 0
    In range [start + n * repeat_time, start + n * repeat_time + duration) return 1
    In range [start + n * repeat_time + duration, start + (n+1) * repeat_time) return 0
    """
    t = time()
    if start <= t < end:
        return 1 if (t-start) % repeat_time < duration else 0
    else:
        return 0


def bounded_normal(minimum, maximum, mean, std, seed):
    """ Implements vensim's BOUNDED NORMAL function """
    # np.random.seed(seed)  # we could bring this back later, but for now, ignore
    return stats.truncnorm.rvs(minimum, maximum, loc=mean, scale=std)


def lookup(x, xs, ys):
    """ Provides the working mechanism for lookup functions the builder builds """
    if not isinstance(xs, np.ndarray):
        return np.interp(x, xs, ys)
    resultarray = np.ndarray(np.shape(x))
    for i, j in np.ndenumerate(x):
        resultarray[i] = np.interp(j, np.array(xs)[i], np.array(ys)[i])
    return resultarray


def if_then_else(condition, val_if_true, val_if_false):
    return np.where(condition, val_if_true, val_if_false)


def active_initial(expr, initval):
    """
    Implements vensim's ACTIVE INITIAL function
    Parameters
    ----------
    expr
    initval

    Returns
    -------

    """
    if time() == initial_time():
        return initval
    else:
        return expr



#        def pos(number):  # dont divide by 0
#            return np.maximum(number, 0.000001)

#
# def tuner(number, factor):
#     if factor>1:
#         if number == 0:
#             return 0
#         else:
#             return max(number,0.000001)**factor
#     else:
#         return (factor*number)+(1-factor)
#
#
# def shorthander(orig,dct,refdct,dictionary):
#     if refdct == 0:
#         return orig
#     elif len(refdct) == 1:
#         return orig
#     def getnumofelements(element,dictionary):
#         if element=="":
#             return 0
#         position=[]
#         elements=element.replace('!','').replace('','').split(',')
#         for element in elements:
#             if element in dictionary.keys():
#                 if isinstance(dictionary[element],list):
#                     position.append((getnumofelements(dictionary[element][-1],dictionary))[0])
#                 else:
#                     position.append(len(dictionary[element]))
#             else:
#                 for d in dictionary.itervalues():
#                     try:
#                         (d[element])
#                     except: pass
#                     else:
#                         position.append(len(d))
#         return position
#     def getshape(refdct):
#         return tuple(getnumofelements(','.join([names for names,keys in refdct.iteritems()]),dictionary))
#     def tuplepopper(tup,pop):
#         tuparray=list(tup)
#         for i in pop:
#             tuparray.remove(i)
#         return tuple(tuparray)
#     def swapfunction(dct,refdct,counter=0):
#         if len(dct)<len(refdct):
#             tempdct = {}
#             sortcount=0
#             for i in sorted(refdct.values()):
#                 if refdct.keys()[refdct.values().index(i)] not in dct:
#                     tempdct[refdct.keys()[refdct.values().index(i)]]=sortcount
#                     sortcount+=1
#             finalval=len(tempdct)
#             for i in sorted(dct.values()):
#                 tempdct[dct.keys()[dct.values().index(i)]]=finalval+i
#         else:
#             tempdct=dct.copy()
#         if tempdct==refdct:
#             return '(0,0)'
#         else:
#             for sub,pos in tempdct.iteritems():
#                 if refdct.keys()[refdct.values().index(counter)]==sub:
#                     tempdct[tempdct.keys()[tempdct.values().index(counter)]]=pos
#                     tempdct[sub]=counter
#                     return '(%i,%i);'%(pos,counter)+swapfunction(tempdct,refdct,counter+1)
# #############################################################################
#     if len(dct)<len(refdct):
#         dest=getshape(refdct)
#         copyoforig=np.ones(tuplepopper(dest,np.shape(orig))+np.shape(orig))*orig
#     else:
#         copyoforig=orig
#     process=swapfunction(dct,refdct).split(';')
#     for i in process:
#         j=re.sub(r'[\(\)]','',i).split(',')
#         copyoforig=copyoforig.swapaxes(int(j[0]),int(j[1]))
#     return copyoforig
#
# def sums(expression,count=0):
#     operations = ['+','-','*','/']
#     merge=[]
#     if count == len(operations):
#         return 'np.sum(%s)'%expression
#     for sides in expression.split(operations[count]):
#         merge.append(sum(sides,count+1))
#     return operations[count].join(merge)
#


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