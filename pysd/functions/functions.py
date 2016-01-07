import numpy as np
import scipy.stats as stats


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


    def if_then_else(self,condition,val_if_true,val_if_false):
        return np.where(condition,val_if_true,val_if_false)
    
    
    def pos(self,number):
        return np.maximum(number, 0.000001)
    
    
    def tuner(self,number,factor):
        if factor>1:
            if number==0:
                return 0
            else:
                return max(number,0.000001)**factor
        else:
            return (factor*number)+(1-factor)


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

    def lookup(self, x, xs, ys):
        """ Provides the working mechanism for lookup functions the builder builds """
        return np.interp(x, xs, ys)

