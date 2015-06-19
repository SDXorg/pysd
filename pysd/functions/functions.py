import numpy as np
import scipy.stats as stats


class Functions():

    def __init__(self, component_class):
        self.components = component_class


    def if_then_else(self, condition, val_if_true, val_if_false):
        """Replicates vensim's IF THEN ELSE function. """
        if condition:
            return val_if_true
        else:
            return val_if_false
        
        
    def step(self, value, tstep):
        t = self.components.t
        return value if t >=tstep else 0


    def pulse(self, start, duration):
        t = self.components.t
        return 1 if t>=start and t<start+duration else 0

    # Warning: I'm not totally sure if this is correct
    def pulse_train(self, start, duration, repeattime, end):
        t = self.components.t
        return 1 if t>=start and (t-start)%repeattime < duration else 0

    def ramp(self, slope, start, finish):
        t = self.components.t
        if t<start:
            return 0
        elif t>finish:
            return slope * (start-finish)
        else:
            return slope * (t-start)
        
    def bounded_normal(self, minimum, maximum, mean, std, seed):
        np.random.seed(seed)
        return stats.truncnorm.rvs(minimum, maximum, loc=mean, scale=std)
    
    def lookup(self, x, xs, ys):
        return np.interp(x, xs, ys)
