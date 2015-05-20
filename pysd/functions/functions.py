
import numpy as np

def if_then_else(condition, val_if_true, val_if_false):
    if condition:
        return val_if_true
    else:
        return val_if_false
    
    
def step(value, tstep):
    t = self.components.state['t']
    return value if t >=tstep else 0


def pulse(start, duration):
    t = self.components.state['t']
    return 1 if t>=start and t<start+duration else 0

# I'm not totally sure if this is correct
def pulse_train(start, duration, repeattime, end):
    t = self.components.state['t']
    return 1 if t>=start and (t-start)%repeattime < duration else 0

def ramp(slope, start, finish):
    t = self.components.state['t']
    if t<start:
        return 0
    elif t>finish:
        return slope * (start-finish)
    else:
        return slope * (t-start)
    
def bounded_normal(minimum, maximum, mean, std, seed):
    #This is mostly for vensim
    #we may support 'seed' later, but not now. Vensim expects it, though
    while True: #keep trying until the return condition is met
        value = np.random.normal(loc=mean, scale=std)
        if value > minimum and value < maximum:
            return value


class lookup():
    """Different from the standard pysd pattern: requires a value"""
    #at some point in the future, we should add bounds checking.
    # also, should intelligently handle the error where a value is not submitted,
    # as this is a likely mistake in the pysd pattern
    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys
        
    def sample(self, x):
        return np.interp(x, self.xs, self.ys)

    def __call__(self, x): #this lets us use a simple calling expression on the class. May be a bad idea?
        """ different from the standard pysd pattern, requires a value """
        return self.sample(x)