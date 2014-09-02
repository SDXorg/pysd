
# coding: utf-8

# In[5]:

import numpy as np

def if_then_else(condition, val_if_true, val_if_false):
    if condition:
        return val_if_true
    else:
        return val_if_false
    
    
def step(value, tstep, t):
    return value if t >=tstep else 0


def pulse(start, duration, t):
    return 1 if t>=start and t<start+duration else 0

# I'm not totally sure if this is correct
def pulse_train(start, duration, repeattime, end, t):
    return 1 if t>=start and (t-start)%repeattime < duration else 0

def ramp(slope, start, finish, t):
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


# In[ ]:



