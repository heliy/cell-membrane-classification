#coding: UTF-8

import numpy as np

from data import getSampleDot, shift

def cato(x):
    if x[0] < x[1]:
        return 0
    else:
        return 1

def test(model, num=300, times=30):
    c = 0
    for i in range(times):
        dots = getSampleDot(label=0, num=num)
        dots = np.array(list(dots))        
        y = model.predict(shift(dots))
        for x in y:
            if cato(y) == 0:
                c += 1
        dots = getSampleDot(label=255, num=num)
        dots = np.array(list(dots))        
        y = model.predict(shift(dots))
        for x in y:
            if cato(y) == 1:
                c += 1
    return c/(num*times)
