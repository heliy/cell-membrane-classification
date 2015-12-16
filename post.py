#coding: UTF-8

import numpy as np

from data import getSampleDot, shift

def cato(x):
    if x[0] < x[1]:
        return 0
    else:
        return 1

def test(model, num=500, times=30):
    c = 0
    for i in range(times):
        d = 0
        dots = getSampleDot(label=0, num=num)
        dots = np.array(list(dots))        
        y = model.predict(shift(dots))
        for x in y:
            if cato(x) == 0:
                d += 1
        dots = getSampleDot(label=255, num=num)
        dots = np.array(list(dots))        
        y = model.predict(shift(dots))
        for x in y:
            if cato(x) == 1:
                d += 1
        c += d
        print(i, ": ", d/(num*2))
    return c/(num*times*2)
