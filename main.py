#coding:UTF-8

import sys

from pre import batch, teVolume
from dcc import build_cnn, train, predict
from models import *

if __name__ == '__main__':
    arg = sys.argv[1]
    if arg == "pre 95 train":
        batch(batch_size=15000)
    elif arg == "pre 95 test":
        batch("test", teVolume, None, 95, 7680, 1.1)
    elif arg == "dcc n1":
        model = build_cnn(n1)
        model = train(model, n1, 5000)
        predict(model, n1)
    elif arg == "dcc n2":
        model = build_cnn(n2)
        model = train(model, n2, 5000)
        predict(model, n2)        
    elif arg == "dcc n3":
        model = build_cnn(n3)
        model = train(model, n3, 5000)
        predict(model, n3)
    elif arg == "dcc n4":
        model = build_cnn(n4)
        model = train(model, n4, 5000)
        predict(model, n4)
    elif arg == "merge":
        pass

