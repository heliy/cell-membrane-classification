#coding:UTF-8

import sys

import pre
from dcc import build_cnn, train, predict
from models import *

if __name__ == '__main__':
    arg = sys.argv[1]
    if arg == "pre 95 train":
        pre.batch(batch_size=15000)
    elif arg == "pre 95 test":
        pre.batch("test", teVolume, None, 95, 7680, 1.1, 9)
    elif arg == 'balance 65 255':
        pre.dir_prefix = "data/pre_balanced"
        pre.append_sample(prefix="test255", LABEL=255, window_size=65, ratio=0.23, begin=0)
    elif arg == 'balance 65 0':
        pre.dir_prefix = "data/pre_balanced"
        pre.append_sample(prefix="test0", LABEL=0, window_size=65, ratio=0.8, begin=0)
    elif arg == 'balance 95 255':
        pre.dir_prefix = "data/pre_balanced"
        pre.append_sample(prefix="test255", ratio=0.23, begin=0)
    elif arg == 'balance 65 0':
        pre.dir_prefix = "data/pre_balanced"
        pre.append_sample(prefix="test0", LABEL=0, ratio=0.8, begin=0)
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

