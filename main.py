#coding:UTF-8

import sys
import os
import numpy as np

import pre
# from dcc import build_cnn, train, predict
from models import *
import post

# label == 0 -> 1 -> is mem
# label == 255 -> 0 -> is not mem

if __name__ == '__main__':
    arg = sys.argv[1]
    # if arg == "pre 95 train":
    #     pre.batch(batch_size=15000)
    if arg == "pre 95 test":
        pre.batch("test", pre.teVolume, None, 95, 7680, 1.1, 710)
    if arg == 'post 65 train':
        pre.dir_prefix = "/media/mmr6-raid5/hly/cell-mem-data/test_65/"        
        pre.batch("L1train", pre.trVolume[0].reshape((1, 512, 512)), None, 65, 4096, 1.1)
    if arg == 'post 65 test':
        pre.dir_prefix = "/media/mmr6-raid5/hly/cell-mem-data/test_65/"        
        pre.batch("L1test", pre.teVolume[0].reshape((1, 512, 512)), None, 65, 4096, 1.1)
    if arg == 'post 95 train':
        pre.dir_prefix = "/media/mmr6-raid5/hly/cell-mem-data/test_95/"        
        pre.batch("L1train", pre.trVolume[0].reshape((1, 512, 512)), None, 95, 4096, 1.1)
    if arg == 'post 95 test':
        pre.dir_prefix = "/media/mmr6-raid5/hly/cell-mem-data/test_95/"        
        pre.batch("L1test", pre.teVolume[0].reshape((1, 512, 512)), None, 95, 4096, 1.1)
    # elif arg == 'balance 65 255':
    #     # white, not mem
    #     pre.dir_prefix = "/media/mmr6-raid5/hly/cell-mem-data/"
    #     pre.append_sample(prefix="test255", LABEL=255, window_size=65, ratio=0.23, begin=0)
    # elif arg == 'balance 65 0':
    #     # black, is mem
    #     pre.dir_prefix = "data/pre_balanced"
    #     pre.append_sample(prefix="test0", LABEL=0, window_size=65, ratio=0.8, begin=0)
    elif arg == 'balance 95 255':
        pre.dir_prefix = "/media/mmr6-raid5/hly/cell-mem-data/train_95/"
        pre.append_sample(prefix="train255", ratio=0.23, batch_size=3000, begin=0, hajimari=136)
    elif arg == 'balance 95 0':
        pre.dir_prefix = "/media/mmr6-raid5/hly/cell-mem-data/train_95/"
        pre.append_sample(prefix="train0", LABEL=0, ratio=0.8, batch_size=3000, begin=0, hajimari=132)
    # elif arg == "dcc n1":
    #     model = build_cnn(n1)
    #     model = train(model, n1, 5000)
    #     predict(model, n1)
    # elif arg == "dcc n2":
    #     model = build_cnn(n2)
    #     model = train(model, n2, 5000)
    #     predict(model, n2)        
    # elif arg == "dcc n3":
    #     model = build_cnn(n3)
    #     model = train(model, n3, 5000)
    #     predict(model, n3)
    # elif arg == "dcc n4":
    #     model = build_cnn(n4)
    #     model = train(model, n4, 5000)
    #     predict(model, n4)
    elif arg == 'fit para':
        net = post.load_net("models/n1", 0)
        d = "/media/mmr6-raid5/hly/cell-mem-data/train_65/"
        xfiles = filter(lambda x: '_x.' in x,
                       os.listdir(d))
        xfiles = [d+x for x in xfiles]
        yfiles = [x.replace('x', 'y') for x in xfiles]
        prob = post.prob_count(net, xfiles, yfiles)
        np.save("fitparaprob.npy", prob)
    elif arg == 'predict n1':
        net = post.load_net("models/n1")
        d = "/media/mmr6-raid5/hly/cell-mem-data/test_65/"
        files = filter(lambda x: '_x.' in x,
                       os.listdir(d))
        post.predict(net, [d+x for x in files])        
    elif arg == 'predict n2':
        net = post.load_net("models/n2")
        d = "/media/mmr6-raid5/hly/cell-mem-data/test_65/"
        files = filter(lambda x: '_x.' in x,
                       os.listdir(d))
        post.predict(net, [d+x for x in files])        
    elif arg == 'predict n3':
        net = post.load_net("models/n3")
        d = "/media/mmr6-raid5/hly/cell-mem-data/test_95/"
        files = filter(lambda x: '_x.' in x,
                       os.listdir(d))
        post.predict(net, [d+x for x in files])        
    elif arg == 'predict n4':
        net = post.load_net("models/n4")
        d = "/media/mmr6-raid5/hly/cell-mem-data/test_95/"
        files = filter(lambda x: '_x.' in x,
                       os.listdir(d))
        post.predict(net, [d+x for x in files])        
    elif arg == "merge":
        pass

