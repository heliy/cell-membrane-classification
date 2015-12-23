#coding: UTF-8

""" PYTHON 222222222222222 """

import os
import sys

import numpy as np
import cv2

def split_classes(dir, has_label=False):
    files = os.listdir(dir)
    x_files = list(filter(lambda x: '_x.npy' in x, files))
    
    if has_label:
        os.system("mkdir '"+dir+"0'")
        os.system("mkdir '"+dir+"1'")

        ss0, ss1 = 0, 0
        for f in filter(lambda x: 'train0' in x, x_files)
            no = f.split("_")[3]
            for (i, x) in enumerate(np.load(dir+f)):
                name = "%s%s/%s_%d.png" % (dir, '0', no, i)
                cv2.imwrite(name, x)
            ss0 += i
            print(f, ss0)
        for f in filter(lambda x: 'train255' in x, x_files)
            no = f.split("_")[3]
            for (i, x) in enumerate(np.load(dir+f)):
                name = "%s%s/%s_%d.png" % (dir, '1', no, i)
                cv2.imwrite(name, x)
            ss1 += i
            print(f, ss1)
        print(ss0, ss1)
    if not has_label:
        for (no, f) in enumerate(x_files):
            X = np.load(dir+f)
            for (i, x) in zip(range(X.shape[0]), X):
                cv2.imwrite(dir+str(no)+"_"+str(i)+".png", x)
            
