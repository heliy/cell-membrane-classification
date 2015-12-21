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
        for (no, f) in enumerate(x_files):
            X = np.load(dir+f)
            Y = np.load(dir+f.replace("_x.", "_y."))[:, 0]
            print(X.shape, Y.shape)
            for (i, x, y) in zip(range(X.shape[0]), X, Y):
                name = dir+str(int(y))+"/"+str(no)+"_"+str(i)+".png"
                print(name)
                cv2.imwrite(name, x)
    if not has_label:
        for (no, f) in enumerate(x_files):
            X = np.load(dir+f)
            for (i, x) in zip(range(X.shape[0]), X):
                cv2.imwrite(dir+str(no)+"_"+str(i)+".png", x)
            
