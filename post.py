#coding: UTF-8

" py 2222222222222222222222222222222222222 "


import os

import numpy as np
from scipy.optimize import leastsq

import caffe

def load_net(model_dir, window_size=65, use_GPU=True):
    model_file = list(filter(lambda i: '.caffemodel' in i, os.listdir(model_dir)))[0]
    use_GPU and caffe.set_mode_gpu() or caffe.set_mode_cpu()
    net = caffe.Classifier(os.path.join(model_dir, 'deploy.prototxt'),
                           os.path.join(model_dir, model_file),
                           image_dims=(window_size, window_size),
                           input_scale=1, raw_scale=255)
    return net

def gen(data):
    for x in data:
        yield x

def __predict(net, npyfile):
    X = np.load(npyfile)
    print(npyfile, X.shape[0])
    window_size = X.shape[1]
    X = X.reshape((X.shape[0], window_size, window_size, 1))
    return net.predict(list(gen(X)))


def predict(net, npy_files):
    i = 1
    for f in npy_files:
        print("%d / %d" % (i, len(npy_files)))
        np.save(f.replace(".npy", "_result.npy"), __predict(net, f))
    
def leastsq_fit(X, Y):
    def residuals(p, y, x):
        p3, p2, p1, p0 = p
        err = y - (p3*(x**3)+p2*(x**2)+p1*x+p0)
        return err
    p0 = [0.001, 0.001, 1, 0.0001]
    plsq = leastsq(residuals, p0, args=(Y, X))

    def eva(x):
        p = plsq[0]
        return p[3]*(x**3)+p[2]*(x**2)+p[1]*x+p[0]

    return eva

