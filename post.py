#coding: UTF-8

" py 2222222222222222222222222222222222222 "


import os

import numpy as np
from scipy.optimize import leastsq

import caffe

def load_net(model_dir, window_size=65, gpu_id=True):
    model_file = list(filter(lambda i: '.caffemodel' in i, os.listdir(model_dir)))[0]
    if gpu_id is None:
        caffe.set_mode_gpu()
        caffe.setdevice(gpu_id)
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(os.path.join(model_dir, 'deploy.prototxt'), os.path.join(model_dir, model_file),
                    caffe.TEST)
    return net

def gen(data):
    for x in data:
        yield x

def __predict(net, npyfile):
    X = np.load(npyfile)
    print(npyfile, X.shape)
    window_size = X.shape[1]
    X = X.reshape((X.shape[0], window_size, window_size, 1))
    return net.predict(list(gen(X)))


def predict(prefix, net, npy_files):
    i = 1
    for f in npy_files:
        print("%d / %d" % (i, len(npy_files)))
        np.save(f.replace(".npy", "_result_"+prefix+".npy"), __predict(net, f))

batch_size = 64
        
def batch_predict(postfix, net, npy_files):
    files_total = len(npy_files)
    for (file_no, f) in enumerate(npy_files):
        print("file: %s, %d / %d" % (f, file_no, files_total))
        X = np.load(f)
        window = X.shape[1]
        Y = np.zeros((X.shape[0], 2))
        if X.shape[0] % batch_size == 0:
            batch_num = int(X.shape[0]/batch_size)
        else:
            batch_num = int(X.shape[0]/batch_size)+1
        for batch_no in range(batch_num):
            print("batch: %d / %d" % (batch_no, batch_num))
            batch = X[batch_no*batch_size:(batch_no+1)*batch_size, :, :]
            batch = batch.reshape((batch_size, 1, window, window))
            net.blobs['data'].data[...] = batch
            Y[batch_no*batch_size:(batch_no+1)*batch_size] = net.forward()
        name = f.replace(".npy", "_result_"+postfix+".npy")
        print("save in %s ..." % (name))
        np.save(name, Y)
        
    
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

