#coding: UTF-8

" py 2222222222222222222222222222222222222 "


import os

import numpy as np
from scipy.optimize import leastsq

import caffe

def load_net(model_dir, gpu_id=None):
    model_file = list(filter(lambda i: '.caffemodel' in i, os.listdir(model_dir)))[0]
    if gpu_id is not None:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
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

batch_size = 256

def batch_predict(net, X):
    window = X.shape[1]
    Y = np.zeros((X.shape[0], 2))
    batch = np.zeros((batch_size, 1, window, window))
    if X.shape[0] % batch_size == 0:
        batch_num = int(X.shape[0]/batch_size)
    else:
        batch_num = int(X.shape[0]/batch_size)+1
    for batch_no in range(batch_num):
        x = X[batch_no*batch_size:(batch_no+1)*batch_size, :, :]
        batch[:x.shape[0], :, :] = x.reshape((x.shape[0], 1, window, window))
        net.blobs['data'].data[...] = batch    
        Y[batch_no*batch_size:batch_no*batch_size+x.shape[0]] = net.forward()['prob'][:x.shape[0]]
    return Y

def save_predict(postfix, net, npy_files):
    files_total = len(npy_files)
    for (file_no, f) in enumerate(npy_files):
        print("file: %s, %d / %d" % (f, file_no, files_total))
        X = np.load(f)
        Y = __batch_predict(X)
        name = f.replace(".npy", "_result_"+postfix+".npy")
        print("save in %s ..." % (name))
        np.save(name, Y)
        
    
def leastsq_fit(X, Y, p0=[0.01, 0.01, 0.01, 0.01]):
    def residuals(p, y, x):
        p3, p2, p1, p0 = p
        err = y - (p3*(x**3)+p2*(x**2)+p1*x+p0)
        return err
    plsq = leastsq(residuals, p0, args=(Y, X))

    def eva(x):
        p = plsq[0]
        return p[3]*(x**3)+p[2]*(x**2)+p[1]*x+p[0]

    return eva

def prob_eval(xfiles, yfiles, net_dir='models/n1', y_index=1, gpu_id=0):
    '''
    y_index = 1 for n1/n2, = 0 for n3/n4
       as when we train the net, the first col in prob is different Orz.
    '''
    prob_dict = {}
    net = load_net(net_dir, gpu_id)

    @np.vectorize
    def input_to_dict(predicted_y, real_y):
        predict = int(predicted_y*10000)/10000.
        if predict not in prob_dict:
            prob_dict[predict] = [0, 0]
        prob_dict[predict][real_y] += 1
    
    for (x, y) in zip(xfiles, yfiles):
        X = np.load(x)
        Y = np.load(y)
        y = batch_predict(net, X)
        input_to_dict(y[:, y_index], Y[:, 0])

    X = prob_dict.keys()
    Y = np.array([float(prob_dict[x][1])/sum(prob_dict[x]) for x in X])
    return leastsq_fit(X, Y)

def threshold_filter(narray, threshold=0.01):
    '''if the value in narray < threshold, it will be setted as threshold'''
    idx = narray < threshold
    narray[idx] = threshold
    return narray

