#coding: UTF-8

import os

import numpy as np
from scipy.optimize import leastsq

import cv2
import caffe

from pre import gauss2D, expend

def load_net(model_dir, gpu_id=None):
    ''' load caffe model from model_dir '''
    model_file = list(filter(lambda i: '.caffemodel' in i, os.listdir(model_dir)))[0]
    if gpu_id is not None:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(os.path.join(model_dir, 'deploy.prototxt'), os.path.join(model_dir, model_file),
                    caffe.TEST)
    return net

# batch size for inputting images
batch_size = 256

def batch_predict(net, X):
    '''
    using caffe model to predict data in X, input with batch images
    '''
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
    ''' using caffe model to prefict data in npyfile '''
    files_total = len(npy_files)
    for (file_no, f) in enumerate(npy_files):
        print("file: %s, %d / %d" % (f, file_no, files_total))
        X = np.load(f)
        Y = batch_predict(net, X)
        name = f.replace(".npy", "_result_"+postfix+".npy")
        print("save in %s ..." % (name))
        np.save(name, Y)
        
    
def prob_count(net, xfiles, yfiles, scale=10**7):
    '''
    count the probabilities from caffe model
    '''
    probs_count_0 = np.zeros((scale+1, 2))
    probs_count_1 = np.zeros((scale+1, 2))
    total = len(xfiles)
    
    for (i, x, y) in zip(range(total), xfiles, yfiles):
        print("%s %d / %d" % (x, i, total))
        X = np.load(x)
        Y = np.load(y)
        predict = batch_predict(net, X)
        predict = (predict*scale).astype('int')
        # col 0, not mem
        probs = predict[:, 0][Y[:, 0] == 0]
        for p in np.unique(probs):
            probs_count_0[p][0] += np.where(probs == p)[0].shape[0]
        # col 0, is mem
        probs = predict[:, 0][Y[:, 0] == 1]
        for p in np.unique(probs):
            probs_count_0[p][1] += np.where(probs == p)[0].shape[0]
        # col 1, not mem
        probs = predict[:, 1][Y[:, 0] == 0]
        for p in np.unique(probs):
            probs_count_1[p][0] += np.where(probs == p)[0].shape[0]
        # col 1, is mem
        probs = predict[:, 1][Y[:, 0] == 1]
        for p in np.unique(probs):
            probs_count_1[p][1] += np.where(probs == p)[0].shape[0]

    # X = np.arange(0, 1, 1./scale)
    # probs_count += 1./scale
    # Y = probs_count[:, 1]/probs_count.sum(axis=1)
    return probs_count_0, probs_count_1

def poly_fit(X, Y, deg=3):
    ''' polynomial fitting '''
    def residuals(p, y, x):
        e = p.dot((np.array([np.power(x, i) for i in range(deg+1)])))
        err = y - e
        return err
    p0 = [0.1]*(deg+1)
    plsq = leastsq(residuals, p0, args=(Y, X))
    return plsq[0]

prob_eval_p = np.array([ 0.61293427,  1.05949205, -1.61071239,  0.92506701])

def to_poly_prob(X, p=prob_eval_p):
    ''' Caculate fitting '''
    k = X.flatten()
    deg = p.shape[0]
    l = [np.power(k, i).reshape((k.shape[0], 1)) for i in range(deg)]
    x = np.hstack(tuple(l))
    return (p.dot(x.T)).reshape(X.shape)

def log_fit(X, Y, p0=1):
    def residuals(p, y, x):
        err = y - p*np.log(x+1)
        return err
    plsq = leastsq(residuals, p0, args=(Y, X))
    return plsq[0]

def prob_fit(probs_count, threshold=3, scale=10**3, p_index=1):
    ''' polynomial fitting for probs '''
    pc = probs_count.reshape((scale, probs_count.shape[0]/scale, 2)).sum(1)
    idx = pc.sum(axis=1) >= threshold
    X = np.arange(0, 1, 1./pc.shape[0])[idx]
    Y = pc[idx, p_index]/pc[idx].sum(1)
    return poly_fit(X, Y)

def threshold_filter(narray, threshold=0.01):
    '''if the value in narray < threshold, it will be setted as threshold'''
    idx = narray < threshold
    narray[idx] = threshold
    return narray

def merge_result(npy_files, shape=[30, 512, 512]):
    ''' get the result from numpy.array files'''
    files_dict = {}
    for f in npy_files:
        no = int(f.split("/")[-1].split("_")[3])
        files_dict[no] = f
    result = np.zeros(tuple(shape+[2]))
    page_num = shape[0]
    loc = 0
    keys = files_dict.keys()
    keys.sort()
    for k in keys:
        f = files_dict[k]
        print(k, f)
        x = np.load(f)
        assert x.shape[0] % page_num == 0
        for n in range(int(x.shape[0]/page_num)):
            result[:, int(loc/shape[1]), int(loc%shape[2])] = x[n*page_num:(n+1)*page_num]
            loc += 1
    return result

def longest_increasing_idx(x):
    X = x.flatten()
    f = np.zeros(X.shape)
    f[0] = 1
    for i in range(1, X.shape[0]):
        if X[i] >= X[i-1]:
            f[i] = f[i-1]+1
        else:
            mae = np.where(X[:i] <= X[i])[0]
            f[i] = mae.shape == (0,) and 1 or f[mae[-1]]+1
    l = f.max()
    idx = np.zeros(X.shape)
    for i in range(X.shape[0]-1, -1, -1):
        if f[i] == l:
            maxidx = i
            idx[maxidx] = True
            break
        else:
            idx[i] = False
    for i in range(maxidx-1, -1, -1):
        if f[i] == f[maxidx]-1 and X[i] <= X[maxidx]:
            idx[i] = True
            maxidx = i
        else:
            idx[i] = False
    return idx.astype('bool')

def output2prob(a, threshold=0.0005):
    ''' from output to probabilities '''
    r = to_poly_prob(a)
    r[a < threshold] = 0
    return r

def prob_mean_filter(a, window_size=3):
    assert window_size %2 == 1
    half = int(window_size/2)
    grounds = expend(a, window_size)
    filtered = np.zeros(a.shape)
    X, Y, Z = np.where(filtered == 0)
    for (x, y, z) in zip(X, Y, Z):
        filtered[x, y, z] = (grounds[x, window_size+y-half:window_size+y+half+1, window_size+z-half:window_size+z+half+1]).mean()
        if filtered[x, y, z] != 0:
            print(filtered[x, y, z])
    return filtered

def save_prob_png(prefix, a):
    for i in range(a.shape[0]):
        name = "%s_%d.png" % (prefix, i)
        cv2.imwrite(name, a[i])

def weights_ave(tup):
    n = len(tup)
    means = [i.mean() for i in tup]
    t = reduce(lambda x, y: x*y, means)
    weights = np.array([t/i for i in means])
    weights /= weights.sum()
    print(weights)
    new = np.empty(tup[0].shape)
    for i in range(n):
        new += weights[i]*tup[i]
    return new

