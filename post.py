#coding: UTF-8

" py 2222222222222222222222222222222222222 "


import os

import numpy as np
from scipy.optimize import leastsq

# import caffe

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
        
    
def leastsq_fit(X, Y, p0=[0.01, 0.01, 0.01]):
    def residuals(p, y, x):
        p3, p2, p1 = p
        err = y - (p3*(x**3)+p2*(x**2)+p1*x)
        return err
    plsq = leastsq(residuals, p0, args=(Y, X))
    return plsq[0]

def prob_count(net, xfiles, yfiles, scale=10**7):
    '''
    y_index = 1 for n1/n2, = 0 for n3/n4
       as when we train the net, the first col in prob is different Orz.
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

def prob_fit(probs_count, scale=10**5):
    pc = probs_count.reshape((scale, probs_count.shape[0]/scale, 2)).sum(1)
    idx = pc.sum(axis=1) != 0
    X = np.arange(0, 1, 1./pc.shape[0])[idx]
    Y = pc[idx, 1]/pc[idx].sum(1)
    return leastsq_fit(X, Y), X, Y

prob_eval_p0 = np.array([ 10.94442125, -17.46711456,   7.83811808])
prob_eval_p1 = np.array([-3.52704756,  4.08572813])
prob_eval_p2 = np.array([ 2.29793248, -3.79739524,  2.00212631,  0.52655225])

def threshold_filter(narray, threshold=0.01):
    '''if the value in narray < threshold, it will be setted as threshold'''
    idx = narray < threshold
    narray[idx] = threshold
    return narray

def merge_result(npy_files, shape=[30, 512, 512]):
    result = np.zeros(tuple(shape+[2]))
    page_num = shape[0]
    loc = 0
    for f in npy_files:
        x = np.load(f)
        assert x.shape[0] % page_num == 0
        for n in range(int(x.shape[0]/page_num)):
            result[:, int(loc/shape[1]), int(loc%shape[2])] = x[n*page_num:(n+1)*page_num]
            loc += 1
    return result

def to_prob(X, p=prob_eval_p1):
    k = X.flatten()
    x = np.hstack((
                   np.power(k, 3).reshape((k.shape[0], 1)),
                   np.power(k, 2).reshape((k.shape[0], 1)),
                   X.reshape((k.shape[0], 1)),
                   # np.ones((k.shape[0], 1))
                   ))
    return p.dot(x.T).reshape(X.shape)

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

# reference to
# http://stackoverflow.com/questions/15191088/how-to-do-a-polynomial-fit-with-fixed-points
def polyfit_with_fixed_points(n, x, y, xf, yf) :
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]
