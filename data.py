# coding:UTF-8

import random

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata

trLabels = np.load('data/train-labels.npy')
trVolume = np.load('data/train-volume.npy')

def expend(mat, window):
    row, col = mat.shape
    reverse = range(window)[::-1]
    ground = np.zeros((row+window*2, col+window*2))
    
    ground[:window, :window] = np.rot90(mat[1:window+1, 1:window+1], 2)
    ground[:window, window:window+col] = mat[1:window+1, :][reverse]
    ground[:window, window+col:] = np.rot90(mat[1:window+1, col-window-1:-1], 2)
    
    ground[window:window+row, :window] = mat[:, 1:window+1][:, reverse]
    ground[window:window+row, window:window+col] = mat[:, :]
    ground[window:window+row, window+col:] = mat[:, col-window-1:-1][:, reverse]
    
    ground[window+row:, :window] = np.rot90(mat[row-window-1:-1, 1:window+1], 2)
    ground[window+row:, window:window+col] = mat[row-window-1:-1, :][reverse]
    ground[window+row:, window+col:] = np.rot90(mat[row-window-1:-1, col-window-1:-1], 2)
    
    return ground

def random_rotate(mat):
    degree = random.randint(0, 3)
    return np.rot90(mat, degree)

center = 4

def gauss2D(shape=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filters(window_size=95):
    filters = {}
    half = window_size//2
    sigmas = np.zeros((half+center, half+center))
    for i in range(half+center):
        for j in range(half+center):
            sigma = ((7*max(i, j))/half)+0.001
            sigmas[i, j] = sigma
            size = max(i//(2*center), j//(2*center))*2+1
            filters[i*half+j] = gauss2D((size, size), sigma)
    return filters

def foveate(mat, filters):
    window_size = mat.shape[0]
    foveated = mat.copy()
    half = window_size//2
    fix_point = np.array(mat.shape)//2
    mirror = expend(mat, half)
    for i in range(window_size):
        for j in range(window_size):
            ds = np.abs(np.array([i, j]) - fix_point)
            if ds[0] < center and ds[1] < center:
                continue
            filter = filters[int(ds[0]*half+ds[1])]
            size = filter.shape[0]//2
            m = mirror[half+i-size:half+i+size+1, half+j-size:half+j+size+1]
            foveated[i, j] = int(round((m*filter).sum()))
    return foveated

def sampling_function(ground, window_size, ratio=1.3):
    window_large = int(window_size*ratio//2)*2
    X, Y = np.meshgrid(np.linspace(-1, 1, window_large), np.linspace(-1, 1, window_large))
    X_, Y_ = np.meshgrid(np.linspace(-1, 1, window_size), np.linspace(-1, 1, window_size))
    X_, Y_ = (3*X_**3+X_)/4, (3*Y_**3+Y_)/4
    x = np.dstack((X, Y)).reshape((window_large**2, 2))
    xi = np.dstack((X_, Y_)).reshape((window_size**2, 2))
    window = window_large//2
    def func(i, j):
        m = ground[i+window_size-window:i+window_size+window, j+window_size-window:j+window_size+window]
        print(x.shape, m.shape, xi.shape)
        # return x, m, xi
        return griddata(x, m.flat, xi, method='cubic').reshape((window_size, window_size))
    return func

def getSampleDot(labels=trLabels, volume=trVolume, window=47, label=0, num=150000):
    num = num // labels.shape[0]
    for (i, page) in enumerate(labels):
        xs, ys = np.where(page == label)
        ground = expend(volume[i], window)
        for index in random.sample(range(len(xs)), num):
            pass
            # yield crop(ground, window, xs[index]+window, ys[index]+window)

def load_data(window=95, positiveNum=50000, negativeNum=50000, rate=0.2):
    
    def mats(label, num):
        train = np.array(list(getSampleDot(window=window, label=label, num=positiveNum)))
        choose = np.random.rand(len(train)) > rate
        return train[choose], train[np.logical_not(choose)]

    
    trainX0, validX0 = mats(0, positiveNum)
    trainX255, validX255 = mats(255, negativeNum)
    trainY0 = np.repeat([[0, 1]], [trainX0.shape[0]], axis=0)
    validY0 = np.repeat([[0, 1]], [validX0.shape[0]], axis=0)
    trainY255 = np.repeat([[1, 0]], [trainX255.shape[0]], axis=0)
    validY255 = np.repeat([[1, 0]], [validX255.shape[0]], axis=0)

    return (np.concatenate((trainX0, trainX255)),
            np.concatenate((trainY0, trainY255)),
            np.concatenate((validX0, validX255)),
            np.concatenate((validY0, validY255)))

def shift(mx):
    return mx.reshape((mx.shape[0], 1, mx.shape[1], mx.shape[2]))
