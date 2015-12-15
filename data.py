# coding:UTF-8

import numpy as np

import random

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

def crop(mat, window, x, y):
    return mat[x-window:x+window+1, y-window:y+window+1]

def getSampleDot(labels=trLabels, volume=trVolume, window=47, label=0, num=150000):
    num = num // labels.shape[0]
    for (i, page) in enumerate(labels):
        xs, ys = np.where(page == label)
        ground = expend(volume[i], window)
        for index in random.sample(range(len(xs)), num):
            yield crop(ground, window, xs[index]+window, ys[index]+window)

def load_data(window=95, positiveNum=50000, negativeNum=50000, rate=0.2):
    cropWindow = (window-1) // 2
    
    def mats(label, num):
        train = np.array(list(getSampleDot(window=cropWindow, label=label, num=positiveNum)))
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
