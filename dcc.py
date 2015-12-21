#coding: UTF-8

import os

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

import mxnet as mx

from models import n1

train_prefix = "train"
test_prefix = "test"
dir_prefix = 'data/prefile/'

def shift(mx):
    return mx.reshape((mx.shape[0], 1, mx.shape[1], mx.shape[2]))

def mx_build(model_setting=n1):
    conve_layers = model_setting['conve_layers']
    pool_sizes = model_setting['pool_sizes']
    input_shape = model_setting['input_shape']

    data = mx.symbol.Variable('data')
    pre = data
    for (conv, pool) in zip(conve_layers, pool_sizes):
        conv_layer = mx.symbol.Convolution(data=pre, kernel=tuple(conv[1:]), num_filter=conv[0])
        acti_layer = mx.symbol.Activation(data=conv_layer, act_type=model_setting['conve_activa'])
        pool_layer = mx.symbol.Pooling(data=acti_layer, pool_type='max', kernel=tuple(pool))
        pre = mx.symbol.Dropout(data=pool_layer, p=model_setting['dropout'])

    flatten = mx.symbol.Flatten(data=pre)
    pre = flatten
    for units in model_setting['dense_layers'][:-1]:
        fc_layer = mx.symbol.FullyConnected(data=pre, num_hidden=units)
        acti_layer = mx.symbol.Activation(data=conv_layer, act_type=model_setting['dense_activa'])
        pre = mx.symbol.Dropout(data=pool_layer, p=model_setting['dropout'])
        
    last_layer = mx.symbol.FullyConnected(data=pre, num_hidden=model_setting['dense_layers'][-1])
    network = mx.symbol.SoftmaxOutput(data=last_layer, name='softmax')
    return network

def mx_train(network, model_setting=n1, gpus=None, epoch=2000):
    devs = mx.cpu() if gpus is None else [mx.gpu(int(i)) for i in gpus.split(',')]
    window_size = model_setting['window_size']
    filename = "%s%s_%d" % (dir_prefix, train_prefix, window_size)
    
    X = np.load(filename+"_1500_0_x.npy")
    Y = np.load(filename+"_1500_0_y.npy")[:, 0]
    X = shift(X)
    
    train = mx.io.NDArrayIter(
        data = X,
        label = Y,
        # image = filename+"_x",
        # label = filename+"_y",
        batch_size = 500,
        shuffle = True,
        # last_batch_handle = 'discard',
        )
    model = mx.model.FeedForward(
        ctx = devs,
        symbol = network,
        num_epoch = epoch,
        learning_rate = 0.1,
        numpy_batch_size = 20,
        )
    print(X.shape, Y.shape)
    
    model.fit(X=X, y=Y)
    return model

def build_cnn(model_setting=n1):
    model = Sequential()
    conve_layers = model_setting['conve_layers']
    pool_sizes = model_setting['pool_sizes']
    input_shape = model_setting['input_shape']

    model.add(Convolution2D(conve_layers[0][0], conve_layers[0][1], conve_layers[0][2], input_shape=input_shape))
    model.add(Activation(model_setting['conve_activa']))
    model.add(MaxPooling2D(pool_size=pool_sizes[0]))
    model.add(Dropout(model_setting['dropout']))
    for i in range(len(pool_sizes)-1):
        i += 1
        model.add(Convolution2D(conve_layers[i][0], conve_layers[i][1], conve_layers[i][2]))
        model.add(Activation(model_setting['conve_activa']))
        model.add(MaxPooling2D(pool_size=pool_sizes[i]))
        model.add(Dropout(model_setting['dropout']))

    model.add(Flatten())
    for n in model_setting['dense_layers'][:-1]:
        model.add(Dense(n))
        model.add(Activation(model_setting['dense_activa']))
        model.add(Dropout(model_setting['dropout']))
    model.add(Dense(model_setting['dense_layers'][-1]))
    model.add(Activation('softmax'))

    sgd = SGD(nesterov=True)
    model.compile(loss=model_setting['loss'], optimizer=sgd)
    return model

def train(model, model_setting=n1, max_batches=1000, every_batch=20, times=10, epoch=100):
    window_size = model_setting['window_size']
    filename = "%s_%d_" % (train_prefix, window_size)
    files = list(filter(lambda x: filename in x, os.listdir(dir_prefix)))
    x_files = list(filter(lambda x: '_x.npy' in x, files))
    x_files.sort()
    single_batch_size = int(files[0].split("_")[2])
    batches = min(len(x_files)//(every_batch), max_batches)
    
    print("Total", batches*every_batch*single_batch_size, "samples")
    print("Total", batches*every_batch, "batches")
    print("Compressed to", batches, "batches")

    X_shape = (single_batch_size*every_batch, window_size, window_size)
    X = np.zeros(X_shape)
    Y = np.zeros((X_shape[0], 2))
    
    for i in range(times):
        print("Round :", i, "/", times)
        for batch_no in range(batches):
            print("Batch:", batch_no, "/", batches)
            b_x_files = x_files[batch_no*every_batch:(batch_no+1)*every_batch]
            b_y_files = [x.replace("_x.", "_y.") for x in b_x_files]
            print("load data ...")
            for (n, x, y) in zip(range(every_batch), b_x_files, b_y_files):
                X[n*single_batch_size:(n+1)*single_batch_size] = np.load(dir_prefix+x)
                Y[n*single_batch_size:(n+1)*single_batch_size] = np.load(dir_prefix+y)
            X = shift(X)
            print("training ... ")
            model.fit(X, Y, nb_epoch=epoch, verbose=1)
            score = model.evaluate(tx, ty, show_accuracy=True, verbose=1)
            print("Train score: ", score[0])
            print("Train accuracy: ", score[1])
    return model

def predict(model, model_setting=n1):
    window_size = model_setting['window_size']
    filename = "%s_%d_" % (test_prefix, window_size)
    files = list(filter(lambda x: filename in x and "_x." in x, os.listdir(dir_prefix)))
    files.sort()
    single_batch_size = int(files[0].split("_")[2])
    batches = len(files)
    Y = np.zeros((batches*single_batch_size, 2))

    print("Total", batches*single_batch_size, "samples")
    print("Total", batches, "batches")
    
    for batch_no in range(batches):
        print("Batch:", batch_no, "/", batches)
        print("load data ...")
        X = np.load(dir_prefix+files[batch_no])
        X = shift(X)
        print("predict ...")
        Y[batch_no*single_batch_size:(batch_no+1)*single_batch_size] = model.predict(X)

    np.save("data/test_result_"+model_setting['name'], Y)
    return Y
    
