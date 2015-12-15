# coding: UTF-8

# CNN for D.C.Ciresan et al, 2012
# lasagne version

import time

import numpy as np
import theano
from theano import function as F
import theano.tensor as T
import lasagne

from data import getSampleDot
from theanoDcc import filterShapes, poolSizes

batch_size = 500
window = 95
inputShape = (batch_size, 1, window, window)
nonlinearities = [lasagne.nonlinearities.tanh]*(len(filterShapes)+2)

def build_cnn(input_var, input_shape=inputShape, filter_shapes=filterShapes, pool_sizes=poolSizes, full_size=[200, 2], nonlinearities=nonlinearities):
    network = lasagne.layers.InputLayer(shape=input_shape,
                                        input_var=input_var,
                                        )
    print(lasagne.layers.get_output_shape(network))
    for (filter, pool, nonlin) in zip(filter_shapes, pool_sizes, nonlinearities):
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=filter[0], filter_size=tuple(filter[-2:]),
            nonlinearity=nonlin, W=lasagne.init.GlorotUniform(),
            )
        print(lasagne.layers.get_output_shape(network))
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=pool)
        print(lasagne.layers.get_output_shape(network))

    for (full, nonlin) in zip(full_size, nonlinearities[-1*len(full_size):]):
        network = lasagne.layers.DenseLayer(
            network, num_units=full, nonlinearity=nonlin,
            )
        print(lasagne.layers.get_output_shape(network))
    return network

def iterate_minibatches(inputs, targets, batchsize=batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for idx in range(0, len(inputs)-batchsize+1, batchsize):
        if shuffle:
            excerpt = indices[idx: idx+batchsize]
        else:
            excerpt = slice(idx, idx+batchsize)
        yield inputs[excerpt], targets[excerpt]

def train(epoches=1000, learning_rate=0.1):
    print("build CNN...")
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var)

    return network

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)/2
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=learning_rate)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)/2

    train_fn = F([input_var, target_var], loss, updates=updates)
    val_fn = F([input_var, target_var], test_loss)

    print('Loading data...')
    (x_train, y_train, x_val, y_val) = load_dataset()

    print('Starting training...')
    for epoch in range(epoches):
        start_time = time.time()
        
        train_error, train_batches = 0, 0
        for batch in iterate_minibatches(x_train, y_train, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err, val_batches = 0, 0
        for batch in iterate_minibatches(x_val, y_val):
            inputs, targets = batch
            val_err += val_fn(inputs, targets)
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

    return F([input_var, target_var], prediction)
