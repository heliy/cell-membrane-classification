# coding:UTF-8

# References:
# http://deeplearning.net/tutorial/lenet.html
# http://deeplearning.net/tutorial/mlp.html
# http://deeplearning.net/tutorial/logreg.html

import theano
from theano import tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv, sigmoid

import numpy as np

class ConvLayer(object):
    """
    Pool Layer of a convolutional network
    """
    
    def __init__(self, rng, input, imageShape, filterShape, poolSize=(2, 2)):
        assert imageShape[1] == filterShape[1]
        self.input = input
        fanIn = np.prod(filterShape[1:])
        fanOut = (filterShape[0]*np.prod(filterShape[2:]))/np.prod(poolSize)
        Wbound = np.sqrt(6/(fanIn+fanOut))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-Wbound, high=Wbound, size=filterShape),
                dtype=theano.config.floatX
                ), borrow=True
        )

        bValues = np.zeros((filterShape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=bValues, borrow=True)

        convOut = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filterShape,
            image_shape=imageShape
        )

        pooledOut = downsample.max_pool_2d(
            input=convOut,
            ds=poolSize,
            ignore_border=True
        )

        self.output = T.tanh(pooledOut+self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]

class HiddenLayer(object):
    def __init__(self, rng, input, nIn, nOut, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None:
            WValues = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6/(nIn + nOut)),
                    high=np.sqrt(6/(nIn + nOut)),
                    size=(nIn, nOut)
                ),
                dtype=theano.config.floatX
            )
            if activation == sigmoid:
                WValues *= 4

            W = theano.shared(value=WValues, name='W', borrow=True)

        if b is None:
            bValues = np.zeros((nOut,), dtype=theano.config.floatX)
            b = theano.shared(value=bValues, name='b', borrow=True)

        self.W = W
        self.b = b

        linOutput = T.dot(input, self.W) + self.b
        self.output = (
            linOutput if activation is None
            else activation(linOutput)
        )
        self.params = [self.W, self.b]            
        

class LogisticRegression(object):
    def __init__(self, input, nIn, nOut):
        self.W = theano.shared(
            value=np.zeros(
                (nIn, nOut),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros(
                (nOut,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

        
class MLP(object):
    def __init__(self, rng, input, nIn, nHidden, nOut):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=nIn,
            n_out=nHidden,
            activation=T.tanh
        )
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=Hidden,
            n_out=nOut
        )
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.input = input
