#coding: UTF-8

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import numpy as np

rng = np.random.RandomState(23455)
input = T.tensor4(name='input')

w_shp = (2, 3, 9, 9)
w_bound = np.sqrt(3*9*9)
W = theano.shared( np.asarray(
    rng.uniform(
        low=-1.0/w_bound,
        high=1.0/w_bound,
        size=w_shp
    ), dtype=input.dtype)
    , name='W'
    )

b_shp = (2,)
b = theano.shared(
    np.asarray(
        rng.uniform(
            low=-0.5,
            high=0.5,
            size=b_shp
        ), dtype=input.dtype
    ), name='b'
    )

conv_out = conv.conv2d(input, W)

output = T.nnet.sigmod(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

f = theano.function([input], output)

maxpool_shape = (2, 2)
pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=True)
f = theano.function([input], pool_out)

