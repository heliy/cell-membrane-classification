#coding: UTF-8

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

from data import load_data, shift
from models import n4

window_size = 95
input_shape = (1, window_size, window_size)

def build_cnn(model_setting=n4):
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

    model.compile(loss=model_setting['loss'], optimizer=model_setting['optimizer'])
    return model

def train_model(model, num=3000, times=10, epoch=42):
    for i in range(times):
        print(i, ":")
        (tx, ty, vx, vy) = load_data(positiveNum=num//2, negativeNum=num//2)
        tx = shift(tx)
        vx = shift(vx)
        print("training ... ")
        model.fit(tx, ty, nb_epoch=epoch, validation_data=(vx, vy))
        score = model.evaluate(tx, ty, show_accuracy=True)
        print("Train score: ", score[0])
        print("Train accuracy: ", score[1])
        score = model.evaluate(vx, vy, show_accuracy=True)
        print("Valid score: ", score[0])
        print("Valid accuracy: ", score[1])
    return model

def test_model(model, num=4000):
    (tx, ty, vx, vy) = load_data(positiveNum=num//2, negativeNum=num//2, rate=0)
    tx = shift(tx)
    score = model.evaluate(tx, ty, show_accuracy=True)
    print("Test score: ", score[0])
    print("Test accuracy: ", score[1])
    
