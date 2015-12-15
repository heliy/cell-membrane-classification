#coding: UTF-8

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

from data import load_data

def build_cnn(window_size=95):
    model = Sequential()
    
    model.add(Convolution2D(48, 4, 4, input_shape=(1, window_size, window_size)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(48, 5, 5))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(48, 4, 4))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(48, 4, 4))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('sigmoid'))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.1, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def train_model(model, num=800, times=10):
    for i in range(times):
        print(i, ":")
        (tx, ty, vx, vy) = load_data(positiveNum=num//2, negativeNum=num//2)
        tx = tx.reshape((tx.shape[0], 1, tx.shape[1], tx.shape[2]))
        vx = vx.reshape((vx.shape[0], 1, vx.shape[1], vx.shape[2]))
        print("training ... ")
        model.fit(tx, ty, validation_data=(vx, vy))
        score = model.evaluate(tx, ty, show_accuracy=True)
        print("Test score: ", score[0])
        print("Test accuracy: ", score[1])
    return model
    
