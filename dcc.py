#coding: UTF-8

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

from data import load_data, shift

conAct = 'tanh'
denAct = 'sigmoid'

def build_cnn(window_size=95):
    model = Sequential()
    
    model.add(Convolution2D(48, 4, 4, input_shape=(1, window_size, window_size)))
    model.add(Activation(conAct))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(48, 5, 5))
    model.add(Activation(conAct))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(48, 4, 4))
    model.add(Activation(conAct))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(48, 4, 4))
    model.add(Activation(conAct))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation(denAct))
    model.add(Dense(2))
    model.add(Activation(denAct))

    sgd = SGD(lr=0.1, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def train_model(model, num=2000, times=10):
    for i in range(times):
        print(i, ":")
        (tx, ty, vx, vy) = load_data(positiveNum=num//2, negativeNum=num//2)
        tx = shift(tx)
        vx = shift(vx)
        print("training ... ")
        model.fit(tx, ty, nb_epoch=40, validation_data=(vx, vy))
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
    
