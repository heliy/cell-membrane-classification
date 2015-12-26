# coding: UTF-8

# CNN for D.C.Ciresan et al, 2012

import timeit

from theanoUnit import ConvLayer, HiddenLayer, LogisticRegression

import theano
from theano import tensor as T
import numpy as np       
        
filterShapes=[
    (48, 1, 4, 4),
    (48, 48, 5, 5),
    (48, 48, 4, 4),
    (48, 48, 4, 4),
]

poolSizes=[
    (2, 2),
    (2, 2),
    (2, 2),
    (2, 2),
]

def nextImageShape(imageShape, filterShape, poolSize):
    n = np.array(imageShape[2:])-np.array(filterShape[2:])+1
    return list((n/np.array(poolSize)).astype('int'))
        
def buildConvLayers(rng, input, imageShape):
    batchSize = imageShape[0]
    print("Layer 1:")
    print("image shape:", imageShape)
    print("filter shape:", filterShapes[0])
    layers = [ConvLayer(rng, input=input,
                       imageShape=imageShape,
                       filterShape=filterShapes[0],
                       poolSize=poolSizes[0],
                       )]
    for (i, filterShape) in enumerate(filterShapes[1:]):
        input = layers[-1].output
        newShape = [batchSize, filterShapes[i+1][1]]
        newShape.extend(
            nextImageShape(imageShape, filterShapes[i], poolSizes[i])
            )
        newShape = tuple(newShape)
        print("Layer %d:" % (i+1))
        print("image shape:", newShape)
        print("filter shape:", filterShapes[i+1])
        layer = ConvLayer(rng, input=input,
                          imageShape=newShape,
                          filterShape=filterShapes[i+1],
                          poolSize=poolSizes[i+1],
                          )
        imageShape = newShape
        layers.append(layer)
    imageShape = [batchSize, filterShapes[-1][1]]+nextImageShape(
        imageShape, filterShapes[-1], poolSizes[-1]
        )
    print("output shape:", tuple(imageShape))
    return layers, tuple(imageShape)

def buildCNN(window=95, batch_size=500):
    rng = np.random.RandomState(23455)
    x = T.matrix('x')
    y = T.ivector('y')
    imageShape = (batch_size, 1, window, window)
    layer0Input = x.reshape(imageShape)
    print("Convolutional Layers:")
    print("Start building network...")
    convLayers, shape = buildConvLayers(rng, layer0Input, imageShape)
    print("hidden layer:", np.prod(shape[1:]), 200)
    hiddenLayer = HiddenLayer(
        rng,
        input=convLayers[-1].output.flatten(2),
        nIn=np.prod(shape[1:]),
        nOut=200,
        activation=T.tanh
        )
    print("output layer:", 200, 2)
    outputLayer = LogisticRegression(input=hiddenLayer.output, nIn=200, nOut=2)
    cost = outputLayer.negative_log_likelihood(y)
    return (x, y), convLayers + [hiddenLayer, outputLayer], cost


def trainModel(positiveNum, negativeNum, validationRate, window=95,
               batch_size=500, learningRate=0.1, maxEpoch=2000):
    trainXs, trainYs, validXs, validYs = splitValid(window, positiveNum, negativeNum, validationRate)
    nTrains = trainXs.shape[0]//batch_size
    nValids = validXs.shape[0]//batch_size

    trainXs = theano.shared(value=trainXs, name='trainXs')
    trainYs = theano.shared(value=trainYs, name='trainYs')
    validXs = theano.shared(value=validXs, name='validXs')
    validYs = theano.shared(value=validYs, name='validYs')
    
    print("trains: %d * %d" % (nTrains, batch_size))
    print("validations: %d * %d" % (nValids, batch_size))

    (x, y), layers, cost = buildCNN(window, batch_size)
    print("network complete ...")
    params = []
    for layer in layers:
        params += layer.params
        
    grads = T.grad(cost, params)
    index = T.lscalar()

    updates = [
        (i, i - learningRate * j) for i, j in zip(params, grads)
        ]

    train = theano.function(
        [index], cost, updates=updates,
        givens={
            x: trainXs[:, :, index*batch_size: (index+1)*batch_size],
            y: trainYs[:, :, index*batch_size: (index+1)*batch_size]
        }
        )
    
    return None
    validate = theano.function(
        [index], layers[-1].errors(y),
        givens={
            x: validXs[index*batch_size: (index+1)*batch_size],
            y: validYs[index*batch_size: (index+1)*batch_size]
        }        
        )

    print()
    print("... training ...")

    patience = 10000
    patienceIncrease = 2
    improvementThreshold = 0.995
    validationFrequency = min(nTrains, patience/2)
    bestLoss = np.inf
    bestIter, testScore, epoch = 0, 0.0, 0
    done = False

    startTime = timeit.default_timer()
    while(epoch < maxEpoch) and (not done):
        epoch += 1
        for batchIndex in range(nTrains):
            iter = (epoch - 1)*nTrains+batchIndex
            if iter % 100 == 0:
                print('training @ iter = ', iter)
            costValue = train(batchIndex)

            if (iter+1) % validationFrequency == 0:
                validationLosse = np.mean([validate(i) for i in xrange(nValids)])
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, batch_index+1, nTrains, validationLoss*100))

                if validationLoss < bestLoss:
                    if validationLoss < bestLoss*improvementThreshold:
                        patience = max(patience, iter*patienceIncrease)
                    bestLoss = validatonLoss
                    bestIter = iter
            if patience <= iter:
                done = True
                break
            
    endTime = timeit.default_timer()
    print('tain complete')
    print('Best validatation score of %f %% obtained at iteration %i,' %
          (bestLoss*100, bestIter+1))

    return layers
    
input = T.tensor4('input')
