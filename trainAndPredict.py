#!/usr/bin/python3

from idxReader import readIdxFile
from NN import NN
from NNLayer import *
import numpy as np
from gradCheck import gradCheck
from plotCost import plotCost

def makeImagesCols(arr):
    return arr.reshape(arr.shape[0],28*28).T

def makeLabelsCols(arr):
    arr = arr.reshape(1, arr.shape[0])
    res = np.arange(arr.shape[0]*10) % 10
    res = res.reshape(10, arr.shape[0])
    return (arr == res).astype(int)

trainX = readIdxFile('Data/train-images-idx3-ubyte')
trainX = makeImagesCols(trainX)
trainLabels = readIdxFile('Data/train-labels-idx1-ubyte')
trainY = makeLabelsCols(trainLabels)

testX = readIdxFile('Data/t10k-images-idx3-ubyte')
testX = makeImagesCols(testX)
testLabels = readIdxFile('Data/t10k-labels-idx1-ubyte')
testY = makeLabelsCols(testLabels)

def createNN(hiddenUnits, learningRate):
    return NN(hiddenUnits, learningRate = learningRate)

def training(NN, iterations = 100, batchSize = 2**7, plot=False):
    costs = []
    for i in range(1, iterations+1):

        batchIndex = (i*batchSize)%trainX.shape[1]
        useX = trainX[:, batchIndex:batchIndex+batchSize]
        useY = trainY[:, batchIndex:batchIndex+batchSize]

        if i%100 == 0:
            NN.setLearningRate(
                    learningRate/(2**(int(i/100)))
                    )

        NN.forwardPass(useX)
        C = NN.cost(useY)
        costs.append(C)
        NN.backwardPass(useY)
        NN.updateParams()

        # pred = NN.predict(trainX)
        # acc = trainLabels == pred

        predTest = NN.predict(testX)
        accTest = testLabels == predTest

        #print ("Run #{:4d}\tCost: {:.3f}\tError rate: {:.3f}%\tError rate on test: {:.3f}%".format(
        print ("Run #{:4d}\tCost: {:.3f}\tError rate on test: {:.3f}%".format(
            i,
            C,
            100-100*np.sum(accTest)/accTest.shape[0]))
    if plot:
        plotCost(costs)


#linearNN = linear(300, batchSize=256)
#exit()

learningRate = 0.003
linearReg = createNN([], learningRate)
twoLayers = createNN([
    (300, ReLU)
    ],
    learningRate
    )
threeLayers = createNN([
    (300, ReLU),
    (40, ReLU)
    ],
    learningRate
    )
# nLayers = createNN([
#     (300, ReLU),
#     (40, ReLU),
#     (20, ReLU),
#     (16, ReLU),
#     ],
#     learningRate
#     )

training(threeLayers, iterations=1200, batchSize=2**7, plot=True)
#print(gradCheck(tlNN, testX[:,:11], testY[:,:11]))
#FIXME: with ReLU gradCheck seems weird but it might be normal due to singularity at 0
#       more investigation needed


