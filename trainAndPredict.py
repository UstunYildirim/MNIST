#!/usr/bin/python3

from idxReader import readIdxFile
from NN import NN
import numpy as np
from gradCheck import gradCheck

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

def linear(iterations = 100):
    hiddenUnits = [] # empty list corresponds to no hidden units
    linearNN = NN(hiddenUnits)
    for i in range(iterations):

        linearNN.forwardPass(trainX)
        C = linearNN.cost(trainY)
        linearNN.backwardPass(trainY)
        linearNN.updateParams(learningRate=0.1,lambd=0.0)

        pred = linearNN.predict(trainX)
        acc = trainLabels == pred

        predTest = linearNN.predict(testX)
        accTest = testLabels == predTest

        print ("Run #{}\tCost: {:.2f}\tError rate: {:.2f}%\tError rate on test: {:.2f}%".format(
            i,
            C,
            100-100*np.sum(acc)/acc.shape[0],
            100-100*np.sum(accTest)/accTest.shape[0]))
    return linearNN


linearNN = linear(100)
# err = gradCheck(linearNN, testX[:,:123], testY[:,:123])
# print ('GradCheck: ', err)
