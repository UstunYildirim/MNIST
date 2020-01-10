#!/usr/bin/python3

from idxReader import readIdxFile
from NN import NN
from NNLayer import *
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

def linear(iterations = 100, batchSize = 256):
    hiddenUnits = [] # empty list corresponds to no hidden units
    linearNN = NN(hiddenUnits)
    for i in range(iterations):

        batchIndex = (i*batchSize)%trainX.shape[1]
        useX = trainX[:, batchIndex:batchIndex+batchSize]
        useY = trainY[:, batchIndex:batchIndex+batchSize]

        linearNN.forwardPass(useX)
        C = linearNN.cost(useY)
        linearNN.backwardPass(useY)
        linearNN.updateParams(learningRate=0.1*batchSize/trainX.shape[1],lambd=0.0)

        pred = linearNN.predict(trainX)
        acc = trainLabels == pred

        predTest = linearNN.predict(testX)
        accTest = testLabels == predTest

        print ("Run #{:4d}\tCost: {:.2f}\tError rate: {:.2f}%\tError rate on test: {:.2f}%".format(
            i,
            C,
            100-100*np.sum(acc)/acc.shape[0],
            100-100*np.sum(accTest)/accTest.shape[0]))
    return linearNN


# linearNN = linear(2, batchSize=256)
# err = gradCheck(linearNN, testX[:,:10], testY[:,:10])
# print ('GradCheck: ', err)
# exit()

def twoLayers(iterations = 100, batchSize = 256):
    hiddenUnits = [(300,ReLU)] # empty list corresponds to no hidden units
    linearNN = NN(hiddenUnits)
    for i in range(iterations):

        batchIndex = (i*batchSize)%trainX.shape[1]
        useX = trainX[:, batchIndex:batchIndex+batchSize]
        useY = trainY[:, batchIndex:batchIndex+batchSize]

        linearNN.forwardPass(useX)
        C = linearNN.cost(useY)
        linearNN.backwardPass(useY)
        linearNN.updateParams(learningRate=0.1*batchSize/trainX.shape[1],lambd=0.0)

        pred = linearNN.predict(trainX)
        acc = trainLabels == pred

        predTest = linearNN.predict(testX)
        accTest = testLabels == predTest

        print ("Run #{:4d}\tCost: {:.2f}\tError rate: {:.2f}%\tError rate on test: {:.2f}%".format(
            i,
            C,
            100-100*np.sum(acc)/acc.shape[0],
            100-100*np.sum(accTest)/accTest.shape[0]))
    return linearNN

tlNN = twoLayers(3)
#print(gradCheck(tlNN, testX[:,:11], testY[:,:11]))
#FIXME: with ReLU gradCheck seems weird but it might be normal due to singularity at 0
#       more investigation needed


