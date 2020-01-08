from idxReader import readIdxFile
from NN import NN
import numpy as np

trainX = readIdxFile('Data/train-images-idx3-ubyte')
trainY = readIdxFile('Data/train-labels-idx1-ubyte')

testX = readIdxFile('Data/t10k-images-idx3-ubyte')
testY = readIdxFile('Data/t10k-labels-idx1-ubyte')

def makeImagesCols(arr):
    return arr.reshape(arr.shape[0],28*28).T

def makeLabelsCols(arr):
    arr = arr.reshape(1, arr.shape[0])
    res = np.arange(arr.shape[0]*10) % 10
    res = res.reshape(10, arr.shape[0])
    return (arr == res).astype(int)

trainX = makeImagesCols(trainX)
testX = makeImagesCols(testX)

trainY = makeLabelsCols(trainY)
testY = makeLabelsCols(testY)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)
exit()
hiddenUnits = [] # empty list corresponds to no hidden units
linearNN = NN(hiddenUnits)


