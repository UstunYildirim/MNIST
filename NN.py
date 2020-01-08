import numpy as np
from NNLayer import *

class NN():

    def __init__(s, NNStructure): 
        s.NNLayers = []
        # NNStructure is a list of pairs of number of hidden units and activation functions
        # input is a 28x28 image
        # and we classify this image as a digit from 0 to 9
        NNStructure = [(28*28,None)] + \
                NNStructure + \
                [(10,sigmoid)]
        for i in range(1,len(NNStructure)):
            s.NNLayers.append(
                    NNLayer(NNStructure[i-1][0],
                        NNStructure[i][0],
                        activation = NNStructure[i][1]))

    def backwardPass(s, Y):
        a = s.lastActivation
        dA = -(np.divide(Y,a)-np.divide(1-Y,1-a))
        for layer in reversed(s.NNLayers):
            dA = layer.backwardPropogate(dA)
        for layer in s.NNLayers:
            layer.updateParams(learningRate=0.00001,lambd=0.000001)


    def forwardPass(s, inp):
        for layer in s.NNLayers:
            inp = layer.forwardPropogate(inp)
        s.lastActivation = inp.reshape(10,1)
        return s.lastActivation

    def predict(s, inp):
        out = s.forwardPass(inp)
        return np.argmax(out)

