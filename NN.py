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
        m = Y.shape[1]
        A = s.lastActivation
        dA = -1/m*(np.divide(Y,A)-np.divide(1-Y,1-A))
        for layer in reversed(s.NNLayers):
            dA = layer.backwardPropogate(dA)

    def updateParams(s):
        for layer in s.NNLayers:
            layer.updateParams()


    def forwardPass(s, inp):
        for layer in s.NNLayers:
            inp = layer.forwardPropogate(inp)
        s.lastActivation = inp
        return inp

    def predict(s, inp):
        out = s.forwardPass(inp)
        return np.argmax(out, axis=0)

    def cost(s, Y):
        m = Y.shape[1]
        A = s.lastActivation
        C = np.sum(-1/m*(
            np.multiply(Y, np.log(A)) + np.multiply(1-Y,np.log(1-A))
            ).flatten())
        return C
