import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(x,0)

def identity(x):
    return x

class NNLayer():

    def __init__(s, numInputs, numOutputs, randomize=True, activation = sigmoid):
        s.numInputs = numInputs
        s.numOutputs = numOutputs
        s.activation = activation
        if randomize:
            s.W = np.random.randn(s.numOutputs, s.numInputs)*np.sqrt(1/s.numInputs)*1.0e-4
            s.bias = np.zeros((s.numOutputs, 1))
        else:
            s.W = np.zeros((s.numOutputs, s.numInputs))
            s.bias = np.zeros((s.numOutputs, 1))
        s.cache = {}

    def forwardPropogate(s, inp):
        Z = np.dot(s.W,inp)+s.bias
        A = s.activation(Z)
        s.cache['Aprev'] = inp
        s.cache['Z'] = Z
        s.cache['A'] = A
        return A

    def backwardPropogate(s, dA):
        m = dA.shape[1]
        z = s.cache['Z']
        if s.activation == identity:
            gpz = z
        elif s.activation == sigmoid:
            a = s.cache['A']
            gpz = np.multiply(np.square(a),np.exp(-z))
        elif s.activation == ReLU:
            gpz = (z>0).astype(int)
        elif s.activation == np.tanh:
            gpz = np.square(2/(np.exp(z)+np.exp(-z)))
        else:
            raise Exception('Not implemented')
        dZ = np.multiply(gpz, dA)
        aPrev = s.cache['Aprev']
        dW = np.dot(dZ, aPrev.T)
        db = np.sum(dZ, axis=1).reshape(s.bias.shape)
        dAprev = np.dot(s.W.T,dZ)
        s.cache['dW'] = dW
        s.cache['db'] = db
        s.cache['m'] = m
        return dAprev

    def updateParams(s, learningRate = 0.001, lambd = 0.001):
        s.W    = (1-lambd)*s.W - learningRate * s.cache['dW'] / s.cache['m']
        s.bias = s.bias        - learningRate * s.cache['db'] / s.cache['m']
    
