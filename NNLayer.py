import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(x,0)

def identity(x):
    return x

class NNLayer():

    def __init__(s,
            numInputs,
            numOutputs,
            learningRate,
            lambd,
            decay1,
            decay2,
            eps,
            randomize=True,
            activation = sigmoid
            ):
        s.numInputs = numInputs
        s.numOutputs = numOutputs
        s.activation = activation
        if randomize:
            s.W = np.random.randn(s.numOutputs, s.numInputs)*np.sqrt(2/s.numInputs)
            s.bias = np.zeros((s.numOutputs, 1))
        else:
            s.W = np.zeros((s.numOutputs, s.numInputs))
            s.bias = np.zeros((s.numOutputs, 1))
        s.cache = {'dW': 0, 'db': 0,
                'VdW': 0, 'Vdb': 0,
                'SdW': 0, 'Sdb': 0}
        s.learningRate = learningRate
        s.lambd = lambd
        s.decay1 = decay1
        s.decay2 = decay2
        s.eps = eps
        s.iteration = 0

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
        s.cache['VdW'] = s.decay1 * s.cache['VdW'] + (1-s.decay1) * dW
        s.cache['Vdb'] = s.decay1 * s.cache['Vdb'] +  (1-s.decay1) * db
        s.cache['SdW'] = s.decay2 * s.cache['SdW'] + (1-s.decay2) * np.square(dW)
        s.cache['Sdb'] = s.decay2 * s.cache['Sdb'] +  (1-s.decay2) * np.square(db)
        return dAprev

    def updateParams(s):
        s.iteration += 1
        VdWcorr = s.cache['VdW'] / (1-s.decay1**s.iteration)
        Vdbcorr = s.cache['Vdb'] / (1-s.decay1**s.iteration)
        SdWcorr = s.cache['SdW'] / (1-s.decay2**s.iteration)
        Sdbcorr = s.cache['Sdb'] / (1-s.decay2**s.iteration)
        s.W    = (1-s.lambd)*s.W - s.learningRate * np.divide(VdWcorr, (np.sqrt(SdWcorr)+s.eps))
        s.bias = s.bias          - s.learningRate * np.divide(Vdbcorr, (np.sqrt(Sdbcorr)+s.eps))
    
