from copy import deepcopy
import numpy as np

def l2norm(x):
    return np.linalg.norm(x)

def gradCheck(NN, X, Y, eps=1.0e-5):
    NN.forwardPass(X)
    C = NN.cost(Y)
    NN.backwardPass(Y)
    grads = []
    for l in NN.NNLayers:
        grads.append((l.cache['dW'],l.cache['db']))

    numGrads = []
    for l in range(len(NN.NNLayers)):
        numdW = np.zeros(NN.NNLayers[l].W.shape)
        numdb = np.zeros(NN.NNLayers[l].bias.shape)
        for i in range(numdW.shape[0]):
            for j in range(numdW.shape[1]):
                copyNN1 = deepcopy(NN)
                copyNN2 = deepcopy(NN)

                copyNN1.NNLayers[l].W[i,j] += eps
                copyNN2.NNLayers[l].W[i,j] -= eps

                copyNN1.forwardPass(X)
                copyNN2.forwardPass(X)
                C1 = copyNN1.cost(Y)
                C2 = copyNN2.cost(Y)
                numdW[i,j] = (C1-C2)/(2*eps)
        for i in range(numdb.shape[0]):
            copyNN1 = deepcopy(NN)
            copyNN2 = deepcopy(NN)

            copyNN1.NNLayers[l].bias[i] += eps
            copyNN2.NNLayers[l].bias[i] -= eps

            copyNN1.forwardPass(X)
            copyNN2.forwardPass(X)
            C1 = copyNN1.cost(Y)
            C2 = copyNN2.cost(Y)
            numdb[i] = (C1-C2)/(2*eps)
        numGrads.append((numdW, numdb))

    
    totalVecLen = 0
    for l in range(len(NN.NNLayers)):
        totalVecLen += grads[l][0].flatten().shape[0] + grads[l][1].shape[0]

    gr = np.zeros(totalVecLen)
    nGr = np.zeros(totalVecLen)
    i = 0
    ip = 0
    for l in range(len(NN.NNLayers)):
        ip += grads[l][0].flatten().shape[0]
        gr[i:ip] = grads[l][0].flatten()
        nGr[i:ip] = numGrads[l][0].flatten()
        i = ip
        ip += grads[l][1].shape[0]
        gr[i:ip] = grads[l][1].flatten()
        nGr[i:ip] = numGrads[l][1].flatten()
        i = ip

    numer = l2norm(nGr-gr)
    denom = l2norm(nGr)+l2norm(gr)
    return numer/denom
