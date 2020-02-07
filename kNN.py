import numpy as np

class kNN():

    def __init__(s,X,Y):
        assert(len(X.shape)==2)
        assert(len(Y.shape)==1)
        s.X = X
        s.Y = Y
        s.k = None

    def optimizeK(s, Xtest, Ytest, kValues, withWeight=False):
        m = Xtest.shape[0]
        kMax = max(kValues)

        correctPredictions = np.zeros((m, len(kValues)))

        for i in range(m):
            kVotes = s.getKnn(Xtest[i], kMax)
            for j, k in enumerate(kValues):
                p = s.predict(Xtest[i], k, withWeight=withWeight, kVotes=kVotes[:k])
                correctPredictions[i,j] = (p == Ytest[i])

        accs = np.sum(correctPredictions, axis = 0)/m
        i = np.argmax(accs)
        return (kValues[i], accs)

    def errors(s, Xtest, Ytest, k, withWeight=False, returnFirstErrorId=False):
        m = Xtest.shape[0]

        if returnFirstErrorId: #for debugging purposes
            for i in range(m):
                p = s.predict(Xtest[i], k, withWeight)
                if p != Ytest[i]:
                    return i

        preds = s.predictMany(Xtest, k, withWeight)
        boolMask = Ytest != preds
        return Xtest[boolMask]

    def predictMany(s, Xs, k, withWeight=False):
        m = Xs.shape[0]
        return [s.predict(Xs[i], k, withWeight) for i in range(m)]

    def predict(s, X, k, withWeight=False, returnVotes=False, kVotes=None):
        if kVotes == None:
            kVotes = s.getKnn(X, k)

        count = s.voteCounting(kVotes, withWeight)

        v,c = max([(v,c) for (c,v) in count.items()])
        if returnVotes:
            return (c, count)
        return c


    def voteCounting(s, kVotes, withWeight=False):
        count = {}
        if withWeight:
            numVotes = len(kVotes)
            for i,v in enumerate(kVotes):
                if v in count:
                    count[v] = count[v] + numVotes-i
                else:
                    count[v] = numVotes-i
        else:
            for v in kVotes:
                if v in count:
                    count[v] = count[v] + 1
                else:
                    count[v] = 1
        return count

    def getKnn(s, X, k):
        m = s.Y.shape[0]
        X = X.reshape(1,X.shape[0])
        kMinDists = []
        kVotes = []
        d = s.X-X
        nrms = np.sum(np.square(d),axis=1)
        for i in range(m):
            nrm = nrms[i]
            for j in range(len(kMinDists)):
                if nrm < kMinDists[j]:
                    kMinDists.insert(j, nrm)
                    kVotes.insert(j,s.Y[i])
                    break
            if len(kMinDists) < k:
                kMinDists.append(nrm)
                kVotes.append(s.Y[i])
            elif len(kMinDists) > k:
                kMinDists.pop()
                kVotes.pop()
        return kVotes
