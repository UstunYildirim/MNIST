import numpy as np

class kNN():

    def __init__(s,X,Y):
        assert(len(X.shape)==2)
        assert(len(Y.shape)==1)
        s.X = X
        s.Y = Y
        s.k = None

    def optimizeK(s, Xtest, Ytest, kValues, withWeight=False):
        accs = []
        for k in kValues:
            preds = np.array(s.predictMany(Xtest, k, withWeight))
            acc = np.sum(preds == Ytest)/Ytest.shape[0]
            accs.append(acc)
        i = np.argmax(accs)
        return (kValues[i], accs)

    def errors(s, Xtest, Ytest, k, withWeight=False):
        m = Xtest.shape[0]
        preds = s.predictMany(Xtest, k, withWeight)
        boolMask = Ytest != preds
        return Xtest[boolMask]

    def predictMany(s, Xs, k, withWeight=False):
        m = Xs.shape[0]
        return [s.predict(Xs[i], k, withWeight) for i in range(m)]

    def predict(s, X, k, withWeight=False, returnVotes=False):
        m = s.Y.shape[0]
        kMinDists = []
        kVotes = []
        for i in range(m):
            nrm = np.linalg.norm(X - s.X[i,:])
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


        maxV = 0
        for k,n in count.items():
            if n > maxV:
                maxV = n
                selectedClass = k
        if returnVotes:
            return (k, count)
        return k
