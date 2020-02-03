import numpy as np

class kNN():

    def __init__(s,X,Y):
        assert(len(X.shape)==2)
        assert(len(Y.shape)==1)
        s.X = X
        s.Y = Y
        s.k = None

    def optimizeK(s, Xtest, Ytest, kValues):
        accs = []
        for k in kValues:
            preds = np.array(s.predictMany(Xtest, k))
            acc = np.sum(preds == Ytest)/Ytest.shape[0]
            accs.append(acc)
        i = np.argmax(accs)
        return (kValues[i], accs)

    def errors(s, Xtest, Ytest, k):
        m = Xs.shape[0]
        boolMask = Ytest != [s.predict(Xs[i], k) for i in range(m)]
        return Xtest[boolMask]

    def predictMany(s, Xs, k):
        m = Xs.shape[0]
        return [s.predict(Xs[i], k) for i in range(m)]

    def predict(s, X, k):
        m = s.Y.shape[0]
        kMinNorms = []
        kVotes = []
        for i in range(m):
            nrm = np.linalg.norm(X - s.X[i,:])
            for j in range(len(kMinNorms)):
                if nrm < kMinNorms[j]:
                    kMinNorms.insert(j, nrm)
                    kVotes.insert(j,s.Y[i])
                    break
            if len(kMinNorms) < k:
                kMinNorms.append(nrm)
                kVotes.append(s.Y[i])
            elif len(kMinNorms) > k:
                kMinNorms.pop()
                kVotes.pop()

        count = {}
        for v in kVotes:
            if v in count:
                count[v] = count[v] + 1
            else:
                count[v] = 1

        maxV = 0
        for k,n in reversed(count.items()):
            if n > maxV:
                maxV = n
                selectedClass = k
        return k
