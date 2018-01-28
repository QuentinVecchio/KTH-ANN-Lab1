import numpy as np


class SingleLayerNN():
    def __init__(self, lr=0.001, nb_eboch=20, batch_size=20):
        self.batch_size = batch_size
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.W = []

    def TLU(self, Y):
        output = []
        for y in Y[0]:
            if (y > 0):
                output.append(1)
            else:
                output.append(-1)
        return output

    def fit(self, X, T):  # X = (len(X[0]) + 1, n)
        X = np.vstack([X, [1] * len(X[0])])
        WHistory = []
        eHistory = []
        self.W = np.reshape(np.random.normal(0, 0.1, len(X)), (1, len(X)))
        WHistory.append(self.W)

        for step in range(self.nb_eboch):
            p = np.random.permutation(len(X[0]))
            X = X.T[p].T
            T = T[p]
            batchIndex_list =[]
            if(self.batch_size == -1):
                batchIndex_list.append([0,len(X[0])])
            else:
                for i in range(int((len(X[0]) * 1.0) / self.batch_size)):
                    batchIndex_list.append([i * self.batch_size, (i + 1) * self.batch_size])

            for batchIndex in batchIndex_list:

                start, end = batchIndex
                batch = X.T[start : end].T
                WX = np.dot(self.W, batch)# Prediction : (1, len(X) + 1) * (len(X) + 1, n) =(1, n)
                aux = T.T[start:end].T - WX
                e = T.T[start:end].T - self.TLU(WX)
                diff = self.lr * np.dot(aux, batch.T)
                self.W = self.W + diff

                eHistory.append(np.mean(abs(e)))
                WHistory.append(self.W)

        WX = np.dot(self.W, X)
        e = T - self.TLU(WX)
        eHistory.append(np.mean(abs(e)))
        return WHistory, eHistory

    def predict(self, X):
        X = np.vstack([X, [1] * len(X[0])])
        return self.TLU(np.dot(self.W, X))


class Perceptron():
    def __init__(self, lr=0.001, nb_eboch=200, batch_size=20):
        self.batch_size = batch_size
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.W = []

    def TLU(self, Y):
        output = []
        for y in Y[0]:
            if (y > 0):
                output.append(1)
            else:
                output.append(-1)
        return output

    def fit(self, X, T):  # X = (len(X[0]) + 1, n)
        X = np.vstack([X, [1] * len(X[0])])

        WHistory = []
        eHistory = []

        self.W = np.reshape(np.random.normal(0, 0.1, len(X)), (1, len(X)))
        WHistory.append(self.W)

        for step in range(self.nb_eboch):
            p = np.random.permutation(len(X[0]))
            X = X.T[p].T
            T = T[p]
            batchIndex_list = []
            if (self.batch_size == -1):
                batchIndex_list.append([0, len(X[0])])
            else:
                for i in range(int((len(X[0]) * 1.0) / self.batch_size)):
                    batchIndex_list.append([i * self.batch_size, (i + 1) * self.batch_size])

            for batchIndex in batchIndex_list:
                start, end = batchIndex
                batch = X.T[start: end].T
                WX = np.dot(self.W, batch)  # Prediction : (1, len(X[0]) + 1) * (len(X[0]) + 1, n) =(1, n)
                e = T.T[start:end].T - self.TLU(WX)
                diff = self.lr * np.dot(e, batch.T)
                self.W = self.W + diff

                eHistory.append(np.mean(abs(e)))
                WHistory.append(self.W)
        return WHistory, eHistory

    def predict(self, X):
        X = np.vstack([X, [1] * len(X[0])])
        return self.TLU(np.dot(self.W, X))
