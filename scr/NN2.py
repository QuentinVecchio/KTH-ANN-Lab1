import numpy as np
import math

class SingleLayerNN2():
    def __init__(self, lr=0.1, nb_eboch=20, hidden_layer_size=10, batch_size=20):
        self.batch_size = batch_size
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.W = []
        self.V = []
        self.hidden_layer_size = hidden_layer_size

    def phi(self, x):
        return 2.0 / (1.0 + np.exp(-x)) - 1


    def phiPrime(self,x):
        phix = self.phi(x)
        return np.multiply((1 + phix),(1 - phix)) / 2.0

    def posneg(self, WX):
        output = []
        for x in WX.T:
            if (x < 0):
                output.append(-1)
            else:
                output.append(1)
        return output

    def fit(self, X, Y):  # X = (len(X[0]) + 1, n)
        X = np.vstack([X, [1] * len(X[0])])

        eHistory = []

        self.W = np.reshape(np.random.normal(0, 1, self.hidden_layer_size * len(X)), (self.hidden_layer_size, len(X)))
        self.V = np.reshape(np.random.normal(0, 1, self.hidden_layer_size), (1, self.hidden_layer_size))

        for step in range(self.nb_eboch):
            p = np.random.permutation(len(X[0]))
            X = X.T[p].T
            Y = Y[p]
            batchIndex_list = []
            if (self.batch_size == -1):
                batchIndex_list.append([0, len(X[0])])
            else:
                for i in range(int((len(X[0]) * 1.0) / self.batch_size)):
                    batchIndex_list.append([i * self.batch_size, (i + 1) * self.batch_size])

            for batchIndex in batchIndex_list:
                start, end = batchIndex
                batch = X.T[start: end].T
                Hstar = np.dot(self.W, batch)  # size: (hls, len(X)) * (len(X), n) =(hls, n)
                H = self.phi(Hstar)  # size: (hls, n)
                Ostar = np.dot(self.V, H)  # size: (1, hls) * (hls, n) =(1, n)
                O = self.phi(Ostar)  # size: (hls, len(X)) * (len(X), n) =(1, n)
                e = self.posneg(O) - Y.T[start:end].T
                deltaO = np.multiply((O - Y.T[start:end].T),self.phiPrime(Ostar))
                deltaH = np.multiply(np.dot(self.V.T, deltaO),self.phiPrime(Hstar))
                deltaW = - self.lr * np.dot(deltaH, batch.T)
                deltaV = - self.lr * np.dot(deltaO, H.T)
                self.V += deltaV
                self.W += deltaW
                eHistory.append(np.mean(abs(e)))
        return eHistory

    def predict(self, X):
        X = np.vstack([X, [1] * len(X[0])])
        Hstar = np.dot(self.W, X)  # size: (hls, len(X)) * (len(X), n) =(hls, n)
        H = self.phi(Hstar)  # size: (hls, n)
        Ostar = np.dot(self.V, H)  # size: (1, hls) * (hls, n) =(1, n)
        O = self.phi(Ostar)  # size: (hls, len(X)) * (len(X), n) =(1, n)
        return self.posneg(O)
