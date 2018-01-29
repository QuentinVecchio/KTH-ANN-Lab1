import numpy as np
import math
import copy

class MultipleLayer():
    def __init__(self, lr=0.0001, nb_eboch=5, hidden_layer_size=2, batch_size=200):
        self.batch_size = batch_size
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.W = []
        self.V = []
        self.hidden_layer_size = hidden_layer_size

    def phi(self, x):
        return 2.0 / (1.0 + np.exp(-x)) - 1.0


    def phiPrime(self,x):
        return np.multiply((1.0 + x),(1.0 - x)) / 2.0

    def posneg(self, WX):
        output = []
        for t in WX.T:
            out = []
            for x in t:
                if (x < 0):
                    out.append(-1)
                else:
                    out.append(1)
            output.append(out)
        return output

    def fit(self, X, Y):  # X = (len(X[0]) + 1, n)
        X = np.vstack([X, [1.0] * len(X[0])])
        WHistory = []
        eHistory = []

        self.W = np.reshape(np.random.normal(0, 1, self.hidden_layer_size * len(X)), (self.hidden_layer_size, len(X)))
        self.V = np.reshape(np.random.normal(0, 1, (self.hidden_layer_size+1) * Y.shape[0]), (Y.shape[0], self.hidden_layer_size+1))

        for step in range(self.nb_eboch):
            # p = np.random.permutation(len(X[0]))
            # X = X.T[p].T
            # Y = Y[p]
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
                H = np.vstack([H, [1.0] * len(H[0])]) # Add bias
                Ostar = np.dot(self.V, H)  # size: (size Output, hls+bias) * (hls+bias, n) =(size Output, n)
                O = self.phi(Ostar)  # size: (hls, len(X)) * (len(X), n) =(1, n)
                e = self.posneg(O) - Y.T[start:end].T
                # print(np.mean(O - Y.T[start:end].T))
                deltaO = np.multiply((O - Y.T[start:end].T),self.phiPrime(O))
                deltaH = np.multiply(np.dot(self.V.T, deltaO),self.phiPrime(H))
                deltaH = deltaH[:-1,:]# Remove Bias row
                deltaW = - self.lr * np.dot(deltaH, batch.T)
                deltaV = - self.lr * np.dot(deltaO, H.T)
                self.V += deltaV
                self.W += deltaW
                WHistory.append(copy.copy(self.W))
                eHistory.append(np.mean(abs(e/2.0)))

        return WHistory, eHistory

    def predict(self, X):
        X = np.vstack([X, [1] * len(X[0])])
        Hstar = np.dot(self.W, X)  # size: (hls, len(X)) * (len(X), n) =(hls, n)
        H = self.phi(Hstar)  # size: (hls, n)
        H = np.vstack([H, [1] * len(H[0])]) # Add bias
        Ostar = np.dot(self.V, H)  # size: (1, hls) * (hls, n) =(1, n)
        O = self.phi(Ostar)  # size: (hls, len(X)) * (len(X), n) =(1, n)
        return self.posneg(O)
