import numpy as np
import math

class SingleLayerNN2():
    def __init__(self, lr=0.1, nb_eboch=20, hidden_layer_size=10):
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

        self.W = np.reshape(np.random.normal(0, 1, self.hidden_layer_size * len(X)), (self.hidden_layer_size, len(X)))
        self.V = np.reshape(np.random.normal(0, 1, self.hidden_layer_size), (1, self.hidden_layer_size))
        for step in range(self.nb_eboch):

            Hstar = np.dot(self.W, X)  # size: (hls, len(X)) * (len(X), n) =(hls, n)
            H = self.phi(Hstar)  # size: (hls, n)
            Ostar = np.dot(self.V, H)  # size: (1, hls) * (hls, n) =(1, n)
            O = self.phi(Ostar)  # size: (hls, len(X)) * (len(X), n) =(1, n)
            deltaO = np.multiply((O - Y),self.phiPrime(Ostar))
            deltaH = np.multiply(np.dot(self.V.T,deltaO),self.phiPrime(Hstar))
            deltaW = - self.lr * np.dot(deltaH, X.T)
            deltaV = - self.lr * np.dot(deltaO, H.T)
            self.V += deltaV
            self.W += deltaW

    def predict(self, X):
        X = np.vstack([X, [1] * len(X[0])])
        Hstar = np.dot(self.W, X)  # size: (hls, len(X)) * (len(X), n) =(hls, n)
        H = self.phi(Hstar)  # size: (hls, n)
        Ostar = np.dot(self.V, H)  # size: (1, hls) * (hls, n) =(1, n)
        O = self.phi(Ostar)  # size: (hls, len(X)) * (len(X), n) =(1, n)
        return self.posneg(O)
