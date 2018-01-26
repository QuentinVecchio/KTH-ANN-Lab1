import numpy as np


class SingleLayerNN():
    def __init__(self, lr=0.001, nb_eboch=400):
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.W = []

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
        WHistory = []
        self.W = np.reshape(np.random.normal(0, 1, len(X)), (1, len(X)))
        for step in range(self.nb_eboch):
            WX = np.dot(self.W, X) # Prediction : (1, len(X[0]) + 1) * (len(X[0]) + 1, n) =(1, n)
            aux = WX - Y
            diff = - self.lr * np.dot(aux, X.T)
            self.W = self.W + diff
            WHistory.append(self.W)
        return WHistory

    def predict(self, X):
        X = np.vstack([X, [1] * len(X[0])])
        return self.posneg(np.dot(self.W, X))


class Perceptron():
    def __init__(self, lr=0.001, nb_eboch=400):
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.W = []

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
        WHistory = []
        self.W = np.reshape(np.random.normal(0, 1, len(X)), (1, len(X)))
        for step in range(self.nb_eboch):
            WX = self.posneg(np.dot(self.W, X)) # Prediction : (1, len(X[0]) + 1) * (len(X[0]) + 1, n) =(1, n)
            signe =  - (WX - Y) / 2

            diff = self.lr * np.dot(X, signe.T).T
            self.W = self.W + diff
            WHistory.append(self.W)
        return WHistory

    def predict(self, X):
        X = np.vstack([X, [1] * len(X[0])])
        return self.posneg(np.dot(self.W, X))
