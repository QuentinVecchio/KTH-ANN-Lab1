import numpy as np


class SingleLayerNN():
    def __init__(self, hidden_layers_size=0, lr=0.1, nb_eboch=20):
        self.hidden_layers_size = hidden_layers_size
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.W = []

    def fit(self, X, Y):  # X = (len(X[0]) + 1, n)
        X = np.vstatck([X, [1]*len(X[0])])
        W = np.reshape(np.random.normal(0, 1, len(X[0])), (1, len(X[0])))
        for step in range(self.nb_eboch):
            WX = np.multiply(W, X)  # Prediction : (1, len(X[0]) + 1) * (len(X[0]) + 1, n) =(1, n)
            diff = self.lr * np.multiply(WX - Y, X.T)
            W = W + diff

    def train(self, X):
        return np.multiply(W, X)[:-1]

class Perceptron():
    def __init__(self, lr=0.1, nb_eboch=20):
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.W = []

    def fit(self, X, Y):  # X = (len(X[0]) + 1, n)
        X = np.vstatck([X, [1]*len(X[0])])
        W = np.reshape(np.random.normal(0, 1, len(X[0])), (1, len(X[0])))
        for step in range(self.nb_eboch):
            WX = np.multiply(W, X)  # Prediction : (1, len(X[0]) + 1) * (len(X[0]) + 1, n) =(1, n)
            diff = self.lr * (Y - WX) * X
            W = W + diff

    def train(self, X):
        return np.multiply(W, X)[:-1]
