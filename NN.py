import numpy as np


class NN1():
    def __init__(self, hidden_layers_size=0, lr=0.1, nb_eboch=20):
        self.hidden_layers_size = hidden_layers_size
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.W = []

    def fit(self, X, Y):  # X = (len(X[0]) + 1, n)
        W = np.reshape(np.random.normal(0, 1, len(X[0]) + 1), (1, len(X[0]) + 1)) # biais
        for step in range(self.nb_eboch):
            WX = np.multiply(W, X)  # Prediction : (1, len(X[0]) + 1) * (len(X[0]) + 1, n) =(1, n)
            diff = np.multiply(WX - Y, X.T) * self.lr
            W = W + diff






