import numpy as np


class SingleLayerNN():
    def __init__(self, hidden_layers_size=0, lr=0.1, nb_eboch=20):
        self.hidden_layers_size = hidden_layers_size
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
        self.W = np.reshape(np.random.normal(0, 1, len(X)), (1, len(X)))
        for step in range(self.nb_eboch):
            WX = np.matmul(self.W, X)  # Prediction : (1, len(X)) * (len(X), n) =(1, n)
            WX = self.posneg(WX)
            aux = WX - Y
            diff = self.lr * np.matmul(aux, X.T)
            self.W = self.W - diff

    def predict(self, X):
        X = np.vstack([X, [1] * len(X[0])])
        return self.posneg(np.matmul(self.W, X))


class Perceptron():
    def __init__(self, lr=0.1, nb_eboch=20):
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
        self.W = np.reshape(np.random.normal(0, 1, len(X[0])), (1, len(X[0])))
        for step in range(self.nb_eboch):
            WX = self.posneg(np.matmul(self.W, X))  # Prediction : (1, len(X[0]) + 1) * (len(X[0]) + 1, n) =(1, n)
            diff = self.lr * (Y - WX) * X
            self.W = self.W + diff

    def predict(self, X):
        X = np.vstack([X, [1] * len(X[0])])
        return np.matmul(self.W, X)
