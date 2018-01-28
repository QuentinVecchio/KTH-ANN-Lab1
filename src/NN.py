import numpy as np

class SingleLayerNN():
    def __init__(self, lr=0.001, nb_eboch=200):
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
            # Prediction : (1, len(X[0]) + 1) * (len(X[0]) + 1, n) =(1, n)
            WX = np.dot(self.W, X)

            aux = T - WX
            e = T - self.TLU(WX)

            diff = self.lr * np.dot(aux, X.T)
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
    def __init__(self, lr=0.001, nb_eboch=200):
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
            # Prediction : (1, len(X[0]) + 1) * (len(X[0]) + 1, n) =(1, n)
            WX = np.dot(self.W, X)

            e = T - self.TLU(WX)

            diff = self.lr * np.dot(e, X.T)
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
