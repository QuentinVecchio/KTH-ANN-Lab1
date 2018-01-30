from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import numpy as np

class NN():
    def __init__(self, name, lr = 0.01, hidden_states = (5,), max_ite = 1000, momentum = 0.9, alpha = 0.01, early_stopping = "false", learning_rate = 'invscaling'):
        self.name = name
        self.prediction = []
        self.clf = MLPRegressor(solver='sgd', learning_rate_init=lr,
                     hidden_layer_sizes=hidden_states, max_iter=max_ite, momentum=momentum, alpha=alpha, early_stopping=early_stopping, learning_rate=learning_rate)

    def fit(self, X, Y):
        self.clf.fit(X, Y)

    def predict(self, Y):
        self.prediction = self.clf.predict(Y)
        return self.prediction

    def getName(self):
        return self.name

    def computeMSE(self, Y):
        self.MSE = np.mean((self.prediction - Y)**2)

    def getMSE(self):
        return self.MSE

    def getPrediction(self):
        return self.prediction
