from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
class NN():
    def __init__(self, lr = 0.001, hidden_states = (5, ), max_ite = 2000, momentum = 0.9):
        self.clf = MLPRegressor(solver='lbfgs', learning_rate_init=lr,
                     hidden_layer_sizes=hidden_states, max_iter=max_ite, momentum=momentum)
    def fit(self, X, Y):
        self.clf.fit(X, Y)

    def predict(self, Y):
        return self.clf.predict(Y)
