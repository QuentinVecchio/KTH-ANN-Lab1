from sklearn.neural_network import MLPClassifier

class NN():
    def __init__(self, lr = 0.001, hidden_states = (5, ), max_ite = 200, momentum = 0.9):
        self.clf = MLPClassifier(solver='lbfgs', learning_rate_init=lr,
                     hidden_layer_sizes=hidden_states, max_iter=max_ite, momentum=momentum)
    def fit(X, Y):
        self.clf.fit(X, Y)

    def predict(Y):
        self.clf.predict(Y)
