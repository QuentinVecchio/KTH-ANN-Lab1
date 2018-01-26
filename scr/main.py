from NN import SingleLayerNN
from NN import Perceptron
import graph
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def generateDataSet():
    X = list(np.random.multivariate_normal([-1.5, -1.5], [[1, 0], [0, 1]], 100))
    X += list(np.random.multivariate_normal([1.5, 1.5], [[1, 0], [0, 1]], 100))
    Y = [1] * 100 + [-1] * 100
    p = np.random.permutation(len(X))
    return (np.array(X)[p]).T, np.mat(np.array(Y)[p])


def showData2D(X, Y):
    X = X.T
    Y = (Y+1)//2
    Y = (Y.tolist())[0]

    fig = plt.figure(figsize=(10, 10))
    colors = ['red', 'blue']

    plt.scatter(X[:, 0], X[:, 1], c = [colors[i] for i in Y])
    plt.show()


def test():
    # nn = NN1()
    X, Y = generateDataSet()
    modelSingleLayer = SingleLayerNN()
    WHistory = modelSingleLayer.fit(X, Y)
    X1, Y2 = generateDataSet()
    print(Y2)
    print(modelSingleLayer.predict(X1) - Y2)
    graph.plotDecisionBoundary(X,Y,WHistory)


if __name__ == '__main__':
    test()
