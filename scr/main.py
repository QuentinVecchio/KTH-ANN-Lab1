from NN import SingleLayerNN
from NN import Perceptron
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def generateDataSet():
    X = list(np.random.multivariate_normal([-3, -3], [[1, 0], [0, 1]], 100))
    X += list(np.random.multivariate_normal([3, 3], [[1, 0], [0, 1]], 100))
    Y = [1] * 100 + [-1] * 100
    p = np.random.permutation(len(X))
    return np.array(X)[p], np.array(Y)[p]


def showData2D(X, Y):
    fig = plt.figure(figsize=(10, 10))
    colors = ['red', 'blue']
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()


def test():
    # nn = NN1()
    X, Y = generateDataSet()
    modelSingleLayer = Perceptron()
    modelSingleLayer.fit(X.T, Y.T)
    X1, Y2 = generateDataSet()
    print(Y2)
    print(modelSingleLayer.predict(X1.T) - Y2)



if __name__ == '__main__':
    test()
