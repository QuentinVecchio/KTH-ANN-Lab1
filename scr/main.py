from NN import SingleLayerNN
from NN import Perceptron
from NN2 import SingleLayerNN2
import graph
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

N = 1000
def generateDataSet():
    X = list(np.random.multivariate_normal([-1, -1], [[1, 0], [0, 1]], N))
    X += list(np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], N))
    Y = [1] * N + [-1] * N
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
    modelSingleLayer = SingleLayerNN2()
    modelSingleLayer.fit(X, Y)

    X1, Y1 = generateDataSet()
    cumsum =  np.sum( abs(modelSingleLayer.predict(X1) - Y1)/ 2.0 )
    print(cumsum / Y1.shape[1])
    
if __name__ == '__main__':
    test()
