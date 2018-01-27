import NN
import graph
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import NN2

N = 100


def generateDataSet():
    X = list(np.random.multivariate_normal([-1, 0], [[1, 0], [0, 1]], N))
    X += list(np.random.multivariate_normal([2, 3], [[1, 0], [0, 1]], N))
    T = [1] * N + [-1] * N
    p = np.random.permutation(len(X))
    return (np.array(X)[p]).T, np.array(T)[p]


def test():
    X, T = generateDataSet()

    #singleLayerNN = NN.SingleLayerNN()
    #perceptron = NN.Perceptron()
    singleLayerNN =NN2.SingleLayerNN2()

    #WHistory, eHistory = singleLayerNN.fit(X, T)
    #graph.plotDecisionBoundary(X, T, WHistory[-1])
    #graph.plotDecisionBoundaryAnim(X, T, WHistory)
    #graph.plotError(eHistory)

    #WHistory, eHistory = perceptron.fit(X, T)
    #graph.plotDecisionBoundary(X, T, WHistory[-1])
    #graph.plotDecisionBoundaryAnim(X, T, WHistory)
    #graph.plotError(eHistory)

    eHistory = singleLayerNN.fit(X, T)
    graph.plotError(eHistory)
if __name__ == '__main__':
    test()
