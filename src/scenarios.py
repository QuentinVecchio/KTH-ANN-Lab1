import NN
import NN2
import graph
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import NN2

def generateDataSet(N, V):
    X = list(np.random.multivariate_normal([V, V], [[1, 0], [0, 1]], N))
    X += list(np.random.multivariate_normal([-V, -V], [[1, 0], [0, 1]], N))
    T = [1] * N + [-1] * N
    p = np.random.permutation(len(X))
    return (np.array(X)[p]).T, np.array(T)[p]


def scenario3_1_2():
    N = 1000
    X, T = generateDataSet(N, 2)
    singleLayerNN = NN.SingleLayerNN(batch_size=-1)
    perceptron = NN.Perceptron(batch_size=-1)
    WHistorySNN, eHistorySNN = singleLayerNN.fit(X, T)
    WHistoryPer, eHistoryPer = perceptron.fit(X, T)
    graph.plotNNInformations("Single Layer NN", X,T,  WHistorySNN[-1], eHistorySNN)
    graph.plotNNInformations("Perceptron", X,T,  WHistoryPer[-1], eHistoryPer)

def scenario3_1_3():
    N = 1000
    X, T = generateDataSet(N, 1)
    singleLayerNN = NN.SingleLayerNN(batch_size=-1)
    perceptron = NN.Perceptron(batch_size=-1)
    WHistorySNN, eHistorySNN = singleLayerNN.fit(X, T)
    WHistoryPer, eHistoryPer = perceptron.fit(X, T)
    graph.plotNNInformations("Single Layer NN", X,T,  WHistorySNN[-1], eHistorySNN)
    graph.plotNNInformations("Perceptron", X,T,  WHistoryPer[-1], eHistoryPer)

def scenario3_2_1():
    N = 500
    V = 5
    X = list(np.random.multivariate_normal([V, V], [[1, 0], [0, 1]], N)) # Blue
    X += list(np.random.multivariate_normal([-V, -V], [[1, 0], [0, 1]], N)) # Blue
    X += list(np.random.multivariate_normal([V, -V], [[1, 0], [0, 1]], N)) # Red
    X += list(np.random.multivariate_normal([-V, V], [[1, 0], [0, 1]], N)) # Red
    T = [1] * 2*N + [-1] * 2*N
    p = np.random.permutation(len(X))
    X, T = (np.array(X)[p]).T, np.array(T)[p]
    multipleLayerNN = NN2.SingleLayerNN2(batch_size=-1)
    WHistoryMNN, eHistoryMNN = multipleLayerNN.fit(X, T)
    graph.plotNNInformations("Multiple Layer NN", X, T, WHistoryMNN[-1], eHistoryMNN)

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
