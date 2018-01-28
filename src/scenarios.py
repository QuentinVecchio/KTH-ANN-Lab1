import NN
import graph
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def generateDataSet(N):
    X = list(np.random.multivariate_normal([-1, 0], [[1, 0], [0, 1]], N))
    X += list(np.random.multivariate_normal([2, 3], [[1, 0], [0, 1]], N))
    T = [1] * N + [-1] * N
    p = np.random.permutation(len(X))
    return (np.array(X)[p]).T, np.array(T)[p]


def scenario3_1_2():
    N = 100
    X, T = generateDataSet(N)
    singleLayerNN = NN.SingleLayerNN()
    perceptron = NN.Perceptron()
    WHistorySNN, eHistorySNN = singleLayerNN.fit(X, T)
    WHistoryPer, eHistoryPer = perceptron.fit(X, T)
    graph.plotNNInformations("Single Layer NN", X,T,  WHistorySNN[20], eHistorySNN)
    # graph.plotDecisionBoundary("Single Layer NN", 200, X, T, WHistorySNN[20])
    # graph.plotError("Learning Curve Single Layer NN", 100, eHistorySNN)
    # graph.plotDecisionBoundary("Decision Boundary Perceptron", 101, X, T, WHistoryPer[20])
    # graph.plotError("Learning Curve Perceptron", 102, eHistoryPer)

def test():
    X, T = generateDataSet()

    singleLayerNN = NN.SingleLayerNN()
    perceptron = NN.Perceptron()

    WHistory, eHistory = singleLayerNN.fit(X, T)
    graph.plotDecisionBoundary(X, T, WHistory[20])
    #graph.plotDecisionBoundaryAnim(X, T, WHistory)
    #graph.plotError(eHistory)

    WHistory, eHistory = perceptron.fit(X, T)
    graph.plotDecisionBoundary(X, T, WHistory[20])
    #graph.plotDecisionBoundaryAnim(X, T, WHistory)
    #graph.plotError(eHistory)
