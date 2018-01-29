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
    N = 100
    V = 5
    X = list(np.random.multivariate_normal([V, V], [[1, 0], [0, 1]], N)) # Blue
    X += list(np.random.multivariate_normal([-V, -V], [[1, 0], [0, 1]], N)) # Blue
    X += list(np.random.multivariate_normal([V, -V], [[1, 0], [0, 1]], N)) # Red
    X += list(np.random.multivariate_normal([-V, V], [[1, 0], [0, 1]], N)) # Red
    T = [1] * 2*N + [-1] * 2*N
    p = np.random.permutation(len(X))
    X, T = (np.array(X)[p]).T, np.array(T)[p]
    multipleLayerNN = NN2.MultipleLayer(batch_size=-1, nb_eboch=1000, lr=0.01, hidden_layer_size=2)
    WHistoryMNN, eHistoryMNN = multipleLayerNN.fit(X, T)
    graph.plotNNInformations("Multiple Layer NN", X, T, WHistoryMNN[-1], eHistoryMNN)
    graph.plotDecisionBoundaryAnim("Multiple Layer NN Anim", X, T, WHistoryMNN)

def scenario3_2_2():
    N = 8
    H = 3
    X = np.ones((N, N)) * -1
    indices_diagonal = np.diag_indices(N)
    X[indices_diagonal] = 1
    multipleLayerNN = NN2.MultipleLayer(hidden_layer_size=H, batch_size=-1, nb_eboch=10000, lr=0.01)
    WHistory, eHistory = multipleLayerNN.fit(X, X)
    predict = multipleLayerNN.predict(X)
    for i,p in enumerate(predict):
        print("Encoder predicts : ")
        print(p)
        print("Good answer was : ")
        print(X[i])
        if np.array_equal(X[i],p):
            print("GOOD!!!")
        else:
            print("FAIL!!!")
        print("--------------------------")
    graph.plotError("Encoder Learning Curve", eHistory)

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
