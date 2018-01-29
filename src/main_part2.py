import NN_part2
import numpy as np
import random as rd
import graph_part2
import time

beta = 0.2
gamma = 0.1
n = 10
tau = 25

X = []
T = []

def computeX(nb):
    x = [1.5]
    for i in range(nb):
        if i - tau >= 0:
            x.append(x[i] + (beta * x[i - tau])/(1 + x[i - tau]**n) - gamma * x[i])
        else:
            x.append(x[i] - gamma * x[i])
    return x


def generateDataSet(noise = False, sigma = 0):
    # Generating the dataset

    T = np.arange(301, 1501)
    X = computeX(1506)
    if noise:
        X += np.random.normal(0, sigma, len(X))
    inputs = [[X[t-20], X[t-15], X[t-10], X[t-5], X[t]] for t in T]
    outputs = [X[t+5] for t in T]

    #graph_part2.plotRecursiveFunction(T, X[301:1501], 800, 1000)

    testSet = inputs[-200:]
    testValue = outputs[-200:]

    inputs = inputs[:1000]
    outputs = outputs[:1000]

    z = list(zip(inputs, outputs))
    rd.shuffle(z)
    inputs, outputs = zip(*z)

    trainSet = inputs[:800]
    trainValue = outputs[:800]

    validationSet = inputs[-200:]
    validationValue = outputs[-200:]

    return trainSet, trainValue, testSet, testValue, validationSet, validationValue


if __name__ == '__main__':
    trainSet, trainValue, testSet, testValue, validationSet, validationValue = generateDataSet()

    # 4.3.1
    """
    networks = []
    MSE = []

    #Add networks to the list
    for lr in [0.001, 0.01, 0.1]:
        for node in range(3, 9):
            for momentum in [0, 0.7, 0.9]:
                for alpha in [0.00001, 0.0001]:
                    for early_stopping in [True, False]:
                        for learning_rate in ['constant']:
                            networks.append(NN_part2.NN("lr: "+str(lr)+" node: "+str(node)+" momentum: "+str(momentum)+" alpha: "+str(alpha)+" early_stopping: "+str(early_stopping)+ " learning_rate: "+str(learning_rate), lr = lr, hidden_states = (node,), max_ite = 1000, momentum = momentum, alpha = alpha, early_stopping = early_stopping, learning_rate = learning_rate))

    print("Compute all networks, nb: "+str(len(networks)))
    for net in networks:
        print("Train new network " + str(len(MSE)))
        net.fit(trainSet,trainValue)
        net.predict(validationSet)
        net.computeMSE(validationValue)
        MSE.append(net.getMSE())

    print("Computation over")
    print("")
    MSE, networks = (list(t) for t in zip(*sorted(zip(MSE, networks))))
    print("MSE on the validation set:")
    for net in networks:
        print(net.getName() + " with MSE: " + str(net.getMSE()))

    print("")
    print("Best network on validation set: " + networks[0].getName())
    networks[0].predict(testSet)
    networks[0].computeMSE(testValue)
    print(networks[0].getName() + " on test set: " + str(networks[0].getMSE()))

    #graph_part2.plotPredictions(networks[0].getPrediction(), testValue)
    """

    #4.3.2
    for sigma in [0.03, 0.09, 0.18]:
        networks = []
        MSE = []
        trainSet, trainValue, testSet, testValue, validationSet, validationValue = generateDataSet(noise = True, sigma = sigma)

        print("Variance of the noise: " + str(sigma))

        #Add networks to the list
        for lr in [0.01, 0.1]:
            for node in range(5, 7):
                for node2 in range(3, 9):
                    for momentum in [0.9]:
                        for alpha in [0.00001, 0.0001, 0.001]:
                            for early_stopping in [True]:
                                for learning_rate in ['constant']:
                                    networks.append(NN_part2.NN("lr: "+str(lr)+" node1: "+str(node)+" node2: "+str(node2)+" momentum: "+str(momentum)+" alpha: "+str(alpha)+" early_stopping: "+str(early_stopping)+ " learning_rate: "+str(learning_rate), lr = lr, hidden_states = (node,node2), max_ite = 1000, momentum = momentum, alpha = alpha, early_stopping = early_stopping, learning_rate = learning_rate))

        print("Compute all networks, nb: "+str(len(networks)))
        for net in networks:
            print("Train new network " + str(len(MSE)))
            net.fit(trainSet,trainValue)
            net.predict(validationSet)
            net.computeMSE(validationValue)
            MSE.append(net.getMSE())

        print("Computation over")
        print("")
        MSE, networks = (list(t) for t in zip(*sorted(zip(MSE, networks))))
        print("MSE on the validation set:")
        for net in networks:
            print(net.getName() + " with MSE: " + str(net.getMSE()))

        print("")
        print("Best network on validation set: " + networks[0].getName())
        time_start = time.clock()
        networks[0].fit(trainSet,trainValue)
        networks[0].predict(testSet)
        networks[0].computeMSE(testValue)
        time_elapsed = (time.clock() - time_start)
        print(networks[0].getName() + " on test set: " + str(networks[0].getMSE()))
        print("computation time for 2 layers: "+ str(time_elapsed))

        print("")
        net = NN_part2.NN("lr: "+str(0.1)+" node1: "+str(6)+" momentum: "+str(0.9)+" alpha: "+str(0.00001)+" early_stopping: "+str(True)+ " learning_rate: constant", lr = 0.1, hidden_states = (6,), max_ite = 1000, momentum = 0.9, alpha = 0.00001, early_stopping = True, learning_rate = "constant")
        print("Use best 1 hidden layer network")
        time_start = time.clock()
        net.fit(trainSet,trainValue)
        net.predict(testSet)
        net.computeMSE(testValue)
        time_elapsed = (time.clock() - time_start)
        print(net.getName() + " on test set: " + str(net.getMSE()))
        print("computation time for 1 layer: "+ str(time_elapsed))

        print("")
        print("")
        print("")
        print("")
        print("")
        #graph_part2.plotPredictions(networks[0].getPrediction(), testValue)
