import NN_part2
import numpy as np
import random as rd
import graph_part2

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


def generateDataSet():
    # Generating the dataset

    T = np.arange(301, 1501)
    X = computeX(1506)
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

    networks = []
    MSE = []

    #Add networks to the list
    networks.append(NN_part2.NN("test"))

    print("Compute all networks")
    for net in networks:
        print("Train new network")
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
