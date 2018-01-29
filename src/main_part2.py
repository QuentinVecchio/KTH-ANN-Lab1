import NN_part2
import numpy as np
import random as rd

beta = 0.2
gamma = 0.1
n = 10
tau = 25

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

    MLP = NN_part2.NN()
    MLP.fit(trainSet,trainValue)
    out = MLP.predict(testSet)
