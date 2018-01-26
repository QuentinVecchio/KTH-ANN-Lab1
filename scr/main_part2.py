import NN_2
import numpy as np

beta = 0.2
gamma = 0.1
n = 10
tau = 25

def computeX(nb):
    x = [0]
    for i in range(nb):
        if i < tau - 1:
            x.append(x[i] + (beta * x[i - tau])/(1 + x[i - tau]**n) - gamma * x[i])
        else:
            x.append(x[i] - gamma * x[i])
    return x


def generateDataSet():
    # Generating the dataset

    t = np.arange(301, 1501)
    x = computeX(1505)



if __name__ == '__main__':
    generateDataSet()
