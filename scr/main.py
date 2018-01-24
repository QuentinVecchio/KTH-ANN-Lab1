from NN import NN1
import numpy as np
def generateDataSet():
    X = list(np.random.multivariate_normal([-3, -3], [[1, 0], [0, 1]], 1000))
    X += list(np.random.multivariate_normal([3, 3], [[1, 0], [0, 1]], 1000))
    Y = [1] * 1000 + [-1] * 1000
    p = np.random.permutation(len(X))
    return np.array(X)[p], np.array(Y)[p]
def test():
    #nn = NN1()
    X, Y = generateDataSet()
    print(Y)
if __name__ == '__main__':
    test()