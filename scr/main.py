from NN import NN1
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def generateDataSet():
    X = list(np.random.multivariate_normal([-3, -3], [[1, 0], [0, 1]], 1000))
    X += list(np.random.multivariate_normal([3, 3], [[1, 0], [0, 1]], 1000))
    Y = [1] * 1000 + [-1] * 1000
    p = np.random.permutation(len(X))
    return np.array(X)[p], np.array(Y)[p]
def showData2D(X,Y):
    fig = plt.figure(figsize=(10, 10))
    colors = ['red', 'blue']
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()

def test():
    # nn = NN1()
    X, Y = generateDataSet()
    showData2D(X,Y)

if __name__ == '__main__':
    test()
