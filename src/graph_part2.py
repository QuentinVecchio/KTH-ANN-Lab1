import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plotRecursiveFunction(T, X, t1, t2):
    plt.plot(T[t1:t2], X[t1:t2], 'ro')
    plt.title("Mackey-Glass time series")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()

def plotPredictions(T, X, t1, t2):
    plt.plot(T[t1:t2], X[t1:t2], 'ro')
    plt.title("Mackey-Glass time series")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()
