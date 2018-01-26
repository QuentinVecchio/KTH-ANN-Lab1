import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def plotDataset(X, T):
    X = X.T
    T = (T + 1) // 2

    fig = plt.figure(figsize=(10, 10))
    colors = ['red', 'blue']

    plt.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in T])
    plt.show()


def plotDecisionBoundary(X, T, W):
    X = X.T
    T = (T + 1) // 2

    fig = plt.figure(figsize=(10, 10))
    colors = ['red', 'blue']

    plt.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in T])

    x = np.linspace(-5, 5, 50)
    print(W)

    y = - (W[0][2] + W[0][0] * x) / W[0][1]
    plt.plot(x, y)
    plt.show()


def plotDecisionBoundaryAnim(X, T, WHistory):
    X = X.T
    T = (T + 1) // 2

    fig, ax = plt.subplots()

    colors = ['red', 'blue']
    ax.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in T])

    x = 1
    y = 0
    bias = 2

    ymin, ymax = plt.ylim()
    w = [WHistory[0].item(x), WHistory[0].item(y)]
    a = -w[x] / w[y]
    xx = np.linspace(ymin, ymax)
    yy = a * xx - (WHistory[0].item(bias)) / w[y]
    line, = ax.plot(yy, xx, 'k-')

    ttl = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
        ttl.set_text('Decision Boundary Iteration ' + str(i % len(WHistory)))
        w = [WHistory[i].item(x), WHistory[i].item(y)]
        a = -w[x] / w[y]
        yy = a * xx - (WHistory[i].item(bias)) / w[y]
        line.set_xdata(yy)  # update the data
        return ttl, line

    def init():
        ttl.set_text('Decision Boundary Iteration ' + str(0))
        w = [WHistory[0].item(x), WHistory[0].item(y)]
        a = -w[x] / w[y]
        yy = a * xx - (WHistory[0].item(bias)) / w[y]
        line.set_xdata(yy)
        return ttl, line,

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(WHistory), blit=True)

    plt.show()


def plotError(eHistory):
    plt.ylim([-0.1, 1.1])
    plt.xlim([-len(eHistory) * 0.1, len(eHistory) + len(eHistory) * 0.1])
    plt.plot(range(len(eHistory)), eHistory, 'red')
    plt.scatter(range(len(eHistory)), eHistory)
    plt.show()
