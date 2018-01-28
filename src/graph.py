import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def plotNNInformations(title, X, T, W, LearningCurve):
    X = X.T
    T = (T + 1) // 2

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    colors = ['red', 'blue']

    plt.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in T])

    x = np.linspace(-5, 5, 50)
    print(W)

    y = - (W[0][2] + W[0][0] * x) / W[0][1]
    plt.title("Decision Boundary " + title)
    plt.plot(x, y)


    plt.subplot(2, 1, 2)
    plt.ylim([-0.1, 1.1])
    plt.xlim([-len(LearningCurve) * 0.1, len(LearningCurve) + len(LearningCurve) * 0.1])
    lines, = plt.plot(range(len(LearningCurve)), LearningCurve)
    plt.setp(lines, linewidth=2, color='r')
    plt.title("Learning Curve  " + title)
    plt.show()

def plotDataset(X, T):
    X = X.T
    T = (T + 1) // 2

    fig = plt.figure(figsize=(10, 10))
    colors = ['red', 'blue']

    plt.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in T])
    plt.show()


def plotDecisionBoundary(title, X, T, W):
    X = X.T
    T = (T + 1) // 2

    fig = plt.figure(figsize=(10, 10))
    colors = ['red', 'blue']

    plt.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in T])

    x = np.linspace(-5, 5, 50)
    print(W)

    y = - (W[0][2] + W[0][0] * x) / W[0][1]
    plt.plot(x, y)
    plt.title(title)
    plt.show()


def plotDecisionBoundaryAnim(title, X, T, WHistory):
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

    plt.title(title)
    plt.show()


def plotError(title, eHistory):
    plt.ylim([-0.1, 1.1])
    plt.xlim([-len(eHistory) * 0.1, len(eHistory) + len(eHistory) * 0.1])
    plt.plot(range(len(eHistory)), eHistory, 'red')
    plt.scatter(range(len(eHistory)), eHistory)
    plt.title(title)
    plt.show()
