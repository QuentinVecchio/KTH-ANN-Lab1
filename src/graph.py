import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def plotNNInformations(title, X, T, W, LearningCurve):
    X = X.T
    T = (T + 1) // 2

    x = 1
    y = 0
    bias = 2

    fig = plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    colors = ['red', 'blue']

    plt.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in T])

    #x = np.linspace(-5, 5, 50)

    ymin, ymax = plt.ylim()
    plt.title("Decision Boundary " + title)
    for w in W:
        print(w)
        a = -w[x] / w[y]
        xx = np.linspace(ymin, ymax)
        yy = a * xx - (w.item(bias)) / w[y]
        plt.plot(yy, xx, 'black')


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
    colorsNeuron = ['green', 'blue']
    ax.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in T])

    x = 1
    y = 0
    bias = 2
    lines = []

    ymin, ymax = plt.ylim()

    for index, W in enumerate(WHistory[0]):
        print(index)
        w = [W.item(x), W.item(y)]
        a = -w[x] / w[y]
        xx = np.linspace(ymin, ymax)
        yy = a * xx - (W.item(bias)) / w[y]
        line, = ax.plot(yy, xx)
        lines.append(line)

    lines.append(ax.text(0.05, 0.9, '', transform=ax.transAxes))

    def animate(i):
        lines[-1].set_text('Decision Boundary Iteration ' + str(i % len(WHistory)))
        for index, W in enumerate(WHistory[i]):
            w = [W.item(x), W.item(y)]
            a = -w[x] / w[y]
            yy = a * xx - (W.item(bias)) / w[y]
            lines[index].set_xdata(yy)

        return tuple(lines)


    def init():
        lines[-1].set_text('Decision Boundary Iteration ' + str(0))
        for index, W in enumerate(WHistory[0]):
            w = [W.item(x), W.item(y)]
            a = -w[x] / w[y]
            yy = a * xx - (W.item(bias)) / w[y]
            lines[index].set_xdata(yy)

        return tuple(lines)

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(WHistory), blit=True, interval=25)

    plt.title(title)
    plt.show()


def plotError(title, eHistory):
    plt.ylim([-0.1, 1.1])
    plt.xlim([-len(eHistory) * 0.1, len(eHistory) + len(eHistory) * 0.1])
    lines, = plt.plot(range(len(eHistory)), eHistory)
    plt.setp(lines, linewidth=2, color='r')
    plt.title(title)
    plt.show()
