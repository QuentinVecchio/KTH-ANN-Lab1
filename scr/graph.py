import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plotDecisionBoundary(X, Y, WHistory):
    print(WHistory)
    print(WHistory[0].item(0))
    X = X.T
    Y = (Y+1)//2
    Y = (Y.tolist())[0]

    fig, ax = plt.subplots()

    colors = ['red', 'blue']
    ax.scatter(X[:, 0], X[:, 1], c = [colors[i] for i in Y])

    ymin, ymax = plt.ylim()
    w = [WHistory[0].item(0), WHistory[0].item(1)]
    a = -w[0] / w[1]
    xx = np.linspace(ymin, ymax)
    yy = a * xx - (WHistory[0].item(2)) / w[1]
    line, = ax.plot(yy,xx, 'k-')

    ttl = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
        ttl.set_text('Decision Boundary Iteration ' + str(i%len(WHistory)))
        w = [WHistory[i].item(0), WHistory[i].item(1)]
        a = -w[0] / w[1]
        yy = a * xx - (WHistory[i].item(2)) / w[1]
        line.set_ydata(yy)  # update the data
        return ttl,line

    def init():
        ttl.set_text('Decision Boundary Iteration ' + str(0))
        w = [WHistory[0].item(0), WHistory[0].item(1)]
        a = -w[0] / w[1]
        yy = a * xx - (WHistory[0].item(2)) / w[1]
        line.set_ydata(yy)
        return ttl,line,

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(WHistory), blit=True)

    plt.show()
