
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plotCost(costs):
    t = range(1,len(costs)+1)

    fig, ax = plt.subplots()
    ax.plot(t, costs)

    ax.set(xlabel='Iteration #', ylabel='Average Cost',
        title='Cost vs # of iterations')
    ax.grid()

    #fig.savefig("test.png")
    plt.show()
