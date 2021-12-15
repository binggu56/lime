import matplotlib.pyplot as plt
import SimpleOscillator as so

def generatePoints(startN, endN, m):
    xpoints = range(startN, endN)
    ypoints = [so.overlap(n, m) for n in range(startN, endN)]
    return [xpoints, ypoints]

def plotPoints(xpoints, ypoints):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p = ax.plot(xpoints, ypoints, 'b')
    return fig

L = generatePoints(0, 10, 0)
fig = plotPoints(L[0], L[1])
fig.show()
