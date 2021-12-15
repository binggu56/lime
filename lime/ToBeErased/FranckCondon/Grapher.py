import matplotlib.pyplot as plt


def makeOverlapGraph(ListOfLns, Lm, f, extraTitle):
    """ Returns a figure containing a plot of f(n,m).
        ListOfLns is a list of lists Ln such that Ln[0] = n, Ln[1] = wn_wavenumbers
        Lm is a list such that Lm such that Lm[0] = m, Lm[1] - wm_wavenumbers
        f is the function to be plotted that takes Ln and Lm as inputs
        extraTitle is a string of extra text to be included in the graph's title
    """
    xpoints = [L[0] for L in ListOfLns] # Getting a list of n values
    f_OneInput = lambda Ln : f(Ln, Lm)
    ypoints = map(f_OneInput, ListOfLns)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("<n|"+ str(Lm[0])+ "> "+ extraTitle)
    ax.set_xlabel("n")
    ax.set_ylabel("<n|"+ str(Lm[0])+ ">")
    p = ax.plot(xpoints, ypoints, 'b')
    return fig

def makeOverlapFig(startN, endN, m, wn, wm, f):
    ListOfLns = []
    for n in range(startN, endN+1):
        ListOfLns += [[n, wn]]
    Lm = [m, wm]
    fig = makeOverlapGraph(ListOfLns, Lm, f, "approximating with wn ="+ str(wn)+", wm ="+ str(wm))
    return fig
