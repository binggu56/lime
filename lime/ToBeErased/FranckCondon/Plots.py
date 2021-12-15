import DiffFreqs as DF
import matplotlib as mpl
import matplotlib.pyplot as plt
import Gaussian as G
import numpy as np

plt.ion()
def genSpectrum(energies, intensities, widths):
    """ Gaussianifies the points on the spectrum using the input widths
    """

    maxE = max(energies)
    minE = min(energies)
    print("maxE", maxE)
    print("minE", minE)
    energyRange = np.linspace(minE-1000, maxE+1000, 10000)

    intensityRange = [0]*len(energyRange)
    print("Number of points to plot:", len(energyRange))

    # for i in range(len(energies)):
#         print "E: ", energies[i], " I: ", intensities[i]
    for i in range(0,len(energies)):
       # print "Gaussian for intensity i", intensities[i]
        if intensities[i]:
            gauss = G.gaussianGenerator(intensities[i], widths[i], energies[i])
            for x in range(len(energyRange)):
                intensityRange[x] += gauss(energyRange[x])

    ypoints = [gauss(x) for x in energyRange]
    #print "Intensities Gaussian"
   # print sorted(intensityRange, reverse=True)[:60]
    # print "Finished Gaussian"
    return (energyRange, intensityRange)

# def plotProbs(probs, deltaE, widths):
#     energies = L[0]
#     intensities = L[1]
#     energyRange = np.arange(0, energies[len(energies)-1]*2, energies[0]/10)
#     intensityRange = [0]*len(energyRange)
#     for x in intensities:
#         print x
#     for i in range(0,11):
#         gauss = G.gaussianGenerator(intensities[i], widths[i], energies[i])
#         for x in range(len(energyRange)):
#             intensityRange[x] += gauss(energyRange[x])

def plotSticks(xpoints, ypoints, title):
    # print "Entered plotSticks"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel("Energies")
    ax.set_ylabel("Intensities")
    ax.bar(xpoints, ypoints, 0.03) # last input is bar width
    # print "made graph"
    plt.ylim([0, max(ypoints)])
    plt.xlim([0, max(xpoints)])
    plt.draw()
   # print "draw graph"
    plt.show()
   # print "show graph"

def plotSpectrum(energies, intensities,widths, title):
    (xpoints, ypoints) = genSpectrum(energies, intensities, widths)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel("Energies")
    ax.set_ylabel("Intensities")
    p = ax.plot( xpoints, ypoints)
    
    plt.ylim([0, max(ypoints)])
    #plt.xlim([0, max(xpoints)])
    plt.draw()
    plt.show()
