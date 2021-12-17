import Plots
import DiffFreqs as DF
import SimpleOscillator as SF
import matplotlib.pyplot as plt
import Gaussian as G
import numpy as np

def calculateFCsAndEnergies(deltaE, deltaQ, w_wavenumbers, wprime_wavenumbers, widths):
    """ wprime must be greater than w
        widths is a list of desired width at half height for each peak
    """
    intensities= DF.genIntensities(deltaE, deltaQ, w_wavenumbers, wprime_wavenumbers)
    energies = DF.genEnergies(deltaE, w_wavenumbers, wprime_wavenumbers)
    return [energies, intensities]

dQ= 1
dE = 0.005
w = 501
wprime = 499

wide = [0.01]*11
med = [0.005]*11
skinny = [0.001]*11

[energies, intensities] = calculateFCsAndEnergies(0.005, dQ, w, wprime, skinny)
# L = Plots.genSpectrum(energies, intensities, skinny)

# Plots.plotSpectrum(L[0], L[1], "DeltaQ = " + str(dQ))

# for i in range(0,11):
#     print(DF.diffFreqOverlap([i, 499], [0, 501], 1))

print(DF.diffFreqOverlap([4, 498], [2, 502], d=1))

# raw_input("Press ENTER to exit ")