# A class to represent a mode
# Contains a ground state frequency, excited state frequency,
# a shift of the normal coordinate
# groundFreqWN is in wavenumbers, while groundFreq is in atomic units for frequency

import DiffFreqs as DF
import SimpleOscillator as SO

class Mode:

    def __init__(self, nu_gs, nu_es, dQ):
        """ nu_gs and nu_es are in wavenumbers
            dQ is in bohr
        """
        self.groundFreqWN = nu_gs
        self.excitedFreqWN = nu_es
        self.groundFreq = nu_gs/8065.5/27.2116
        self.excitedFreq = nu_es/8065.5/27.2116
        self.deltaQ = dQ
        self.groundEnergy = self.groundFreqWN / 2; # In Atomic units, planck's constant is one
        self.FrankCondons = []

    def excitedEnergy(self, qNumber):
        """ Input is the quantum number of the excited state
            returns the energy of this state
        """
        return self.excitedFreqWN*(qNumber + 0.5)


    def frankCondon(self, qNumber):
        """ Input is the quantum number of the excited state
            returns the Frank-Condon factor for the transition from the ground state
        """
        if self.excitedFreqWN == self.groundFreqWN:
            return SO.sameFreqOverlap(qNumber, 0, self.groundFreqWN, self.deltaQ)
        return DF.diffFreqOverlap([qNumber, self.excitedFreqWN], [0, self.groundFreqWN],
                            	  self.deltaQ)

    def computeFranckCondons(self, ListOfNs, threshold):
        """ Computes <n|0> for each value in ListOfNs and puts them in self.FrankCondons
            Once <n|0> has past the threshold, the rest of the frankCondon list will be filled
            with 0s (saves time from calculating needlessly small numbers)
        """
        pastThreshold = False
        for n in ListOfNs:
            if pastThreshold:
                print("Mode,  n ", n, "past threshold")
                self.FrankCondons += [0]
            else:
                FC = self.frankCondon(n)
                print("Mode  n ", n, "FC = ", FC)
                if FC < threshold:
                    pastThreshold = True
                    self.FrankCondons += [0]
                else:
                    self.FrankCondons += [FC]

if __name__ == '__main__':
    dho = Mode(1000, 1000, 1)
    print(dho.frankCondon(0))
