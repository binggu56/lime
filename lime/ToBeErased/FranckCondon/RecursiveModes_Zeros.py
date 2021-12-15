#
# RecursiveModes_Zeros
# Instead of cutting a branch of the tree of possible
# combinations of Modes, it sets the value of all nodes
# of that branch to 0. This is slower, but is more correct.
#


import DiffFreqs as DF
import Mode as M
import itertools as it
import numpy as np
import Plots 
from Stack import Stack

class Node:
    def __init__(self, modeIndex, product, energy):
        """ modeIndex is the index of the last mode to be added to the product.
        product is the current product of FrankCondons
        state is a list of the state of each mode. E.g., if this node represented there
        the 2nd state in mode0, and the 4th state in mode 1, then state = [2,4]
        """
        self.modeIndex = modeIndex
        self.product = product
        self.energy = energy

def depthFirstSearch(threshold, Modes, values, E_electronic):
    """ Threshold is the threshold Franck-Condon after which we assign the value 0. 
        Modes is a list of modes.
        values is a list of values the modes can have (e.g. n = 0,1,2,3)

        Returns a tuple (ListOfEnergies, ListOfIntensities).
    """
    ListOfIntensites = []
    ListOfEnergies = []
    fringe = Stack()
    mode0 = Modes[0]
    print "NumModes =", len(Modes)
    for n in values:
        FC = mode0.FrankCondons[n]
        if FC >= threshold:
            energy = E_electronic + mode0.excitedEnergy(n) - mode0.groundEnergy
            node = Node(0, mode0.FrankCondons[n], energy)
            fringe.push(node)
          #  print "Mode 0, n ", n, "FC = ", FC
        #else:
          #  print "Mode 0, n ", n, "FC = ", FC, "below threshold"
    while True:
        if fringe.isEmpty():
           break
        node = fringe.pop()
        if (node.modeIndex+1) < len(Modes):
           # print "NumModes =", len(Modes), "next mode index =", (node.modeIndex+1)
            nextMode = Modes[node.modeIndex+1]
            for n in values:
                FC = nextMode.FrankCondons[n]
                energy = (nextMode.excitedEnergy(n)-
                          nextMode.groundEnergy)
                if node.product >= threshold:
                    newNode = Node(node.modeIndex+1,
                                   nextMode.FrankCondons[n]*node.product,
                                   node.energy+energy)
                    fringe.push(newNode)
                else:
                    newNode = Node(node.modeIndex+1,
                                   0,
                                   node.energy+energy)
                    fringe.push(newNode)
        else:
            
            ListOfIntensites += [node.product**2]
            ListOfEnergies += [node.energy]

    return (ListOfEnergies, ListOfIntensites)



def genMultiModePoints(threshold, Modes, E_electronic, range_ns):
    ListOfNs = range(0,range_ns)
    for mode in Modes:
        mode.computeFranckCondons(ListOfNs, threshold)
    (Energies, Intensities) = depthFirstSearch(threshold, Modes, ListOfNs, E_electronic)

    numpoints = range_ns ** (len(Modes))
    print
    print "Expected Numpoints", numpoints
    print "Actual numpoints", len(Intensities)

    return (Energies, Intensities) 
    
threshold = 10**-5

# m = M.Mode(501, 499, 1)
# Modes = [m]

m1 = M.Mode(500, 450, 1)
m2 = M.Mode(300, 200, 1)
m3 = M.Mode(600, 500, 1)
m4 = M.Mode(500, 450, 1)
m5 = M.Mode(100, 50,  1)
m6 = M.Mode(600, 250, 1)
m7 = M.Mode(500, 250, 0.5)
m8 = M.Mode(550, 250, 10)
m9 = M.Mode(550, 400, 10)
m10 = M.Mode(550, 450, 20)
m11 = M.Mode(560, 450, 2)

# m1 = M.Mode(501, 499, 1)
# m2 = M.Mode(500.1, 300, 0.1)
# m3 = M.Mode(5000.01, 5000, 0.1)

Modes = [m1, m2, m3, m10, m5, m6 ]#, m7, m8, m9, m10, m11]
E_electronic = 0.005
(energies, intensities)  = genMultiModePoints(threshold, Modes, E_electronic, 11)

wide = [0.01]*11
med = [0.005]*11
skinny = [0.001]*11

energies.reverse()
intensities.reverse()

points = Plots.genSpectrum(energies, intensities, skinny)

#Plots.plotSpectrum(points[0], points[1], "N Modes")
#raw_input("Press ENTER to exit ")
