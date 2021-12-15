import DiffFreqs as DF
import Mode as M
import itertools as it
import numpy as np
import pp # Parallel Python

import Stack

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
    ListOfIntensities = []
    ListOfEnergies = []
    jobserver = pp.Server()
    mode0 = Modes[0]
    jobs = [0]*len(values) # will store jobs
    print "NumModes =", len(Modes)
    for n in values:
        FC = mode0.FrankCondons[n]
        if FC >= threshold:
            energy = E_electronic + mode0.excitedEnergy(n) - mode0.groundEnergy
            node = Node(0, mode0.FrankCondons[n], energy)
            jobs[n] = jobserver.submit(oneThread,
                                       (node, threshold, Modes, values, E_electronic),
                                       )
    results = [f() for f in jobs]        
    map(ListOfEnergies.append, [es for (es, ins) in results])
    map(ListOfIntensities.append, [ins for (es, ins) in results])
    return (ListOfEnergies, ListOfIntensities)
    
def oneThread(initNode, threshold, Modes, values, E_electronic):
    print "Calculating N", initNode.energy
    fringe = []
    fringe.append(initNode)
    intensities = []
    energies = []
    
    while True:
        if fringe == []:
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
                    fringe.append(newNode)
        else:          
            intensities += [node.product**2]
            energies += [node.energy]

    return (energies, intensities)



def genMultiModePoints(threshold, Modes, E_electronic, range_ns):
    ListOfNs = range(0,range_ns)
    for mode in Modes:
        mode.computeFranckCondons(ListOfNs, threshold)
    (Energies, Intensities) = depthFirstSearch(threshold, Modes, ListOfNs, E_electronic)

    EXPnumpoints = range_ns ** (len(Modes))
    print
    print "Expected Numpoints", EXPnumpoints
    print "Actual numpoints", len(Intensities)

    return (Energies, Intensities, len(Intensities)) 
    
# threshold = 10**-2

# # m = M.Mode(501, 499, 1)
# # Modes = [m]
# mtest = M.Mode(600, 500, 8.53896647)
# m1 = M.Mode(500, 450, 10)
# m2 = M.Mode(300, 200, 10)
# m3 = M.Mode(600, 500, 10)
# m4 = M.Mode(500, 450, 1)
# m5 = M.Mode(100, 50,  1)
# m6 = M.Mode(600, 250, 1)
# m7 = M.Mode(500, 250, 0.5)
# m8 = M.Mode(550, 250, 10)
# m9 = M.Mode(550, 400, 10)
# m10 = M.Mode(550, 450, 20)
# m11 = M.Mode(560, 450, 2)

# m1 = M.Mode(501, 499, 1)
# m2 = M.Mode(500.1, 300, 0.1)
# m3 = M.Mode(5000.01, 5000, 0.1)

# Modes = [mtest] #[m1, m2, m3, m10, m5]#, m6 ]#, m7, m8, m9, m10, m11]
# E_electronic = 0.005
# (energies, intensities, numpoints)  = genMultiModePoints(threshold, Modes, E_electronic, 11)

# print "E len", len(energies), "I len", len(intensities)

# wide = [0.01]*numpoints
# med = [0.005]*numpoints
# skinny = [0.001]*numpoints

# energies.reverse()
# intensities.reverse()

# #get the gaussians
# #points = Plots.genSpectrum(energies, intensities, skinny)

# points = [energies, intensities]
# # for x in range(len(energies)):
# #     print "E: ", energies[x], " I: ", intensities[x]x
# #Plots.plotSpectrum(points[0], points[1], "N Modes")
# Plots.plotSticks(points[0], points[1], "N Modes")
# raw_input("Press ENTER to exit ")
