import DiffFreqs as DF
import Mode as M
import Plots 


def genMultiModeIntensities(Modes):
	""" Modes is a list of Mode-type objects (see Modes.py)
	"""
	# number of dimensions
	nModes = len(Modes)
	range_ns = 11

	# Threshold value after which we round to 0
	threshold = 10**(-16)

	# For example, if n would range from 0 to 4, and there were 3 modes, 
	# ListofNs would be [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,].
	# This is so that permutations can generate (0,0,0)
	# FALSE: GIVES REPEATS
	ListOfNs = range(range_ns)*nModes;


	# Franck-Condon factors
	FCFactors = []

	# FCFactorInputs will track the states for which FCFactors were calculated
	# (must be tracked because not all states are calculated because of the
    # threshold.)
	states = []
	for mode in Modes:
		mode.computeFranckCondons(ListOfNs, threshold)

	listOfNs = range(range_ns)
	for n in listOfNs:
		FCn = Modes[0].FrankCondons[n]
		if FCn < threshold:
			break
		for m in listOfNs:
			FCm = Modes[1].FrankCondons[m]
			if FCm < threshold:
				break
			for p in listOfNs:
				FCp = Modes[2].FrankCondons[p]
				if FCp < threshold:
					break
				FCFactors += [FCn*FCm*FCp]
				#print "Multiplying ", FCn, FCm, FCp, "to get the square: ", FCn*FCm*FCp
				states += [(n,m,p)]

	#print FCFactors
	
	#FCFactors = map(lambda x: helper(Modes, x), permutations)

	numpoints = range_ns ** (nModes)
	print
	print "Expected Numpoints", numpoints
	print "Actual numpoints", len(FCFactors)
	intensities = map(lambda x: x**2, FCFactors)
	return [intensities, states]

def genMultiModeEnergies(E_el, Modes, states):
	Eground = 0
	for mode in Modes:
		Eground += mode.groundEnergy

	energies = [E_el - Eground]*len(states)
		
	for i in range(len(states)):
		state = states[i]
		for j in range(len(state)):
			energies[i] += Modes[j].excitedEnergy(state[j])
	return energies

m1 = M.Mode(501, 499, 1)
m2 = M.Mode(500.1, 300, 0.1)
m3 = M.Mode(5000.01, 5000, 0.1)

# m1 = M.Mode(500, 450, 1)
# m2 = M.Mode(300, 200, 20)
# m3 = M.Mode(600, 500, 0.5)

Modes = [m1, m2, m3]
E_electronic = 0.005
[intensities, states] = genMultiModeIntensities(Modes)
energies = genMultiModeEnergies(E_electronic, Modes, states)

wide = [0.01]*11
med = [0.005]*11
skinny = [0.001]*11

points = Plots.genSpectrum(energies, intensities, med)
Plots.plotSpectrum(points[0], points[1], "3 Modes")
raw_input("Press ENTER to exit ")
