import DiffFreqs as DF
import Mode as M
import itertools as it
import numpy as np

def helper(Modes, tuple):
	product = 1
	for i in range(len(Modes)):
		product *= Modes[i].FranckCondons[tuple[i]]

def nestedLoopMaker(nModes, range_ns):
	print nModes


def genMultiModeIntensities(Modes):
	""" Modes is a list of Mode-type objects (see Modes.py)
	"""
	# number of dimensions
	nModes = len(Modes)
	range_ns = 5

	# For example, if n would range from 0 to 4, and there were 3 modes, 
	# ListofNs would be [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,].
	# This is so that permutations can generate (0,0,0)
	# FALSE: GIVES REPEATS
	ListOfNs = range(range_ns)*nModes;


	# Franck-Condon factors
	FCFactors = []
	FCFactorParts = []
	for mode in Modes:
		mode.computeFranckCondons(ListOfNs)
		FCFactorParts += [mode.FrankCondons]
	
	#for x in FCFactorParts:
	#	print x

	for n in range(range_ns):
		for m in range(range_ns):
			for p in range(range_ns):
				FCFactors += [Modes[0].FrankCondons[n]* Modes[1].FrankCondons[m] * Modes[0].FrankCondons[p]]


	# use permutations in itertools
	# Each tuple in permutations is (for 3 modes) the tuple(n, m, o) 
	# such that Modes[0] is going to the nth state, Modes[1] is going
	# to the mth state, and Modes[2] is going to the oth state.
	permutations = [[0]*nModes]*range_ns

	print FCFactors
	
	#FCFactors = map(lambda x: helper(Modes, x), permutations)

	numpoints = range_ns ** (nModes)
	print
	print "Expected Numpoints", numpoints
	print "Actual numpoints", len(FCFactors)
	#intensities = map(lambda x: x**2, FCFactors)
	#return intensities

#def genMultiModeEnergies(E_el, Modes):


m1 = M.Mode(500, 450, 1)
m2 = M.Mode(300, 200, 0.1)
m3 = M.Mode(600, 500, 0.5)
Modes = [m1, m2, m3]
genMultiModeIntensities(Modes)


