import DiffFreqs as DF

def genMultiDimIntensities(deltaQs, w_wn, w_primes_wn):
	""" w_wn must be greater than all w_primes_wn
		deltaQs is a list of the deltaQ for each dimension
		w_primes_wn is a list of the frequencies for each dimension
	"""
	# number of dimensions
	ndims = len(w_primes_wn)

	nlevels = 10

	# Franck-Condon factors
	FCfactorParts = [[0]*ndims]*nlevels

	# row is the energy level we are going to (from ground), col is dimension
	ground = [0, w_wn]
	for row in range(nlevels):
		for col in range(ndims):
			FCfactorParts[row][col] = DF.diffFreqOverlap([row, w_primes_wn[col]], ground, deltaQs[col])
	FCFactors = map(lambda x: reduce(lambda y,z: y*z, x), FCfactorParts)
	intensities = map(lambda x: x**2, FCfactors)
	return intensities


print genMultiDimIntensities([0.2, 0.5, 1], 590, [500, 505, 510])

