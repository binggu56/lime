import math

def gaussian(x, mu, sigma):
    """ mu will be the 'x' location of the the peak, sigma is width
    """
    exponent = -(x-float(mu))**2/(2*sigma**2)
    coef = sigma*math.sqrt(2*math.pi)
    return math.exp(exponent)/coef 

def gaussianGenerator(A, y, dX):
	""" y is the width of the peak at half height, A is the peak height,
		dX is the x location of the peak
	"""
	alpha = math.log(2)*8/(y**2)
	coef = A #*((alpha/math.pi)**0.25)
	return lambda x: coef*math.exp((-alpha*(x-dX)**2)/2)



