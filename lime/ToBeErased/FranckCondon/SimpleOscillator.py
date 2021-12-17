import numpy as np
import sympy as sym
#import math
from DiffFreqs import factorial
from scipy.special import genlaguerre
# from mpmath import laguerre

def laguerre(a, n):
    """
        Generates the nth laguerre polynomial with superscript a
        using the Rodrigues formula.
    """
    # function = lambda x: ((x ** -a)*(math.exp(x))/(math.factorial(n))) #incomplete
    x = sym.Symbol('x')
    subFunction = (sym.exp(-x) * (x ** (a+n))).diff(x, n)
    L = ((x ** -a)*(sym.exp(x))/(sym.factorial(n))) * subFunction
    L = sym.simplify(L)
    #print "Laguerre", L
    l = sym.lambdify(x, L)
    return l


def sameFreqOverlap(n, m, w_wavenumbers, deltaQ):
    """
        n must be greater than m, w_wavenumbers is the frequency in
        wavenumbers.
        deltaQ is the change in normal coordinate in bohr.

        Since sameFreqOverlap will only be called with m as the ground
        state (m=0 -> 0th laguerre), we only need the 0th laguerre, which
        is equal to 1 for all a, so I have removed the call to the laguerre
        function.
    """
    w = w_wavenumbers/8065.5/27.2116
    # F is the (massless) force constant for the mode
    F = w ** 2

    # Convert dQ from amu to multiples of electron mass:
    # convertedQSquared = deltaQ**2/(6.0221413*(10**23) * 9.10938291*(10**-28))
    convertedQSquared = deltaQ**2
    # X is defined as such in Siders, Marcus 1981
    X = (F * (convertedQSquared)) / ( 2 * w)
    #print "X is", X
    L = laguerre(n-m, m)
    # L = genlaguerre(m, alpha=n-m)
    #L = lambda x : x
    exp1 = (float(n)-m)/2
    exp2 = -X/float(2)
    #print "L(X)", L(X)
    P = (X**(exp1) * (factorial(m)/float(factorial(n)))**0.5 * np.exp(exp2) *
          L(X))
    
    # customized
    # P = (X**(exp1) * (factorial(m)/float(factorial(n)))**0.5 * np.exp(exp2) *
    #       laguerre(n-m, m, X))
    return P

if __name__ == '__main__':
    print(sameFreqOverlap(4, 2, 500, 1))
    