from Hermite import iHermite

from math import factorial
from math import sqrt, exp
from scipy.special import hermite

def diffFreqOverlap(Ln, Lm, d):
    """ Ln and Lm are lists where L[0] is the state number, L[1]
        is the frequency in wavenumbers.
        d is the change in normal coordinate (in bohr)
    """

    # If the excited state frequency (Ln[1]) is greater than the ground state
    # frequency (Lm[1]) then we must swap Ln and Lm for the program, but then
    # take the absolute value of the result.
    if (Ln[1] > Lm[1]):
        Ln, Lm = Lm, Ln

    n = Ln[0]
    m = Lm[0]
    wn_wavenumbers = Ln[1]
    wm_wavenumbers = Lm[1]

    wn = wn_wavenumbers/8065.5/27.2116
    wm = wm_wavenumbers/8065.5/27.2116
    f = wn/wm
    # w = wm

    # F is the (massless) force constant for the mode. But which w?
    # F = w ** 2

    #convertedQSquared = deltaQ**2/(6.02214*(10**23) * 9.1094*(10**-28))
    convertedQSquared = d**2



    # X is defined as such in Siders, Marcus 1981 Average frequency?
    X = convertedQSquared / 2
    P0 = (-1)**(m+n) # Should data be alternating plus minus?

    P1 = sqrt(2*sqrt(f)/(1.0+f))
    P2 = ((2**(m+n) * factorial(m) * factorial(n))**(-0.5)
          * exp(-X*f/(1.0+f)))
    P3 = (abs((1-f)/(1+f)))**((float(m)+n)/2.0)

    l = min(m,n)
    P4 = 0
    for i in range(l+1):
        #In Siders and Marcus's 'F_n(x)' is equal to my G(x)
        G = iHermite(n-i)
        # G = hermite(n-i)
        H = hermite(m-i)
        P4 += ((factorial(m)*factorial(n) /
                (float(factorial(i))*float(factorial(m-i))*float(factorial(n-i)))) *
               (4*sqrt(f)/abs(1-f))**i
               * H(sqrt(abs(2*X*f/(1-f**2))))
               * G(f*sqrt(abs(2*X/(1-f**2))))
               )
    fc = P0*P1*P2*P3*P4
    return fc

def genIntensities( deltaE, deltaQ, w_wavenumbers, wprime_wavenumbers):
    """ wprime must be greater than w"""
    wprime = wprime_wavenumbers/8065.5/27.2116
    w = w_wavenumbers/8065.5/27.2116
    intensityFunction = lambda n: (diffFreqOverlap([n, wprime_wavenumbers], [0, w_wavenumbers], deltaQ))**2
    intensities = map(intensityFunction, range(0,11))
    return intensities

def genEnergies(deltaE, w_wavenumbers, wprime_wavenumbers):
    wprime = wprime_wavenumbers/8065.5/27.2116
    w = w_wavenumbers/8065.5/27.2116
    energyFunction = lambda n: (deltaE + (n+0.5)*(wprime) - 0.5*w)
    energies = map(energyFunction, range(0, 11))
    return energies


if __name__ == '__main__':
    fc = diffFreqOverlap([2, 499], [4, 501], d=1)
    print(fc)