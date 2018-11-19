import numpy as np

def fermi(E, Ef = 0.0, T = 1e-4):
    """
    Fermi-Dirac distribution function
    INPUT:
        E : Energy
        Ef : Fermi energy
        T : temperture (in units of energy, i.e., kT)
    OUTPUT:
        f(E): Fermi-Dirac distribution function at energy E

    """
#    if E > Ef:
#        return 0.0
#    else:
#        return 1.0
    #else:
    return 1./(1. + np.exp((E-Ef)/T))


def heaviside(x):
    return 0.5 * (np.sign(x) + 1)