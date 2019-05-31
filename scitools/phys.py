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

def commutator(A,B):
    assert(A.shape == B.shape)
    return A.dot(B) - B.dot(A)


def anticommutator(A,B):
    assert(A.shape == B.shape)
    return A.dot(B) + B.dot(A)

def dagger(a):
    return a.conjugate().transpose()

def coth(x):
    return 1./np.tanh(x)

def pauliz():
     return np.array([[1.0,0.0],[0.0,-1.0]], dtype=np.complex128)

def paulix():
    return np.array([[0.0,1.0],[1.0,0.0]], dtype=np.complex128)

def pauliy():
    return np.array([[0.0,-1j],[1j,0.0]], dtype=np.complex128)

# spin-half matrices 
sz = np.array([[0.5,0.0],[0.0,-0.5]])

sx = np.array([[0.0,1.0],[1.0,0.0]])

sy = np.array([[0.0,-1j],[1j,0.0]]) 

s0 = np.identity(2)