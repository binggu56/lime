import numpy as np
from scipy.sparse import csr_matrix
import numba 

    
def rk4(rho, fun, dt, *args):
    """
    Runge-Kutta method
    """
    dt2 = dt/2.0

    k1 = fun(rho, *args )
    k2 = fun(rho + k1*dt2, *args)
    k3 = fun(rho + k2*dt2, *args)
    k4 = fun(rho + k3*dt, *args)

    rho += (k1 + 2*k2 + 2*k3 + k4)/6. * dt

    return rho

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

@numba.autojit
def transform_basis(A, v):
    """
    transformation rule: A_{ab} = <a|i><i|A|j><j|b> = Anew = v^\dag A v
    input:
        A: matrix of operator A in old basis
        v: basis transformation matrix
    output:
        Anew: matrix A in the new basis
    """
    Anew = dag(v).dot(A.dot(v))
    Anew = csr_matrix(A)

    return Anew

def heaviside(x):
    return 0.5 * (np.sign(x) + 1)

def commutator(A,B):
    assert(A.shape == B.shape)
    return A.dot(B) - B.dot(A)

def comm(A,B):
    assert(A.shape == B.shape)
    return A.dot(B) - B.dot(A)

def anticomm(A,B):
    assert(A.shape == B.shape)
    return A.dot(B) + B.dot(A)

def anticommutator(A,B):
    assert(A.shape == B.shape)
    return A.dot(B) + B.dot(A)

def dagger(a):
    return a.conjugate().transpose()

def dag(a):
    return a.conjugate().transpose()

def coth(x):
    return 1./np.tanh(x)

def pauliz():
     return np.array([[1.0,0.0],[0.0,-1.0]], dtype=np.complex128)

def paulix():
    return np.array([[0.0,1.0],[1.0,0.0]], dtype=np.complex128)

def pauliy():
    return np.array([[0.0,-1j],[1j,0.0]], dtype=np.complex128)

def pauli():
    # spin-half matrices 
    sz = np.array([[1.0,0.0],[0.0,-1.0]])
    
    sx = np.array([[0.0,1.0],[1.0,0.0]])
    
    sy = np.array([[0.0,-1j],[1j,0.0]]) 
    
    s0 = np.identity(2)
    
    return s0, sx, sy, sz

def ham_ho(freq, n, ZPE=True):
    """
    input:
        freq: fundemental frequency in units of Energy
        n : size of matrix
        ZPE: boolean, if ZPE is included in the Hamiltonian 
    output:
        h: hamiltonian of the harmonic oscilator
    """
    if ZPE:
        energy = (np.arange(n) + 0.5) * freq
    else:
        energy = np.arange(n) * freq

    return np.diagflat(energy)

sz = np.array([[1.,0.0],[0.0,-1.0]], dtype=np.complex128)
sx = np.array([[0.0,1.0],[1.0,0.0]])
sy = np.array([[0.0,-1j],[1j,0.0]]) 
s0 = np.identity(2)