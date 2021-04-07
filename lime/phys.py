from __future__ import absolute_import

import numpy as np
from numpy import exp
from scipy.sparse import csr_matrix, lil_matrix, identity, kron, linalg, spdiags

import numba
import sys

from lime.units import au2fs, au2ev


def lowering(dims=2):
    if dims == 2:
        sm = csr_matrix(np.array([[0.0, 1.0],[0.0,0.0]], dtype=np.complex128))
    else:
        raise ValueError('dims can only be 2.')
    return sm


def raising(dims=2):
    """
    raising operator for spin-1/2
    Parameters
    ----------
    dims: integer
        Hilbert space dimension

    Returns
    -------
    sp: 2x2 array
        raising operator
    """
    if dims == 2:
        sp = csr_matrix(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128))
    else:
        raise ValueError('dims can only be 2.')
    return sp


def sinc(x):
    '''
    sinc(x) = sin(x)/x

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return np.sinc(x/np.pi)

def norm2(f, dx=1, dy=1):
    '''
    L2 norm of the 2D array f

    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.
    dy : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return np.trace(dag(f).dot(f))*dx*dy

def get_index(array, value):
    '''
    get the index of element in array closest to value
    '''
    if value < np.min(array) or value > np.max(array):
        print('Warning: the value is out of the range of the array!')

    return np.argmin(np.abs(array-value))


def rgwp(x, x0=0., sigma=1.):
    '''
    real Gaussian wavepacket

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    x0 : float
        central position
    sigma : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    psi = 1./np.sqrt(np.sqrt(np.pi) * sigma) * np.exp(-(x-x0)**2/2./sigma**2)
    return psi

def gwp(x, sigma=1., x0=0., p0=0.):
    '''
    complex Gaussian wavepacket

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    sigma : TYPE, optional
        DESCRIPTION. The default is 1..
    x0 : TYPE, optional
        DESCRIPTION. The default is 0..
    p0 : TYPE, optional
        DESCRIPTION. The default is 0..

    Returns
    -------
    psi : TYPE
        DESCRIPTION.

    '''
    psi = np.sqrt(np.sqrt(1./np.pi/sigma**2)) * \
        np.exp(-(x-x0)**2/2./sigma**2 + 1j * p0 * (x-x0))
    return psi

def thermal_dm(n, u):
    """
    return the thermal density matrix for a boson
    n: integer
        dimension of the Fock space
    u: float
        reduced temperature, omega/k_B T
    """
    nlist = np.arange(n)
    diags = exp(- nlist * u)
    diags /= np.sum(diags)
    rho = lil_matrix(n)
    rho.setdiag(diags)
    return rho.tocsr()

def liouvillian(rho, H, c_ops):
    """
    lindblad quantum master eqution
    """
    rhs = -1j * comm(H, rho)
    for c_op in c_ops:
        rhs += lindbladian(c_op, rho)
    return rhs

def lindbladian(l, rho):
    """
    lindblad superoperator: l rho l^\dag - 1/2 * {l^\dag l, rho}
    l is the operator corresponding to the disired physical process
    e.g. l = a, for the cavity decay and
    l = sm for polarization decay
    """
    return l.dot(rho.dot(dag(l))) - 0.5 * anticomm(dag(l).dot(l), rho)

def ket2dm(psi):
    return np.einsum("i, j -> ij", psi.conj(), psi)

def norm(psi):
    '''
    normalization of the wavefunction

    Parameters
    ----------
    psi : 1d array, complex
        DESCRIPTION.

    Returns
    -------
    float, L2 norm

    '''
    return dag(psi).dot(psi).real


def destroy(N):
    """
    Annihilation operator for bosons.

    Parameters
    ----------
    N : int
        Size of Hilbert space.

    Returns
    -------
    2d array complex

    """

    a = lil_matrix((N, N))
    a.setdiag(np.sqrt(np.arange(1, N)), 1)

    return a.tocsr()


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

def lorentzian(x, x0=0., width=1.):
    '''


    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    x0 : float
        center of the Lorentzian

    width : float
        Half-wdith half-maximum

    Returns
    -------
    None.

    '''
    return 1./np.pi * width/(width**2 + (x-x0)**2)




def transform(A, v):
    """
    transformation rule: A_{ab} = <a|i><i|A|j><j|b> = Anew = v^\dag A v
    input:
        A: matrix of operator A in old basis
        v: basis transformation matrix
    output:
        Anew: matrix A in the new basis
    """
    Anew = dag(v).dot(A.dot(v))
    #Anew = csr_matrix(A)

    return Anew

def basis_transform(A, v):
    """
    transformation rule: A_{ab} = <a|i><i|A|j><j|b> = Anew = v^\dag A v
    input:
        A: matrix of operator A in old basis
        v: basis transformation matrix
    output:
        Anew: matrix A in the new basis
    """
    Anew = dag(v).dot(A.dot(v))

    return csr_matrix(Anew)

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

def ham_ho(freq, n, ZPE=False):
    """
    Hamiltonian for harmonic oscilator

    input:
        freq: fundemental frequency in units of Energy
        n : size of matrix
        ZPE: boolean, if ZPE is included in the Hamiltonian
    output:
        h: hamiltonian of the harmonic oscilator
    """
    if ZPE:
        h = lil_matrix((n,n))
        h = h.setdiag((np.arange(n) + 0.5) * freq)
    else:
        h = lil_matrix((n, n)).setdiag(np.arange(n) * freq)

    return h

def boson(omega, n, ZPE=False):
    if ZPE:
        h = lil_matrix((n,n))
        h.setdiag((np.arange(n) + 0.5) * omega)
    else:
        h = lil_matrix((n, n))
        h.setdiag(np.arange(n) * omega)
    return h

def quadrature(n):
    """
    Quadrature operator of a photon mode

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    a = destroy(n)
    return 1./np.sqrt(2) * (a + dag(a))

def obs_dm(rho, d):
    """
    observables for operator d
    """

    dAve = d.dot(rho).diagonal().sum()

    return dAve

def obs(psi, a):
    """


    Parameters
    ----------
    psi : 1d array
        wavefunction.
    a : 2d array
        operator a.

    Returns
    -------
    complex
        Expectation of operator a.

    """
    return dag(psi).dot(a.dot(psi))

def resolvent(omega, Ulist, dt):
    """
    compute the resolvent 1/(omega - H) from the Fourier transform of the propagator
    omega: float
        frequency to evaluate the resolvent
    Ulist: list of matrices
        propagators
    dt: time-step used in the computation of U
    """
    N = len(Ulist)
    t = np.array(np.arange(N) * dt)
    return sum(np.exp(1j * omega * t) * Ulist)


def propagator(h, Nt, dt):
    """
    compute the resolvent for the multi-point correlation function signals
    U(t) = e^{-i H t}
    """

    # propagator
    U = identity(h.shape[-1], dtype=complex)

    # set the ground state energy to 0
    print('Computing the propagator ...\n Please make sure that the ground-state energy is 0.')
    Ulist = [U]

    for k in range(Nt):
        U = rk4(U, tdse, dt, h)
        Ulist.append(U)

    return Ulist


def tdse(wf, h):
    return -1j * h.dot(wf)

def quantum_dynamics(ham, psi0, dt=0.001, Nt=1, obs_ops=None, nout=1,\
                    t0=0.0, output='obs.dat'):
    '''
    Laser-driven dynamics in the presence of laser pulses

    Parameters
    ----------
    ham : 2d array
        Hamiltonian of the molecule
    psi0: 1d array
        initial wavefunction
    dt : float
        time step.
    Nt : int
        timesteps.
    obs_ops: list
        observable operators to compute
    nout: int
        Store data every nout steps

    Returns
    -------
    None.

    '''

    # initialize the density matrix
    #wf = csr_matrix(wf0).transpose()
    psi = psi0

    #nstates = len(psi0)

    #f = open(fname,'w')
    if obs_ops is not None:
        fmt = '{} '* (len(obs_ops) + 1)  + '\n'
    #fmt_dm = '{} '* (nstates + 1)  + '\n'

    #f_dm = open('psi.dat', 'w') # wavefunction
    f_obs = open(output, 'w') # observables

    t = t0

    #f_dip = open('dipole'+'{:f}'.format(pulse.delay * au2fs)+'.dat', 'w')

    for k1 in range(int(Nt/nout)):

        for k2 in range(nout):
            psi = rk4(psi, tdse, dt, ham)

        t += dt * nout

        # compute observables
        Aave = np.zeros(len(obs_ops), dtype=complex)

        for j, A in enumerate(obs_ops):
            Aave[j] = obs(psi, A)

        #print(Aave)

#        f_dm.write(fmt_dm.format(t, *psi))
        f_obs.write(fmt.format(t, *Aave))

    np.savez('psi', psi)
    #f_dm.close()
    f_obs.close()

    return

def driven_dynamics(ham, dip, psi0, pulse, dt=0.001, Nt=1, obs_ops=None, nout=1,\
                    t0=0.0):
    '''
    Laser-driven dynamics in the presence of laser pulses

    Parameters
    ----------
    ham : 2d array
        Hamiltonian of the molecule
    dip : TYPE
        transition dipole moment
    psi0: 1d array
        initial wavefunction
    pulse : TYPE
        laser pulse
    dt : TYPE
        time step.
    Nt : TYPE
        timesteps.
    obs_ops: list
        observable operators to compute
    nout: int
        Store data every nout steps

    Returns
    -------
    None.

    '''

    # initialize the density matrix
    #wf = csr_matrix(wf0).transpose()
    psi = psi0

    nstates = len(psi0)

    #f = open(fname,'w')
    fmt = '{} '* (len(obs_ops) + 1)  + '\n'
    fmt_dm = '{} '* (nstates + 1)  + '\n'

    f_dm = open('psi.dat', 'w') # wavefunction
    f_obs = open('obs.dat', 'w') # observables

    t = t0

    #f_dip = open('dipole'+'{:f}'.format(pulse.delay * au2fs)+'.dat', 'w')

    for k1 in range(int(Nt/nout)):

        for k2 in range(nout):

            ht = pulse.field(t) * dip + ham
            psi = rk4(psi, tdse, dt, ht)

        t += dt * nout

        # compute observables
        Aave = np.zeros(len(obs_ops), dtype=complex)
        for j, A in enumerate(obs_ops):
            Aave[j] = obs(psi, A)

        #print(Aave)

        f_dm.write(fmt_dm.format(t, *psi))
        f_obs.write(fmt.format(t, *Aave))

    f_dm.close()
    f_obs.close()

    return

def driven_dissipative_dynamics(ham, dip, rho0, pulse, dt=0.001, Nt=1, \
                                obs_ops=None, nout=1):
    '''
    Laser-driven dynamics in the presence of laser pulses

    Parameters
    ----------
    ham : 2d array
        Hamiltonian of the molecule
    dip : TYPE
        transition dipole moment
    rho0: 2d array complex
        initial density matrix
    pulse : TYPE
        laser pulse
    dt : float
        DESCRIPTION.
    Nt : TYPE
        DESCRIPTION.
    obs_ops: list
        observable operators to compute
    nout: int
        Store data every nout steps

    Returns
    -------
    None.

    '''
    return


####################
# spin chains
####################
def multi_spin(onsite, nsites):
    """
    construct the hamiltonian for a multi-spin system
    params:
        onsite: array, transition energy for each spin
        nsites: number of spins
    """

    s0, sx, sy, sz = pauli()

    head = onsite[0] * kron(sz, tensor_power(s0, nsites-1))
    tail = onsite[-1] * kron(tensor_power(s0, nsites-1), sz)
    ham = head + tail

    for i in range(1, nsites-1):
        ham += onsite[i] * kron(tensor_power(s0, i), kron(sz, tensor_power(s0, nsites-i-1)))

    lower_head = kron(sm, tensor_power(s0, nsites-1))
    lower_tail = kron(tensor_power(s0, nsites-1), sm)
    lower = lower_head + lower_tail

    for i in range(1, nsites-1):
        lower += kron(tensor_power(s0, i), kron(sm, tensor_power(s0, nsites-i-1)))



    return ham, lower

def multiboson(omega, nmodes, J=0, truncate=2):
    """
    construct the hamiltonian for a multi-spin system

    Parameters
    ----------
    omegas : 1D array
        resonance frequenies of the boson modes
    nmodes : integer
        number of boson modes
    J : float
        hopping constant
    truncation : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    ham : TYPE
        DESCRIPTION.
    lower : TYPE
        DESCRIPTION.

    """

    N = truncate
    h0 = boson(omega, N)
    idm = identity(N)
    a = destroy(N)
    adag = dag(a)
    x = csr_matrix(a  + adag)


    if nmodes == 1:

        return h0

    elif nmodes == 2:

        ham = kron(idm, h0) + kron(h0, idm) + J * kron(x, x)
        return ham

    elif nmodes > 2:

        head = kron(h0, tensor_power(idm, nmodes-1))
        tail = kron(tensor_power(idm, nmodes-1), h0)
        ham = head + tail

        for i in range(1, nmodes-1):
             ham += kron(tensor_power(idm, i), \
                                     kron(h0, tensor_power(idm, nmodes-i-1)))

        hop_head = J * kron(kron(x, x), tensor_power(idm, nmodes-2))
        hop_tail = J * kron(tensor_power(idm, nmodes-2), kron(x, x))

        ham += hop_head + hop_tail

        for i in range(1, nmodes-2):
            ham += J * kron(tensor_power(idm, i), \
                                kron(kron(x, x), tensor_power(idm, nmodes-i-2)))

        # connect the last mode to the first mode

    # lower_head = kron(a, tensor_power(idm, nmodes-1))
    # lower_tail = kron(tensor_power(idm, nmodes-1), a)
    # lower = lower_head + lower_tail

    # for i in range(1, nmodes-1):
    #     lower += kron(tensor_power(idm, i), kron(a, tensor_power(idm, nmodes-i-1)))


        return ham

def tensor_power(a, n:int):
    """
    kron(a, kron(a, ...))
    """
    if n == 1:
        return csr_matrix(a)
    else:
        tmp = a
        for i in range(n-1):
            tmp = kron(tmp, a)

        return tmp

#def exact_diagonalization(nsites):
#    """
#    exact-diagonalization of the Dicke model
#    """
#    nsites = 10
#    omega0 = 2.0
#
#    g = 0.1
#
#    onsite = np.random.randn(nsites) * 0.08 + omega0
#    print('onsite energies', onsite)
#
#    ham, dip = multi_spin(onsite, nsites)
#
#    cav = Cavity(n_cav=2, freq = omega0)
#    print(cav.get_ham())
#
#    mol = Mol(ham, dip=dip)
#
#    pol = Polariton(mol, cav, g)
#
#
#
#    # spectrum of the sytem
#
#    #fig, ax = plt.subplots()
#    #w = linalg.eigsh(ham, nsites+1, which='SA')[0]
#    #for i in range(1, len(w)):
#    #    ax.axvline(w[i] - w[0])
#
#
#    # polaritonic spectrum
#    fig, ax = plt.subplots()
#
#    set_style()
#
#    nstates = nsites + 2
#    w, v = pol.spectrum(nstates)
#
#    num_op = cav.get_num()
#    print(num_op)
#    num_op = kron(identity(ham.shape[0]), num_op)
#
#    n_ph = np.zeros(nstates)
#    for j in range(nstates):
#        n_ph[j] = v[:,j].conj().dot(num_op.dot(v[:,j]))
#
#    print(n_ph)
#
#    for i in range(1, len(w)):
#        ax.axvline(w[i] - w[0], ymin=0 , ymax = n_ph[i], lw=3)
#
#    #ax.set_xlim(1,3)
#    ax.axvline(omega0 + g * np.sqrt(nsites), c='r', lw=1)
#    ax.axvline(omega0 - g * np.sqrt(nsites), color='r', lw=1)
#    ax.set_ylim(0, 0.5)
#
#    fig.savefig('polariton_spectrum.eps', transparent=True)
