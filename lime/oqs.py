'''
modules for open quantum systems

@author: Bing Gu
@email: bingg@uci.edu
'''

import numpy as np
import numba
import sys

from .phys import anticommutator, comm, anticomm, dag, ket2dm, \
    obs_dm, destroy, rk4, liouvillian

from .units import au2fs

from scipy.sparse import csr_matrix


# class Redfield:
#     def __init__(self):
#         self.h_sys = None
#         self.c_ops = None
#         self.obs_ops = None
#         return

#     def configure(self, h_sys, c_ops, obs_ops):
#         self.c_ops = c_ops
#         self.obs_ops = obs_ops
#         self.h_sys = h_sys
#         return

#     def run(self, rho0, dt, Nt, store_states=False, nout=1):
#         '''
#         propogate the dynamics

#         Parameters
#         ----------
#         rho0 : TYPE
#             DESCRIPTION.
#         dt : TYPE
#             DESCRIPTION.
#         Nt : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         '''

#         c_ops = self.c_ops
#         h0 = self.h_sys
#         obs_ops = self.obs_ops

#         rho = _redfield(rho0, c_ops, h0, Nt, dt,obs_ops, integrator='SOD')

#         return rho


class Oqs:
    def __init__(self, ham):
        '''
        open quantum systems class

        Returns
        -------
        None.

        '''
        self.hamiltonian = ham
        self.h_sys = ham
        self.nstates = ham.shape[-1]
        #self.rho = rho0
        self.obs_ops = None

    def set_hamiltonian(self, h):
        self.hamiltonian = h
        return

    def set_c_ops(self, c_ops):
        self.c_ops = c_ops
        return

    def set_obs_ops(self, obs_ops):
        """
        set observable operators
        """
        self.obs_ops = obs_ops
        return

    def configure(self, c_ops, obs_ops):
        self.c_ops = c_ops
        self.obs_ops = obs_ops
        return

    def heom(self, env, nado=5, fname=None):
        nt = self.nt
        dt = self.dt
        return heom(self.oqs, env, self.c_ops, nado, nt, dt, fname)

    def redfield(self, env, dt, Nt, integrator='SOD'):
        nstates = self.nstates
        rho0 = self.rho0
        c_ops = self.c_ops
        h0 = self.hamiltonian
        obs_ops = self.obs_ops

        redfield(nstates, rho0, c_ops, h0, Nt, dt,obs_ops, env, integrator='SOD')

    def tcl2(self, env, rho0, dt, Nt, integrator='SOD'):
        nstates = self.nstates
        c_ops = self.c_ops
        h0 = self.hamiltonian
        obs_ops = self.obs_ops

        redfield(nstates, rho0, c_ops, h0, Nt, dt,obs_ops, env, integrator='SOD')

        return

    def lindblad(self, rho0, dt, Nt):
        """
        lindblad quantum master equations

        Parameters
        ----------
        rho0: 2D array
            initial density matrix
        """
        c_ops = self.c_ops
        obs_ops = self.obs_ops
        h0 = self.hamiltonian
        lindblad(rho0, h0, c_ops, Nt, dt, obs_ops)
        return

    def steady_state(self):
        return

    def correlation_2p_1t(self, rho0, ops, dt, Nt, method='lindblad', output='cor.dat'):
        '''
        two-point correlation function <A(t)B>

        Parameters
        ----------
        rho0 : TYPE
            DESCRIPTION.
        ops : TYPE
            DESCRIPTION.
        dt : TYPE
            DESCRIPTION.
        Nt : TYPE
            DESCRIPTION.
        method : TYPE, optional
            DESCRIPTION. The default is 'lindblad'.
        output : TYPE, optional
            DESCRIPTION. The default is 'cor.dat'.

        Returns
        -------
        None.

        '''

        H = self.hamiltonian
        c_ops = self.c_ops

        correlation_2p_1t(H, rho0, ops=ops, c_ops=c_ops, dt=dt, Nt=Nt, \
                          method=method, output=output)

        return

    # def tcl2(self):
    #     """
    #     second-order time-convolutionless quantum master equation
    #     """
    #     pass

#@numba.jit
def correlation_2p_1t(H, rho0, ops, c_ops, dt, Nt, method='lindblad', output='cor.dat'):
    """
    compute the time-translation invariant two-point correlation function in the
    density matrix formalism using quantum regression theorem

        <A(t)B> = Tr[ A U(t) (B rho0)  U^\dag(t)]

    input:
    ========
    H: 2d array
        full Hamiltonian

    rho0: initial density matrix

    ops: list of operators [A, B] for computing the correlation functions

    method: str
        dynamics method e.g. lindblad, redfield, heom

    args: dictionary of parameters for dynamics

    Returns
    ========
    the density matrix is stored in 'dm.dat'
    'cor.dat': the correlation function is stored
    """
    #nstates =  H.shape[-1] # number of states in the system

    # initialize the density matrix
    A, B = ops
    rho = B.dot(rho0)

    f = open(output, 'w')
    # f_dm = open('dm.dat', 'w')
    # fmt = '{} ' * (H.size + 1) + '\n' # format to store the density matrix

    # dynamics

    t = 0.0
    # Nt = len(tlist)
    # dt = tlist[1] - tlist[0]

    # sparse matrix
    H = csr_matrix(H)
    rho = csr_matrix(rho)

    A = csr_matrix(A)

    c_ops_sparse = [csr_matrix(c_op) for c_op in c_ops]

    if method == 'lindblad':

        for k in range(Nt):

            t += dt

            rho = rk4(rho, lindblad, dt, H, c_ops_sparse)

            # cor = A.dot(rho).diagonal().sum()
            cor = obs_dm(rho, A)

            # store the reduced density matrix
            f.write('{} {} \n'.format(t, cor))

            # f_dm.write(fmt.format(t, *np.ravel(rho)))

            # f_dm.write(fmt.format(t, *np.ravel(rho.toarray())))

    else:
        sys.exit('The method {} has not been implemented yet! Please \
                 try lindblad.'.format(method))

    f.close()
    # f_dm.close()

    return

class Env:
    def __init__(self, temperature, cutoff, reorg):
        self.temperature = temperature
        self.cutoff = cutoff
        self.reorg = reorg

    def set_bath_ops(self, bath_ops):
        """


        Parameters
        ----------
        bath_ops : list of 2d arrays
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.bath_ops = bath_ops
        return

    def corr(self):
        """
        compute the correlation function for bath operators
        """
        pass

    def spectral_density(self):
        """
        spectral density
        """
        pass

def heom(oqs, env, c_ops, nado, nt, dt, fname):
    '''

    terminator : ado[:,:,nado] = 0

    INPUT:
        T: in units of energy, kB * T, temperature of the bath
        reorg: reorganization energy
        nado : auxiliary density operators, truncation of the hierachy
        fname: file name for output

    '''
    nst = oqs.nstate
    ado = np.zeros((nst, nst, nado), dtype=np.complex128)     # auxiliary density operators
    ado[:,:,0] = oqs.rho0 # initial density matrix



    gamma = env.cutoff # cutoff frequency of the environment, larger gamma --> more Makovian
    T = env.temperature
    reorg = env.reorg
    print('Temperature of the environment = {}'.format(T))
    print('High-Temperature check gamma/(kT) = {}'.format(gamma/T))

    if gamma/T > 0.8:
        print('WARNING: High-Temperature Approximation may fail.')

    print('Reorganization energy = {}'.format(reorg))

    # D(t) = (a + ib) * exp(- gamma * t)
    a = np.pi * reorg * T  # initial value of the correlation function D(0) = pi * lambda * kB * T
    b = 0.0
    print('Amplitude of the fluctuations = {}'.format(a))

    #sz = np.zeros((nstate, nstate), dtype=np.complex128)
    sz = c_ops # collapse opeartor

    #H = Hamiltonian()
    H = oqs.hamiltonian

    f = open(fname,'w')
    fmt = '{} '* 5 + '\n'

    # propagation time loop - HEOM
    t = 0.0
    for k in range(nt):

        t += dt # time increments

        ado[:,:,0] += -1j * commutator(H, ado[:,:,0]) * dt - \
            commutator(sz, ado[:,:,1]) * dt

        for n in range(nado-1):
            ado[:,:,n] += -1j * commutator(H, ado[:,:,n]) * dt + \
                        (- commutator(sz, ado[:,:,n+1]) - n * gamma * ado[:,:,n] + n * \
                        (a * commutator(sz, ado[:,:,n-1]) + \
                         1j * b * anticommutator(sz, ado[:,:,n-1]))) * dt

        # store the reduced density matrix
        f.write(fmt.format(t, ado[0,0,0], ado[0,1,0], ado[1,0,0], ado[1,1,0]))

        #sz += -1j * commutator(sz, H) * dt

    f.close()
    return ado[:,:,0]

@numba.jit
def func(rho, h0, c_ops, l_ops):
    """
    right-hand side of the master equation
    """
    rhs = -1j * commutator(h0, rho)

    for i in range(len(c_ops)):
        c_op = c_ops[i]
        l_op = l_ops[i]
        rhs -=  commutator(c_op, l_op.dot(rho) - rho.dot(dag(l_op)))
    return rhs



def basis(N, j):
    """
    Parameters
    ----------
    N: int
        Size of Hilbert space for a multi-level system.
    j: int
        The j-th basis function.

    Returns
    -------
    1d complex array
        j-th basis function for the Hilbert space.
    """
    b = np.zeros(N, dtype=complex)
    if N < j:
        sys.exit('Increase the size of the Hilbert space.')
    else:
        b[j] = 1.0

    return b

def coherent(N, alpha):
    """Generates a coherent state with eigenvalue alpha.

    Constructed using displacement operator on vacuum state.

    Modified from Qutip.

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    alpha : float/complex
        Eigenvalue of coherent state.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the state. Using a non-zero offset will make the
        default method 'analytic'.

    method : string {'operator', 'analytic'}
        Method for generating coherent state.

    Returns
    -------
    state : qobj
        Qobj quantum object for coherent state

    Examples
    --------
    >>> coherent(5,0.25j)
    Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
    Qobj data =
    [[  9.69233235e-01+0.j        ]
     [  0.00000000e+00+0.24230831j]
     [ -4.28344935e-02+0.j        ]
     [  0.00000000e+00-0.00618204j]
     [  7.80904967e-04+0.j        ]]

    Notes
    -----
    Select method 'operator' (default) or 'analytic'. With the
    'operator' method, the coherent state is generated by displacing
    the vacuum state using the displacement operator defined in the
    truncated Hilbert space of size 'N'. This method guarantees that the
    resulting state is normalized. With 'analytic' method the coherent state
    is generated using the analytical formula for the coherent state
    coefficients in the Fock basis. This method does not guarantee that the
    state is normalized if truncated to a small number of Fock states,
    but would in that case give more accurate coefficients.

    """

    x = basis(N, 0)
    a = destroy(N)
    D = np.expm(alpha * dag(a) - np.conj(alpha) * a)

    return D * x

    # elif method == "analytic" or offset > 0:

    #     data = np.zeros([N, 1], dtype=complex)
    #     n = arange(N) + offset
    #     data[:, 0] = np.exp(-(abs(alpha) ** 2) / 2.0) * (alpha ** (n)) / \
    #         _sqrt_factorial(n)
    #     return Qobj(data)

    # else:
    #     raise TypeError(
    #         "The method option can only take values 'operator' or 'analytic'")



def coherent_dm(N, alpha):
    """Density matrix representation of a coherent state.

    Constructed via outer product of :func:`qutip.states.coherent`

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    alpha : float/complex
        Eigenvalue for coherent state.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the state.

    method : string {'operator', 'analytic'}
        Method for generating coherent density matrix.

    Returns
    -------
    dm : qobj
        Density matrix representation of coherent state.

    Examples
    --------
    >>> coherent_dm(3,0.25j)
    Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 0.93941695+0.j          0.00000000-0.23480733j -0.04216943+0.j        ]
     [ 0.00000000+0.23480733j  0.05869011+0.j          0.00000000-0.01054025j]
     [-0.04216943+0.j          0.00000000+0.01054025j  0.00189294+0.j\
        ]]

    Notes
    -----
    Select method 'operator' (default) or 'analytic'. With the
    'operator' method, the coherent density matrix is generated by displacing
    the vacuum state using the displacement operator defined in the
    truncated Hilbert space of size 'N'. This method guarantees that the
    resulting density matrix is normalized. With 'analytic' method the coherent
    density matrix is generated using the analytical formula for the coherent
    state coefficients in the Fock basis. This method does not guarantee that
    the state is normalized if truncated to a small number of Fock states,
    but would in that case give more accurate coefficients.

    """
    #if method == "operator":
    psi = coherent(N, alpha)

    return ket2dm(psi)

    # elif method == "analytic":
    #     psi = coherent(N, alpha, offset=offset, method='analytic')
    #     return psi * psi.dag()

    # else:
    #     raise TypeError(
    #         "The method option can only take values 'operator' or 'analytic'")
#def dipole_cor(Nt, dt):
#    """
#    compute the dipole auto-correlation function using quantum regression theorem
#
#    """
#    ns = n_el * n_vt * n_vc * n_cav  # number of states in the system
#
#    # initialize the density matrix
#    rho0 = np.zeros((ns, ns), dtype=np.complex128)
#    rho0[0,0] = 1.0
#
#    d = np.zeros((ns, ns), dtype = np.complex128)
#    d[0, 2] = d[2,0] = 1.0
#    d[3, 0] = d[3,0] = 1.0
#
#    rho0 = np.matmul(d, rho0)
#
#
#    # dissipative operators
#    S1 = np.zeros((ns,ns), dtype=np.complex128)
#    S2 = np.zeros((ns,ns), dtype = np.complex128)
#
#    S1[2,2] = 1.0
#    S2[3,3] = 1.0
#
#    Lambda1 = getLambda(S1)
#    Lambda2 = getLambda(S2)
#
#    # short time approximation
#    # Lambda = 0.5 * reorg * T * ((hop - Delta)/cutfreq**2 * sigmay + 1./cutfreq * sigmaz)
#
#    h0 = Hamiltonian()
#    print(h0)
#
#    f = open('cor.dat', 'w')
#    f_dm = open('den_mat.dat', 'w')
#
#    t = 0.0
#    dt2 = dt/2.0
#
#
#    # first-step
#    rho_half = rho0 + func(rho0, h0, S1, S2, Lambda1, Lambda2) * dt2
#    rho1 = rho0 + func(rho_half, h0, S1, S2, Lambda1, Lambda2) * dt
#
#    rho_old = rho0
#    rho = rho1
#
#
#    for k in range(Nt):
#
#        t += dt
#
#        rho_new = rho_old + func(rho, h0, S1, S2, Lambda1, Lambda2) * 2. * dt
#
#        # update rho_old
#        rho_old = rho
#        rho = rho_new
#
#        cor = np.trace(np.matmul(d, rho))
#        # store the reduced density matrix
#        f.write('{} {} \n'.format(t, cor))
#        f_dm.write('{} {} \n'.format(t, rho[2,0]))
#
#
#    f.close()
#    f_dm.close()
#
#    return

def corr(t, temp, cutoff, reorg, SD='Drude'):
    """
    bath correlation function C(t) = <x(t)x>. For the Drude spectral density,
    in the high-temperature limit, C(t) ~ pi * reorg * T * e^{-cutoff * t}
    """
    # numerical treatment
    #NP = 1000
    #maxfreq = 1.0
    #omega = np.linspace(1e-4, maxfreq, NP)
    #dfreq = omega[1] - omega[0]

#    cor = sum(spec_den(omega) * (coth(omega/2./T) * np.cos(omega * t) - \
#                       1j * np.sin(omega * t))) * dfreq

    # test correlation function
#    reorg = 500. # cm^{-1}
#    reorg *= wavenumber2hartree

#    T = 0.03 # eV
#    T /= au2ev

#    td = 10 # fs
#    td /= au2fs

    ### analytical

    # Drude spectral density at high-temperature approximation
    if SD == 'Drude':
        cor = np.pi * temp * reorg * np.exp(- cutoff * t)

    return cor


def make_lambda(ns, h0, S, T, cutfreq, reorg):

    tmax = 1000.0
    print('time range to numericall compute the bath correlation function = [0, {}]'.format(tmax))
    print('Decay of correlation function at {} fs = {} \n'.format(tmax * au2fs, \
          corr(tmax, T, cutfreq, reorg)/corr(0., T, cutfreq, reorg)))
    print('!!! Please make sure this is much less than 1 !!!')

    time = np.linspace(0, tmax, 10000)
    dt = time[1] - time[0]
    dt2 = dt/2.0


    l = np.zeros((ns, ns), dtype=np.complex128)

    #t = time[0]
    #phi = hop * t - Delta * np.sin(omegad * t)/omegad
    #Lambda += corr(t) * (np.sin(2. * phi) * sigmay + np.cos(2. * phi) * sigmaz) * dt2

    #h0 = Hamiltonian()
    Sint = S.copy()

    for k in range(len(time)):

        t = time[k]

        Sint += -1j * commutator(S, h0) * (-dt)
        l += dt * Sint * corr(t, T, cutfreq, reorg)

#    t = time[len(time)-1]
#    phi = hop * t + Delta * np.sin(omegad * t)/omegad
#    Lambda += corr(t) * (np.sin(2. * phi) * sigmay + np.cos(2. * phi) * sigmaz) * dt2
#    Lambda = cy * sigmay + cz * sigmaz

    return l



def redfield(nstates, rho0, c_ops, h0, Nt, dt,obs_ops, env):
    """
    time propagation of the Redfield equation with second-order differencing
    input:
        nstate: total number of states
        h0: system Hamiltonian
        Nt: total number of time steps
        dt: tiem step
        c_ops: list of collapse operators
        obs_ops: list of observable operators
        rho0: initial density matrix
    """
    t = 0.0

    print('Total number of states in the system = {}'.format(nstates))

    # initialize the density matrix
    rho = rho0

    # properties of the environment
    T = env.T
    cutfreq = env.cutfreq
    reorg = env.reorg

    #f = open(fname,'w')
    fmt = '{} '* (len(obs_ops) + 1) + '\n'

    # construct system-bath operators in H_SB

    # short time approximation
    # Lambda = 0.5 * reorg * T * ((hop - Delta)/cutfreq**2 * sigmay + 1./cutfreq * sigmaz)

    # constuct the Lambda operators needed in Redfield equation
    l_ops = []
    for c_op in c_ops:
        l_ops.append(getLambda(nstates, h0, c_op, T, cutfreq, reorg))

    f_dm = open('den_mat.dat', 'w')
    f_obs = open('obs.dat', 'w')

    t = 0.0
    dt2 = dt/2.0

    # first-step
    rho_half = rho0 + func(rho0, h0, c_ops, l_ops) * dt2
    rho1 = rho0 + func(rho_half, h0, c_ops, l_ops) * dt

    rho_old = rho0
    rho = rho1

    for k in range(Nt):

        t += dt

        rho_new = rho_old + func(rho, h0, c_ops, l_ops) * 2. * dt

        # update rho_old
        rho_old = rho
        rho = rho_new

        # dipole-dipole auto-corrlation function
        #cor = np.trace(np.matmul(d, rho))

        # store the reduced density matrix
        f_dm.write('{} '* (nstates**2 + 1) + '\n'.format(t, *rho))

        # take a partial trace to obtain the rho_el
        obs = np.zeros(len(obs_ops))
        for i, obs_op in enumerate(obs_ops):
            obs[i] = observe(obs_op, rho)

        f_obs.write(fmt.format(t * au2fs, *obs))


    f_obs.close()
    f_dm.close()

    return rho

@numba.jit
def observe(A, rho):
    """
    compute expectation value of the operator A
    """
    return A.dot(rho).diagonal().sum()


def lindblad(rho0, h0, c_ops, Nt, dt, obs_ops):
    """
    time propagation of the lindblad quantum master equation
    with second-order differencing

    Input
    -------
    h0: 2d array
            system Hamiltonian
    Nt: total number of time steps

    dt: time step
        c_ops: list of collapse operators
        obs_ops: list of observable operators
        rho0: initial density matrix

    Returns
    =========
    rho: 2D array
        density matrix at time t = Nt * dt
    """

    nstates = h0.shape[-1]
    # initialize the density matrix
    rho = rho0

    f_dm = open('den_mat.dat', 'w')
    fmt_dm = '{} ' * (nstates**2 + 1) + '\n'

    f_obs = open('obs.dat', 'w')
    fmt = '{} '* (len(obs_ops) + 1) + '\n'


    t = 0.0
    dt2 = dt/2.0

    # first-step
    # rho_half = rho0 + liouvillian(rho0, h0, c_ops) * dt2
    # rho1 = rho0 + liouvillian(rho_half, h0, c_ops) * dt

    # rho_old = rho0
    # rho = rho1

    for k in range(Nt):

        t += dt

        # rho_new = rho_old + liouvillian(rho, h0, c_ops) * 2. * dt
        # # update rho_old
        # rho_old = rho
        # rho = rho_new

        rho = rk4(rho, liouvillian, dt, h0, c_ops)

        # dipole-dipole auto-corrlation function
        #cor = np.trace(np.matmul(d, rho))

        # store the reduced density matrix
        f_dm.write(fmt_dm.format(t, *rho.toarray().ravel()))

        # take a partial trace to obtain the rho_el
        # compute observables
        observables = np.zeros(len(obs_ops), dtype=complex)

        for i, obs_op in enumerate(obs_ops):
            observables[i] = obs_dm(rho, obs_op)

        f_obs.write(fmt.format(t, *observables))


    f_obs.close()
    f_dm.close()

    return rho



