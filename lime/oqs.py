'''
modules for open quantum systems

@author: Bing Gu
@email: bingg@uci.edu
'''

import numpy as np
from .phys import commutator, anticommutator, comm, anticomm, dag
import numba 


class Env:
    def __init__(self, temperature, cutoff, reorg):
        self.temperature = temperature
        self.gamma = cutoff
        self.reorg = reorg
        self.c_ops = None
    
class Oqs:
    def __init__(self):
        self.hamiltonian = None
        self.nstates = None
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
        
    def lindblad(self):
        """
        lindblad quantum master equations
        """
        c_ops = self.c_ops
        obs_ops = self.obs_ops 
        
        # todo 
        #lindblad(nstates, rho0, c_ops, obs_ops, h0, dt, Nt)
        pass 
    
    def tcl2(self):
        """
        second-order time-convolutionless quantum master equation 
        """
        pass 

class Env:
    def __init__(self, temperature, cutoff, reorg):
        self.temperature = temperature
        self.cutoff = cutoff
        self.reorg = reorg

    def set_bath_ops(self, bath_ops):
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

@numba.autojit
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

#def obs(rho, d):
#    """
#    observables during propagation
#    """
#
#    cor = np.trace(np.matmul(d, rho))
#
#    return cor


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

@numba.autojit
def observe(A, rho):
    """
    compute expectation value of the operator A
    """
    return A.dot(rho).diagonal().sum()

def lindbladian(l, rho):
    """
    lindblad superoperator: l rho l^\dag - 1/2 * {l^\dag l, rho}
    l is the operator corresponding to the disired physical process 
    e.g. l = a, for the cavity decay and 
    l = sm for polarization decay 
    """
    return l.dot(rho.dot(dag(l))) - 0.5 * anticomm(dag(l).dot(l), rho)


