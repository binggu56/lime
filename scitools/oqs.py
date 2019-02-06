'''
modules for open quantum systems
'''

import numpy as np
from .phys import commutator, anticommutator

class Oqs:
    def __init__(self, nstate, rho0, nt, dt):
        #self.hamiltonian = h
        self.nstate = None
        self.rho = rho0
        self.dt = dt
        self.nt = nt

    def set_hamiltonian(self, h):
        self.hamiltonian = h
        return

    def set_c_ops(self, c_ops):
        self.c_ops = c_ops
        return

    def heom(self, env, nado=5, fname=None):
        nt = self.nt
        dt = self.dt
        return heom(self.oqs, env, self.c_ops, nado, nt, dt, fname)

    def redfield(self):
        pass

    def markovian(self):
        pass

class Env:
    def __init__(self, temperature, cutoff, reorg):
        self.temperature = temperature
        self.cutoff = cutoff
        self.reorg = reorg

    def set_c_ops(self, c_ops):
        self.c_ops = c_ops
        return


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