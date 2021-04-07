#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:26:02 2019

@author: binggu
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, kron, identity, linalg
import scipy
import sys


from .units import au2fs, au2k, au2ev
from .phys import dag, coth, ket2dm, comm, anticomm, pauli
from .mol import Mol

class Composite(Mol):
    def __init__(self, A, B):
        """

        Parameters
        ----------
        A : object
            Quantum system.
        B : object
            Quantum system.

        Returns
        -------
        None.

        """

        self.A = A
        self.B = B
        self.idm = kron(A.idm, B.idm)  # identity matrix
        self.ida = A.idm
        self.idb = B.idm
        self.H = None

    def getH(self, a_ops=None, b_ops=None, g=0):
        """


        Returns
        -------
        2d array
            Hamiltonian of the full system.

        """

        H = kron(self.A.H, self.idb) + kron(self.ida, self.B.H)

        if a_ops == None:

            print('Warning: there is no coupling between the two subsystems.')

            self.H = H
            return H
        else:
            for i, a_op in enumerate(a_ops):
                b_op = b_ops[i]
                H += g[i] * kron(a_op, b_op)

            self.H = H
            return H

    def promote(self, o, subspace='A'):
        """
        promote an operator in subspace to the full Hilbert space
        E.g. A = A \otimes I_B
        """
        if subspace == 'A':
            return kron(o, self.B.idm)

        elif subspace == 'B':
            return kron(self.A.idm, o)
        else:
            raise ValueError('The subspace option can only be A or B.')

    def promote_ops(self, ops, subspaces=None):
        if subspaces is None:
            subspaces = ['A'] * len(ops)

        new_ops = []
        for i, op in enumerate(ops):
            new_ops.append(self.promote(op, subspaces[i]))

        return new_ops

    def spectrum(self):
        if self.H == None:
            sys.exit('Call getH to contrust the full Hamiltonian first.')
        else:
            eigvals, eigvecs = np.linalg.eigh(self.H.toarray())

            return eigvals, eigvecs

def ham_ho(freq, N, ZPE=False):
    """
    input:
        freq: fundemental frequency in units of Energy
        n : size of matrix
    output:
        h: hamiltonian of the harmonic oscilator
    """

    if ZPE:
        energy = np.arange(N + 0.5) * freq
    else:
        energy = np.arange(N) * freq

    H = lil_matrix((N, N))
    H.setdiag(energy)

    return H.tocsr()


def fft(t, x, freq=np.linspace(0, 0.1)):

    t = t/au2fs

    dt = (t[1] - t[0]).real

    sp = np.zeros(len(freq), dtype=np.complex128)

    for i in range(len(freq)):
        sp[i] = x.dot(np.exp(1j * freq[i] * t - 0.002*t)) * dt

    return sp

# def dag(H):
#     return H.conj().T

# def coth(x):
#     return 1./np.tanh(x)

# def ket2dm(psi):
#     return np.einsum("i, j -> ij", psi.conj(), psi)

# def obs(A, rho):
#     """
#     compute observables
#     """
#     return A.dot( rho).diagonal().sum()


def rk4_step(a, fun, dt, *args):

    dt2 = dt/2.0

    k1 = fun(a, *args)
    k2 = fun(a + k1*dt2, *args)
    k3 = fun(a + k2*dt2, *args)
    k4 = fun(a + k3*dt, *args)

    a += (k1 + 2*k2 + 2*k3 + k4)/6. * dt
    return a


class Pulse:
    def __init__(self, delay, sigma, omegac, amplitude=0.01, cep=0.):
        """
        Gaussian pulse A * exp(-(t-T)^2/2 / sigma^2)
        A: amplitude
        T: time delay
        sigma: duration
        """
        self.delay = delay
        self.sigma = sigma
        self.omegac = omegac  # central frequency
        self.unit = 'au'
        self.amplitude = amplitude
        self.cep = cep

    def envelop(self, t):
        return np.exp(-(t-self.delay)**2/2./self.sigma**2)

    def spectrum(self, omega):
        """
        Fourier transform of the Gaussian pulse
        """
        omegac = self.omegac
        sigma = self.sigma
        a = self.amplitude
        return a * sigma * np.sqrt(2.*np.pi) * np.exp(-(omega-omegac)**2 * sigma**2/2.)

    def field(self, t):
        '''
        electric field
        '''
        omegac = self.omegac
        delay = self.delay
        a = self.amplitude
        sigma = self.sigma
        return a * np.exp(-(t-delay)**2/2./sigma**2)*np.cos(omegac * (t-delay))


class Cavity():
    def __init__(self, freq, n_cav, Q=None):
        self.freq = freq
        self.resonance = freq
        self.n_cav = n_cav
        self.n = n_cav

        self.idm = identity(n_cav)
        self.create = self.get_create()

        self.annihilate = self.get_annihilate()
        self.H = self.getH()
        if Q is not None:
            self.kappa = freq/2./Q

#    @property
#    def hamiltonian(self):
#        return self._hamiltonian
#
#    @hamiltonian.setter
#    def hamiltonian(self):
#        self._hamiltonian = ham_ho(self.resonance, self.n)

    def get_ham(self, zpe=False):
        return self.getH(zpe)

    def getH(self, zpe=False):
        self.H = ham_ho(self.freq, self.n_cav)
        return self.H

    def get_nonhermitianH(self):
        '''
        non-Hermitian Hamiltonian for the cavity mode

        Params:
            kappa: decay constant

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        ncav = self.n_cav
        return ham_ho(self.freq, ncav) - 1j * self.kappa * np.identity(ncav)

    def ham(self, ZPE=False):
        return ham_ho(self.freq, self.n_cav)

    def get_create(self):
        n_cav = self.n_cav
        c = lil_matrix((n_cav, n_cav))
        c.setdiag(np.sqrt(np.arange(1, n_cav)), -1)
        return c.tocsr()

    def get_annihilate(self):
        n_cav = self.n_cav
        a = lil_matrix((n_cav, n_cav))
        a.setdiag(np.sqrt(np.arange(1, n_cav)), 1)

        return a.tocsr()

    def get_dm(self):
        """
        get initial density matrix for cavity vacuum state
        """
        vac = np.zeros(self.n_cav)
        vac[0] = 1.
        return ket2dm(vac)

    def get_num(self):
        """
        number operator
        """
        ncav = self.n_cav
        a = lil_matrix((ncav, ncav))
        a.setdiag(range(ncav), 0)

        return a.tocsr()

    def num(self):
        """
        number operator
        input:
            N: integer
                number of states
        """
        N = self.n_cav
        a = lil_matrix((N, N))
        a.setdiag(range(N), 0)
        return a.tocsr()


class Polariton:
    def __init__(self, mol, cav, RWA=False):
        #self.g = g
        self.mol = mol
        self.cav = cav
        self._ham = None
        self.dip = None
        self.cav_leak = None
        self.H = None
        self.size = mol.nstates * cav.n_cav

        #self.dm = kron(mol.dm, cav.get_dm())

    def getH(self, g, RWA=False):

        mol = self.mol
        cav = self.cav

        hmol = mol.get_ham()
        hcav = cav.get_ham()

        Icav = cav.idm
        Imol = mol.idm

        if RWA == True:

            hint = g * (kron(mol.raising(), cav.get_annihilate()) +
                        kron(mol.lowering(), cav.get_create()))

        else:

            hint = g * kron(mol.dip, cav.get_create() + cav.get_annihilate())

        self.H = kron(hmol, Icav) + kron(Imol, hcav) + hint

        return self.H

    def get_nonhermitianH(self, g, RWA=False):

        mol = self.mol
        cav = self.cav

        hmol = mol.get_nonhermitianH()
        hcav = cav.get_nonhermitianH()

        Icav = cav.idm
        Imol = mol.idm

        if RWA == True:

            hint = g * (kron(mol.raising(), cav.get_annihilate()) +
                        kron(mol.lowering(), cav.get_create()))

        else:

            hint = g * kron(mol.dip, cav.get_create() + cav.get_annihilate())

        H = kron(hmol, Icav) + kron(Imol, hcav) + hint

        return H

    def get_ham(self, RWA=False):
        return self.getH(RWA)

    def setH(self, h):
        self.H = h
        return

    def get_dip(self, basis='product'):
        '''
        transition dipole moment in the direct product basis

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return kron(self.mol.dip, self.cav.idm)

    def get_dm(self):
        return kron(self.mol.dm, self.cav.get_dm())

    def get_cav_leak(self):
        """
        damp operator for the cavity mode
        """
        if self.cav_leak == None:
            self.cav_leak = kron(self.mol.idm, self.cav.annihilate)

        return self.cav_leak

    def eigenstates(self, nstates=1, sparse=True):
        """
        compute the polaritonic spectrum

        Parameters
        ----------
        nstates : int, optional
            number of eigenstates. The default is 1.
        sparse : TYPE, optional
            if the Hamiltonian is sparse. The default is True.

        Returns
        -------
        evals : TYPE
            DESCRIPTION.
        evecs : TYPE
            DESCRIPTION.
        n_ph : TYPE
            photonic fractions in polaritons.

        """

        if self.H == None:
            sys.exit('Please call getH to compute the Hamiltonian first.')

        if sparse:

            evals, evecs = linalg.eigsh(self.H, nstates, which='SA')
            # number of photons in polariton states
            num_op = self.cav.num()
            num_op = kron(self.mol.idm, num_op)

            #print(num_op.shape, evecs.shape)

            n_ph = np.zeros(nstates)
            for j in range(nstates):
                n_ph[j] = np.real(evecs[:, j].conj().dot(
                    num_op.dot(evecs[:, j])))

            return evals, evecs, n_ph

        else:

            """
            compute the full polaritonic spectrum with numpy
            """
            h = self.H.toarray()
            evals, evecs = scipy.linalg.eigh(h, subset_by_index=[0, nstates])
            # number of photons in polariton states
            num_op = self.cav.num()
            num_op = kron(self.mol.idm, num_op)

            n_ph = np.zeros(nstates)
            for j in range(nstates):
                n_ph[j] = np.real(evecs[:, j].conj().dot(
                    num_op.dot(evecs[:, j])))

            return evals, evecs, n_ph

    def rdm_photon(self):
        """
        return the reduced density matrix for the photons
        """


def QRM(omega0, omegac, ncav=2):
    '''
    Quantum Rabi model / Jaynes-Cummings model

    Parameters
    ----------
    omega0 : float
        atomic transition frequency
    omegac : float
        cavity frequency
    g : float
        cavity-molecule coupling strength

    Returns
    -------
    rabi: object

    '''
    s0, sx, sy, sz = pauli()

    hmol = 0.5 * omega0 * (-sz + s0)

    mol = Mol(hmol, sx)  # atom
    cav = Cavity(omegac, ncav)  # cavity

    return Polariton(mol, cav)


if __name__ == '__main__':
    mol = QRM(1, 1)
