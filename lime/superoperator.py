#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:01:00 2020

@author: Bing

Modules for computing signals with superoperator formalism

Instead of performing open quantum dynamics, the Liouvillian is directly diagonalized

Possible improvements:
    1. merge the Qobj class with QUTIP
"""
import numpy as np
import matplotlib.pyplot as plt

#from scipy.linalg import kron, identity
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigs


from lime.phys import dag, pauli

# from qutip import Qobj as Basic
from Qobj import Qobj

# class Qobj(Basic):
#     def __init__(self, data=None, dims=None):
#         """
#         Class for quantum operators: is this useful?

#         Parameters
#         ----------
#         n : int
#             size of Hilbert space.

#         Returns
#         -------
#         None.

#         """
#         Basic.__init__(self, dims=dims, inpt=data)
#         return

#     def dot(self, b):

#         return Qobj(np.dot(self.data, b.data))

#     def to_vector(self):
#         return operator_to_vector(self.data)

#     def to_super(self, type='commutator'):

#         return operator_to_superoperator(self.data, type=type)

#     def to_linblad(self, gamma=1.):
#         l = self.data
#         return gamma * (kron(l, l.conj()) - \
#                 operator_to_superoperator(dag(l).dot(l), type='anticommutator'))


def liouville_space(N):
    """
    constuct liouville space out of N Hilbert space basis |ij>
    """
    return

def operator_to_vector(rho):
    """
    transform an operator/density matrix to an superoperator in Liouville space

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    return rho.toarray().flatten()

def operator_to_superoperator(a, type='commutator'):
    """
    promote an operator/density matrix to an superoperator in Liouville space

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    N = np.shape(a.toarray())[-1]

    idm = identity(N)

    if type == 'commutator':

        return kron(a, idm) - kron(idm, a.T)

    elif type == 'left':

        # elementwise operator for defining the commutator

        # for n in range(N2):
        #     i, j = divmod(n, N)
        #     for m in range(N2):
        #         k, l = divmod(m, N)
        #         am[n, m] = a[i, k] * idm[j,l]

        return kron(a, idm)

    elif type == 'right':

        return kron(idm, a.T)

    elif type == 'anticommutator':

        return kron(a, idm) + kron(idm, a.T)

    else:

        raise ValueError('Error: superoperator {} does not exist.'.format(type))

def left(a):

    idm = identity(a.toarray().shape[-1])
    return kron(a, idm)

def right(a):
    idm = identity(a.toarray().shape[-1])
    return kron(idm, a.T)


def resolvent(omega, L):
    '''
    Resolvent of the Lindblad quantum master equation

    Parameters
    ----------
    omega : TYPE
        DESCRIPTION.
    L : 2d array
        full liouvillian
    Returns
    -------
    None.

    '''
    idm = np.identity(L.shape[0])

    return np.linalg.inv(omega * idm - L)

def correlation_2p_1t(omegas, rho0, ops, L):
    a, b = ops
    out = np.zeros(len(omegas))

    for j in range(len(omegas)):
        omega = omegas[j]
        r = resolvent(omega, L)
        out[j] = operator_to_vector(a.T).dot(r.dot(operator_to_vector(b.rho0)))

    return out


s0, sx, sy, sz = pauli()

sx = Qobj(sx)

nstates = 10

h = Qobj(np.diagflat(np.arange(10)))

dip = np.zeros(h.shape)
dip[0,:] = dip[:,0] = np.random.rand(nstates)

dip =  Qobj(dip)
c_op = dip

l = h.to_super() + 1j * c_op.to_linblad(gamma=0.02)


ntrans = 3 * nstates # number of transitions
eigvals1, U1 = eigs(l, k=ntrans, which='LR')

def sort(eigvals, eigvecs):

    idx = np.argsort(eigvals)

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    return eigvals, eigvecs

eigvals1, U1 = sort(eigvals1, U1)

print(eigvals1.real)

omegas = np.linspace(0.1 , 10.5, 200)

rho0  =  Qobj(dims=[10,10])
rho0.data[0,0] = 1.0

ops = [sz, sz]

# out = correlation_2p_1t(omegas, rho0, ops, L)
# print(eigvecs)
eigvals2, U2 = eigs(dag(l), k=ntrans, which='LR')


eigvals2, U2 = sort(eigvals2, U2)

#idx = np.where(eigvals2.real > 0.2)[0]
#print(idx)


norm = [np.vdot(U2[:,n], U1[:,n]) for n in range(ntrans)]

la = np.zeros(len(omegas), dtype=complex) # linear absorption
for j, omega in enumerate(omegas):

    for n in range(ntrans):

        la[j] += np.vdot(dip.to_vector(), U1[:,n]) * \
             np.vdot(U2[:,n], dip.dot(rho0).to_vector()) \
             /(omega - eigvals1[n]) / norm[n]


fig, ax = plt.subplots()
# ax.scatter(eigvals1.real, eigvals1.imag)
ax.plot(omegas, -2 * la.imag)


