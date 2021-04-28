#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:16:53 2020

@author: Bing Gu

Basic module for molecules
"""
from __future__ import absolute_import

from typing import Union, Iterable

import numpy as np
import scipy.integrate
from numpy.core._multiarray_umath import ndarray
from scipy.sparse import csr_matrix, lil_matrix, identity, kron, linalg

import numba
import sys

from lime.units import au2fs, au2ev
from lime.signal import sos
from lime.phys import dag, driven_dynamics, quantum_dynamics, \
    obs, rk4, tdse, basis


class Result:
    def __init__(self, description=None, psi0=None, rho0=None, dt=None, Nt=None):
        self.description = description
        self.dt = dt
        self.timesteps = Nt
        self.observables = None
        self.rholist = None
        self._psilist = None
        self.rho0 = rho0
        self.psi0 = psi0
        self.times = None
        return

    @property
    def psilist(self):
        return self._psilist

    @psilist.setter
    def psilist(self, psilist):
        self._psilist = psilist

    def expect(self):
        return self.observables

    # def times(self):
    #     if dt is not None & Nt is not None:
    #         return np.arange(self.Nt) * self.dt
    #     else:
    #         sys.exit("ERROR: Either dt or Nt is None.")


class Mol:
    def __init__(self, ham, edip=None, gamma=None):
        """
        Class for multi-level systems.

        Parameters
        ----------
        ham : TYPE
            DESCRIPTION.
        dip : TYPE
            DESCRIPTION.
        rho : TYPE, optional
            DESCRIPTION. The default is None.
        tau: 1d array
            lifetime of energy levels
        Returns
        -------
        None.

        """
        self.H = ham
        self.h = ham
        #        self.initial_state = psi0
        self.edip = edip
        self.dip = self.edip
        self.nstates = ham.shape[0]
        #        self.ex = np.tril(dip)
        #        self.deex = np.triu(dip)
        self.idm = identity(ham.shape[0])
        self.size = ham.shape[0]
        self.dim = ham.shape[0]
        # if tau is not None:
        #     self.lifetime = tau
        #     self.decay = 1./tau
        self.gamma = gamma

    def set_dip(self, dip):
        self.dip = dip
        return

    def set_dipole(self, dip):
        self.dip = dip
        return

    def set_edip(self, edip):
        self.edip = edip
        return

    def setH(self, H):
        '''
        Set model Hamiltonian

        Parameters
        ----------
        H : 2d array
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.H = H
        return

    def ground(self):
        return basis(self.dim, 0)

    def set_decay_for_all(self, gamma):
        """
        Set the decay rate for all excited states.

        Parameters
        ----------
        gamma : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.gamma = [gamma] * self.nstates
        self.gamma[0] = 0  # ground state decay is 0
        return

    def get_ham(self):
        return self.H

    def getH(self):
        return self.H

    def get_nonhermitianH(self):
        H = self.H
        if self.gamma is not None:
            return H - 1j * np.diagflat(self.gamma)
        else:
            return H

    def get_dip(self):
        return self.dip

    def get_edip(self):
        return self.edip

    def get_dm(self):
        return self.dm

    def set_lifetime(self, tau):
        self.lifetime = tau

    def eigenenergies(self):
        return np.linalg.eigvals(self.H)

    def eigenstates(self, k=6):
        """

        Parameters
        ----------
        k: integer
            number of eigenstates to compute, < dim

        Returns
        -------
        eigvals: vector
        eigvecs: 2d array
        """
        if self.H is None:
            raise ValueError('Call getH to compute H first.')

        if k < self.dim:
            eigvals, eigvecs = linalg.eigs(self.H, k=k, which='SR')
            idx = eigvals.argsort()[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            return eigvals, eigvecs

        if k == self.dim:
            return np.linalg.eigh(self.H.toarray())

    def driven_dynamics(self, pulse, dt=0.001, Nt=1, obs_ops=None, nout=1, t0=0.0):
        '''
        wavepacket dynamics in the presence of laser pulses

        Parameters
        ----------
        pulse : TYPE
            DESCRIPTION.
        dt : TYPE, optional
            DESCRIPTION. The default is 0.001.
        Nt : TYPE, optional
            DESCRIPTION. The default is 1.
        obs_ops : TYPE, optional
            DESCRIPTION. The default is None.
        nout : TYPE, optional
            DESCRIPTION. The default is 1.
        t0: float
            initial time

        Returns
        -------
        None.

        '''
        H = self.H
        dip = self.dip
        psi0 = self.initial_state

        if psi0 is None:
            sys.exit("Error: Initial wavefunction not specified!")

        driven_dynamics(H, dip, psi0, pulse, dt=dt, Nt=Nt, \
                        e_ops=obs_ops, nout=nout, t0=t0)

        return

    def quantum_dynamics(self, psi0, dt=0.001, Nt=1, obs_ops=None, nout=1, t0=0.0):
        '''
        quantum dynamics under time-independent hamiltonian

        Parameters
        ----------
        pulse : TYPE
            DESCRIPTION.
        dt : TYPE, optional
            DESCRIPTION. The default is 0.001.
        Nt : TYPE, optional
            DESCRIPTION. The default is 1.
        obs_ops : TYPE, optional
            DESCRIPTION. The default is None.
        nout : TYPE, optional
            DESCRIPTION. The default is 1.
        t0: float
            initial time

        Returns
        -------
        None.

        '''
        ham = self.H

        quantum_dynamics(ham, psi0, dt=dt, Nt=Nt, \
                         obs_ops=obs_ops, nout=nout, t0=t0)

        return

    def evolve(self, psi0, pulse=None, dt=0.001, Nt=1, obs_ops=None, nout=1, t0=0.0):
        '''
        quantum dynamics under time-independent hamiltonian

        Parameters
        ----------
        pulse : TYPE
            DESCRIPTION.
        dt : TYPE, optional
            DESCRIPTION. The default is 0.001.
        Nt : TYPE, optional
            DESCRIPTION. The default is 1.
        obs_ops : TYPE, optional
            DESCRIPTION. The default is None.
        nout : TYPE, optional
            DESCRIPTION. The default is 1.
        t0: float
            initial time

        Returns
        -------
        None.

        '''
        if psi0 is None:
            raise ValueError("Initial wavefunction not specified!")

        H = self.H

        if pulse is None:
            return quantum_dynamics(H, psi0, dt=dt, Nt=Nt, \
                                    obs_ops=obs_ops, nout=nout, t0=t0)
        else:

            edip = self.edip
            return driven_dynamics(H, edip, psi0, pulse, dt=dt, Nt=Nt, \
                                   e_ops=obs_ops, nout=nout, t0=t0)

    # def heom(self, env, nado=5, c_ops, obs_ops, fname=None):
    #     nt = self.nt
    #     dt = self.dt
    #     return heom(self.oqs, env, c_ops, nado, nt, dt, fname)

    # def redfield(self, env, dt, Nt, c_ops, obs_ops, rho0, integrator='SOD'):
    #     nstates = self.nstates

    #     h0 = self.H

    #     redfield(nstates, rho0, c_ops, h0, Nt, dt,obs_ops, env, integrator='SOD')
    #     return

    # # def tcl2(self, env, c_ops, obs_ops, rho0, dt, Nt, integrator='SOD'):

    # #     nstates = self.nstates
    # #     h0 = self.H

    # #     redfield(nstates, rho0, c_ops, h0, Nt, dt,obs_ops, env, integrator='SOD')

    # #     return

    # def lindblad(self, c_ops, obs_ops, rho0, dt, Nt):
    #     """
    #     lindblad quantum master equations

    #     Parameters
    #     ----------
    #     rho0: 2D array
    #         initial density matrix
    #     """

    #     h0 = self.H
    #     lindblad(rho0, h0, c_ops, Nt, dt, obs_ops)
    #     return

    # def steady_state(self):
    #     return

    # def correlation_2p_1t(self, rho0, ops, dt, Nt, method='lindblad', output='cor.dat'):
    #     '''
    #     two-point correlation function <A(t)B>

    #     Parameters
    #     ----------
    #     rho0 : TYPE
    #         DESCRIPTION.
    #     ops : TYPE
    #         DESCRIPTION.
    #     dt : TYPE
    #         DESCRIPTION.
    #     Nt : TYPE
    #         DESCRIPTION.
    #     method : TYPE, optional
    #         DESCRIPTION. The default is 'lindblad'.
    #     output : TYPE, optional
    #         DESCRIPTION. The default is 'cor.dat'.

    #     Returns
    #     -------
    #     None.

    #     '''

    #     H = self.hamiltonian
    #     c_ops = self.c_ops

    #     correlation_2p_1t(H, rho0, ops=ops, c_ops=c_ops, dt=dt, Nt=Nt, \
    #                       method=method, output=output)

    #     return

    # def tcl2(self):
    #     """
    #     second-order time-convolutionless quantum master equation
    #     """
    #     pass

    def linear_absorption(self, omegas, method='SOS', gamma=1. / au2ev, \
                          c_ops=None, normalize=True):
        '''
        Linear absorption of the model

        Parameters
        ----------
        omegas : TYPE
            DESCRIPTION.
        method : TYPE, optional
            DESCRIPTION. The default is 'SOS'.
        normalize : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if method == 'SOS':
            eigvals = self.eigvals()
            dip = self.dip

            return sos.linear_absorption(omegas, eigvals, dip=dip, gamma=gamma, \
                                         normalize=normalize)

        elif method == 'superoperator':
            pass

        else:
            raise ValueError('The method {} has not been implemented. \
                             Try "SOS"'.format(method))


class SESolver:
    def __init__(self, H, isherm=True):
        """
        Basic class for time-dependent Schrodinger equation.

        Parameters
        ----------
        H : array
            Hamiltonian.

        Returns
        -------
        None.

        """
        self.H = H
        self._isherm = isherm
        return

    def evolve(self, psi0, dt=0.001, Nt=1,
               e_ops=None, nout=1, t0=0.0, edip=None, pulse=None):
        '''
        quantum dynamics under time-independent and time-dependent hamiltonian

        Parameters
        ----------
        psi0: 1d array
            initial state
        pulse : TYPE
            DESCRIPTION.

        dt : TYPE, optional
            DESCRIPTION. The default is 0.001.
        Nt : TYPE, optional
            DESCRIPTION. The default is 1.
        obs_ops : TYPE, optional
            DESCRIPTION. The default is None.
        nout : TYPE, optional
            DESCRIPTION. The default is 1.
        t0: float
            initial time. The default is 0.

        Returns
        -------
        None.

        '''
        # if psi0 is None:
        #     raise ValueError("Initial state not specified!")

        H = self.H

        if pulse is None:
            return _quantum_dynamics(H, psi0, dt=dt, Nt=Nt,
                                     e_ops=e_ops, nout=nout, t0=t0)
        else:
            if edip is None:
                raise ValueError('Electric dipole not specified for \
                                 laser-driven dynamics.')

            return driven_dynamics(H, edip, psi0, pulse, dt=dt, Nt=Nt, \
                                   e_ops=e_ops, nout=nout, t0=t0)

    def propagator(self, dt, Nt):
        H = self.H
        return _propagator(H, dt, Nt)

    def correlation_2p_1t(self):
        pass

    def correlation_3p_1t(self, psi0, oplist, dt, Nt):
        """
        <AB(t)C>

        Parameters
        ----------
        psi0
        oplist
        dt
        Nt

        Returns
        -------

        """
        H = self.H
        a_op, b_op, c_op = oplist

        corr_vec = np.zeros(Nt, dtype=complex)

        psi_ket = _quantum_dynamics(H, c_op @ psi0, dt=dt, Nt=Nt).psilist
        psi_bra = _quantum_dynamics(H, dag(a_op) @ psi0, dt=dt, Nt=Nt).psilist

        for j in range(Nt):
            corr_vec[j] = np.vdot(psi_bra[j], b_op @ psi_ket[j])

        return corr_vec

    def correlation_3p_2t(self, psi0, oplist, dt, Nt, Ntau):
        """
        <A(t)B(t+tau)C(t)>
        Parameters
        ----------
        oplist: list of arrays
            [a, b, c]
        psi0: array
            initial state
        dt
        nt: integer
            number of time steps for t
        ntau: integer
            time steps for tau

        Returns
        -------

        """
        H = self.H
        psi_t = _quantum_dynamics(H, psi0, dt=dt, Nt=Nt).psilist

        a_op, b_op, c_op = oplist

        corr_mat = np.zeros([Nt, Ntau], dtype=complex)

        for t_idx, psi in enumerate(psi_t):
            psi_tau_ket = _quantum_dynamics(H, c_op @ psi, dt=dt, Nt=Ntau).psilist
            psi_tau_bra = _quantum_dynamics(H, dag(a_op) @ psi, dt=dt, Nt=Ntau).psilist

            corr_mat[t_idx, :] = [np.vdot(psi_tau_bra[j], b_op @ psi_tau_ket[j])
                                  for j in range(Ntau)]

        return corr_mat

    def correlation_4p_1t(self, psi0, oplist, dt=0.005, Nt=1):
        """
        <AB(t)C(t)D>

        Parameters
        ----------
        psi0
        oplist
        dt
        Nt

        Returns
        -------

        """
        a_op, b_op, c_op, d_op = oplist
        return self.correlation_3p_1t(psi0, [a_op, b_op @ c_op, d_op], dt, Nt)

    def correlation_4p_2t(self, psi0, oplist, dt=0.005, Nt=1, Ntau=1):
        """

        Parameters
        ----------
        psi0 : vector
            initial state
        oplist : list of arrays
        """
        a_op, b_op, c_op, d_op = oplist
        return self.correlation_3p_2t(psi0, [a_op, b_op @ c_op, d_op], dt, Nt, Ntau)


def _propagator(H, dt, Nt):
    """
    compute the resolvent for the multi-point correlation function signals
    U(t) = e^{-i H t}
    Parameters
    -----------
    t: float or list
        times
    """

    # propagator
    U = identity(H.shape[-1], dtype=complex)

    # set the ground state energy to 0
    print('Computing the propagator. '
          'Please make sure that the ground-state energy is 0.')
    Ulist = []
    for k in range(Nt):
        Ulist.append(U)
        U = rk4(U, tdse, dt, H)

    return Ulist


def _quantum_dynamics(H, psi0, dt=0.001, Nt=1, e_ops=[], t0=0.0,
                      nout=1, store_states=True, output='obs.dat'):
    """
    Quantum dynamics for a multilevel system.

    Parameters
    ----------
    e_ops: list of arrays
        expectation values to compute.
    H : 2d array
        Hamiltonian of the molecule
    psi0: 1d array
        initial wavefunction
    dt : float
        time step.
    Nt : int
        timesteps.
    e_ops: list
        observable operators to compute
    nout: int
        Store data every nout steps

    Returns
    -------
    None.

    """

    psi = psi0.copy()

    if e_ops is not None:
        fmt = '{} ' * (len(e_ops) + 1) + '\n'

    f_obs = open(output, 'w')  # observables

    t = t0

    # f_obs.close()

    if store_states:

        result = Result(dt=dt, Nt=Nt, psi0=psi0)

        observables = np.zeros((Nt // nout, len(e_ops)), dtype=complex)
        psilist = [psi0.copy()]

        # compute observables for t0
        observables[0, :] = [obs(psi, e_op) for e_op in e_ops]

        for k1 in range(1, Nt // nout):

            for k2 in range(nout):
                psi = rk4(psi, tdse, dt, H)

            t += dt * nout

            # compute observables
            observables[k1, :] = [obs(psi, e_op) for e_op in e_ops]
            # f_obs.write(fmt.format(t, *e_list))

            psilist.append(psi.copy())

        # f_obs.close()

        result.psilist = psilist
        result.observables = observables

        return result

    else:  # not save states
        for k1 in range(int(Nt / nout)):
            for k2 in range(nout):
                psi = rk4(psi, tdse, dt, H)

            t += dt * nout

            # compute observables
            e_list = [obs(psi, e_op) for e_op in e_ops]
            f_obs.write(fmt.format(t, *e_list))

        f_obs.close()

        return psi


def _ode_solver(H, psi0, dt=0.001, Nt=1, e_ops=[], t0=0.0,
                nout=1, store_states=True, output='obs.dat'):
    """
    Integrate the TDSE for a multilevel system using Scipy.

    Parameters
    ----------
    e_ops: list of arrays
        expectation values to compute.
    H : 2d array
        Hamiltonian of the molecule
    psi0: 1d array
        initial wavefunction
    dt : float
        time step.
    Nt : int
        timesteps.
    e_ops: list
        observable operators to compute
    nout: int
        Store data every nout steps

    Returns
    -------
    None.

    """

    psi = psi0.copy()
    t_eval = np.arange(Nt // nout) * dt * nout

    def fun(t, psi): return -1j * H.dot(psi)

    tf = t0 + Nt * dt
    t_span = (t0, tf)
    sol = scipy.integrate.solve_ivp(fun, t_span=t_span, y0=psi,
                                    t_eval=t_eval)

    result = Result(dt=dt, Nt=Nt, psi0=psi0)
    result.times = sol.t
    print(sol.nfev)

    observables = np.zeros((Nt // nout, len(e_ops)), dtype=complex)
    for i in range(len(t_eval)):
        psi = sol.y[:, i]
        observables[i, :] = [obs(psi, e_op) for e_op in e_ops]

    result.observables = observables
    # if store_states:
    #
    #     result = Result(dt=dt, Nt=Nt, psi0=psi0)
    #
    #     observables = np.zeros((Nt // nout, len(e_ops)), dtype=complex)
    #     psilist = [psi0.copy()]
    #
    #     # compute observables for t0
    #     observables[0, :] = [obs(psi, e_op) for e_op in e_ops]
    #
    #     for k1 in range(1, Nt // nout):
    #
    #         for k2 in range(nout):
    #             psi = rk4(psi, tdse, dt, H)
    #
    #         t += dt * nout
    #
    #         # compute observables
    #         observables[k1, :] = [obs(psi, e_op) for e_op in e_ops]
    #         # f_obs.write(fmt.format(t, *e_list))
    #
    #         psilist.append(psi.copy())
    #
    #     # f_obs.close()
    #
    #     result.psilist = psilist
    #     result.observables = observables

    return result


def driven_dynamics(ham, dip, psi0, pulse, dt=0.001, Nt=1, e_ops=None, nout=1, \
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
    e_ops: list
        observable operators to compute
    nout: int
        Store data every nout steps

    Returns
    -------
    None.

    '''

    # initialize the density matrix
    # wf = csr_matrix(wf0).transpose()
    psi = psi0

    nstates = len(psi0)

    # f = open(fname,'w')
    fmt = '{} ' * (len(e_ops) + 1) + '\n'
    fmt_dm = '{} ' * (nstates + 1) + '\n'

    f_dm = open('psi.dat', 'w')  # wavefunction
    f_obs = open('obs.dat', 'w')  # observables

    t = t0

    # f_dip = open('dipole'+'{:f}'.format(pulse.delay * au2fs)+'.dat', 'w')

    for k1 in range(int(Nt / nout)):

        for k2 in range(nout):
            ht = pulse.field(t) * dip + ham
            psi = rk4(psi, tdse, dt, ht)

        t += dt * nout

        # compute observables
        Aave = np.zeros(len(e_ops), dtype=complex)
        for j, A in enumerate(e_ops):
            Aave[j] = obs(psi, A)

        f_dm.write(fmt_dm.format(t, *psi))
        f_obs.write(fmt.format(t, *Aave))

    f_dm.close()
    f_obs.close()

    return


if __name__ == '__main__':
    from lime.phys import pauli
    import time

    s0, sx, sy, sz = pauli()
    H = (-0.05) * sz - 0. * sx

    solver = SESolver(H)
    Nt = 1000
    dt = 4
    psi0 = (basis(2, 0) + basis(2, 1)) / np.sqrt(2)
    start_time = time.time()

    result = solver.evolve(psi0, dt=dt, Nt=Nt, e_ops=[sx])
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()

    times = np.arange(Nt) * dt
    r = _ode_solver(H, psi0, dt=dt, Nt=Nt, nout=2, e_ops=[sx])

    print("--- %s seconds ---" % (time.time() - start_time))

    # test correlation functions
    # corr = solver.correlation_3p_1t(psi0=psi0, oplist=[s0, sx, sx],
    #                                 dt=0.05, Nt=Nt)
    #
    # times = np.arange(Nt)
    #
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(times, result.observables[:, 0])
    ax.plot(r.times, r.observables[:, 0], '--')
    # ax.plot(times, corr.real)
    plt.show()
