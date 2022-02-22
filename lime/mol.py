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

import proplot as plt

from lime.units import au2fs, au2ev
from lime.signal import sos
from lime.phys import dag, quantum_dynamics, \
    obs, rk4, tdse, basis, isdiag

from lime.units import au2wavenumber


class Result:
    def __init__(self, description=None, psi0=None, rho0=None, dt=None, \
                 Nt=None, times=None):
        self.description = description
        self.dt = dt
        self.timesteps = Nt
        self.observables = None
        self.rholist = None
        self._psilist = None
        self.rho0 = rho0
        self.psi0 = psi0
        self.times = np.arange(Nt) * dt
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
    def __init__(self, ham, edip=None, edip_rms=None, gamma=None):
        """
        Class for multi-level systems.

        All signals computed using SOS formula can be directly called from the
        Mol objective.

        More sophisticated ways to compute the spectra should be done with
        specific method, e.g., MESolver.

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
        self.nonhermH = None
        self.h = ham
        #        self.initial_state = psi0
        self.edip = edip
        self.dip = self.edip
        if edip is not None:
            self.edip_x = edip[:,:, 0]
            self.edip_y = edip[:, :, 1]
            self.edip_z = edip[:, :, 2]
            self.edip_rms = np.sqrt(np.abs(edip[:, :, 0])**2 + \
                                    np.abs(edip[:, :, 1])**2 + \
                                        np.abs(edip[:,:,2])**2)
        else:
            self.edip_rms = edip_rms

        self.nstates = ham.shape[0]

        # self.raising = np.tril(edip)
        # self.lowering = np.triu(edip)

        self.idm = identity(ham.shape[0])
        self.size = ham.shape[0]
        self.dim = ham.shape[0]
        # if tau is not None:
        #     self.lifetime = tau
        #     self.decay = 1./tau
        self.gamma = gamma
        self.mdip = None
        self.dephasing = 0.

    # def raising(self):
    #     return self.raising

    # def lowering(self):
    #     return self.lowering

    def set_dip(self, dip):
        self.dip = dip
        return

    def set_dipole(self, dip):
        self.dip = dip
        return

    def set_edip(self, edip, pol=None):
        self.edip_rms = edip
        return

    def set_mdip(self, mdip):
        self.mdip = mdip
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

    def set_dephasing(self, gamma):
        """
        set the pure dephasing rate for all coherences

        Parameters
        ----------
        gamma : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.dephasing = gamma

    def set_decay(self, gamma):
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
        self.gamma = gamma
        return


    def get_ham(self):
        return self.H


    def getH(self):
        return self.H

    def get_nonhermitianH(self):
        H = self.H

        if self.gamma is not None:

            self.nonhermH = H - 1j * np.diagflat(self.gamma)

        else:
            raise ValueError('Please set gamma first.')

        return self.nonhermH

    def get_nonhermH(self):
        H = self.H

        if self.gamma is not None:

            self.nonhermH = H - 1j * np.diagflat(self.gamma)

        else:
            raise ValueError('Please set gamma first.')

        return self.nonhermH

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

    def eigvals(self):
        if isdiag(self.H):
            return np.diagonal(self.H)
        else:
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
            raise ValueError("Please specify initial wavefunction psi0.")

        H = self.H

        if pulse is None:
            return quantum_dynamics(H, psi0, dt=dt, Nt=Nt, \
                                    obs_ops=obs_ops, nout=nout, t0=t0)
        else:

            edip = self.edip
            return driven_dynamics(H, edip, psi0, pulse, dt=dt, Nt=Nt, \
                                   e_ops=obs_ops, nout=nout, t0=t0)

    # def heom(self, env, nado=5, c_ops=None, obs_ops=None, fname=None):
    #     nt = self.nt
    #     dt = self.dt
    #     return _heom(self.oqs, env, c_ops, nado, nt, dt, fname)

    # def redfield(self, env, dt, Nt, c_ops, obs_ops, rho0, integrator='SOD'):
    #     nstates = self.nstates

    #     h0 = self.H

    #     _redfield(nstates, rho0, c_ops, h0, Nt, dt,obs_ops, env, integrator='SOD')
    #     return

    # def tcl2(self, env, c_ops, obs_ops, rho0, dt, Nt, integrator='SOD'):

    #     nstates = self.nstates
    #     h0 = self.H

    #     redfield(nstates, rho0, c_ops, h0, Nt, dt,obs_ops, env, integrator='SOD')

    #     return

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

    def tcl2(self):
        """
        second-order time-convolutionless quantum master equation
        """
        pass

    def absorption(self, omegas, method='sos', **kwargs):
        '''
        Linear absorption of the model.

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
        if method == 'sos':
            # eigvals = self.eigvals()
            # dip = self.dip
            # gamma = self.gamma

            return sos.absorption(self, omegas, **kwargs)

        elif method == 'superoperator':

            raise NotImplementedError('The method {} has not been implemented. \
                              Try "sos"'.format(method))

    def PE(self, pump, probe, t2=0.0, **kwargs):
        '''
        alias for photon_echo

        '''
        return self.photon_echo(pump, probe, t2=t2, **kwargs)

    def photon_echo(self, pump, probe, t2=0.0, **kwargs):
        '''
        2D photon echo signal at -k1+k2+k3

        Parameters
        ----------
        pump : TYPE
            DESCRIPTION.
        probe : TYPE
            DESCRIPTION.
        t2 : TYPE, optional
            DESCRIPTION. The default is 0.0.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        # E = self.eigvals()
        # edip = self.edip

        return sos.photon_echo(self, pump=pump, probe=probe, \
                                t2=t2, **kwargs)

    def PE2(self, omega1, omega2, t3=0.0, **kwargs):
        '''
        2D photon echo signal at -k1+k2+k3
        Transforming t1 and t2 to frequency domain.

        Parameters
        ----------
        pump : TYPE
            DESCRIPTION.
        probe : TYPE
            DESCRIPTION.
        t2 : TYPE, optional
            DESCRIPTION. The default is 0.0.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''

        return sos.photon_echo_t3(self, omega1=omega1, omega2=omega2, \
                                t3=t3, **kwargs)

    def DQC(self):
        pass

    def TPA(self):
        pass

    def ETPA(self):
        pass

    def cars(self, shift, omega1, t2=0., plt_signal=False, fname=None):

        edip = self.edip_rms
        E = self.eigvals()

        S = sos.cars(E, edip, shift, omega1, t2=t2)

        if not plt_signal:

            return S

        else:

            fig, ax = plt.subplots(figsize=(8,4))

            ax.contourf(shift*au2wavenumber, omega1*au2ev, S.imag.T, lw=0.6,\
                        cmap='spectral')

            ax.format(xlabel='Raman shift (cm$^{-1}$)', ylabel=r'$\Omega_1$ (eV)')

            fig.savefig(fname+'.pdf')

            return S, fig, ax

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

    def correlation_2op_1t(self):
        pass

    def correlation_3op_1t(self, psi0, oplist, dt, Nt):
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

    def correlation_3op_2t(self, psi0, oplist, dt, Nt, Ntau):
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

    def correlation_4op_1t(self, psi0, oplist, dt=0.005, Nt=1):
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
        return self.correlation_3op_1t(psi0, [a_op, b_op @ c_op, d_op], dt, Nt)

    def correlation_4op_2t(self, psi0, oplist, dt=0.005, Nt=1, Ntau=1):
        """

        Parameters
        ----------
        psi0 : vector
            initial state
        oplist : list of arrays
        """
        a_op, b_op, c_op, d_op = oplist
        return self.correlation_3op_2t(psi0, [a_op, b_op @ c_op, d_op], dt, Nt, Ntau)


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
        Ulist.append(U.copy())
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

    # f_obs = open(output, 'w')  # observables
    # f_wf = open('psi.dat', 'w')

    t = t0

    # f_obs.close()

    if store_states:

        result = Result(dt=dt, Nt=Nt, psi0=psi0)

        observables = np.zeros((Nt // nout, len(e_ops)), dtype=complex)
        psilist = [psi0.copy()]

        # compute observables for t0
        print(psi.shape, e_ops[0].shape)

        observables[0, :] = [obs(psi.toarray(), e_op) for e_op in e_ops]

        for k1 in range(1, Nt // nout):

            for k2 in range(nout):
                psi = rk4(psi, tdse, dt, H)

            t += dt * nout

            # compute observables
            observables[k1, :] = [obs(psi.toarray(), e_op) for e_op in e_ops]

            # f_obs.write(fmt.format(t, *e_list))

            psilist.append(psi.copy())

        # f_obs.close()

        result.psilist = psilist
        result.observables = observables

        return result

    else:  # not save states

        f_obs = open(output, 'w')  # observables

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


def driven_dynamics(H, edip, psi0, pulse, dt=0.001, Nt=1, e_ops=None, nout=1, \
                    t0=0.0, return_result=True):
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
    psi = psi0.astype(complex)

    nstates = len(psi0)

    # f = open(fname,'w')

    if e_ops is None:
        e_ops = []

    t = t0

    # f_dip = open('dipole'+'{:f}'.format(pulse.delay * au2fs)+'.dat', 'w')
    if return_result:

        result = Result(dt=dt, Nt=Nt, psi0=psi0)

        observables = np.zeros((Nt // nout, len(e_ops)), dtype=complex)
        psilist = [psi0.copy()]

        # compute observables for t0
        observables[0, :] = [obs(psi, e_op) for e_op in e_ops]

        for k1 in range(1, Nt // nout):

            for k2 in range(nout):

                ht = -pulse.field(t) * edip + H
                psi = rk4(psi, tdse, dt, ht)

            t += dt * nout

            # compute observables
            observables[k1, :] = [obs(psi, e_op) for e_op in e_ops]
            # f_obs.write(fmt.format(t, *e_list))

            psilist.append(psi.copy())

        # f_obs.close()

        result.psilist = psilist
        result.observables = observables

        return result

    else:

        fmt = '{} ' * (len(e_ops) + 1) + '\n'
        fmt_dm = '{} ' * (nstates + 1) + '\n'

        f_dm = open('psi.dat', 'w')  # wavefunction
        f_obs = open('obs.dat', 'w')  # observables

        for k1 in range(int(Nt / nout)):

            for k2 in range(nout):
                ht = pulse.field(t) * edip + H
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

def mls(dim=3):

    E = np.array([0., 0.6, 1.0])/au2ev
    N = len(E)

    gamma = np.array([0, 0.002, 0.002])/au2ev
    H = np.diag(E)

    dip = np.zeros((N, N, 3), dtype=complex)
    dip[1,2, :] = [1.+0.5j, 1.+0.1j, 0]

    dip[2,1, :] = np.conj(dip[1, 2, :])
    # dip[1,3, :] = dip[3,1] = 1.

    dip[0, 1, :] = [1.+0.2j, 0.5-0.1j, 0]
    dip[1, 0, :] = np.conj(dip[0, 1, :])

    # dip[3, 3] = 1.
    dip[0, 2, :] = dip[2, 0, :] = [0.5, 1, 0]

    mol = Mol(H, dip)
    mol.set_decay(gamma)

    return mol

def test_sesolver():
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


if __name__ == '__main__':
    from lime.phys import pauli
    import time
    import proplot as plt

    mol = mls()
    mol.set_decay([0, 0.002, 0.002])
    omegas=np.linspace(0, 2, 200)/au2ev
    shift = np.linspace(0, 1)/au2ev

    # mol.absorption(omegas)
    # mol.photon_echo(pump=omegas, probe=omegas, plt_signal=True)
    S = mol.cars(shift=shift, omega1=omegas, plt_signal=True)


