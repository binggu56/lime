#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:16:53 2020

@author: Bing Gu

Basic module for molecules
"""
from __future__ import absolute_import

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, identity, kron, linalg

import numba
import sys

from lime.units import au2fs, au2ev
from lime.signal import sos

from lime.phys import driven_dynamics, quantum_dynamics

class Mol:
    def __init__(self, ham, edip, gamma=None):
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
        self.gamma[0] = 0 # ground state decay is 0
        return

    def get_ham(self):
        return self.H

    def getH(self):
        return self.H

    def get_nonhermitianH(self):
        H = self.H
        return H - 1j * np.diagflat(self.gamma)

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

    def eigenstates(self, sparse=False, k=6):
        if sparse == False:
            return np.linalg.eigh(self.H.toarray())
        else:
            eigvals, eigvecs = linalg.eigs(self.H, k=k, which='SR')
            idx = eigvals.argsort()[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:,idx]

            return eigvals, eigvecs

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
                        obs_ops=obs_ops, nout=nout, t0=t0)

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
                        obs_ops=obs_ops, nout=nout, t0=t0)




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

    def linear_absorption(self, omegas, method='SOS', gamma=1./au2ev, \
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

            return sos.linear_absorption(omegas, eigvals, dip=dip, gamma=gamma,\
                                   normalize=normalize)

        elif method == 'superoperator':
            pass

        else:
            raise ValueError('The method {} has not been implemented. \
                             Try "SOS"'.format(method))

class SESolver:
    def __init__(self, H):
        """
        Basic class for time-depedent Schodinger equation.

        Parameters
        ----------
        H : array
            Hamiltonian.

        Returns
        -------
        None.

        """
        self.H = H
        return

    def solve(self, psi0, edip=None, pulse=None, dt=0.001, Nt=1, \
              e_ops=None, nout=1, t0=0.0):
        '''
        quantum dynamics under time-independent and time-dependent hamiltonian

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
            initial time. The default is 0.

        Returns
        -------
        None.

        '''
        if psi0 is None:
            raise ValueError("Initial state not specified!")

        H = self.H

        if pulse is None:
            return quantum_dynamics(H, psi0, dt=dt, Nt=Nt, \
                        obs_ops=e_ops, nout=nout, t0=t0)
        else:
            if edip is None:
                raise ValueError('Electric dipole not specified for \
                                 laser-driven dynamics.')

            return driven_dynamics(H, edip, psi0, pulse, dt=dt, Nt=Nt, \
                        obs_ops=e_ops, nout=nout, t0=t0)

    def correlation_2p_1t(self):
        pass
