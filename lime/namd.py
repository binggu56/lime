#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Oct 10 11:14:55 2017

@author: Bing Gu

History:
2/12/18 : fix a bug with the FFT frequency

Several possible improvements:
    1. use pyFFTW to replace the Scipy


### not finished
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.fftpack import fft,ifft,fftshift
from scipy.linalg import expm, sinm, cosm
import scipy
from common import dagger

class NAMD:
    def __init__(self, x, psi0, m, ns=2, V_x):
        """
        Non-adiabatic molecular dynamics (NAMD) simulations for one nuclear dof
            and many electronic states.

        Args:
            x: real array of size N
                grid points

            psi0: complex array [N, ns]
                initial wavefunction

            m: float, nuclear mass

            ns: integer, number of states, default 2

            V_x: real array [N, ns**2]
                potential energy surfaces and vibronic couplings
        """
        self.x = x
        self.psi_x = psi0
        self.m = m
        self.V_x = V_x

    def x_evolve(self, dt, psi_x):

        x = self.x
        ns = self.ns
        V_x = self.V_x

        for i in range(len(x)):

            Vmat = np.reshape(V_x[i,:], (ns,ns))
            w, U = scipy.linalg.eigh(Vmat)

            #print(np.dot(U.conj().T, Vmat.dot(U)))

            V = np.diagflat(np.exp(- 1j * w * dt))

            psi_x[i,:] = np.dot(U,V.dot(dagger(U))).dot(self.psi_x[i,:])

        return psi_x


    def k_evolve(self, dt, k, psi_x):
        """
        one time step for exp(-i * K * dt)
        """
        m = self.m

        for n in range(2):

            psi_k = fft(psi_x[:,n])

            psi_k *= np.exp(-0.5 * 1j / m * (k * k) * dt)

            psi_x[:,n] = ifft(psi_k)

        return psi_x

    def evolution(self, dt, psi_x, Nsteps = 1):

        """
        Perform a series of time-steps via the time-dependent
        Schrodinger Equation.

        Parameters
        ----------
        dt : float
            the small time interval over which to integrate

        Nsteps : float, optional
            the number of intervals to compute.  The total change
            in time at the end of this method will be dt * Nsteps.
            default is N = 1
        """
        if dt > 0.0:
            f = open('density_matrix.dat', 'w')
        else:
            f = open('density_matrix_backward.dat', 'w')

        x = self.x
        N = len(x)
        dx = x[1] - x[0]

        psi_x = self.psi0

        dt2 = 0.5 * dt

        k = scipy.fftpack.fftfreq(N, dx)
        k[:] = 2.0 * np.pi * k[:]

        if Nsteps > 0:
            self.x_evolve(dt2, x, V_x, psi_x)

        for i in range(Nsteps - 1):

            t += dt

            psi_x = self.k_evolve(dt, k, psi_x)
            psi_x = self.x_evolve(dt, x, V_x, psi_x)

            rho = self.density_matrix(psi_x, dx)
            f.write('{} {} {} {} {} \n'.format(t, *rho))

        psi_x = self.k_evolve(dt, k, psi_x)
        psi_x = self.x_evolve(dt2, psi_x)

        t += dt

        f.close()

        return psi_x

    def density_matrix(psi_x,dx):
        """
        compute purity from the wavefunction
        """
        rho00 = np.sum(np.abs(psi_x[:,0])**2)*dx
        rho01 = np.vdot(psi_x[:,1], psi_x[:,0])*dx
        rho11 = 1. - rho00
        return rho00, rho01, rho01.conj(), rho11

######################################################################
# Helper functions for gaussian wave-packets

def gauss_x(x, a, x0, k0):
    """
    a gaussian wave packet of width a, centered at x0, with momentum k0
    """
    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k0))

def gauss_k(k,a,x0,k0):
    """
    analytical fourier transform of gauss_x(x), above
    """
    return ((a / np.sqrt(np.pi))**0.5
            * np.exp(-0.5 * (a * (k - k0)) ** 2 - 1j * (k - k0) * x0))


######################################################################
def theta(x):
    """
    theta function :
      returns 0 if x<=0, and 1 if x>0
    """
    x = np.asarray(x)
    y = np.zeros(x.shape)
    y[x > 0] = 1.0
    return y

def square_barrier(x, width, height):
    return height * (theta(x) - theta(x - width))




######################################################################


