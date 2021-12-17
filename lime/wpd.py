#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:44:15 2021

Wave packet dynamics solver for wavepacket dynamics with N vibrational modes
(N = 1 ,2)

For linear coordinates, use SPO method
For curvilinear coordinates, use RK4 method

@author: Bing Gu
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi
from numba import jit
from scipy.fftpack import fft2, ifft2, fftfreq
from numpy.linalg import inv, det

from lime.phys import rk4
from lime.units import au2fs

class Solver():
    def __init__(self):
        self.obs_ops = None
        self.grid = None

    def set_obs_ops(self, obs_ops):
        self.obs_ops = obs_ops
        return

class SPO_1d(Solver):
    def __init__(self):
        self.x = None
        self.v = None

    def set_grid(self, xmin=-1, xmax=1, npts=32):
        self.x = np.linspace(xmin, xmax, npts)

    def set_potential(self, potential):
        self.v = potential(self.x)
        return

    def evolve(self, psi0, dt, Nt=1):
        psi = psi0
        return psi


class SPO_2d():
    def __init__(self):
        self.x = None
        self.y = None

    def set_grid(self, x, y):
        self.x = x
        self.y = y

    def set_potential(self, v):
        return v

    def evolve(self, psi0, dt, Nt=1):
        psi = psi0
        return psi

class SPO_3d():
    def __init__(self):
        self.x = None
        self.y = None

    def set_grid(self, x, y):
        self.x = x
        self.y = y

    def set_potential(self, v):
        return v

    def evolve(self, psi0, dt, Nt=1):
        psi = psi0
        return psi

def gwp(x, a, x0=0., p0=0.):
    """
    a Gaussian wave packet centered at x0, with momentum k0
    """
    return (a/np.sqrt(np.pi))**(-0.25)*\
        np.exp(-0.5 * a * (x - x0)**2 + 1j * (x-x0) * p0)

@jit
def gauss_x_2d(sigma, x0, y0, kx0, ky0):
    """
    generate the gaussian distribution in 2D grid
    :param x0: float, mean value of gaussian wavepacket along x
    :param y0: float, mean value of gaussian wavepacket along y
    :param sigma: float array, covariance matrix with 2X2 dimension
    :param kx0: float, initial momentum along x
    :param ky0: float, initial momentum along y
    :return: gauss_2d: float array, the gaussian distribution in 2D grid
    """
    gauss_2d = np.zeros((len(x), len(y)), dtype=np.complex128)

    for i in range(len(x)):
        for j in range(len(y)):
            delta = np.dot(np.array([x[i]-x0, y[j]-y0]), inv(sigma))\
                      .dot(np.array([x[i]-x0, y[j]-y0]))
            gauss_2d[i, j] = (np.sqrt(det(sigma))
                              * np.sqrt(np.pi) ** 2) ** (-0.5) \
                              * np.exp(-0.5 * delta + 1j
                                       * np.dot(np.array([x[i], y[j]]),
                                                  np.array([kx0, ky0])))

    return gauss_2d


@jit
def potential_2d(x_range_half, y_range_half, couple_strength, couple_type):
    """
    generate two symmetric harmonic potentials wrt the origin point in 2D
    :param x_range_half: float, the displacement of potential from the origin
                                in x
    :param y_range_half: float, the displacement of potential from the origin
                                in y
    :param couple_strength: the coupling strength between these two potentials
    :param couple_type: int, the nonadiabatic coupling type. here, we used:
                                0) no coupling
                                1) constant coupling
                                2) linear coupling
    :return: v_2d: float list, a list containing for matrices:
                               v_2d[0]: the first potential matrix
                               v_2d[1]: the potential coupling matrix
                                        between the first and second
                               v_2d[2]: the potential coupling matrix
                                        between the second and first
                               v_2d[3]: the second potential matrix
    """
    v_2d = [0, 0, 0, 0]
    v_2d[0] = (xv + x_range_half) ** 2 / 2.0 + (yv + y_range_half) ** 2 / 2.0
    v_2d[3] = (xv - x_range_half) ** 2 / 2.0 + (yv - y_range_half) ** 2 / 2.0

    # x_cross = sympy.Symbol('x_cross')
    # mu = sympy.solvers.solve(
    #     (x_cross - x_range_half) ** 2 / 2.0 -
    #     (x_cross + x_range_half) ** 2 / 2.0,
    #     x_cross)

    if couple_type == 0:
        v_2d[1] = np.zeros(np.shape(v_2d[0]))
        v_2d[2] = np.zeros(np.shape(v_2d[0]))
    elif couple_type == 1:
        v_2d[1] = np.full((np.shape(v_2d[0])), couple_strength)
        v_2d[2] = np.full((np.shape(v_2d[0])), couple_strength)
    elif couple_type == 2:
        v_2d[1] = couple_strength * (xv+yv)
        v_2d[2] = couple_strength * (xv+yv)
    # elif couple_type == 3:
    #     v_2d[1] = couple_strength \
    #                 * np.exp(-(x - float(mu[0])) ** 2 / 2 / sigma ** 2)
    #     v_2d[2] = couple_strength \
    #                 * np.exp(-(x - float(mu[0])) ** 2 / 2 / sigma ** 2)
    else:
        raise 'error: coupling type not existing'

    return v_2d


@jit
def diabatic(x, y):
    """
    PESs in diabatic representation
    :param x_range_half: float, the displacement of potential from the origin
                                in x
    :param y_range_half: float, the displacement of potential from the origin
                                in y
    :param couple_strength: the coupling strength between these two potentials
    :param couple_type: int, the nonadiabatic coupling type. here, we used:
                                0) no coupling
                                1) constant coupling
                                2) linear coupling
    :return:
        v:  float 2d array, matrix elements of the DPES and couplings
    """
    nstates = 2

    v = np.zeros((nstates, nstates))

    v[0,0] = (x + 4.) ** 2 / 2.0 + (y + 3.) ** 2 / 2.0
    v[1,1] = (x - 4.) ** 2 / 2.0 + (y - 3.) ** 2 / 2.0

    v[0, 1] = v[1, 0] = 0

    return v

# @jit
# def x_evolve_half_2d(dt, v_2d, psi_grid):
#     """
#     propagate the state in grid basis half time step forward with H = V
#     :param dt: float
#                 time step
#     :param v_2d: float array
#                 the two electronic states potential operator in grid basis
#     :param psi_grid: list
#                 the two-electronic-states vibrational state in grid basis
#     :return: psi_grid(update): list
#                 the two-electronic-states vibrational state in grid basis
#                 after being half time step forward
#     """

#     for i in range(len(x)):
#         for j in range(len(y)):
#             v_mat = np.array([[v_2d[0][i, j], v_2d[1][i, j]],
#                              [v_2d[2][i, j], v_2d[3][i, j]]])

#             w, u = scipy.linalg.eigh(v_mat)
#             v = np.diagflat(np.exp(-0.5 * 1j * w / hbar * dt))
#             array_tmp = np.array([psi_grid[0][i, j], psi_grid[1][i, j]])
#             array_tmp = np.dot(u.conj().T, v.dot(u)).dot(array_tmp)
#             psi_grid[0][i, j] = array_tmp[0]
#             psi_grid[1][i, j] = array_tmp[1]
#             #self.x_evolve = self.x_evolve_half * self.x_evolve_half
#             #self.k_evolve = np.exp(-0.5 * 1j * self.hbar / self.m * \
#             #               (self.k * self.k) * dt)


@jit
def x_evolve_2d(dt, psi, v):
    """
    propagate the state in grid basis half time step forward with H = V
    :param dt: float
                time step
    :param v_2d: float array
                the two electronic states potential operator in grid basis
    :param psi_grid: list
                the two-electronic-states vibrational state in grid basis
    :return: psi_grid(update): list
                the two-electronic-states vibrational state in grid basis
                after being half time step forward
    """


    vpsi = np.exp(- 1j * v * dt) * psi


    return vpsi


def k_evolve_2d(dt, kx, ky, psi):
    """
    propagate the state in grid basis a time step forward with H = K
    :param dt: float, time step
    :param kx: float, momentum corresponding to x
    :param ky: float, momentum corresponding to y
    :param psi_grid: list, the two-electronic-states vibrational states in
                           grid basis
    :return: psi_grid(update): list, the two-electronic-states vibrational
                                     states in grid basis
    """

    psi_k = fft2(psi)
    mx, my = mass

    Kx, Ky = np.meshgrid(kx, ky)

    kin = np.exp(-1j * (Kx**2/2./mx + Ky**2/2./my) * dt)

    psi_k = kin * psi_k
    psi = ifft2(psi_k)

    return psi


def dpsi(psi, kx, ky, ndim=2):
    '''
    Momentum operator operates on the wavefunction

    Parameters
    ----------
    psi : 2D complex array
        DESCRIPTION.
    ndim : int, default 2
        coordinates dimension
    Returns
    -------
    kpsi : (nx, ny, ndim)
        DESCRIPTION.

    '''

    # Fourier transform of the wavefunction
    psi_k = fft2(psi)

    # momentum operator in the Fourier space
    kxpsi = np.einsum('i, ij -> ij', kx, psi_k)
    kypsi = np.einsum('j, ij -> ij', ky, psi_k)

    kpsi = np.zeros((nx, ny, ndim), dtype=complex)

    # transform back to coordinate space
    kpsi[:,:,0] = ifft2(kxpsi)
    kpsi[:,:,1] = ifft2(kypsi)

    return kpsi

def dxpsi(psi):
    '''
    Momentum operator operates on the wavefunction

    Parameters
    ----------
    psi : 2D complex array
        DESCRIPTION.

    Returns
    -------
    kpsi : (nx, ny, ndim)
        DESCRIPTION.

    '''

    # Fourier transform of the wavefunction
    psi_k = fft2(psi)

    # momentum operator in the Fourier space
    kxpsi_k = np.einsum('i, ij -> ij', kx, psi_k)

    # transform back to coordinate space
    kxpsi = ifft2(kxpsi_k)

    return kxpsi

def dypsi(psi):
    '''
    Momentum operator operates on the wavefunction

    Parameters
    ----------
    psi : 2D complex array
        DESCRIPTION.

    Returns
    -------
    kpsi : (nx, ny, ndim)
        DESCRIPTION.

    '''

    # Fourier transform of the wavefunction
    psi_k = fft2(psi)

    # momentum operator in the Fourier space
    kxpsi_k = np.einsum('i, ij -> ij', kx, psi_k)

    # transform back to coordinate space
    kxpsi = ifft2(kxpsi_k)

    return kxpsi


def adiabatic_2d(x, y, psi0, v, dt, Nt=0, coords='linear', mass=None, G=None):
    """
    propagate the adiabatic dynamics at a single surface

    :param dt: time step
    :param v: 2d array
                potential matrices in 2D
    :param psi: list
                the initial state
    mass: list of 2 elements
        reduced mass

    Nt: int
        the number of the time steps, Nt=0 indicates that no propagation has been done,
                   only the initial state and the initial purity would be
                   the output

    G: 4D array nx, ny, ndim, ndim
        G-matrix

    :return: psi_end: list
                      the final state

    G: 2d array
        G matrix only used for curvilinear coordinates
    """
    #f = open('density_matrix.dat', 'w')
    t = 0.0
    dt2 = dt * 0.5

    psi = psi0.copy()

    nx, ny = psi.shape

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    kx = 2. * np.pi * fftfreq(nx, dx)
    ky = 2. * np.pi * fftfreq(ny, dy)

    if coords == 'linear':
        # Split-operator method for linear coordinates

        psi = x_evolve_2d(dt2, psi,v)

        for i in range(Nt):
            t += dt
            psi = k_evolve_2d(dt, kx, ky, psi)
            psi = x_evolve_2d(dt, psi, v)

    elif coords == 'curvilinear':

        # kxpsi = np.einsum('i, ijn -> ijn', kx, psi_k)
        # kypsi = np.einsum('j, ijn -> ijn', ky, psi_k)

        # tpsi = np.zeros((nx, ny, nstates), dtype=complex)
        # dxpsi = np.zeros((nx, ny, nstates), dtype=complex)
        # dypsi = np.zeros((nx, ny, nstates), dtype=complex)

        # for i in range(nstates):

        #     dxpsi[:,:,i] = ifft2(kxpsi[:,:,i])
        #     dypsi[:,:,i] = ifft2(kypsi[:,:,i])

        for k in range(Nt):
            t += dt
            psi = rk4(psi, hpsi, dt, kx, ky, v, G)

        #f.write('{} {} {} {} {} \n'.format(t, *rho))
        #purity[i] = output_tmp[4]



    # t += dt
    #f.close()

    return psi

def KEO(psi, kx, ky, G):
    '''
    compute kinetic energy operator K * psi

    Parameters
    ----------
    psi : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
#    kpsi = dpsi(psi, kx, ky)

    # Fourier transform of the wavefunction
    psi_k = fft2(psi)

    # momentum operator in the Fourier space
    kxpsi = np.einsum('i, ij -> ij', kx, psi_k)
    kypsi = np.einsum('j, ij -> ij', ky, psi_k)

    nx, ny = len(kx), len(ky)
    kpsi = np.zeros((nx, ny, 2), dtype=complex)

    # transform back to coordinate space
    kpsi[:,:,0] = ifft2(kxpsi)
    kpsi[:,:,1] = ifft2(kypsi)

#   ax.contour(x, y, np.abs(kpsi[:,:,1]))

    tmp = np.einsum('ijrs, ijs -> ijr', G, kpsi)
    #G = metric_tensor(x[i], y[j]) # 2 x 2 matrix metric tensor at (x, y)

    # Fourier transform of the wavefunction
    phi_x = tmp[:,:,0]
    phi_y = tmp[:,:,1]

    phix_k = fft2(phi_x)
    phiy_k = fft2(phi_y)

    # momentum operator in the Fourier space
    kxphi = np.einsum('i, ij -> ij', kx, phix_k)
    kyphi = np.einsum('j, ij -> ij', ky, phiy_k)

    # transform back to coordinate space
    kxphi = ifft2(kxphi)
    kyphi = ifft2(kyphi)

    # psi += -1j * dt * 0.5 * (kxphi + kyphi)

    return 0.5 * (kxphi + kyphi)

def PEO(psi, v):
    """
    V |psi>
    :param dt: float
                time step
    :param v_2d: float array
                the two electronic states potential operator in grid basis
    :param psi_grid: list
                the two-electronic-states vibrational state in grid basis
    :return: psi_grid(update): list
                the two-electronic-states vibrational state in grid basis
                after being half time step forward
    """


    vpsi = v * psi
    return vpsi

def hpsi(psi, kx, ky, v, G):

    kpsi = KEO(psi, kx, ky, G)
    vpsi = PEO(psi, v)

    return -1j * (kpsi + vpsi)

######################################################################
# Helper functions for gaussian wave-packets


def gauss_k(k,a,x0,k0):
    """
    analytical fourier transform of gauss_x(x), above
    """
    return ((a / np.sqrt(np.pi))**0.5
            * np.exp(-0.5 * (a * (k - k0)) ** 2 - 1j * (k - k0) * x0))

@jit
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

@jit
def density_matrix(psi_grid):
    """
    compute electronic purity from the wavefunction
    """
    rho00 = np.sum(np.multiply(np.conj(psi_grid[0]), psi_grid[0]))*dx*dy
    rho01 = np.sum(np.multiply(np.conj(psi_grid[0]), psi_grid[1]))*dx*dy
    rho11 = np.sum(np.multiply(np.conj(psi_grid[1]), psi_grid[1]))*dx*dy

    purity = rho00**2 + 2*rho01*rho01.conj() + rho11**2

    return rho00, rho01, rho01.conj(), rho11, purity



if __name__ == '__main__':

    # specify time steps and duration
    ndim = 2 # 2D problem, DO NOT CHANGE!
    dt = 0.01
    print('time step = {} fs'.format(dt * au2fs))

    num_steps = 2000


    nx = 2 ** 6
    ny = 2 ** 6
    xmin = -8
    xmax = -xmin
    ymin = -8
    ymax = -ymin
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # k-space grid
    kx = 2. * np.pi * fftfreq(nx, dx)
    ky = 2. * np.pi * fftfreq(ny, dy)

    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots()
    v = 0.5 * (X**2 + Y**2)

    # for i in range(nx):
    #     for j in range(ny):
    #         v[i,j] = diabatic(x[i], y[j])[0,0]

    #ax.imshow(v)

    # specify constants
    mass = [1.0, 1.0]  # particle mass

    x0, y0, kx0, ky0 = -3, -1, 2.0, 0

    #coeff1, phase = np.sqrt(0.5), 0

    print('x range = ', x[0], x[-1])
    print('dx = {}'.format(dx))
    print('number of grid points along x = {}'.format(nx))
    print('y range = ', y[0], y[-1])
    print('dy = {}'.format(dy))
    print('number of grid points along y = {}'.format(ny))

    sigma = np.identity(2) * 0.5

    psi0 =   gauss_x_2d(sigma, x0, y0, kx0, ky0)

    fig, ax = plt.subplots()
    ax.contour(x, y, np.abs(psi0).T)

    #psi = psi0

    # propagate

    # store the final wavefunction
    #f = open('wft.dat','w')
    #for i in range(N):
    #    f.write('{} {} {} \n'.format(x[i], psi_x[i,0], psi_x[i,1]))
    #f.close()


    G = np.zeros((nx, ny, ndim, ndim))
    G[:,:,0, 0] = G[:,:,1, 1] = 1.

    fig, ax = plt.subplots()
    extent=[xmin, xmax, ymin, ymax]

    psi1 = adiabatic_2d(x, y, psi0, v, dt=dt, Nt=num_steps, coords='curvilinear',G=G)
    ax.contour(x,y, np.abs(psi1).T)

    fig, ax = plt.subplots()

    psi2 = adiabatic_2d(psi0, v, mass=mass, dt=dt, Nt=num_steps)
    ax.contour(x,y, np.abs(psi2).T)
