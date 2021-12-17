#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 15:27:12 2021

@author: Bing Gu

EM field pragators, largely from pyGDM2 package

"""

import copy

import numpy as np
import math
import cmath
import numba
from numba import cuda
# =============================================================================
# numba compatible propagators (CPU + CUDA)
# =============================================================================

@numba.njit(cache=True)
def G0(R1, R2, lamda, eps):
    """
    free space propagator for a dipole source at R2

    G_NF =
    G_MF =
    G_FF =

    Parameters
    ----------
    R1 : 1d array or list
        observation point
    R2 : TYPE
        source
    lamda : TYPE
        wavelength
    eps : TYPE
        dielectric constant

    Returns
    -------
    Dyadic Green's tensor G_{ij}, i, j = {x, y, z}

    """
    Dx = R2[0] - R1[0]
    Dy = R2[1] - R1[1]
    Dz = R2[2] - R1[2]
    lR = math.sqrt(Dx**2 + Dy**2 + Dz**2)

    k = 2*np.pi / lamda
    cn2 = cmath.sqrt(eps)
    ck0 = -1j * k * cn2
    k2 = k*k*eps

    r25 = math.pow((Dx*Dx+Dy*Dy+Dz*Dz), 2.5)
    r2 = math.pow((Dx*Dx+Dy*Dy+Dz*Dz), 2.0)
    r15 = math.pow((Dx*Dx+Dy*Dy+Dz*Dz), 1.5)

#!C-------------------------------------------------------------------
    T1XX = (2*Dx*Dx-Dy*Dy-Dz*Dz) / r25
    T2XX = (2*Dx*Dx-Dy*Dy-Dz*Dz) / r2
    T3XX = -1*(Dy*Dy+Dz*Dz) / r15
#!C-------------------------------------------------------------------
    T1XY = 3*Dx*Dy / r25
    T2XY = 3*Dx*Dy / r2
    T3XY = Dx*Dy / r15
#!C-------------------------------------------------------------------
    T1XZ = 3*Dx*Dz / r25
    T2XZ = 3*Dx*Dz / r2
    T3XZ = Dx*Dz / r15
#!C-------------------------------------------------------------------
    T1YY = (2*Dy*Dy-Dx*Dx-Dz*Dz) / r25
    T2YY = (2*Dy*Dy-Dx*Dx-Dz*Dz) / r2
    T3YY = -(Dx*Dx+Dz*Dz) / r15
#!C-------------------------------------------------------------------
    T1YZ = 3*Dy*Dz / r25
    T2YZ = 3*Dy*Dz / r2
    T3YZ = Dy*Dz / r15
#!C------------------------------------------------------------------
    T1ZZ = (2*Dz*Dz-Dx*Dx-Dy*Dy) / r25
    T2ZZ = (2*Dz*Dz-Dx*Dx-Dy*Dy) / r2
    T3ZZ = -(Dx*Dx+Dy*Dy) / r15

    CFEXP = cmath.exp(1j*k*cn2*lR)


    ## setting up the tensor
    xx = CFEXP*(T1XX + ck0*T2XX - k2*T3XX)/eps
    yy = CFEXP*(T1YY+ck0*T2YY-k2*T3YY)/eps
    zz = CFEXP*(T1ZZ+ck0*T2ZZ-k2*T3ZZ)/eps

    xy = CFEXP*(T1XY+ck0*T2XY-k2*T3XY)/eps
    xz = CFEXP*(T1XZ+ck0*T2XZ-k2*T3XZ)/eps

    yz = CFEXP*(T1YZ+ck0*T2YZ-k2*T3YZ)/eps

    yx = xy
    zx = xz
    zy = yz

    return xx, yy, zz, xy, xz, yx, yz, zx, zy

## --- "1-2-3" slab propagator, via method of mirror charges for 2 surfaces
@numba.njit(cache=True)
def Gs123(R1, R2, lamda, eps1, eps2, eps3, spacing):
    Dx = R2[0] - R1[0]
    Dy = R2[1] - R1[1]
    Dz = R2[2] + R1[2]

    cdelta12 = (eps1-eps2)/(eps1+eps2)
    cdelta23 = (eps3-eps2)/(eps3+eps2)

#!************* Interface: (1,2) *******************************
    r25 = math.pow((Dx*Dx+Dy*Dy+Dz*Dz), 2.5)

    SXX12 = cdelta12*(Dz*Dz+Dy*Dy-2*Dx*Dx) / r25
    SYY12 = cdelta12*(Dz*Dz+Dx*Dx-2*Dy*Dy) / r25
    SZZ12 = cdelta12*(2*Dz*Dz-Dx*Dx-Dy*Dy) / r25
    SXY12 = cdelta12*(-3*Dx*Dy) / r25
    SXZ12 = cdelta12*(3*Dx*Dz) / r25
    SYZ12 = cdelta12*(3*Dy*Dz) / r25

#!************* Interface: (2,3) *******************************
    GZ = Dz - 2*spacing
    rgz25 = math.pow((Dx*Dx+Dy*Dy+GZ*GZ), 2.5)

    SXX23 = cdelta23*(GZ*GZ+Dy*Dy-2*Dx*Dx) / rgz25
    SYY23 = cdelta23*(GZ*GZ+Dx*Dx-2*Dy*Dy) / rgz25
    SZZ23 = cdelta23*(2*GZ*GZ-Dx*Dx-Dy*Dy) / rgz25
    SXY23 = cdelta23*(-3*Dx*Dy) / rgz25
    SXZ23 = cdelta23*(3*Dx*GZ) / rgz25
    SYZ23 = cdelta23*(3*Dy*GZ) / rgz25
#!**************************************************************

    xx = SXX12+SXX23
    yy = SYY12+SYY23
    zz = SZZ12+SZZ23

    xy = SXY12+SXY23
    xz = SXZ12+SXZ23

    yx = xy
    yz = SYZ12+SYZ23

    zx = -1*xz
    zy = -1*yz

    return xx, yy, zz, xy, xz, yx, yz, zx, zy


## --- the full propagator: vacuum + surface term -- nearfield approximation!
@numba.njit(cache=True)
def G(R1, R2, lamda, eps1, eps2, eps3, spacing):
    xx, yy, zz, xy, xz, yx, yz, zx, zy = G0(R1, R2, lamda, eps2)
    xxs,yys,zzs,xys,xzs,yxs,yzs,zxs,zys = Gs123(R1, R2, lamda,
                                                    eps1, eps2, eps3, spacing)

    return xx+xxs, yy+yys, zz+zzs, \
           xy+xys, xz+xzs, yx+yxs, \
           yz+yzs, zx+zxs, zy+zys

def G0_1D(z1, z2, k, eps):
    """
    Green's function for 1D homogenous media

    Parameters
    ----------
    z1 : TYPE
        DESCRIPTION.
    z2 : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    eps : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    d = abs(z1-z2)
    k = k * sqrt(eps)
    return np.exp(1j * k * d) * 1j/(2. * k)


if __name__ == '__main__':
    from math import sqrt
    from lime.units import c0

    R2 = np.array([0, 0, 0])
    lamda = 0.5
    R1s = np.linspace(0.2, 3, 100)
    g0 = np.zeros(len(R1s), dtype=complex)
    N = len(R1s)

    # for i in range(N):
    #     R1 = np.array([R1s[i], 0., 0])
    #     g0[i] = G0(R1, R2, lamda, 1.0)[0]

    # from lime.style import subplots
    # fig, ax = subplots()
    # ax.plot(R1s, g0.real)

    # 1D
    nz = 128 * 2
    d = 100 # length, nm

    z = np.linspace(-d, d, nz)
    dz = z[1] - z[0]

    nmodes = np.arange(1, 10)
    # lamda  = 2 * L/nmodes

    eps0 = np.array([1.] * nz) # background
    eps1 = np.zeros(nz, dtype=complex) # scattering region
    eps1[:] = 4.

    wavelengths = np.linspace(100, 400, 200) # nm
    ks = 2. * np.pi/wavelengths
    #ks = np.linspace(0.005, 0.04, 150)
    gk = np.zeros(len(ks), dtype=complex)

    # g0 = np.zeros((nz, nz), dtype=complex)

    for n, k in enumerate(ks):

        e0 = k**2 * np.diagflat(eps0)
        e1 = k**2 * np.diagflat(eps1)

        I = np.identity(nz)
        Z1, Z2 = np.meshgrid(z, z)
        # for i in range(nz):
        #     for j in range(i+1):
        #         g0[i,j] = G0_1D(z[i], z[j], k, eps=1)
        #         g0[j,i] = g0[i,j]

        g0 = G0_1D(Z1, Z2, k, eps=1)

        g = np.linalg.inv(I - g0 @ e1 * dz) @ g0

        gk[n] = g[nz//2+1, nz//2+1]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    c1 = c0/sqrt(5.)

    ax.plot(ks * c0, gk.imag)

    ax.vlines(c1 * np.pi * nmodes/d, ymin=0, ymax=8)


