#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:04:43 2021

Three-state two-mode linear vibronic model of pyrazine (S0/S1/S2)

@author: bing
"""


import numpy as np
import numba
from numpy import cos, sin
from scipy.sparse import identity, coo_matrix, lil_matrix, csr_matrix, kron

import sys
# sys.path.append(r'C:\Users\Bing\Google Drive\lime')
#sys.path.append(r'/Users/bing/Google Drive/lime')

from lime.phys import boson, dag, sort
from lime.style import set_style
from lime.units import au2ev, wavenumber2hartree, wavenum2au


def pos(n_vib):
    """
    position matrix elements <n|Q|n'>
    """
    X = np.zeros((n_vib, n_vib))

    for i in range(1, n_vib):
        X[i, i-1] = np.sqrt(i/2.)
    for i in range(n_vib-1):
        X[i, i+1] = np.sqrt((i+1)/2.)

    return X


def vibronic(n_el, n_vc, n_vt):
    """
    contruct vibronic-cavity basis for polaritonic states, that is a
    direct product of electron-vibration-photon space
    """

    #n_basis = n_el * n_cav * n_vc * n_vt

    freq_vc = 952. * wavenumber2hartree
    freq_vt = 597. * wavenumber2hartree

    Eshift = np.array([0.0, 31800.0, 39000]) * wavenumber2hartree
    kappa = np.array([0.0, -847.0, 1202.]) * wavenumber2hartree
    coup = 2110.0 * wavenumber2hartree # inter-state coupling lambda


    # indentity matrices in each subspace
    I_el = identity(n_el)
    I_vc = identity(n_vc)
    I_vt = identity(n_vt)

    h_vt = boson(freq_vt, n_vt, ZPE=False)
    h_vc = boson(freq_vc, n_vc, ZPE=False)

    h_el = np.diagflat(Eshift)

    # the bare term in the system Hamiltonian
    h0 = kron(h_el, kron(I_vc, I_vt)) + kron(I_el, kron(h_vc, I_vt)) +\
         kron(I_el, kron(I_vc, h_vt))

    X = pos(n_vt)

    h1 = kron(np.diagflat(kappa), kron(I_vc, X))

    Xc = pos(n_vc)

    trans_el = np.zeros((n_el, n_el)) # electronic excitation operator
    #deex = np.zeros((n_el, n_el)) # deexcitation
    #deex[2, 1] = 1.0
    #ex[1, 2] = 1.0
    trans_el[1,2] = trans_el[2,1] = 1.0 #= ex + deex

    #h_fake = kron(np.diagflat(kappa), kron(I_cav, kron(Xc, I_vt)))

    ###
#    h_m = np.zeros((n_basis, n_basis))
#
#    for m, b0 in enumerate(basis_set):
#        for n, b1 in enumerate(basis_set):
##            h_m[m,n] = h_el[b0.n_el, b1.n_el] * h_cav[b0.n_cav, b1.n_cav] * h_vc[b0.n_vc, b1.n_vc]
#            h_m[m,n] = trans_el[b0.n_el, b1.n_el] * I_cav[b0.n_cav, b1.n_cav] \
#                    * Xc[b0.n_vc, b1.n_vc] * I_vt[b0.n_vt, b1.n_vt]

    h2 = coup * kron(trans_el, kron(Xc, I_vt), format='csr')


    h_s = h0 + h1 + h2

    # polaritonic states can be obtained by diagonalizing H_S
    # v is the basis transformation matrix, v[i,j] = <old basis i| polaritonic state j>
    #eigvals, v = np.linalg.eigh(h_s)

    # collapse operators in dissipative dynamics
    # Sc = kron(I_el, kron(Xc, I_vt), format='csr')
    # St = kron(I_el, kron(I_vc, X), format='csr')

    return h_s

def DPES(x, y, nstates=2):
    """
    Diabatic PES

    Parameters
    ----------
    x : TYPE
        qc coupling mode coordinate
    y : TYPE
        qt tuning mode coordinate

    Returns
    -------
    2D array
        molecular Hamiltonian

    """
    freq_vc = 1000. * wavenum2au
    freq_vt = 1000. * wavenum2au

    Eshift = np.array([0, 1000]) * wavenum2au
    kappa = np.array([-1000.0, 1000.]) * wavenum2au

    V0 = freq_vc * x**2/2. + freq_vt * y**2/2 + kappa[0] * y
    V1 = freq_vc * x**2/2 + freq_vt * y**2/2 + kappa[1] * y

    coup = 2000 * x * wavenum2au

    V = np.array([[V0, coup], [coup, V1]])

    return V


def get_apes(x, y):
    """
    calculate adiabatic PES from diabatic PES

    input:
        R: 1d array with length n_dof
    output:
        V: same size as R, potential energy
    """



    #A0 = np.zeros(len(x))
    #A1 = np.zeros(len(x))

    #for i in range(len(x)):
    V = DPES(x, y)

    E, U = np.linalg.eigh(V)
    E, U = sort(E, U)

    return E, U


def cut():
    x = 0
    y = np.linspace(-8,6,100)

    dpes = DPES(x, y)

    fig, ax = plt.subplots(figsize=(4,4))
    set_style(13)

    for surface in dpes:
        ax.plot(y, surface * au2ev, lw=2)
    #ax.plot(y, (dpes[1] - dpes[0]) * au2ev, label='0-1')
    #ax.plot(y, (dpes[2]- dpes[0]) * au2ev, label='0-2')

    #ax.legend()
    #ax.set_ylim(4.31, 4.32)
    #ax.grid()
    ax.set_ylabel('Energy (eV)')
    ax.set_xlabel('Tuning mode')

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig('dpes.pdf', dpi=1200, transparent=True)
    plt.show()


def plot3d():

    #data = [go.Surface(z=apes)]
    #fig = go.Figure(data = data)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5,4))
    set_style(fontsize=14)

    ax = fig.add_subplot(111, projection='3d')
    #py.iplot(fig)

    x = np.linspace(-4, 4)
    y = np.linspace(-4, 4)

    apes = np.zeros((len(x), len(y)))
    apes1 = np.zeros((len(x), len(y)))
    apes2 = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            apes1[i,j], apes2[i,j] = get_apes(x[i], y[j])[0]

    X, Y = np.meshgrid(x, y)

    for surface in [apes1, apes2]:
        ax.plot_surface(X, Y, surface * au2ev, rstride=1, cstride=1, cmap='viridis',\
                    edgecolor='k',
                    linewidth=0.1)

    #surf(ground)
#    ax.plot_surface(X, Y, apes1 * au2ev, rstride=6, cstride=6, cmap='viridis', edgecolor='k'\
#                    , linewidth=0.5)
#
#    ax.plot_surface(X, Y, apes2 * au2ev, rstride=6, cstride=6, cmap='viridis', edgecolor='k'\
#                    , linewidth=0.5)

    ax.view_init(10, -60)
    ax.set_zlim(0, 4)
    ax.set_xlabel(r'Couping mode')
    ax.set_ylabel(r'Tuning mode')

    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel('Energy (eV)', rotation=90)

    #fig.subplots_adjust(top=0.95, bottom=0.16,left=0.16, right=0.9)

    plt.savefig('apes_3d.pdf')

    plt.show()


def geometric_phase():

    # polar coordinates in the nuclear space
    import proplot as plt

    r = 1
    N = 100
    thetas = np.linspace(np.pi/2, 5/2*np.pi, N)
    dtheta = thetas[1] - thetas[0]

    U = np.zeros((2, 2, len(thetas)))
    apes = np.zeros((2, N))

    for j, theta in enumerate(thetas):
        x = r * cos(theta)
        y = r * sin(theta)
        apes[:, j], u = get_apes(x, y)

        U[:,:, j] = u

        print('DAT mixing angle =', np.arccos(u[0, 0])/np.pi)


    # ax.format(ylim=(-1, 1))

    # overlap matrix
    nac = np.zeros(N) # derivative coupling <I(R)|J(R+dR)>

    adt = 0. # ADT mixing angle

    for j in range(N-1):
        nac[j] = (dag(U[:,:, j]).dot(U[:, :, j+1]))[0, 1]
        adt += nac[j]
        print('ADT mixing angle', adt/np.pi)

    nac[-1] = dag(U[:,:, -1]).dot(U[:, :, 0])[0, 1]

    print(np.sum(nac)/np.pi)

    # APEs
    fig, ax = plt.subplots()
    ax.plot(thetas, apes[0, :])
    ax.plot(thetas, apes[1, :])


    fig, ax = plt.subplots()
    ax.plot(thetas, U[0, 0, :])
    ax.plot(thetas, nac)


    return


def ADT(apes, nac):
    angle = 0.


def contour():

    #data = [go.Surface(z=apes)]
    #fig = go.Figure(data = data)


    x = np.linspace(-6, 6, 200)
    y = np.linspace(-4, 4, 200)

    apes = np.zeros((len(x), len(y)))
    apes1 = np.zeros((len(x), len(y)))
    apes2 = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            apes[i,j], [apes1[i,j], apes2[i,j]] = get_apes(x[i], y[j])

    X, Y = np.meshgrid(x, y)

    for j, surface in enumerate([apes, apes1, apes2]):

        # fig, ax = plt.subplots()

        fig, ax = matplot(x, y, surface.T * au2ev, cmap='inferno')

    #ax.contour(apes1)
    #surf(ground)
#    ax.plot_surface(X, Y, apes1 * au2ev, rstride=6, cstride=6, cmap='viridis', edgecolor='k'\
#                    , linewidth=0.5)
#
#    ax.plot_surface(X, Y, apes2 * au2ev, rstride=6, cstride=6, cmap='viridis', edgecolor='k'\
#                    , linewidth=0.5)

        ax.set_xlabel(r'Tuning mode $Q_\mathrm{t}$')
        ax.set_ylabel(r'Coupling mode $Q_\mathrm{c}$')

        #ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        #ax.set_zlabel('Energy (eV)', rotation=90)

        fig.subplots_adjust(top=0.95, bottom=0.16,left=0.16, right=0.95)

        plt.savefig('apes{}_contour.pdf'.format(j))

    return

def mayavi(surfaces):


    from mayavi import mlab

    apes, apes1 = surfaces

    fig = mlab.figure()

    surf2 = mlab.surf(apes * au2ev, warp_scale=20)
    surf3 = mlab.surf(apes1 * au2ev, warp_scale=20)
    #mlab.surf(ground * au2ev, warp_scale=20)



    mlab.axes(xlabel = 'Coupling mode', ylabel = 'Tuning mode')
    #mlab.view(60, 74, 17, [-2.5, -4.6, -0.3])

    mlab.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import rcParams

    rcParams['axes.labelpad'] = 6
    rcParams['xtick.major.pad']='2'
    rcParams['ytick.major.pad']='2'

    # mayavi()
    # contour()
    # cut()
    geometric_phase()
