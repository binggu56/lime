'''
SOS formula for computing the nonlinear signals
'''

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

import sys
sys.path.append(r'C:\Users\Bing\Google Drive\lime')
sys.path.append(r'/Users/bing/Google Drive/lime')

from lime.phys import lorentzian
from lime.units import au2ev


def linear_absorption(omegas, transition_energies, dip, output=None, \
                      gamma=1./au2ev, normalize=False):
    '''
    SOS for linear absorption signal S = 2 pi |mu_{fg}|^2 delta(omega - omega_{fg}).
    The delta function is replaced with a Lorentzian function.

    Parameters
    ----------
    omegas : 1d array
        the frequency range for the signal
    transition_energies : TYPE
        DESCRIPTION.
    dip : 2d array
        dipole matrix
    output : TYPE, optional
        DESCRIPTION. The default is None.
    gamma : float, optional
        Lifetime broadening. The default is 1./au2ev.
    normalize : TYPE, optional
        Normalize the maximum intensity of the signal as 1. The default is False.

    Returns
    -------
    signal : 1d array
        linear absorption signal at omegas

    '''

    signal = 0.0


    for j, transition_energy in enumerate(transition_energies):

        signal += dip[j]**2 * lorentzian(omegas, transition_energy, gamma)


    if normalize:
        signal /= max(signal)

    if output is not None:

        fig, ax = plt.subplots(figsize=(4,3))

        ax.plot(omegas * au2ev, signal)

        scale = 1./np.sum(dip**2)

        for transition_energy in transition_energies:
            ax.axvline(transition_energy * au2ev, 0., dip[j]**2 * scale, color='grey')


        ax.set_xlim(min(omegas) * au2ev, max(omegas) * au2ev)
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('Absorption')
        #ax.set_yscale('log')
        # ax.set_ylim(0, 1)

        fig.subplots_adjust(wspace=0, hspace=0, bottom=0.15, \
                            left=0.17, top=0.96, right=0.96)
        fig.savefig(output, dpi=1200, transparent=True)

    return signal

def TPA(E, dip, omegap, g_idx, e_idx, f_idx, gamma, degenerate=True):
    """
    TPA signal with classical light
    """
    if degenerate:
        omega1 = omegap * 0.5
        omega2 = omegap - omega1

    i = 0

    signal = 0

    for f in f_idx:

        tmp = 0.0

        for m in e_idx:

             p1 = dip[f, m] * dip[m, i] / (omega1 - (E[m] - E[i]) + 1j * gamma[m])
             p2 = dip[f, m] * dip[m, i] /(omega2 - (E[m] - E[i]) + 1j * gamma[m])
             # if abs(p1) > 10:
             #      print('0 -> photon a -> {} -> photon b -> {}'.format(m, f), p1)
             #      print('0 -> photon b -> {} -> photon a -> {}'.format(m, f), p2)

             tmp += (p1 + p2)

        signal += np.abs(tmp)**2 * lorentzian(omegap - E[f] + E[i], width=gamma[f])

    return signal


def TPA2D(E, dip, omegaps, omega1s, g_idx, e_idx, f_idx, gamma):
    """
    2D two-photon-absorption signal with classical light scanning the omegap = omega1 + omega2 and omega1
    """

    g = 0

    signal = np.zeros((len(omegaps), len(omega1s)))

    for i, omegap in enumerate(omegaps):

        for j, omega1 in enumerate(omega1s):

            omega2 = omegap - omega1

            for f in f_idx:

                tmp = 0.

                for m in e_idx:

                     tmp += dip[f, m] * dip[m, g] * ( 1./(omega1 - (E[m] - E[g]) + 1j * gamma[m])\
                      + 1./(omega2 - (E[m] - E[g]) + 1j * gamma[m]) )

                signal[i,j] += np.abs(tmp)**2 * lorentzian(omegap - E[f] + E[g], width=gamma[f])

    return signal

def TPA2D_time_order(E, dip, omegaps, omega1s, g_idx, e_idx, f_idx, gamma):
    """
    2D two-photon-absorption signal with classical light scanning the omegap = omega1 + omega2 and omega1
    """

    g = 0

    signal = np.zeros((len(omegaps), len(omega1s)))

    for i in  range(len(omegaps)):
        omegap = omegaps[i]

        for j in range(len(omega1s)):
            omega1 = omega1s[j]

            omega2 = omegap - omega1

            for f in f_idx:

                tmp = 0.
                for m in e_idx:
                     tmp += dip[f, m] * dip[m, g] * 1./(omega1 - (E[m] - E[g]) + 1j * gamma[m])

                signal[i,j] += np.abs(tmp)**2 * lorentzian(omegap - E[f] + E[g], width=gamma[f])

    return signal

def gaussian(x, width):
    return np.exp(-(x/width)**2)

def GF(E, a, b, t):
    '''
    Retarded propagator of the element |a><b| for time t

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if t >= 0:
        # N = len(evals)
        # propagator = np.zeros((N,N), dtype=complex)

        # for a in range(N):
        #     for b in range(N):
        #         propagator[a, b] = np.exp(-1j * (evals[a] - evals[b]) * t - (gamma[a] + gamma[b])/2. * t)

        # return propagator
        return  -1j * np.exp(-1j * (E[a] - E[b]) * t)
    else:
        return 0.

@jit
def G(omega, E, a, b):
    '''
    Green's function in the frequency domain, i.e., FT of the retarded propagator

    Parameters
    ----------
    omega : TYPE
        DESCRIPTION.
    evals : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return 1./(omega - (E[a]- E[b]))


def ESA(evals, dip, omega1, omega3, tau2, g_idx, e_idx, f_idx, gamma):
    '''
    Excited state absorption component of the photon echo signal.
    In Liouville sapce, gg -> ge -> e'e -> fe -> ee

    Parameters
    ----------
    evals : TYPE
        DESCRIPTION.
    dip : TYPE
        DESCRIPTION.
    omega3 : TYPE
        DESCRIPTION.
    t2 : TYPE
        DESCRIPTION.
    omega1 : TYPE
        DESCRIPTION.
    g_idx: list of integers
        index for ground state (manifold)
    e_idx: list of integers
        index for e-states
    f_idx: list of integers
        index of f-states

    Returns
    -------
    signal : 2d array (len(pump), len(probe))
        DESCRIPTION.

    '''

    signal = np.zeros((len(omega1), len(omega3)), dtype=complex)
    a = 0 # initial state

    for i in range(len(omega1)):
        pump = omega1[i]

        for j in range(len(omega3)):
            probe = omega3[j]

            # sum-over-states
            for b in e_idx:

                G_ab = 1./(pump - (evals[a]-evals[b]) + 1j * (gamma[a] + gamma[b])/2.0)

                for c in e_idx:
                    U_cb = -1j * np.exp(-1j * (evals[c] - evals[b]) * tau2 - (gamma[c] + gamma[b])/2. * tau2)

                    for d in f_idx:

                        G_db = 1./(probe - (evals[d]-evals[b]) + 1j * (gamma[d] + gamma[b])/2.0)

                        signal[i,j] += dip[b,a] * dip[c,a] * dip[d,c]* dip[b,d] * \
                            G_db * U_cb * G_ab

    # 1 interaction in the bra side
    sign = -1
    return sign * signal


def GSB(evals, dip, omega1, omega3, tau2, g_idx, e_idx, gamma):
    '''
    gg -> ge -> gg' -> e'g' -> g'g'

    Parameters
    ----------
    evals : TYPE
        DESCRIPTION.
    dip : TYPE
        DESCRIPTION.
    omega3 : TYPE
        DESCRIPTION.
    t2 : TYPE
        DESCRIPTION.
    omega1 : TYPE
        DESCRIPTION.
    g_idx: list of integers
        indexes for ground manifold
    e_idx: list of integers
        indexes for excited state manifold

    Returns
    -------
    chi : TYPE
        DESCRIPTION.

    '''

    signal = np.zeros((len(omega1), len(omega3)), dtype=complex)
    a = 0
    for i in range(len(omega1)):
        pump = omega1[i]

        for j in range(len(omega3)):
            probe = omega3[j]
            # sum-over-states
            for b in e_idx:

                G_ab = 1./(pump - (evals[a]-evals[b]) + 1j * (gamma[a] + gamma[b])/2.0)

                for c in g_idx:
                    U_ac = -1j * np.exp(-1j * (evals[a] - evals[c]) * tau2 - (gamma[a] + gamma[c])/2. * tau2)

                    for d in e_idx:

                        G_dc = 1./(probe - (evals[d]-evals[c]) + 1j * (gamma[d] + gamma[c])/2.0)

                        signal[i,j] += dip[a,b] * dip[b,c] * dip[c,d]* dip[d,a] * \
                            G_dc * U_ac * G_ab
    return signal


def SE(evals, dip, omega1, omega3, tau2, g_idx, e_idx, gamma):
    '''
    Stimulated emission gg -> ge -> e'e -> g'e -> g'g' in the impulsive limit.
    The signal wave vector is ks = -k1 + k2 + k3

    Parameters
    ----------
    evals : TYPE
        DESCRIPTION.
    dip : TYPE
        DESCRIPTION.
    omega3 : TYPE
        DESCRIPTION.
    t2 : TYPE
        DESCRIPTION.
    omega1 : TYPE
        DESCRIPTION.
    g_idx: list of integers
        indexes for ground manifold
    e_idx: list of integers
        indexes for excited state manifold

    Returns
    -------
    chi : TYPE
        DESCRIPTION.

    '''

    signal = np.zeros((len(omega1), len(omega3)), dtype=complex)
    a = 0
    for i in range(len(omega1)):
        pump = omega1[i]

        for j in range(len(omega3)):
            probe = omega3[j]
            # sum-over-states
            for b in e_idx:

                G_ab = 1./(pump - (evals[a]-evals[b]) + 1j * (gamma[a] + gamma[b])/2.0)

                for c in e_idx:
                    U_cb = -1j * np.exp(-1j * (evals[c] - evals[b]) * tau2 - (gamma[c] + gamma[b])/2. * tau2)

                    for d in g_idx:

                        G_cd = 1./(probe - (evals[c]-evals[d]) + 1j * (gamma[c] + gamma[d])/2.0)

                        signal[i,j] += dip[a,b] * dip[c,a] * dip[d,c]* dip[b, d] * \
                            G_cd * U_cb * G_ab
    return signal


def DQC_R1(evals, dip, omega1=None, omega2=[], omega3=None, tau1=None, tau3=None,\
           g_idx=[0], e_idx=None, f_idx=None, gamma=None):
    '''
    Double quantum coherence, diagram 1:
        gg -> eg -> fg -> fe' -> e'e' in the impulsive limit.
    The signal wave vector is ks = k1 + k2 - k3

    Parameters
    ----------
    evals : TYPE
        DESCRIPTION.
    dip : TYPE
        DESCRIPTION.
    omega3 : TYPE
        DESCRIPTION.
    t2 : TYPE
        DESCRIPTION.
    omega1 : TYPE
        DESCRIPTION.
    g_idx: list of integers
        indexes for ground manifold
    e_idx: list of integers
        indexes for excited state manifold

    Returns
    -------
    chi : TYPE
        DESCRIPTION.

    '''

    a = 0
    if omega3 is None and tau3 is not None:

        signal = np.zeros((len(omega1), len(omega2)), dtype=complex)

        for i in range(len(omega1)):
            pump = omega1[i]

            for j in range(len(omega2)):
                probe = omega2[j]

                # sum-over-states
                for b in e_idx:

                    G_ba = 1./(probe - (evals[b]-evals[a]) + 1j * (gamma[b] + gamma[a])/2.0)


                    for c in f_idx:
                        G_ca = 1./(probe - (evals[c]-evals[a]) + 1j * (gamma[c] + gamma[a])/2.0)


                        for d in e_idx:

                            U_cd = -1j * np.exp(-1j * (evals[c] - evals[d]) * tau3 - (gamma[c] + gamma[d])/2. * tau3)


                            signal[i,j] += dip[b, a] * dip[c,b] * dip[d,a]* dip[d,c] * \
                                G_ba * G_ca * U_cd

    elif omega1 is None and tau1 is not None:

        signal = np.zeros((len(omega2), len(omega3)), dtype=complex)


        for i in range(len(omega2)):
            pump = omega2[i]

            for j in range(len(omega3)):
                probe = omega3[j]

                # sum-over-states
                for b in e_idx:

                    U_ba =  -1j * np.exp(-1j * (evals[b] - evals[a]) * tau1 - (gamma[b] + gamma[a])/2. * tau1)


                    for c in f_idx:
                        G_ca = 1./(pump - (evals[c]-evals[a]) + 1j * (gamma[c] + gamma[a])/2.0)


                        for d in e_idx:

                            G_cd = 1./(probe - (evals[c]-evals[d]) + 1j * (gamma[c] + gamma[d])/2.0)


                            signal[i,j] += dip[b, a] * dip[c,b] * dip[d,a]* dip[d,c] * \
                                U_ba * G_ca * G_cd

    # one interaction in the bra side
    sign = -1
    return sign * signal

def DQC_R2(evals, dip, omega1=None, omega2=[], omega3=None, tau1=None, tau3=None,\
           g_idx=[0], e_idx=None, f_idx=None, gamma=None):
    '''
    Double quantum coherence, diagram 2:
        gg -> eg -> fg -> eg -> gg in the impulsive limit.
    The signal wave vector is ks = k1 + k2 - k3

    Parameters
    ----------
    evals : TYPE
        DESCRIPTION.
    dip : TYPE
        DESCRIPTION.
    omega1 : TYPE, optional
        DESCRIPTION. The default is None.
    omega2 : TYPE, optional
        DESCRIPTION. The default is [].
    omega3 : TYPE, optional
        DESCRIPTION. The default is None.
    tau1 : TYPE, optional
        DESCRIPTION. The default is None.
    tau3 : TYPE, optional
        DESCRIPTION. The default is None.
    g_idx : TYPE, optional
        DESCRIPTION. The default is [0].
    e_idx : TYPE, optional
        DESCRIPTION. The default is None.
    f_idx : TYPE, optional
        DESCRIPTION. The default is None.
    gamma : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    signal : TYPE
        DESCRIPTION.

    '''


    a = 0

    if omega3 is None and tau3 is not None:

        signal = np.zeros((len(omega1), len(omega2)), dtype=complex)

        for i in range(len(omega1)):
            pump = omega1[i]
            for j in range(len(omega2)):
                probe = omega2[j]

                # sum-over-states
                for b in e_idx:

                    G_ba = 1./(pump - (evals[b]-evals[a]) + 1j * (gamma[b] + gamma[a])/2.0)


                    for c in f_idx:
                        G_ca = 1./(probe - (evals[c]-evals[a]) + 1j * (gamma[c] + gamma[a])/2.0)


                        for d in e_idx:

                            U_da =  -1j * np.exp(-1j * (evals[d] - evals[a]) * tau3 - (gamma[d] + gamma[a])/2. * tau3)


                            signal[i,j] += dip[b, a] * dip[c,b] * dip[d,c]* dip[a,d] * \
                                G_ba * G_ca * U_da

    elif omega1 is None and tau1 is not None:

        signal = np.zeros((len(omega2), len(omega3)), dtype=complex)

        for i in range(len(omega2)):
            pump = omega2[i]
            for j in range(len(omega3)):
                probe = omega3[j]

                # sum-over-states
                for b in e_idx:

                    U_ba = np.exp(-1j * (evals[b] - evals[a]) * tau1 - (gamma[b] + gamma[a])/2. * tau1)

                    for c in f_idx:
                        G_ca = 1./(pump - (evals[c]-evals[a]) + 1j * (gamma[c] + gamma[a])/2.0)

                        for d in e_idx:

                            G_da = 1./(probe - (evals[d]-evals[a]) + 1j * (gamma[d] + gamma[a])/2.0)

                            signal[i,j] += dip[b, a] * dip[c,b] * dip[d,c]* dip[a,d] * \
                                U_ba * G_ca * G_da
    else:
        raise Exception('Input Error! Please specify either omega1, tau3 or omega3, tau1.')

    # positive sign due to 0 interactions at the bra side
    sign = 1
    return sign * signal

# def spontaneous_photon_echo(E, dip, pump, probe, tau2=0.0, normalize=True):
#     """

#     Compute the spontaneous photon echo signal.

#     Parameters
#     ----------
#     E : TYPE
#         DESCRIPTION.
#     dip : TYPE
#         DESCRIPTION.
#     pump: 1d array
#         pump frequency of the first pulse
#     probe: 1d array
#         probe frequency of the third pulse
#     tau2: float
#         time-delay between the second and third pulses. The default is 0.0.

#     Returns
#     -------
#     None.

#     """

#     signal = np.zeros((len(pump), len(probe)))

#     for i in range(len(pump)):
#         for j in range(len(probe)):

#             signal[i,j] = response2_freq(E, dip, probe[j], tau2, pump[i]) + \
#                           response3_freq(E, dip, probe[j], tau2, pump[i])

#     if normalize:
#         signal /= abs(signal).max() # normalize

#     return signal



# fig, ax = plt.subplots(figsize=(4.2,3.2))
# E = [0., 1., 1.1]
# gamma = [0, 0.02, 0.02]

# from lime.phys import paulix
# from matplotlib import cm
# dip =np.zeros((3,3))
# dip[1,2] = dip[2,1] = 1.
# dip[0,1] = dip[1, 0] = 1.
# dip[0,2] = dip[2,0] = 1.

# pump = np.linspace(0.8, 1.2, 100)
# probe = np.linspace(0.8, 1.2, 100)
# omega_min = min(pump)
# omega_max = max(pump)


# SPE = SE(E, dip, omega1=-pump, omega3=probe, tau2=1e-3, g_idx=[0], e_idx= [1,2],\
#           gamma=gamma)


# im = ax.imshow(SPE.real/abs(SPE).max(), interpolation='bilinear', cmap=cm.RdBu,
#                origin='lower', extent=[omega_min, omega_max, omega_min, omega_max],
#                vmax=1, vmin=-1, aspect=1) #-abs(SPE).max())

# ax.axhline(y=1.1, color='w', linestyle='--', linewidth=0.5, alpha=0.5)
# ax.axhline(y=0.9, color='w', linestyle='--', linewidth=0.5, alpha=0.5)

# ax.axvline(x=1.1, color='w', linestyle='--', linewidth=0.5, alpha=0.5)
# ax.axvline(x=0.9, color='w', linestyle='--', linewidth=0.5, alpha=0.5)
# #im = ax.contour(SPE,
# #               origin='lower', extent=[0.8, omega_max, omega_min, omega_max],
# #               vmax=1, vmin=0) #-abs(SPE).max())

# ax.set_xlabel(r'$-\Omega_1/\omega_\mathrm{c}$')
# ax.set_ylabel(r'$\Omega_3/\omega_\mathrm{c}$')
# #ax.set_title(r'$T_2 = $' + str(t2))

# plt.colorbar(im)

# fig.subplots_adjust(hspace=0.0,wspace=0.0,bottom=0.14,left=0.0,top=0.95,right=0.98)

# #plt.savefig(fname[:-4]+'.pdf', transparent=True)
