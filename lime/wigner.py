#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 22:23:15 2021

@author: bing
"""
from __future__ import absolute_import

# from time import clock
from numpy import abs, arange, shape, array, ceil, zeros, conj, ix_,\
 transpose, append, fft, real, float64, linspace, sqrt
import numpy as np
from scipy.signal import hilbert
from scipy import interpolate
from math import log, ceil, floor

def wvd(audioFile, t=None, N=None, trace=0, make_analytic=True):
    # if make_analytic:
    #     x = hilbert(audioFile[1])
    # else:
    #     x = array(audioFile[1])

    x = array(audioFile)
    if x.ndim == 1: [xrow, xcol] = shape(array([x]))
    else: raise ValueError("Signal x must be one-dimensional.")

    if t is None: t = arange(len(x))
    if N is None: N = len(x)

    if (N <= 0 ): raise ValueError("Number of Frequency bins N must be greater than zero.")

    if t.ndim == 1: [trow, tcol] = shape(array([t]))
    else: raise ValueError("Time indices t must be one-dimensional.")

    # if xrow != 1:
    #     raise ValueError("Signal x must have one row.")
    # elif trow != 1:
    #     raise ValueError("Time indicies t must have one row.")
    # elif nextpow2(N) != N:
        # print "For a faster computation, number of Frequency bins N should be a power of two."

    tfr = zeros([N, tcol], dtype='complex')
    # if trace: print "Wigner-Ville distribution",
    for icol in range(0, tcol):
        ti = t[icol]
        taumax = min([ti, xcol-ti-1, int(round(N/2.0))-1])
        tau = arange(-taumax, taumax+1)
        indices = ((N+tau)%N)
        tfr[ix_(indices, [icol])] = transpose(array(x[ti+tau] * conj(x[ti-tau]), ndmin=2))
        tau=int(round(N/2))+1
        if ((ti+1) <= (xcol-tau)) and ((ti+1) >= (tau+1)):
            if(tau >= tfr.shape[0]): tfr = append(tfr, zeros([1, tcol]), axis=0)
            tfr[ix_([tau], [icol])] = array(0.5 * (x[ti+tau] * conj(x[ti-tau]) + x[ti-tau] * conj(x[ti+tau])))
        # if trace: disprog(icol, tcol, 10)

    tfr = real(fft.fft(tfr, axis=0))
    f = 0.5*arange(N)/float(N)
    return (transpose(tfr), t, f )


def wigner(signal):
    """
    Wigner transform of an input signal.
    W(t, w) = int dtau x(t + tau/2) x^*(t - tau/2) e^{i w tau}

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """

    N = len(signal)
    tausec = N//2
    winlength = tausec - 1

    taulens = np.min(np.c_[np.arange(N),
                           N - np.arange(N) - 1,
                     winlength * np.ones(N)], axis=1)

    conj_signal = np.conj(signal)

    tfr = zeros((N, N), dtype=complex)

    for icol in range(N):

        taumax = taulens[icol]

        tau = np.arange(-taumax, taumax + 1).astype(int)

        indices = np.remainder(N + tau, N).astype(int)

        tfr[indices, icol] = signal[icol + tau] * conj_signal[icol - tau]

        if (icol <= N - tausec) and (icol >= tausec + 1):
            tfr[tausec, icol] = signal[icol + tausec, 0] * \
                np.conj(signal[icol - tausec, 0]) + \
                signal[icol - tausec, 0] * conj_signal[icol + tausec, 0]

    tfr = np.fft.fft(tfr, axis=0)
    tfr = np.real(tfr)
    freqs = 0.5 * np.arange(N, dtype=float) / N

    return tfr, freqs


from lime.optics import Pulse, interval
from lime.units import au2fs, au2ev
#from lime.phys import

pulse = Pulse(sigma=4/au2fs, omegac=2/au2ev, beta=0.2)


times = np.linspace(-14, 14, 256)/au2fs

efield = pulse.efield(times)

# wvd(efield.real, times, N=None, trace=0, make_analytic=False)

wvd = wigner(efield)[0]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(wvd)
