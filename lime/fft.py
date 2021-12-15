


import numpy as np
#import numba

def fft(x, f):
    """
    customized fourier transform of function f
    g = int dt f(t) * exp(- i * freq * t)
    return:
        freq: frequencies where f are evaluated
        g: the fourier transform of f
    """
    nx = len(f)
    dx = x[1] - x[0]

    g = np.fft.fft(f)
    g = np.fft.fftshift(g)
    g *= dx

    freq = 2. * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))

    return freq, g * np.exp(-1j * freq * x[0])

def ifft(x, f):
    """
    customized fourier transform of function f
    g = int dt f(t) * exp(i * freq * t)
    return:
        freq: frequencies where f are evaluated
        g: the fourier transform of f
    """
    nx = len(f)
    dx = x[1] - x[0]

    g = np.fft.ifft(f)
    g = np.fft.ifftshift(g)
    g = g * dx /2./np.pi * len(x)
    freq = 2. * np.pi * np.fft.ifftshift(np.fft.fftfreq(nx, d=dx))

    return freq, g * np.exp(1j * freq * x[0])

def fft2(f, dx=1, dy=1):
    """
    customized FFT for 2D function
    input:
        f: 2d array,
            input array
    return:
        freq: 1d array
            frequencies
        g: 2d array
            fourier transform of f
    """
    nx, ny = f.shape

    g = np.fft.fft2(f)
    g = np.fft.fftshift(g)

    g = g * dx * dy

    freqx = 2. * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    freqy = 2. * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dy))

    return freqx, freqy, g

def dft(x, f, k):
    '''
    Discrete Fourier transfrom at specified momentum
    '''

    dx = (x[1] - x[0]).real

    g = np.zeros(len(k), dtype=np.complex128)

    for i in range(len(k)):
        g[i] = np.sum(f * np.exp(-1j * k[i] * x)) * dx

    return g


def dft2(x, y, f, kx, ky):
    '''
    Discrete Fourier transfrom at specified momentum
    '''

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y)
    g = np.zeros((len(kx), len(ky)), dtype=complex)

    for i in range(len(kx)):
        for j in range(len(ky)):
            g[i, j] = np.sum(f * np.exp(-1j * kx[i] * X - 1j * ky[j] * Y)) * dx * dy

    return g
