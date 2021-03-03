#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:26:02 2019

@author: binggu
"""

import numpy as np 
from scipy.sparse import lil_matrix, csr_matrix, kron, identity, linalg 

from .units import au2fs, au2k, au2ev


class Pulse:
    def __init__(self, delay, sigma, omegac, amplitude=0.01, cep=0.):
        """
        Gaussian pulse A * exp(-(t-T)^2/2 / sigma^2)
        A: amplitude 
        T: time delay 
        sigma: duration 
        """
        self.delay = delay
        self.sigma = sigma
        self.omegac = omegac # central frequency
        self.unit = 'au'
        self.amplitude = amplitude
        self.cep = cep
        self.bandwidth = 1./sigma
        self.duration = 2. * sigma 

    def envelop(self, t):
        return np.exp(-(t-self.delay)**2/2./self.sigma**2)

    def spectrum(self, omega):
        """
        Fourier transform of the Gaussian pulse
        """
        omegac = self.omegac
        sigma = self.sigma
        a = self.amplitude
        return a * sigma * np.sqrt(2.*np.pi) * np.exp(-(omega-omegac)**2 * sigma**2/2.)

    def field(self, t):
        '''
        electric field
        '''
        omegac = self.omegac
        delay = self.delay
        a = self.amplitude
        sigma = self.sigma
        return a * np.exp(-(t-delay)**2/2./sigma**2)*np.cos(omegac * (t-delay))

