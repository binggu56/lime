#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:27:19 2018

@author: binggu
"""

import numpy as np

def corr(eps):
    """
    calculate the autocorrelation function in variable MEAN.
    """
    f = open('corr.out')

    cor = np.zeros(nstep)

    npts = nstep * nsample

    for idly in range(nstep):

        mean = 0.0
        std = 0.

        for i in range(nsample):# i=1,nreal
            for j in range(nstep - idly):
                mean += eps[j,i] * eps[j+idly, i]

        cor[idly] = mean/float(npts)
        #smean=sngl(mean)          #single precision speeds up calculations
    f.close()
    return cor

def cnoise(nstep, nsample, dt=0.001, tau=0.0025, ave=0., D=0.0025, ):
    """
    store several series of Gaussian noise values in array EPS.

    This is based on the algorithm in R. F. Fox et al. Phys. Rev. A 38, 11 (1988).
    The generated noise satisfy <eps(t) eps(s)> = D/tau * exp(-|t-s|/tau), and
    the initial distribution is Gaussian N(0, sigma) with sigma**2 = D/tau

    INPUT:
        dt: timestep, default 0.001
        tau: correlation time, default 0.0025
        ave: average value, default 0.0
        D: strength of the noise, default 0.0025
    OUTPUT:
        eps: eps[nstep, nsample] colored Gaussian noise
    """
    sigma = np.sqrt(D/tau) # variance

    # initialize

    eps = np.zeros((nstep, nsample))
    eps[0,:] = np.random.rand(nsample) * sigma


    E = np.exp(-dt/tau)

    for j in range(nsample):

        for i in range(nstep-1):

            a = np.random.rand()
            b = np.random.rand()
            h = np.sqrt(-2. * D / tau * (1. - E**2) * np.log(a)) * np.cos(2. * np.pi * b)
            eps[i+1, j] = eps[i, j] * E + h

    return eps

# get input parameters (typical values shown)
#        open(1,file='fastcnoise.dat')
#        read(1,*)nreal             !number of realizations=1000
#        read(1,*)nstep             !max delay in corr. func=10
#        read(1,*)dt                !time step size=.5
#        read(1,*)cortim            !corr. time in the same units as DT=5
#        read(1,*)ave               !average value for gaussian distri
#        read(1,*)sigma             !sigma for gaussian distribution
#        close(1)
#        allocate(eps(nreal,-1:nstep*2))
#        allocate(flag(nreal))
#        flag='N'



# store the noise
#f = open('fastcnoise.out', 'w')
#f.write('{} '.format( sigma*np.sqrt(tau)*eps(:,j) + ave
#f.close()



#! calculate the error in autocorrelation function in variable STD.
#        do i=1,nreal
#          do j=0,nstep
#           std=std+dble((eps(i,j)*eps(i,j+idly)-smean)**2.)
#          enddo
#        enddo
#        std=sqrt(std)/dble(npts-1.)
#        write(2,*)idly,mean,std            !output results
#       enddo


