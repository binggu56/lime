#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:01:00 2020

@author: Bing Gu 

Modules for computing signals with superoperator formalism. 

This is valid for open quantum system whose dynamics can be described by 
TIME-INDEPENDENT Liouvillian. 

The difference vs. the time-correlation function approach is: 
Instead of performing real-time open quantum dynamics, the Liouvillian 
is directly diagonalized.   

"""
import numpy as np 
from scipy.sparse.linalg import eigs


from lime.phys import dag, lindblad_superoperator, operator_to_superoperator

def liouvillian(H, c_ops):
    '''
    Construct the Liouvillian out of the Hamiltonian and collapse operators 

    Parameters
    ----------
    H : TYPE
        DESCRIPTION.
    c_ops : TYPE
        DESCRIPTION.

    Returns
    -------
    l : TYPE
        DESCRIPTION.

    '''
    dissipator = 0. 
    for c_op in c_ops:
        dissipator = dissipator + lindblad_superoperator(c_op)
                
    l = operator_to_superoperator(H) + 1j * dissipator 

    return l 

def linear_absorption(omegas, ham, dip, rho0, c_ops=[], ntrans=1):
    """
    Note that the eigenvectors of the liouvillian L and L^\dag have to 
    be ordered in parallel!

    Parameters
    ----------
    omegas : TYPE
        DESCRIPTION.
    ham : TYPE
        DESCRIPTION.
    dip : TYPE
        DESCRIPTION.
    rho0 : 2d array
        initial density matrix 
    c_ops : TYPE, optional
        DESCRIPTION. The default is [].
    ntrans : int, optional
        Number of transitions to be computed. The default is 1.

    Returns
    -------
    signal : TYPE
        DESCRIPTION.

    """
    
    # dissipation 
    dissipator = 0. 
    for c_op in c_ops:
        dissipator = dissipator + lindblad_superoperator(c_op)
                
    liouvillian = operator_to_superoperator(ham) + 1j * dissipator 
    
    
    eigvals1, U1 = eigs(liouvillian, k=ntrans, which='LR')
    eigvals1, U1 = sort(eigvals1, U1)

    eigvals2, U2 = eigs(dag(liouvillian), k=ntrans, which='LR')
    eigvals2, U2 = sort(eigvals2, U2)
    
    norm = [np.vdot(U2[:,n], U1[:,n]) for n in range(ntrans)]

    signal = np.zeros(len(omegas), dtype=complex)  
    
    tmp = [np.vdot(dip.flatten(), U1[:,n]) * np.vdot(U2[:,n], \
          dip.dot(rho0).flatten()) / norm[n] for n in range(ntrans)] 
    
    for j in range(len(omegas)):
        omega = omegas[j] 
        signal[j] += sum(tmp / (omega - eigvals1))   
    
    
    return  -2. * signal.imag 

def sort(eigvals, eigvecs):
    
    idx = np.argsort(eigvals)
    
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]
    
    return eigvals, eigvecs

