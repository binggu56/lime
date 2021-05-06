#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:16:49 2020

@author: bing

Entangled two-photon absorption 
"""

import numpy as np 
from lime.phys import MLS, sz, sx

nlevel = 3 
tau = [1.e-5, ] * nlevel
H = np.diag([0, 0.5, 0.8])



model = MLS(H, dip, tau)

def etpa(tpw, model):
    """

    Parameters
    ----------
    tpw :  2d array
        two-photon wavefunction.
    model : 
        multilevel system.

    Returns
    -------
    None.

    """
    en = molel.eigvals() 
    

