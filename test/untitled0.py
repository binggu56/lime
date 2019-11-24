#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 23:40:36 2019

@author: binggu
"""



import sys 
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.append('/Users/binggu/Google Drive/lime')
#sys.path.append(r'C:\Users\Bing\Google Drive\lime')
from lime.cavity import Mol, Cavity, Polariton
from lime.phys import sz, sx, s0 
from lime.style import set_style

tls = Mol(sz, sx)
cav = Cavity(1.,4)

gs = np.linspace(0, 0.8, 200)

nstates = 4

energy = np.zeros((nstates, len(gs)))


for i in range(len(gs)):
    g = gs[i]
    pol = Polariton(tls, cav, g)

    energy[:,i] = np.sort(np.array(pol.spectrum(nstates, g))[0].real)

fig, ax = plt.subplots(figsize=(4,3))
set_style()
for n in range(nstates):
    ax.plot(gs, energy[n,:], lw=3) 

#ax.set_ylim(-1,1)
ax.set_xlim(0, max(gs))
ax.set_xlabel(r'$g/\omega_\mathrm{c}$')
ax.set_ylabel(r'Energy/$\omega_\mathrm{c}$')

fig.subplots_adjust(hspace=0.0,wspace=0.0,bottom=0.18,left=0.16,top=0.95,right=0.90)

plt.savefig('rabi.pdf',dpi=1200)
plt.show()
