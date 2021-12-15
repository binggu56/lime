#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 17:25:18 2021

Test pyscf dft calculations

@author: Bing Gu

"""

from pyscf import gto, dft
from pyscf.tools import cubegen

import numpy as np
from pprint import pprint
#p 6-31g(d) 5d ub3lyp tda(nstates=60,full,alltransitiondensities)

mol = gto.M(
atom='''
  H                 0    0    0.00000000
  H                 0.00000000   -0.00000000    0.74
''',
basis='321g', charge=0, spin=0)

mf = dft.RKS(mol)
mf.xc = 'B3LYP'
mf.kernel()


# save mo coeffcients
np.save('mo_coeff', mf.mo_coeff)

j = 1
cubegen.orbital(mol, 'mo'+str(j)+'.cube', mf.mo_coeff[:,j])
