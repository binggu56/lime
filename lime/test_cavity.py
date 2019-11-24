# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:59:19 2019

@author: Bing
"""
import sys 
sys.path.append(r'C:\Users\Bing\Google\ Drive\scitools')
#sys.path.append(r'/Users/binggu/Dropbox/scitools')
from cavity import Cavity

cav = Cavity(1, 2)

print(cav.hamiltonian * 0.1)

from oqs import Oqs

oqs = Oqs()

print(oqs.c_ops)
