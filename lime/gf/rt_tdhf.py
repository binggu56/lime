#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:05:25 2022

Real time time-dependent Hartree-Fock

@author: Bing Gu
"""
import numpy as np
import scipy.linalg
from scipy.optimize import newton

from pyscf.lib import logger
import pyscf.ao2mo
import pyscf
from functools import reduce


from lime.phys import eig_asymm, is_positive_def
from lime.optics import Pulse


def center_of_mass(mol):
    mass = mol.atom_mass_list()
    coords = mol.atom_coords()
    mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()
    # coords = coords - mass_center
    return mass_center


def charge_center(mol):
    charge_center = (np.einsum('z,zx->x', mol.atom_charges(), mol.atom_coords())
                 / mol.atom_charges().sum())
    return charge_center

def _tdhf(mo_coeff, hcore, r, pulse):
    pass

class TDHF:
    def __init__(mf, pulse):
        pass

if __name__ == '__main__':
    from pyscf import scf, gto
    mol = gto.Mole()
    mol.verbose = 3
    #mol.atom = [['Ne' , (0., 0., 0.)]]
    #mol.basis = {'Ne': '6-31G'}
    # This is from G2/97 i.e. MP2/6-31G*
    mol.atom = [['H' , (0,      0., 0.)],
                ['H', (1.1, 0., 0.)]]
                # ['F' , (0.91, 0., 0.)]]


    mol.basis = 'STO-3G'
    mol.build()

    mol.set_common_origin(charge_center(mol))

    mf = scf.RHF(mol)

    mf.kernel()

    # 1-particle RDM
    C = mf.mo_coeff[:, mf.mo_occ > 0]
    rdm1 = np.conj(C).dot(C.T)

    print(mf.mo_energy)
    hcore = mf.get_hcore()

    r = mol.intor('int1e_r') # AO-matrix elements of r
    eri = mol.intor('int2e')
    print(eri.shape)
    # fock matrix including the drive
    # f = hcore + se + r * pulse(t)


    # propagate

def self_energy_hf(eri):
    pass
