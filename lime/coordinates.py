# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:28:20 2022

Cartesian to internal coordinates transformation

Based on QCL https://github.com/ben-albrecht/qcl/blob/master/qcl/

@author: Bing Gu
"""

from __future__ import print_function
from __future__ import division

import math
from numpy import sin, cos, pi
from numpy.linalg import norm
import numpy as np

# Suppress scientific notation printouts and change default precision
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# try:
#     from cclib.parser.data import ccData
#     from cclib.parser.utils import PeriodicTable
# except ImportError:
#     print("Failed to load cclib!")
#     raise


# class ccData_xyz(ccData):
#     """
#     ccData subclass for xyzfiles
#     TODO: Checks for previous steps before continuing,
#     i.e. check for dist_matrix before building conn_vector
#     Includes some hot new attributes and class methods
#     """

#     def __init__(self, attributes={}):
#         """Adding some new attributes for xyzfiles"""

#         self.newcoords = None
#         self.distancematrix = None

#         # Internal Coordinate Connectivity
#         self.connectivity = None
#         self.angleconnectivity = None
#         self.dihedralconnectivity = None

#         # Internal Coordinates
#         self.distances = None
#         self.angles = None
#         self.dihedrals = None

#         self._attrtypes['comment'] = str
#         self._attrlist.append('comment')
#         self._attrtypes['filename'] = str
#         self._attrlist.append('filename')
#         self._attrtypes['elements'] = list
#         self._attrlist.append('elements')

#         #self._attrtypes['distancematrix'] = np.ndarray
#         #self._attrlist.append('distancematrix')
#         #self._attrtypes['connectivity'] = list
#         #self._attrlist.append('connectivity')

#         super(ccData_xyz, self).__init__(attributes=attributes)

#         # Initialize new data types if attributes were parsed as an original ccdata_xyz
#         if not hasattr(self, 'elements'):
#             pt = PeriodicTable()
#             self.comment = '\n'
#             self.filename = ''
#             self.elements = []
#             for atomno in self.atomnos:
#                 self.elements.append(pt.element[atomno])

#     def _build_distance_matrix(self):
#         """Build distance matrix between all atoms
#            TODO: calculate distances only as needed for efficiency"""
#         coords = self.atomcoords[-1]
#         self.distancematrix = np.zeros((len(coords), len(coords)))
#         for i in range(len(coords)):
#             for j in [x for x in range(len(coords)) if x > i]:
#                 self.distancematrix[i][j] = norm(coords[i] - coords[j])
#                 self.distancematrix[j][i] = self.distancematrix[i][j]

#     def build_zmatrix(self):
#         """
#        'Z-Matrix Algorithm'
#         Build main components of zmatrix:
#         Connectivity vector
#         Distances between connected atoms (atom >= 1)
#         Angles between connected atoms (atom >= 2)
#         Dihedral angles between connected atoms (atom >= 3)
#         """
#         self._build_distance_matrix()

#         # self.connectivity[i] tells you the index of 2nd atom connected to atom i
#         self.connectivity = np.zeros(len(self.atomnos)).astype(int)

#         # self.angleconnectivity[i] tells you the index of
#         #    3rd atom connected to atom i and atom self.connectivity[i]
#         self.angleconnectivity = np.zeros(len(self.atomnos)).astype(int)

#         # self.dihedralconnectivity tells you the index of 4th atom connected to
#         #    atom i, atom self.connectivity[i], and atom self.angleconnectivity[i]
#         self.dihedralconnectivity = np.zeros(len(self.atomnos)).astype(int)

#         # Starts with r1
#         self.distances = np.zeros(len(self.atomnos))
#         # Starts with a2
#         self.angles = np.zeros(len(self.atomnos))
#         # Starts with d3
#         self.dihedrals = np.zeros(len(self.atomnos))

#         atoms = range(1, len(self.atomnos))
#         for atom in atoms:
#             # For current atom, find the nearest atom among previous atoms
#             distvector = self.distancematrix[atom][:atom]
#             distmin = np.array(distvector[np.nonzero(distvector)]).min()
#             nearestindices = np.where(distvector == distmin)[0]
#             nearestatom = nearestindices[0]

#             self.connectivity[atom] = nearestatom
#             self.distances[atom] = distmin

#             # Compute Angles
#             if atom >= 2:
#                 atms = [0, 0, 0]
#                 atms[0] = atom
#                 atms[1] = self.connectivity[atms[0]]
#                 atms[2] = self.connectivity[atms[1]]
#                 if atms[2] == atms[1]:
#                     for idx in range(1, len(self.connectivity[:atom])):
#                         if self.connectivity[idx] in atms and not idx in atms:
#                             atms[2] = idx
#                             break

#                 self.angleconnectivity[atom] = atms[2]

#                 self.angles[atom] = self._calc_angle(atms[0], atms[1], atms[2])

#             # Compute Dihedral Angles
#             if atom >= 3:
#                 atms = [0, 0, 0, 0]
#                 atms[0] = atom
#                 atms[1] = self.connectivity[atms[0]]
#                 atms[2] = self.angleconnectivity[atms[0]]
#                 atms[3] = self.angleconnectivity[atms[1]]
#                 if atms[3] in atms[:3]:
#                     for idx in range(1, len(self.connectivity[:atom])):
#                         if self.connectivity[idx] in atms and not idx in atms:
#                             atms[3] = idx
#                             break

#                 self.dihedrals[atom] =\
#                     self._calc_dihedral(atms[0], atms[1], atms[2], atms[3])
#                 if math.isnan(self.dihedrals[atom]):
#                     # TODO: Find explicit way to denote undefined dihedrals
#                     self.dihedrals[atom] = 0.0

#                 self.dihedralconnectivity[atom] = atms[3]

#     def _calc_angle(self, atom1, atom2, atom3):
#         """Calculate angle between 3 atoms"""
#         coords = self.atomcoords[-1]
#         vec1 = coords[atom2] - coords[atom1]
#         uvec1 = vec1 / norm(vec1)
#         vec2 = coords[atom2] - coords[atom3]
#         uvec2 = vec2 / norm(vec2)
#         return np.arccos(np.dot(uvec1, uvec2))*(180.0/pi)

#     def _calc_dihedral(self, atom1, atom2, atom3, atom4):
#         """
#            Calculate dihedral angle between 4 atoms
#            For more information, see:
#                http://math.stackexchange.com/a/47084
#         """
#         coords = self.atomcoords[-1]
#         # Vectors between 4 atoms
#         b1 = coords[atom2] - coords[atom1]
#         b2 = coords[atom2] - coords[atom3]
#         b3 = coords[atom4] - coords[atom3]

#         # Normal vector of plane containing b1,b2
#         n1 = np.cross(b1, b2)
#         un1 = n1 / norm(n1)

#         # Normal vector of plane containing b1,b2
#         n2 = np.cross(b2, b3)
#         un2 = n2 / norm(n2)

#         # un1, ub2, and m1 form orthonormal frame
#         ub2 = b2 / norm(b2)
#         um1 = np.cross(un1, ub2)

#         # dot(ub2, n2) is always zero
#         x = np.dot(un1, un2)
#         y = np.dot(um1, un2)

#         dihedral = np.arctan2(y, x)*(180.0/pi)
#         if dihedral < 0:
#             dihedral = 360.0 + dihedral
#         return dihedral

#     def build_xyz(self):
#         """ Build xyz representation from z-matrix"""
#         coords = self.atomcoords[-1]
#         self.newcoords = np.zeros((len(coords), 3))
#         for i in range(len(coords)):
#             self.newcoords[i] = self._calc_position(i)
#         self.atomcoords[-1] = self.newcoords

#     def _calc_position(self, i):
#         """Calculate position of another atom based on internal coordinates"""

#         if i > 1:
#             j = self.connectivity[i]
#             k = self.angleconnectivity[i]
#             l = self.dihedralconnectivity[i]

#             # Prevent doubles
#             if k == l and i > 0:
#                 for idx in range(1, len(self.connectivity[:i])):
#                     if self.connectivity[idx] in [i, j, k] and not idx in [i, j, k]:
#                         l = idx
#                         break

#             avec = self.newcoords[j]
#             bvec = self.newcoords[k]

#             dst = self.distances[i]
#             ang = self.angles[i] * pi / 180.0

#             if i == 2:
#                 # Third atom will be in same plane as first two
#                 tor = 90.0 * pi / 180.0
#                 cvec = np.array([0, 1, 0])
#             else:
#                 # Fourth + atoms require dihedral (torsional) angle
#                 tor = self.dihedrals[i] * pi / 180.0
#                 cvec = self.newcoords[l]

#             v1 = avec - bvec
#             v2 = avec - cvec

#             n = np.cross(v1, v2)
#             nn = np.cross(v1, n)

#             n /= norm(n)
#             nn /= norm(nn)

#             n *= -sin(tor)
#             nn *= cos(tor)

#             v3 = n + nn
#             v3 /= norm(v3)
#             v3 *= dst * sin(ang)

#             v1 /= norm(v1)
#             v1 *= dst * cos(ang)

#             position = avec + v3 - v1

#         elif i == 1:
#             # Second atom dst away from origin along Z-axis
#             j = self.connectivity[i]
#             dst = self.distances[i]
#             position = np.array([self.newcoords[j][0] + dst, self.newcoords[j][1], self.newcoords[j][2]])

#         elif i == 0:
#             # First atom at the origin
#             position = np.array([0, 0, 0])

#         return position

#     @property
#     def splitatomnos(self):
#         """Returns tuple of atomnos from reactants joined by atoms 0 and 1"""
#         fragments = [[], []]

#         return fragments


#     def print_distance_matrix(self):
#         """Print distance matrix in formatted form"""

#         # Title
#         print("\nDistance Matrix")

#         # Row Indices
#         for i in range(len(self.distancematrix)):
#             print("%3d" % i, end="  ")

#         print("\n", end="")
#         idx = 0
#         for vector in self.distancematrix:

#             # Column indices
#             print(idx, end=" ")

#             # Actual Values
#             for element in vector:
#                 if not element == 0:
#                     print("%1.2f" % element, end=" ")
#                 else:
#                     print("%1s" % " ", end="    ")
#             print("\n", end="")
#             idx += 1

#     def print_xyz(self):
#         """Print Standard XYZ Format"""
#         if not self.newcoords.any():
#             self.build_xyz()

#         print(len(self.newcoords))

#         if self.comment:
#             print(self.comment, end='')
#         else:
#             print(self.filename, end='')

#         atomcoords = [x.tolist() for x in self.newcoords]
#         for i in range(len(atomcoords)):
#             atomcoords[i].insert(0, self.elements[i])

#         for atom in atomcoords:
#             print("  %s %10.5f %10.5f %10.5f" % tuple(atom))

#     def print_gzmat(self):
#         """Print Gaussian Z-Matrix Format
#         e.g.
#         0  3
#         C
#         O  1  r2
#         C  1  r3  2  a3
#         Si 3  r4  1  a4  2  d4
#         ...
#         Variables:
#         r2= 1.1963
#         r3= 1.3054
#         a3= 179.97
#         r4= 1.8426
#         a4= 120.10
#         d4=  96.84
#         ...
#         """
#         pt = PeriodicTable()

#         print('#', self.filename, "\n")
#         print(self.comment)

#         print(self.comment, end='')
#         for i in range(len(self.atomnos)):
#             idx = str(i+1)+" "
#             if i >= 3:
#                 print(pt.element[self.atomnos[i]], "",
#                       self.connectivity[i]+1, " r"+idx,
#                       self.angleconnectivity[i]+1, " a"+idx,
#                       self.dihedralconnectivity[i]+1, " d"+idx.rstrip())
#             elif i == 2:
#                 print(pt.element[self.atomnos[i]], "",
#                       self.connectivity[i]+1, " r"+idx,
#                       self.angleconnectivity[i]+1, " a"+idx.rstrip())
#             elif i == 1:
#                 print(pt.element[self.atomnos[i]], "",
#                       self.connectivity[i]+1, " r"+idx.rstrip())
#             elif i == 0:
#                 print(pt.element[self.atomnos[i]])

#         print("Variables:")

#         for i in range(1, len(self.atomnos)):
#             idx = str(i+1)+"="
#             if i >= 3:
#                 print("%s" % "r"+idx, "%6.4f" % self.distances[i])
#                 print("%s" % "a"+idx, "%6.2f" % self.angles[i])
#                 print("%s" % "d"+idx, "%6.2f" % self.dihedrals[i])
#             elif i == 2:
#                 print("%s" % "r"+idx, "%6.4f" % self.distances[i])
#                 print("%s" % "a"+idx, "%6.2f" % self.angles[i])
#             elif i == 1:
#                 print("%s" % "r"+idx, "%6.4f" % self.distances[i])

#     def print_zmat(self):
#         """Print Standard Z-Matrix Format"""
#         #TODO

#         """
#         0 1
#         O
#         O 1 1.5
#         H 1 1.0 2 120.0
#         H 2 1.0 1 120.0 3 180.0
#         """

import numpy as np
import scipy.linalg as linalg
from scipy.optimize import newton

from pyscf.lib import logger
import pyscf.ao2mo
import pyscf
from functools import reduce


from lime.phys import eig_asymm, is_positive_def, dag
from lime.optics import Pulse

class Molecule(pyscf.gto.Mole):
    def __init__(self, *args):

        super().__init__(*args)

        # self.atom_coord = mol.atom_coord
        # self.atom_coords = mol.atom_coords()
        # self.natom = self.natm

        self.distmat = None

    def atom_symbols(self):
        return [self.atom_symbol(i) for i in range(self.natm)]

    # @property
    # def natom(self):
    #     return self.natm

    def _build_distance_matrix(self):
        """Build distance matrix between all atoms
           TODO: calculate distances only as needed for efficiency"""
        coords = self.atom_coords()
        natom = self.natm

        distancematrix = np.zeros((natom, natom))

        for i in range(natom):
            for j in range(i+1, natom):
                distancematrix[i, j] = np.linalg.norm(coords[:, i]-coords[:, j])
                distancematrix[j, i] = distancematrix[i, j]

        self.distmat =  distancematrix
        return distancematrix

    def _calc_angle(self, atom1, atom2, atom3):
        """
        Calculate angle in radians between 3 atoms

        Parameters
        ----------
        atom1 : TYPE
            DESCRIPTION.
        atom2 : TYPE
            DESCRIPTION.
        atom3 : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        vec1 = self.atom_coord(atom2) - self.atom_coord(atom1)
        uvec1 = vec1 / norm(vec1)
        vec2 = self.atom_coord(atom2) - self.atom_coord(atom3)
        uvec2 = vec2 / norm(vec2)
        return np.arccos(np.dot(uvec1, uvec2))*(180.0/pi)

    def _calc_dihedral(self, atom1, atom2, atom3, atom4):
        """

           Calculate dihedral angle (in radians) between 4 atoms
           For more information, see:
               http://math.stackexchange.com/a/47084

        Parameters
        ----------
        atom1 : TYPE
            DESCRIPTION.
        atom2 : TYPE
            DESCRIPTION.
        atom3 : TYPE
            DESCRIPTION.
        atom4 : TYPE
            DESCRIPTION.

        Returns
        -------
        dihedral : TYPE
            DESCRIPTION.

        """
        r1 = self.atom_coord(atom1)
        r2 = self.atom_coord(atom2)
        r3 = self.atom_coord(atom3)
        r4 = self.atom_coord(atom4)

        # Vectors between 4 atoms
        b1 = r2 - r1
        b2 = r2 - r3
        b3 = r4 - r3

        # Normal vector of plane containing b1,b2
        n1 = np.cross(b1, b2)
        un1 = n1 / norm(n1)

        # Normal vector of plane containing b1,b2
        n2 = np.cross(b2, b3)
        un2 = n2 / norm(n2)

        # un1, ub2, and m1 form orthonormal frame
        ub2 = b2 / norm(b2)
        um1 = np.cross(un1, ub2)

        # dot(ub2, n2) is always zero
        x = np.dot(un1, un2)
        y = np.dot(um1, un2)

        dihedral = np.arctan2(y, x)*(180.0/pi)
        if dihedral < 0:
            dihedral = 360.0 + dihedral
        return dihedral

    def zmat(self, rvar=False, avar=False, dvar=False):
        npart = self.natm

        if self.distmat is None:
            self._build_distance_matrix()

        distmat = self.distmat

        atomnames = self.atom_symbols()

        rlist = []
        alist = []
        dlist = []
        if npart > 0:
            # Write the first atom
            print(atomnames[0])

            if npart > 1:
                # and the second, with distance from first
                n = atomnames[1]
                rlist.append(distmat[0][1])
                if (rvar):
                    r = 'R1'
                else:
                    r = '{:>11.5f}'.format(rlist[0])
                print('{:<3s} {:>4d}  {:11s}'.format(n, 1, r))

                if npart > 2:
                    n = atomnames[2]

                    rlist.append(distmat[0][2])
                    if (rvar):
                        r = 'R2'
                    else:
                        r = '{:>11.5f}'.format(rlist[1])

                    alist.append(self._calc_angle(2, 0, 1))
                    if (avar):
                        t = 'A1'
                    else:
                        t = '{:>11.5f}'.format(alist[0])

                    print('{:<3s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(n, 1, r, 2, t))

                    if npart > 3:
                        for i in range(3, npart):
                            n = atomnames[i]

                            rlist.append(distmat[i-3][i])
                            if (rvar):
                                r = 'R{:<4d}'.format(i)
                            else:
                                r = '{:>11.5f}'.format(rlist[i-1])

                            alist.append(self._calc_angle(i, i-3, i-2))
                            if (avar):
                                t = 'A{:<4d}'.format(i-1)
                            else:
                                t = '{:>11.5f}'.format(alist[i-2])

                            dlist.append(self._calc_dihedral(i, i-3, i-2, i-1))
                            if (dvar):
                                d = 'D{:<4d}'.format(i-2)
                            else:
                                d = '{:>11.5f}'.format(dlist[i-3])
                            print('{:3s} {:>4d}  {:11s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(n, i-2, r, i-1, t, i, d))
        if (rvar):
            print(" ")
            for i in range(npart-1):
                print('R{:<4d} = {:>11.5f}'.format(i+1, rlist[i]))
        if (avar):
            print(" ")
            for i in range(npart-2):
                print('A{:<4d} = {:>11.5f}'.format(i+1, alist[i]))
        if (dvar):
            print(" ")
            for i in range(npart-3):
                print('D{:<4d} = {:>11.5f}'.format(i+1, dlist[i]))

        return

    def jacobian(self, q):
        return

    def metric(self):
        pass

if __name__ == '__main__':
    from pyscf import scf, gto, tdscf
    from lime.units import au2fs, au2ev
    import proplot as plt

    mol = Molecule()
    mol.verbose = 3
    #mol.atom = [['Ne' , (0., 0., 0.)]]
    #mol.basis = {'Ne': '6-31G'}
    # This is from G2/97 i.e. MP2/6-31G*
    mol.atom = [['H' , (0,      0., 0.)],
                ['H', (1.1, 0., 0.)]]
                # ['F' , (0.91, 0., 0.)]]


    mol.basis = 'STO-3G'
    mol.build()

    # print(mol.natm)

    # mole = Molecule(mol)
    # mol.zmat(rvar=True)
    mf = scf.RHF(mol).run()

    td = tdscf.TDRHF(mf)
    td.kernel()




