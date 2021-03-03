# encoding: utf-8
#
#Copyright (C) 2017-2020, P. R. Wiecha
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
python implementations of the Green's Dyads, accelerated using `numba`
"""
from __future__ import print_function
from __future__ import absolute_import

import copy

import numpy as np
import math
import cmath
import numba
from numba import cuda


# =============================================================================
# numba compatible propagators (CPU + CUDA)
# =============================================================================
## --- free space propagator
@numba.njit(cache=True)
def G0(R1, R2, lamda, eps):
    Dx = R2[0] - R1[0]
    Dy = R2[1] - R1[1]
    Dz = R2[2] - R1[2]
    lR = math.sqrt(Dx**2 + Dy**2 + Dz**2)
    
    k = 2*np.pi / lamda
    cn2 = cmath.sqrt(eps)
    ck0 = -1j * k * cn2
    k2 = k*k*eps
    
    r25 = math.pow((Dx*Dx+Dy*Dy+Dz*Dz), 2.5)
    r2 = math.pow((Dx*Dx+Dy*Dy+Dz*Dz), 2.0)
    r15 = math.pow((Dx*Dx+Dy*Dy+Dz*Dz), 1.5)
    
#!C-------------------------------------------------------------------
    T1XX = (2*Dx*Dx-Dy*Dy-Dz*Dz) / r25
    T2XX = (2*Dx*Dx-Dy*Dy-Dz*Dz) / r2
    T3XX = -1*(Dy*Dy+Dz*Dz) / r15
#!C-------------------------------------------------------------------
    T1XY = 3*Dx*Dy / r25
    T2XY = 3*Dx*Dy / r2
    T3XY = Dx*Dy / r15
#!C-------------------------------------------------------------------
    T1XZ = 3*Dx*Dz / r25
    T2XZ = 3*Dx*Dz / r2
    T3XZ = Dx*Dz / r15
#!C-------------------------------------------------------------------
    T1YY = (2*Dy*Dy-Dx*Dx-Dz*Dz) / r25
    T2YY = (2*Dy*Dy-Dx*Dx-Dz*Dz) / r2
    T3YY = -(Dx*Dx+Dz*Dz) / r15
#!C-------------------------------------------------------------------
    T1YZ = 3*Dy*Dz / r25
    T2YZ = 3*Dy*Dz / r2
    T3YZ = Dy*Dz / r15
#!C------------------------------------------------------------------
    T1ZZ = (2*Dz*Dz-Dx*Dx-Dy*Dy) / r25
    T2ZZ = (2*Dz*Dz-Dx*Dx-Dy*Dy) / r2
    T3ZZ = -(Dx*Dx+Dy*Dy) / r15
    
    CFEXP = cmath.exp(1j*k*cn2*lR)
    
    
    ## setting up the tensor
    xx = CFEXP*(T1XX+ck0*T2XX-k2*T3XX)/eps
    yy = CFEXP*(T1YY+ck0*T2YY-k2*T3YY)/eps
    zz = CFEXP*(T1ZZ+ck0*T2ZZ-k2*T3ZZ)/eps
    
    xy = CFEXP*(T1XY+ck0*T2XY-k2*T3XY)/eps
    xz = CFEXP*(T1XZ+ck0*T2XZ-k2*T3XZ)/eps
    
    yz = CFEXP*(T1YZ+ck0*T2YZ-k2*T3YZ)/eps
        
    yx = xy
    zx = xz
    zy = yz
    
    return xx, yy, zz, xy, xz, yx, yz, zx, zy


## --- "1-2-3" slab propagator, via method of mirror charges for 2 surfaces
@numba.njit(cache=True)
def Gs123(R1, R2, lamda, eps1, eps2, eps3, spacing):
    Dx = R2[0] - R1[0]
    Dy = R2[1] - R1[1]
    Dz = R2[2] + R1[2]
    
    cdelta12 = (eps1-eps2)/(eps1+eps2)
    cdelta23 = (eps3-eps2)/(eps3+eps2)
    
#!************* Interface: (1,2) *******************************
    r25 = math.pow((Dx*Dx+Dy*Dy+Dz*Dz), 2.5)
     
    SXX12 = cdelta12*(Dz*Dz+Dy*Dy-2*Dx*Dx) / r25
    SYY12 = cdelta12*(Dz*Dz+Dx*Dx-2*Dy*Dy) / r25
    SZZ12 = cdelta12*(2*Dz*Dz-Dx*Dx-Dy*Dy) / r25
    SXY12 = cdelta12*(-3*Dx*Dy) / r25
    SXZ12 = cdelta12*(3*Dx*Dz) / r25
    SYZ12 = cdelta12*(3*Dy*Dz) / r25

#!************* Interface: (2,3) *******************************
    GZ = Dz - 2*spacing
    rgz25 = math.pow((Dx*Dx+Dy*Dy+GZ*GZ), 2.5)
    
    SXX23 = cdelta23*(GZ*GZ+Dy*Dy-2*Dx*Dx) / rgz25
    SYY23 = cdelta23*(GZ*GZ+Dx*Dx-2*Dy*Dy) / rgz25
    SZZ23 = cdelta23*(2*GZ*GZ-Dx*Dx-Dy*Dy) / rgz25
    SXY23 = cdelta23*(-3*Dx*Dy) / rgz25
    SXZ23 = cdelta23*(3*Dx*GZ) / rgz25
    SYZ23 = cdelta23*(3*Dy*GZ) / rgz25
#!**************************************************************
    
    xx = SXX12+SXX23
    yy = SYY12+SYY23
    zz = SZZ12+SZZ23
    
    xy = SXY12+SXY23
    xz = SXZ12+SXZ23
    
    yx = xy
    yz = SYZ12+SYZ23
    
    zx = -1*xz
    zy = -1*yz
    
    return xx, yy, zz, xy, xz, yx, yz, zx, zy


## --- the full propagator: vacuum + surface term -- nearfield approximation!
@numba.njit(cache=True)
def G(R1, R2, lamda, eps1, eps2, eps3, spacing):
    xx, yy, zz, xy, xz, yx, yz, zx, zy = G0(R1, R2, lamda, eps2)
    xxs,yys,zzs,xys,xzs,yxs,yzs,zxs,zys = Gs123(R1, R2, lamda, 
                                                    eps1, eps2, eps3, spacing)
    
    return xx+xxs, yy+yys, zz+zzs, \
           xy+xys, xz+xzs, yx+yxs, \
           yz+yzs, zx+zxs, zy+zys




@numba.njit(cache=True)
def G0_HE(R1, R2, lamda):
    Dx = R2[0] - R1[0]
    Dy = R2[1] - R1[1]
    Dz = R2[2] - R1[2]
    lR2 = (Dx**2 + Dy**2 + Dz**2)
    
    k0 = 2*np.pi / lamda
    k02 = k0**2
#-----------------------------------------------------------------
    T2XY = -Dz/lR2
    T3XY = -Dz/lR2**1.5
#-----------------------------------------------------------------
    T2XZ = Dy/lR2
    T3XZ = Dy/lR2**1.5
#-----------------------------------------------------------------
    T2YZ = -Dx/lR2
    T3YZ = -Dx/lR2**1.5
#-----------------------------------------------------------------
    CFEXP = cmath.exp(1j*k0*math.sqrt(lR2))
    
    xx = 0
    yy = 0
    zz = 0
    
    xy = CFEXP * (1j*k0*T3XY + k02*T2XY)
    xz = CFEXP * (1j*k0*T3XZ + k02*T2XZ)
    yz = CFEXP * (1j*k0*T3YZ + k02*T2YZ)

    yx = -xy
    zx = -xz
    zy = -yz
    
    return xx, yy, zz, xy, xz, yx, yz, zx, zy






# =============================================================================
# coupling matrix generator
# =============================================================================
@numba.njit(parallel=True, cache=True)
def t_sbs(geo, lamda, cnorm, eps1, eps2, eps3, spacing, alpha, M):
    for i in numba.prange(len(geo)):    # explicit parallel loop
        R1 = geo[i]
        for j in range(len(geo)):
            R2 = geo[j]
            aj = alpha[j]
            
            ## --- vacuum dyad
            if i==j:
                ## self term
                xx = cnorm
                yy = cnorm
                zz = cnorm
                xy = 0
                xz = 0
                yx = 0
                yz = 0
                zx = 0
                zy = 0
            else:
                xx, yy, zz, xy, xz, yx, yz, zx, zy = G0(R1, R2, lamda, eps2)
            
            ## --- 1-2-3 surface dyad
            xxs,yys,zzs,xys,xzs,yxs,yzs,zxs,zys = Gs123(R1, R2, lamda, 
                                                    eps1, eps2, eps3, spacing)
            
            ## combined dyad
            xx, yy, zz, xy, xz, yx, yz, zx, zy = xx+xxs, yy+yys, zz+zzs, \
                                                 xy+xys, xz+xzs, yx+yxs, \
                                                 yz+yzs, zx+zxs, zy+zys
            
            ## return invertible matrix:  delta_ij*1 - G[i,j] * alpha[j]
            M[3*i+0, 3*j+0] = -1*(xx*aj[0,0] + xy*aj[1,0] + xz*aj[2,0])
            M[3*i+1, 3*j+1] = -1*(yx*aj[0,1] + yy*aj[1,1] + yz*aj[2,1])
            M[3*i+2, 3*j+2] = -1*(zx*aj[0,2] + zy*aj[1,2] + zz*aj[2,2])
            M[3*i+1, 3*j+0] = -1*(xx*aj[0,1] + xy*aj[1,1] + xz*aj[2,1])
            M[3*i+2, 3*j+0] = -1*(xx*aj[0,2] + xy*aj[1,2] + xz*aj[2,2])
            M[3*i+0, 3*j+1] = -1*(yx*aj[0,0] + yy*aj[1,0] + yz*aj[2,0])
            M[3*i+2, 3*j+1] = -1*(yx*aj[0,2] + yy*aj[1,2] + yz*aj[2,2])
            M[3*i+0, 3*j+2] = -1*(zx*aj[0,0] + zy*aj[1,0] + zz*aj[2,0])
            M[3*i+1, 3*j+2] = -1*(zx*aj[0,1] + zy*aj[1,1] + zz*aj[2,1])
            if i==j:
                M[3*i+0, 3*j+0] += 1
                M[3*i+1, 3*j+1] += 1
                M[3*i+2, 3*j+2] += 1










# =============================================================================
# asymptotic propagators for far-field
# =============================================================================
@numba.njit(cache=True)
def G0_EE_asymptotic(R1, R2, lamda, eps2): 
    """Asymptotic vacuum Green's Dyad
    
    Parameters
    ----------
    R1 : np.array of 3 float
        dipole position [x,y,z]
    R2 : np.array of 3 float
        evaluation position
    lamda : float
        emitter wavelength in nm
    eps1 : float, complex
        substrate permittivity
    """
    ## transform to spherical coordinates
    lR = np.linalg.norm(R2)
    theta = np.arccos(R2[2] / lR)
    phi = np.arctan2(R2[1], R2[0])
    
    if lR < lamda:
        raise Exception("Distance too close. Trying to evaluate asymtpotic far-field dyad in the near-field region (R < wavelength).")
    
    ## wavenumber
    kvac = 2 * np.pi / lamda                   # in vacuum
    k2 = 2 * np.pi * cmath.sqrt(eps2) / lamda     # in surrounding medium of emitting dipole
    
    ## tensor prefactor
    A = ((kvac**2)*np.exp(1.0j*k2*lR)/lR * 
                      np.exp(-1.0j*k2*np.sin(theta) * 
                     (R1[0]*np.cos(phi) + R1[1]*np.sin(phi))) * 
                      np.exp(-1.0j*k2*np.cos(theta)*R1[2]))
    
    ## matrix elements
    S0_XX = A * (1.-np.sin(theta)**2*np.cos(phi)**2)
    S0_XY = A * (-np.sin(theta)**2*np.cos(phi)*np.sin(phi))
    S0_XZ = A * (-np.sin(theta)*np.cos(theta)*np.cos(phi))
    S0_YX = A * (-np.sin(theta)**2*np.cos(phi)*np.sin(phi))
    S0_YY = A * (1.-np.sin(theta)**2*np.sin(phi)**2)
    S0_YZ = A * (-np.sin(theta)*np.cos(theta)*np.sin(phi))
    S0_ZX = A * (-np.sin(theta)*np.cos(theta)*np.cos(phi))
    S0_ZY = A * (-np.sin(theta)*np.cos(theta)*np.sin(phi))
    S0_ZZ = A * (np.sin(theta)**2)
    
    return S0_XX, S0_YY, S0_ZZ, \
           S0_XY, S0_XZ, S0_YX, \
           S0_YZ, S0_ZX, S0_ZY


@numba.njit(cache=True, fastmath=True)
def Gs_EE_asymptotic(R1, R2, lamda, eps1, eps2):
    """Asymptotic Green's Dyad for dipole above dielectric interface (inside eps2)
    
    Electric field propagator for electric dipole transition.
    
    In contrast to the nearfield propagators, this includes the vacuum contribution!
    
    Parameters
    ----------
    R1 : np.array of 3 float
        dipole position [x,y,z]
    R2 : np.array of 3 float
        evaluation position
    lamda : float
        emitter wavelength in nm
    eps1 : float, complex
        substrate permittivity
    eps2 : float, complex
        emitter environment permittivity (above substrate)
    """
    ## transform to spherical coordinates
    lR = np.linalg.norm(R2)
    theta = np.arccos(R2[2] / lR)
    phi = np.arctan2(R2[1], R2[0])
    
    if lR < lamda:
        raise Exception("Distance too close. Trying to evaluate asymtpotic far-field dyad in the near-field region (R < wavelength).")
    
    ## refractive index of media
    n1 = cmath.sqrt(eps1)
    n2 = cmath.sqrt(eps2)
    ## wavenumber
    kvac = 2 * np.pi / lamda        # in vacuum
    k1 = 2 * np.pi * n1 / lamda     # in substrate
    k2 = 2 * np.pi * n2 / lamda     # in surrounding medium of emitting dipole
    
    ## workaround for positions too close to substrate
    if np.abs(np.cos(theta)) < 1E-10:
        theta -= np.sign(np.cos(theta))*0.001*np.pi
    
# =============================================================================
# Surface propagator above the surface
# =============================================================================
    if np.cos(theta) >= 0:
        ## tensor prefactors
        A0 = (kvac**2 * np.exp(1.0j*k2*lR) / lR * 
                              np.exp(-1.0j*k2*np.sin(theta) * 
                             (R1[0]*np.cos(phi) + R1[1]*np.sin(phi))))
        A_vac = A0 * np.exp(-1.0j*k2*np.cos(theta)*R1[2])
        A_up = -1 * A0 * np.exp(1.0j*k2*np.cos(theta)*R1[2])
        
        
        ## ------------------------------ vacuum contribution
        ## matrix elements
        S0_11 = A_vac * (1.-np.sin(theta)**2 * np.cos(phi)**2)
        S0_12 = A_vac * (-np.sin(theta)**2 * np.cos(phi) * np.sin(phi))
        S0_13 = A_vac * (-np.sin(theta) * np.cos(theta) * np.cos(phi))
        S0_21 = A_vac * (-np.sin(theta)**2 * np.cos(phi) * np.sin(phi))
        S0_22 = A_vac * (1.-np.sin(theta)**2 * np.sin(phi)**2)
        S0_23 = A_vac * (-np.sin(theta) * np.cos(theta) * np.sin(phi))
        S0_31 = A_vac * (-np.sin(theta) * np.cos(theta) * np.cos(phi))
        S0_32 = A_vac * (-np.sin(theta) * np.cos(theta) * np.sin(phi))
        S0_33 = A_vac * (np.sin(theta)**2)
        
        
        ## ------------------------------ surface contribution
        ## -- Fresnel coefficients
        r_p = ((eps1*n2*np.cos(theta) - eps2*cmath.sqrt(eps1 - eps2*np.sin(theta)**2)) /
               (eps1*n2*np.cos(theta) + eps2*cmath.sqrt(eps1 - eps2*np.sin(theta)**2)))
        
        r_s = ((n2*np.cos(theta) - cmath.sqrt(eps1 - eps2*np.sin(theta)**2)) /
               (n2*np.cos(theta) + cmath.sqrt(eps1 - eps2*np.sin(theta)**2)))
        
        ## -- matrix elements
        Sp11 = A_up * (r_p * np.cos(theta)**2 * np.cos(phi)**2 - r_s*np.sin(phi)**2)
        Sp12 = A_up * ((r_p * np.cos(theta)**2 + r_s) * np.sin(phi) * np.cos(phi))
        Sp13 = A_up * (r_p * np.cos(theta) * np.sin(theta) * np.cos(phi))
        Sp21 = Sp12
        Sp22 = A_up * (r_p * np.cos(theta)**2*np.sin(phi)**2 - r_s*np.cos(phi)**2)
        Sp23 = A_up * (r_p * np.cos(theta) * np.sin(theta) * np.sin(phi))
        Sp31 = -Sp13
        Sp32 = -Sp23
        Sp33 = A_up * (-r_p * np.sin(theta)**2)
        
        Sp11 += S0_11
        Sp12 += S0_12
        Sp13 += S0_13
        Sp21 += S0_21
        Sp22 += S0_22
        Sp23 += S0_23
        Sp31 += S0_31
        Sp32 += S0_32
        Sp33 += S0_33
    
        
# =============================================================================
# Surface propagator under the surface 
# =============================================================================
    else:
        ## -- coeff.
        D_eps_eff = cmath.sqrt(eps2 - eps1*np.sin(theta)**2)
        
        delta_s = ((-n1*np.cos(theta) - D_eps_eff)/
                   (-n1*np.cos(theta) + D_eps_eff))
        tau_s = 1. - delta_s
        phi_s = n1*tau_s / D_eps_eff 
        
        delta_p = ((-eps2*n1*np.cos(theta) - eps1*D_eps_eff)/
                   (-eps2*n1*np.cos(theta) + eps1*D_eps_eff))
        tau_p = delta_p + 1.
        phi_p = n1*tau_p * D_eps_eff
    
        A_low = ((kvac**2)*np.exp(1.0j*k1*lR) / lR*np.exp(-1.0j*k1*np.sin(theta)*
              (R1[0]*np.cos(phi) + R1[1]*np.sin(phi)))*np.exp(1.0j*kvac*D_eps_eff*R1[2]))
        
        ## -- matrix elements
        Sp11 = A_low * ((phi_p/eps2*np.cos(phi)**2 + phi_s*np.sin(phi)**2)*np.cos(theta))
        Sp12 = A_low * ((phi_p/eps2-phi_s)*np.cos(theta)*np.sin(phi)*np.cos(phi))
        Sp13 = A_low * (tau_p*eps1/eps2*np.cos(phi)*np.cos(theta)*np.sin(theta))
        Sp21 = A_low * ((phi_p/eps2-phi_s)*np.cos(theta)*np.sin(phi)*np.cos(phi))
        Sp22 = A_low * ((phi_p/eps2*np.sin(phi)**2 + phi_s*np.cos(phi)**2)*np.cos(theta))
        Sp23 = A_low * (tau_p*eps1/eps2*np.sin(phi)*np.cos(theta)*np.sin(theta))
        Sp31 = A_low * (-phi_p/eps2*np.sin(theta)*np.cos(phi))
        Sp32 = A_low * (-phi_p/eps2*np.sin(theta)*np.sin(phi))
        Sp33 = A_low * (-tau_p*eps1/eps2*np.sin(theta)**2)
    
    return Sp11, Sp22, Sp33, \
           Sp12, Sp13, Sp21, \
           Sp23, Sp31, Sp32