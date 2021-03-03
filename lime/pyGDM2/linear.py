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
linear optical effects

"""
from __future__ import print_function
from __future__ import absolute_import

import warnings


import numpy as np
import copy





## (on linux) Set Stacksize for enabling passing large arrays to the fortran subroutines
import platform
if platform.system() == 'Linux':
    import resource
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


#==============================================================================
# GLOBAL PARAMETERS
#==============================================================================




#==============================================================================
# EXCEPTIONS
#==============================================================================








#==============================================================================
# Linear Field processing functions
#==============================================================================
def extinct(sim, field_index):
    """Extinction, scattering and absorption cross-sections
    
    Calculates extinction, scattering and absorption crosssections
    for each wavelength of the GDM simulation
    
    
    Parameters
    ----------
    sim : :class:`.core.simulation`
        simulation description
    
    field_index : int
        index of evaluated self-consistent field to use for calculation. Can be
        obtained for specific parameter-set using :func:`.tools.get_closest_field_index`
        
    
    Returns
    -------
    extinct : float
        extinction cross-section
    
    scatter : float
        scattering cross-section
    
    absorpt : float
        apsorption cross-section
        
    
    Notes
    -----
    For the calculation of the cross-sections from the complex nearfield, 
    see e.g.: 
    Draine, B. T. & Flatau, P. J. **Discrete-dipole approximation for scattering 
    calculations.**
    Journal of the Optical Society of America, A 11, 1491 (1994).
    
    """
    from .pyGDMfor import extinct as forExtinct

    warnings.warn("Info: The fortran code underlying 'extinct' will be replaced " +
          "by a python/numpy implementation in one of the next releases. " + 
          "The new implementation can be tested by using the `linear_py` " +
          "instead of the `linear` module.")
    
    if sim.E is None: 
        raise ValueError("Error: Scattering field inside the structure not yet evaluated. Run `core.scatter` simulation first.")
    if len(np.unique(sim.struct.geometry.T[2]>sim.struct.spacing)) > 1 or \
        len(np.unique(sim.struct.geometry.T[2]<0)) > 1:
            warnings.warn("Structure seems not to lie fully in one layer (center cladding, substrate or top layer)." +
                          "`extinct` assumes the structure to be fully in one of the layers." +
                          " If this is not the case, the results of `extinct` won't be correct!")
    ## environment refractive index
    n_env = sim.struct.n2
    if np.unique(sim.struct.geometry.T[2]<0)[0]:
        n_env = sim.struct.n1
    if np.unique(sim.struct.geometry.T[2]>sim.struct.spacing)[0]:
        n_env = sim.struct.n3
    
    ## --- incident field configuration
    field_params    = sim.E[field_index][0]
    wavelength      = field_params['wavelength']
    wavelengths     = np.asfortranarray([wavelength], dtype=sim.dtypef)
    
    ## -- get polarizability at wavelength
    alpha_spectral = np.transpose([sim.struct.getPolarizability(wavelength)])
    alpha_spectral = np.asfortranarray(alpha_spectral, dtype=sim.dtypec)
    
    ## --- fundamental and scattered field
    E0 = sim.efield.field_generator(sim.struct, **field_params)
    EX0, EY0, EZ0 = np.asfortranarray(E0, dtype=sim.dtypec).T
    E = sim.E[field_index][1]
    EX,  EY,  EZ  = np.asfortranarray(E, dtype=sim.dtypec).T
    
    ## --- use openmp-parallel fortran routine
    ext_sc, abs_sc = forExtinct(
                   elambda=wavelengths, 
                   cap=alpha_spectral, cn2=n_env,
                   cex=EX, cey=EY, cez=EZ, cex0=EX0, cey0=EY0, cez0=EZ0)
    
    sca_sc = ext_sc - abs_sc
    
    return ext_sc, sca_sc, abs_sc



def farfield(sim, field_index, 
             r=10000., tetamin=0, tetamax=np.pi/2., Nteta=10, Nphi=36, 
             polarizerangle='none', return_value='map'):
    """spatially resolved and polarization-filtered far-field scattering 
    
    For a given incident field, calculate the electro-magnetic field 
    (E-component) in the far-field around the nanostructure 
    (on a sphere of radius `r`).
    
    Propagator for scattering into a substrate contributed by C. Majorel
    
    
    Parameters
    ----------
    sim : :class:`.core.simulation`
        simulation description
        
    field_index : int
        index of evaluated self-consistent field to use for calculation. Can be
        obtained for specific parameter-set using :func:`.tools.get_closest_field_index`
    
    r : float, default: 10000.
        radius of integration sphere (distance to coordinate origin in nm)
        
    tetamin, tetamax : float, float; defaults: 0, np.pi/2. (in [0, pi])
        minimum and maximum polar angle for semi-sphere in radians 
        (calculate from `tetamin` to `tetamax`)
        
    Nteta, Nphi : int, int; defaults: 10, 36
        number of polar and azimuthal angles on sphere to calculate,
        (angle ranges: polar = tetamin - tetamax, azimuthal: 0 - 2*Pi)
        
    polarizerangle : float or 'none', default: 'none'
        optional polarization filter angle **in degrees**(!). If 'none' (default), 
        the total field-intensity is calculated (= no polarization filter)
    
    return_value : str, default: 'map'
        Values to be returned. Either 'map' (default) or 'integrated'.
          - "map" : (default) return spatially resolved farfield intensity at each spherical coordinate (5 lists)
          - "efield" : return spatially resolved E-fields at each spherical coordinate (5 lists)
          - "int_Es" : return the integrated scattered field (as float)
          - "int_E0" : return the integrated fundamental field (as float)
          - "int_Etot" : return the integrated total field (as float)
    
    
    Returns
    -------
    return_value == "map" : 5 arrays of shape (Nteta, Nphi) : 
        - tetalist : teta angles
        - philist : phi angles
        - I_sc : intensity of scattered field, 
        - I_tot : intensity of total field (I_tot=|E_sc+E_0|^2), 
        - I0 : intensity of incident field
    
    return_value == "efield" : float
        - tetalist : teta angles
        - philist : phi angles
        - E_sc : complex scattered field at each pos.
        - E_tot : complex total field at each pos. (E_sc+E0)
        - E0 : complex incident field at each pos.
        
    return_value == "int_XX" : float
        integrated total intensity over specified solid angle
        
    
    Notes
    -----
    For details of the asymptotic (non-retarded) far-field propagators for a 
    dipole above a substrate, see e.g.:
    Colas des Francs, G. & Girard, C. & Dereux, A. **Theory of near-field 
    optical imaging with a single molecule as light source.** 
    The Journal of Chemical Physics 117, 4659–4666 (2002)
        
    """    
    from .pyGDMfor import multidipolefarfield as forFarfield
    
    if sim.E is None: 
        raise ValueError("Error: Scattering field inside the structure not yet evaluated. Run `core.scatter` simulation first.")
    
    if sim.struct.n3 != sim.struct.n2:
        raise ValueError("`farfield` does not support a cladding layer yet. " +
                         "It implements only asymptotic Green's tensor for an " +
                         "environment with a substrate.")
    
    if str(polarizerangle).lower() == 'none':
        polarizer = 0
    else:
        polarizer = polarizerangle * np.pi/180.
    
    if np.pi < tetamax < 0:
        raise ValueError("`tetamax` out of range, must be in [0, pi]")
    
    
    field_params    = sim.E[field_index][0]
    wavelength      = field_params['wavelength']
    
    dteta = (tetamax-tetamin)/float(Nteta-1)
    dphi = 2.*np.pi/float(Nphi)
    
#==============================================================================
#     Scattered field
#==============================================================================
    ## --- environment
    eps0, eps1 = sim.struct.n1**2, sim.struct.n2**2
    
    ## --- dipoles: Positions and their electric polarization
    xm, ym, zm = sim.struct.geometry.T    
    alpha = sim.struct.getPolarizability(wavelength)
    scattered_efield = sim.E[field_index][1]
    cpx = np.asfortranarray(scattered_efield.T[0] * alpha, dtype=sim.dtypec)
    cpy = np.asfortranarray(scattered_efield.T[1] * alpha, dtype=sim.dtypec)
    cpz = np.asfortranarray(scattered_efield.T[2] * alpha, dtype=sim.dtypec)
    
    ## --- calculate scattered field
    tetalist, philist, IntensInt, IntensIntPol, IntensPol, Intens, \
                cex_sc, cey_sc, cez_sc, ce_sc_pol = \
                              forFarfield(wavelength, int(Nteta), int(Nphi), 
                                          tetamin, tetamax,
                                          r, eps0, eps1, polarizer,
                                          xm, ym, zm,
                                          cpx, cpy, cpz)

#==============================================================================
#     Fundamental field
#==============================================================================
    ## --- fundamental field - use dummy structure with 
    ## --- spherical map-definition as "geometry"
    xff = (r * np.sin(tetalist) * np.cos(philist)).flatten()
    yff = (r * np.sin(tetalist) * np.sin(philist)).flatten()
    zff = (r * np.cos(tetalist)).flatten()
    MAP = [xff, yff, zff]
    
    field_generator = sim.efield.field_generator
    dummy_struct = copy.deepcopy(sim.struct)
    dummy_struct.geometry = np.transpose(MAP)
    E0 = field_generator(dummy_struct, returnField='E', **field_params)
    ce0x = np.asfortranarray(E0.T[0], dtype=sim.dtypec).reshape((int(Nteta), int(Nphi)))
    ce0y = np.asfortranarray(E0.T[1], dtype=sim.dtypec).reshape((int(Nteta), int(Nphi)))
    ce0z = np.asfortranarray(E0.T[2], dtype=sim.dtypec).reshape((int(Nteta), int(Nphi)))
    
    ## apply correct phase for Z<0
    ce0x.T[np.logical_and(tetalist.T>np.pi/2, tetalist.T<3*np.pi/2)] *= -1
    ce0y.T[np.logical_and(tetalist.T>np.pi/2, tetalist.T<3*np.pi/2)] *= -1
    ce0z.T[np.logical_and(tetalist.T>np.pi/2, tetalist.T<3*np.pi/2)] *= -1
    
    if str(polarizerangle).lower() != 'none':
        ## --- fundamental E-field parallel and perpendicular to scattering plane
        ce0_par  = ( ce0x * np.cos(tetalist) * np.cos(philist) + 
                     ce0y * np.sin(philist) * np.cos(tetalist) - 
                     ce0z * np.sin(tetalist) )
        ce0_perp = ( ce0x * np.sin(philist) - ce0y * np.cos(philist) )
        ## --- fundamental E-field parallel to polarizer
        ce0_pol  = ( ce0_par * np.cos(polarizer - philist) - 
                     ce0_perp * np.sin(polarizer - philist) )

#==============================================================================
#     Intensities with and without fundamental field / polarizer
#==============================================================================
    ## --- total field (no polarizer)
    I_sc  = Intens
    I0    = ( np.abs(ce0x)**2 + np.abs(ce0y)**2 + np.abs(ce0z)**2 )
    I_tot = ( np.abs(cex_sc + ce0x)**2 + 
              np.abs(cey_sc + ce0y)**2 + 
              np.abs(cez_sc + ce0z)**2 )
    
    ## --- optionally: with polarizer
    if str(polarizerangle).lower() != 'none':
        I_sc  = IntensPol
        I0    = ( np.abs(ce0_pol)**2 )
        I_tot = ( np.abs(ce_sc_pol + ce0_pol)**2 )
        
    if return_value.lower() == 'map':
        return tetalist, philist, I_sc, I_tot, I0
    elif return_value.lower() in ['efield', 'fields', 'field']:
        return tetalist, philist, np.transpose([cex_sc, cey_sc, cez_sc]), \
               np.transpose([cex_sc + ce0x, cey_sc + ce0y, cez_sc + ce0z]), \
               np.transpose([ce0x, ce0y, ce0z])
    else:
        d_solid_surf = r**2 * np.sin(tetalist) * dteta*dphi
        if return_value.lower() == 'int_es':
            return np.sum(I_sc*d_solid_surf)
        elif return_value.lower() == 'int_e0':
            return np.sum(I0*d_solid_surf)
        elif return_value.lower() == 'int_etot':
            return np.sum(I_tot*d_solid_surf)
        else:
            raise ValueError("Parameter 'return_value' must be one of ['map', 'int_es', 'int_e0', 'int_etot'].")


def multipole_decomp(sim, field_index, r0=None):
    """multipole decomposition of nanostructure optical response
    
    ** ------- FUNCTION STILL UNDER TESTING ------- **
    
    Multipole decomposition of electromagnetic field inside nanostructure for 
    electric and magnetic dipole and quadrupole moments.


    Parameters
    ----------
    sim : :class:`.core.simulation`
        simulation description
    
    field_index : int
        index of evaluated self-consistent field to use for calculation. Can be
        obtained for specific parameter-set using :func:`.tools.get_closest_field_index`
    
    r0 : array, default: None
        [x,y,z] position of mulipole decomposition development. 
        If `None`, use structure's center of gravity
    
    
    Returns
    -------
    
    p : 3-vector
        electric dipole moment
    
    m : 3-vector
        magnetic dipole moment
        
    q : 3x3 tensor
        electric quadrupole moment
        
    mq : 3x3 tensor
        magnetic quadrupole moment
    
    
    Notes
    -----
    For details about the method, see: 
    Evlyukhin, A. B. et al. *Multipole analysis of light scattering by 
    arbitrary-shaped nanoparticles on a plane surface.*, 
    JOSA B 30, 2589 (2013)
    """
    warnings.warn("Multipole decomposition is a new functionality still under testing. " + 
                  "Please use with caution.")
    
    field_params = sim.E[field_index][0]
    wavelength = field_params['wavelength']
    k0 = 2*np.pi / wavelength
    
    geo = sim.struct.geometry
    E = sim.E[field_index][1]
    ex, ey, ez = E.T
    
    ## --- polarizability of each meshpoint
    alpha = sim.struct.getPolarizability(wavelength)
    
    ## --- use center of gravity for multimode expansion
    if r0 is None:
        r0 = np.average(geo, axis=0)
    Dr = geo - r0
    
    
    ## --- total electric dipole moment:
    ##     P = sum_i p(r_i) = chi V_cell sum_i E(r_i)
    P = alpha[:, None] * E
    p = np.sum(P, axis=0)  # = volume integral
    
    ## --- total electric quadrupole moment: (X=curl)
    ##     Q = 3 * sum_i (Dr p(r_i) + p(r_i) Dr)
    ##       = 3 * sum_i chi Vcell (Dr E(r_i) + E(r_i) Dr)
    Q = 3 * (np.einsum('li,lj->lji', Dr, P) + np.einsum('li,lj->lji', P, Dr))
    q = np.sum(Q, axis=0)
        
    ## --- total magnetic dipole moment: (X=curl)
    ##     M = -iw0/2c  sum_i (r_i - r0) X p(r_i) 
    ##       = -ik0/2 sum_i chi Vcell (r_i - r0) X E(r_i)
    M = -(1j*k0/2.) * np.cross(Dr, P)
    m = np.sum(M, axis=0)
    
    ## --- total magnetic dipole moment: (X=curl)
    ##     M = -2iw0/3c  sum_i (r_i - r0) X p(r_i) 
    ##       = -2ik0/3 sum_i chi Vcell (r_i - r0) X E(r_i)
    MQ = -(2j*k0/3.) * np.einsum('li,lj->lji', np.cross(Dr, P), Dr)
    mq = np.sum(MQ, axis=0)
    
    return p, m, q, mq
    
    
    
def multipole_decomp_extinct(sim, field_index, r0=None, eps_dd=0.1):
    """extinction from multipole decomposition
    
    ** ------- FUNCTION STILL UNDER TESTING ------- **
    
    Returns extinction cross sections for electric and magnetic dipole and 
    quadrupole moments of the multipole decomposition.
    
    
    Parameters
    ----------
    sim : :class:`.core.simulation`
        simulation description
    
    field_index : int
        index of evaluated self-consistent field to use for calculation. Can be
        obtained for specific parameter-set using :func:`.tools.get_closest_field_index`
    
    r0 : array, default: None
        [x,y,z] position of mulipole decomposition development. 
        If `None`, use structure's center of gravity
    
    
    eps_dd : float, default: 0.1
        numerical integration stepsize (in nm). Used for e/m quadrupole.
    
    
    Returns
    -------
    
    sigma_ext_p : float
        electric dipole extinction cross section (in nm^2)
    
    sigma_ext_m : float
        magnetic dipole extinction cross section (in nm^2)
        
    sigma_ext_q : float
        electric quadrupole extinction cross section (in nm^2)
        
    sigma_ext_mq : float
        magnetic quadrupole extinction cross section (in nm^2)
    
    
    Notes
    -----
    For details about the method, see: 
    Evlyukhin, A. B. et al. *Multipole analysis of light scattering by 
    arbitrary-shaped nanoparticles on a plane surface.*, 
    JOSA B 30, 2589 (2013)
    
    """
    ## --- by default, use center of gravity for multimode expansion
    field_params = sim.E[field_index][0]
    wavelength = field_params['wavelength']
    k0 = 2*np.pi / wavelength
    
    geo = sim.struct.geometry
    n2 = sim.struct.n2
    E = sim.E[field_index][1]
    ex, ey, ez = E.T
    
    if r0 is None:
        r0 = np.average(geo, axis=0)
    
    p, m, q, mq = multipole_decomp(sim, field_index, r0=r0)
    
    
    class dummy_struct(object):
        dtypec = sim.struct.dtypec
        dtypef = sim.struct.dtypef
        spacing = sim.struct.spacing
        step = sim.struct.step
        def __init__(self, r0, sim):
            self.n1, self.n2, self.n3 = sim.struct.n1, sim.struct.n2, sim.struct.n3
            self.geometry = np.array(r0)
    
# =============================================================================
#     dipole extinction cross sections
# =============================================================================
    dummy_s = dummy_struct([r0], sim)
    E0 = sim.efield.field_generator(dummy_s, **field_params)
    H0 = sim.efield.field_generator(dummy_s, returnField='H', **field_params)
    E2in = np.sum(np.abs(E0)**2)     # incident intensity
    sigma_ext_p = (4 * np.pi * k0 / n2 * 1/E2in * np.sum(np.conjugate(E0)*p)).imag
    sigma_ext_m = (4 * np.pi * k0 / n2 * 1/E2in * np.sum(np.conjugate(H0)*m)).imag
    
    
# =============================================================================
#     quadrupole extinction cross sections
# =============================================================================
    DX = dummy_struct([r0-np.array([eps_dd,0,0]), r0+np.array([eps_dd,0,0])], sim)
    DY = dummy_struct([r0-np.array([0,eps_dd,0]), r0+np.array([0,eps_dd,0])], sim)
    DZ = dummy_struct([r0-np.array([0,0,eps_dd]), r0+np.array([0,0,eps_dd])], sim)
    EDXcj = np.conjugate(sim.efield.field_generator(DX, **field_params))
    EDYcj = np.conjugate(sim.efield.field_generator(DY, **field_params))
    EDZcj = np.conjugate(sim.efield.field_generator(DZ, **field_params))
    gradE0cj = np.array([[EDXcj[1,0]-EDXcj[0,0], EDYcj[1,0]-EDYcj[0,0], EDZcj[1,0]-EDZcj[0,0]],
                         [EDXcj[1,1]-EDXcj[1,1], EDYcj[1,1]-EDYcj[0,1], EDZcj[1,1]-EDZcj[0,1]],
                         [EDXcj[1,2]-EDXcj[1,2], EDYcj[1,2]-EDYcj[0,2], EDZcj[1,2]-EDZcj[0,2]]])
    sigma_ext_q = (np.pi * k0 / n2 * 1/(3. * E2in) * 
                           np.sum(np.tensordot(gradE0cj + gradE0cj.T, q) )).imag
    
    HDXcj = np.conjugate(sim.efield.field_generator(DX, returnField='H', **field_params))
    HDYcj = np.conjugate(sim.efield.field_generator(DY, returnField='H', **field_params))
    HDZcj = np.conjugate(sim.efield.field_generator(DZ, returnField='H', **field_params))
    gradH0cj = np.array([[HDXcj[1,0]-HDXcj[0,0], HDYcj[1,0]-HDYcj[0,0], HDZcj[1,0]-HDZcj[0,0]],
                         [HDXcj[1,1]-HDXcj[1,1], HDYcj[1,1]-HDYcj[0,1], HDZcj[1,1]-HDZcj[0,1]],
                         [HDXcj[1,2]-HDXcj[1,2], HDYcj[1,2]-HDYcj[0,2], HDZcj[1,2]-HDZcj[0,2]]])
    sigma_ext_mq = (2 * np.pi * k0 / n2 * 1/E2in * 
                           np.sum(np.tensordot(gradH0cj.T, mq) )).imag
    
    return sigma_ext_p, sigma_ext_m, sigma_ext_q, sigma_ext_mq



def nearfield(sim, field_index, r_probe, val_inside_struct="field", 
              return_value='field'):
    """Nearfield distribution in proximity of nanostructre
    
    For a given incident field, calculate the electro-magnetic field in the 
    proximity of the nanostructure (positions defined by `MAP`).
    
    Parameters
    ----------
    sim : :class:`.core.simulation`
        simulation description
    
    field_index : int
        index of evaluated self-consistent field to use for calculation. Can be
        obtained for specific parameter-set using :func:`.tools.get_closest_field_index`
    
    r_probe : tuple (x,y,z) or list of 3-lists/-tuples
        (list of) coordinate(s) to evaluate nearfield on. 
        Format: tuple (x,y,z) or list of 3 lists: [Xmap, Ymap, Zmap] 
        (the latter can be generated e.g. using :func:`.tools.generate_NF_map`)

    val_inside_struct : str, default: "field"
        one of ["field", "0", "zero", "NaN"].
        value to return for positions inside structure. default "field" returns 
        the field, which is however only available on mesh positions and can 
        be very coarse.
        
    return_value : str, default: "field"
        Which quantity to return. either "field" or "intensity" ("f" or "i").

    
    Returns
    -------
    4 lists of 6-tuples, complex : 
        - scattered Efield
        - total Efield (inlcuding fundamental field)
        - scattered Bfield
        - total Bfield (inlcuding fundamental field)
    
    
    the tuples are of shape (X,Y,Z, Ax,Ay,Az) with Ai the corresponding 
    complex field component. if return_value=="intensity" 4-tuples, float (x,y,z,I)
        
    
    Notes
    -----
    For details of the calculation of the scattered field outside the 
    nano-object using the self-consistent field inside the particle, see e.g.: 
    Girard, C. **Near fields in nanostructures.** 
    Reports on Progress in Physics 68, 1883–1933 (2005).

    """
    from .pyGDMfor import nearfield as forNearfield
    
    if sim.E is None: 
        raise ValueError("Error: Scattering field inside the structure not yet evaluated. Run `core.scatter` simulation first.")
    
    if len(np.shape(r_probe)) == 1:
        if len(r_probe) == 3:
            r_probe = [[r_probe[0]], [r_probe[1]], [r_probe[2]]]
        else: 
            raise ValueError("If 'r_probe' is tuple, must consist of *exactly* 3 elements!")
    elif len(np.shape(r_probe)) == 2:
        if np.shape(r_probe)[0] != 3 and np.shape(r_probe)[1] != 3:
            raise ValueError("'r_probe' must consist of *exactly* 3 elements!")
        if np.shape(r_probe)[0] != 3:
            r_probe = np.transpose(r_probe)
    else:
        raise ValueError("wrong format for 'r_probe'. must consist of *exactly* 3 elements, either floats, or lists.")
    
    field_generator = sim.efield.field_generator
    xm, ym, zm      = sim.struct.geometry.T
    
    field_params    = sim.E[field_index][0]
    wavelength      = field_params['wavelength']
    
    cap = sim.struct.getPolarizability(wavelength)
    
    scattered_efield = sim.E[field_index][1]
    Ex = np.asfortranarray(scattered_efield.T[0], dtype=sim.dtypec)
    Ey = np.asfortranarray(scattered_efield.T[1], dtype=sim.dtypec)
    Ez = np.asfortranarray(scattered_efield.T[2], dtype=sim.dtypec)
    
    
    ## -- fundamental field - use dummy structure with map-definition as "geometry"
    dummy_struct = copy.deepcopy(sim.struct)
    dummy_struct.geometry = np.transpose(r_probe)
    E0 = field_generator(dummy_struct, returnField='E', **field_params)
    E0x = np.asfortranarray(E0.T[0], dtype=sim.dtypec)
    E0y = np.asfortranarray(E0.T[1], dtype=sim.dtypec)
    E0z = np.asfortranarray(E0.T[2], dtype=sim.dtypec)
    
    B0 = field_generator(dummy_struct, returnField='B', **field_params)
    B0x = np.asfortranarray(B0.T[0], dtype=sim.dtypec)
    B0y = np.asfortranarray(B0.T[1], dtype=sim.dtypec)
    B0z = np.asfortranarray(B0.T[2], dtype=sim.dtypec)
    
    
    ## -- coordinates to evaluate
    MAPx = np.asfortranarray(r_probe[0], dtype=sim.dtypef)
    MAPy = np.asfortranarray(r_probe[1], dtype=sim.dtypef)
    MAPz = np.asfortranarray(r_probe[2], dtype=sim.dtypef)
    
    ## internal fields treatment behavior
    if val_inside_struct.lower() == "field":
        fill_inside = -1
    elif val_inside_struct.lower() in ["zeros", "0", "zero", "null"]:
        fill_inside = 0
    elif val_inside_struct.lower() in ["nan"]:
        fill_inside = -2
    
    cepx,cepy,cepz,cepx1,cepy1,cepz1, \
        cbpx,cbpy,cbpz,cbpx1,cbpy1,cbpz1 = forNearfield(
                 alambda=wavelength, space=sim.struct.spacing,
                 step=sim.struct.step, xm=xm, ym=ym, zm=zm,
                 cap=cap,
                 cn1=sim.struct.n1, cn2=sim.struct.n2, cn3=sim.struct.n3,
                 csolx=E0x, csoly=E0y, csolz=E0z, 
                 csolbx=B0x, csolby=B0y, csolbz=B0z,
                 cex=Ex, cey=Ey, cez=Ez, 
                 xmap=MAPx, ymap=MAPy, zmap=MAPz,
                 fillinside=fill_inside)
    
    if fill_inside == -2:
        cepx1[cbpz==-999999] = np.nan
        cepy1[cbpz==-999999] = np.nan
        cepz1[cbpz==-999999] = np.nan
        cepx[cbpz==-999999] = np.nan
        cepy[cbpz==-999999] = np.nan
        cepz[cbpz==-999999] = np.nan
        cbpx1[cbpz==-999999] = np.nan
        cbpy1[cbpz==-999999] = np.nan
        cbpz1[cbpz==-999999] = np.nan
        cbpx[cbpz==-999999] = np.nan
        cbpy[cbpz==-999999] = np.nan
        cbpz[cbpz==-999999] = np.nan
    ## --- fill positions inside structure if requested
    val_inside_struct = str(val_inside_struct)
    if val_inside_struct.lower() != "field":
        try:
            from scipy.linalg import norm
            ## -- check if scipy.linalg.norm supports 'axis' kwarg
            import scipy
            if int(scipy.__version__.split('.')[0]) == 0 and int(scipy.__version__.split('.')[1]) < 17:
                no_norm_axis = True
            else:
                no_norm_axis = False
        except ImportError:
            no_norm_axis = True
        if no_norm_axis:
            warnings.warn("No scipy or scipy with version < 0.17.0 installed! Field at positions inside nanostructure cannot be replaced by fill value. *nearfield* will return the field instead.")
        
        if val_inside_struct.lower() == "nan":
            fill_value = np.nan
        else:
            fill_value = 0
        
        for i,R in enumerate(r_probe.T):
            ## --- outside structure
            if abs(sorted(norm(sim.struct.geometry - R, axis=1))[0]) <= 1.05*sim.struct.step:
                cepx[i] = cepy[i] = cepz[i] = fill_value
                cepx1[i] = cepy1[i] = cepz1[i] = fill_value
                cbpx[i] = cbpy[i] = cbpz[i] = fill_value
                cbpx1[i] = cbpy1[i] = cbpz1[i] = fill_value
    
    Es = np.array([MAPx,MAPy,MAPz, cepx,cepy,cepz]).T
    Et = np.array([MAPx,MAPy,MAPz, cepx1,cepy1,cepz1]).T
    Bs = np.array([MAPx,MAPy,MAPz, cbpx,cbpy,cbpz]).T
    Bt = np.array([MAPx,MAPy,MAPz, cbpx1,cbpy1,cbpz1]).T
    
    if return_value.lower() in ['intensity', 'i']:
        from pyGDM2 import tools
        return tools.get_intensity_from_fieldlist(Es), \
               tools.get_intensity_from_fieldlist(Et), \
               tools.get_intensity_from_fieldlist(Bs), \
               tools.get_intensity_from_fieldlist(Bt)
    else:
        return Es, Et, Bs, Bt



def optical_chirality(sim, field_index, r_probe, which_field="t", **kwargs):
    """calculate the optical chirality of the electromagnetic field
    
    ** ------- FUNCTION STILL UNDER TESTING ------- **
    
    Returns the normalized electromagnetic chirality *C / C_LCP*, as defined in [2].
    Normalized to a left circular polarized (LCP) plane wave of amplitude |È0|=1.
    
     - LCP: C = +2 P0 c/omega
     - RCP: C = -2 P0 c/omega
    
    P0 is the time-averaged electric energy density, c the speed of light and omega
    the angular frequency of the time-harmonic electric field. 
    
    |C/C_LCP|>1 means that the local field is "superchiral", 
    hence a chiral molecule is excited with higher selectivity than it would be
    with circular polarized light.
    
    kwargs are passed to :func:`.nearfield`.
    
    Parameters
    ----------
    sim : :class:`.core.simulation`
        simulation description
    
    field_index : int
        index of evaluated self-consistent field to use for calculation. Can be
        obtained for specific parameter-set using :func:`.tools.get_closest_field_index`
    
    r_probe : tuple (x,y,z) or list of 3-lists/-tuples
        (list of) coordinate(s) to evaluate nearfield on. 
        Format: tuple (x,y,z) or list of 3 lists: [Xmap, Ymap, Zmap] 
        (the latter can be generated e.g. using :func:`.tools.generate_NF_map`)
    
    which_field : str, default: 't'
        either of:
            - "s", "Es" for *scattered field*
            - "t", "Et" for *total field* (with illumination, i.e. Et=Es+E0; Bt=Bs+B0)
    
    Returns
    -------
    list of tuples (X,Y,Z, X). Each tuple consists of the position and the 
    (normalized) optical chirality *C / C_LCP*. If evaluated at single 
    position, return only C as float
    
    
    Notes
    -----
    For details on the optical chirality, see:
    
    [1] Tang, Y. & Cohen, A. E.: **Optical Chirality and Its Interaction 
    with Matter**. Phys. Rev. Lett. 104, 163901 (2010)
    
    [2] Meinzer, N., Hendry, E. & Barnes, W. L.: **Probing the chiral nature 
    of electromagnetic fields surrounding plasmonic nanostructures**. 
    Phys. Rev. B 88, 041407 (2013)
    
    A discussion about the proper normalization of C can be found in:
    
    [3] Schäferling M., Yin X., Engheta N., Giessen H. **Correction to Helical 
    Plasmonic Nanostructures as Prototypical Chiral Near-Field Sources**. 
    ACS Photonics 3(10), 2000-2002 (2016)
    
    """
    warnings.warn("Chirality is a beta-functionality still under testing. " +
                  "Please use with caution.")
    
    Es, Et, Bs, Bt = nearfield(sim, field_index, r_probe, **kwargs)
    
    if which_field.lower() in ["s", "es", "scat", "scattered"]:
        Et = Es
        Bt = Bs
    
    ## Ref. [2]: C ~ Im(E* B)
    C = np.concatenate([
                    Et.T[:3], # positions
                    [np.sum(np.multiply(np.conjugate(Et.T[3:]), 
                                                     Bt.T[3:]), axis=0).imag]
                        ]).astype(np.float)
    
    if len(np.shape(r_probe)) == 1 or len(r_probe) == 1:
        return C[3][0]
    else:
        return C
    






# =============================================================================
# heat and temperature
# =============================================================================
def heat(sim, field_index, power_scaling_e0=1.0, return_value='total', return_units='nw'):
    """calculate the total induced heat in the nanostructure
    
    Parameters
    ----------
    sim : :class:`.core.simulation`
        simulation description
    
    field_index : int
        index of evaluated self-consistent field to use for calculation. Can be
        obtained for specific parameter-set using :func:`.tools.get_closest_field_index`
    
    power_scaling_e0 : float, default: 1
        incident laser power scaling. power_scaling_e0 = 1 corresponds 
        to 1 mW per micron^2. See [1].
    
    return_value : str, default: 'total'
        Values to be returned. Either 'total' (default) or 'structure'.
        
        - "total" : return the total deposited heat in nW (float)
        
        - "structure" : return spatially resolved deposited heat at each 
          meshpoint in nW (list of tuples [x,y,z,q])
    
    return_units : str, default: "nw"
        units of returned heat values, either "nw" or "uw" (nano or micro Watts)
    
    
    Returns
    -------
    Q : float *or* list of tuples [x,y,z,q]
    
        - `return_value`="total" : (return float)
          total deposited heat in nanowatt (optionally in microwatt). 

        - `return_value`="structure" : (return list of tuples [x,y,z,q])
          The returned quantity *q* is the total deposited heat 
          at mesh-cell position [x,y,z] in nanowatt. To get the heating 
          power-density, please divide by the mesh-cell volume.
    
    
    Notes
    -----
    For details on heat/temperature calculations and raster-scan simulations, see:
    
    [1] Baffou, G., Quidant, R. & Girard, C.: **Heat generation in plasmonic 
    nanostructures: Influence of morphology**
    Applied Physics Letters 94, 153109 (2009)
    
    [2] Teulle, A. et al.: **Scanning optical microscopy modeling in nanoplasmonics** 
    Journal of the Optical Society of America B 29, 2431 (2012).


    """
    if sim.E is None: 
        raise ValueError("Error: Scattering field inside the structure not yet evaluated. Run `core.scatter` simulation first.")
    
    power_scaling_e0 *= 0.01    # for mW/cm^2
    
    field_params    = sim.E[field_index][0]
    wavelength      = field_params['wavelength']
    k0 = 2*np.pi / wavelength
    
    ## --- Factor allowing to have the released power in nanowatt 
    released_power_scaling = 100.
    
    ## --- polarizabilities and electric fields
    alpha = sim.struct.getPolarizability(wavelength)
    E = sim.E[field_index][1]
    ex, ey, ez = E.T
    
    ## --- total deposited heat
#    I_e = np.abs(ex**2 + ey**2 + ez**2)
    I_e = (np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2)
    
    ## heat at each meshpoint in nW/nm^3
    q = 4.*np.pi *np.imag(alpha) * I_e * k0 * power_scaling_e0 * released_power_scaling
    
    ## --- optional conversion to micro watts
    if return_units.lower() == 'uw':
        q /= 1.0E3
    
    if return_value == 'total':
        return np.sum(q)
    elif return_value in ['structure', 'struct']:
        x,y,z = sim.struct.geometry.T
        return np.concatenate([[x],[y],[z],[q]]).T
    else:
        raise ValueError("`return_value` must be one of ['total', 'structure', 'struct'].")



def temperature(sim, field_index, r_probe, 
                kappa_env=0.6, kappa_subst=None, incident_power=1.0):
    """calculate the temperature rise at locations outside the nano-particle
    
    Calculate the temperature increase close to a optically excited 
    nanostructure using the approach described in [2] and [3] (Ref. [3] 
    introduces a further correction term for particles lying on a substrate)
    
    Parameters
    ----------
    sim : :class:`.core.simulation`
        simulation description
    
    field_index : int
        index of evaluated self-consistent field to use for calculation. Can be
        obtained for specific parameter-set using :func:`.tools.get_closest_field_index`
    
    r_probe : tuple (x,y,z) or list of 3-lists/-tuples
        (list of) coordinate(s) to evaluate nearfield on. 
        Format: tuple (x,y,z) or list of 3 lists: [Xmap, Ymap, Zmap] 
        (the latter can be generated e.g. using :func:`.tools.generate_NF_map`)
    
    kappa_env : float, default: 0.6
        heat conductivity of environment. default: kappa_env = 0.6 (water). 
        (air: kappa=0.024, for more material values see e.g. [4]). In W/mK.
    
    kappa_subst : float, default: None
        heat conductivity of substrate. default: None --> same as environment. 
        Using the mirror-image technique described in [3]. (glass: kappa=0.8)
    
    incident_power : float, default: 1.0
        incident laser power density in mW per micron^2. See also [1].
    
    
    Returns
    -------
    if evaluating at a single position, D_T : float
        temperature increase in Kelvin at r_probe
    
    if evaluating at a list of positions, list of tuples [x, y, z, D_T] 
        where D_T is the temperature increase in Kelvin at (x,y,z), which
        are the positions defined by `r_probe`
    
    Notes
    -----
    For details on heat/temperature calculations and raster-scan simulations, see:
    
    [1] Baffou, G., Quidant, R. & Girard, C.: **Heat generation in plasmonic 
    nanostructures: Influence of morphology**
    Applied Physics Letters 94, 153109 (2009)
    
    [2] Baffou, G., Quidant, R. & Girard, C.: **Thermoplasmonics modeling: 
    A Green’s function approach** 
    Phys. Rev. B 82, 165424 (2010)

    [3] Teulle, A. et al.: **Scanning optical microscopy modeling in nanoplasmonics**
    Journal of the Optical Society of America B 29, 2431 (2012).
    
    
    For the thermal conductivity of common materials, see:
    
    [4] Hugh D Young, Francis Weston Sears: **University Physics**, *chapter 15*,
    8th. edition: Addison-Wesley, 1992
    (see also e.g.: http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/thrcn.html)

    """
    kappa_subst = kappa_subst or kappa_env
    
    if sim.E is None: 
        raise ValueError("Error: Scattering field inside the structure not yet evaluated. Run `core.scatter` simulation first.")
    
    if kappa_subst == kappa_env and sim.struct.n1 != sim.struct.n2:
        warnings.warn("Substrate and environment have different indices but same heat conductivity.")
    if kappa_subst != kappa_env and sim.struct.n1 == sim.struct.n2:
        warnings.warn("Substrate and environment have same ref.index but different heat conductivity.")
    
    incident_power *= 0.01          # for mW/cm^2
    released_power_scaling = 100.   # output in nW
    
    field_params    = sim.E[field_index][0]
    wavelength      = field_params['wavelength']
    
    n2 = sim.struct.n2
    k_env = 2*np.pi* np.real(n2) / wavelength
    
    
    
    ## --- main heat generation function
    def calc_heat_single_position(sim, r_probe):
        ## --- polarizability and field in structure
        alpha = sim.struct.getPolarizability(wavelength)
        E = sim.E[field_index][1]
        ex, ey, ez = E.T
        
        ## --- meshpoint distance to probe, polarizabilities and electric fields
        r_mesh = sim.struct.geometry
        dist_probe = np.sqrt( np.sum( np.power((np.array(r_probe) - r_mesh), 2), axis=1) )
        
        ## --- mirror structure below substrate for heat reflection at substrate
        if kappa_subst != kappa_env:
            r_mesh_mirror = copy.deepcopy(sim.struct.geometry)
            r_mesh_mirror.T[2] *= -1
            dist_probe_mirror = np.sqrt( np.sum( np.power((np.array(r_probe) - r_mesh_mirror), 2), axis=1) ) 
        
        ## --- temperature rise at r_probe
#        I_e = np.abs(ex**2 + ey**2 + ez**2)
        I_e = np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2
#        I_e = ex.real**2 + ex.imag**2 + ey.real**2 + ey.imag**2 + ez.real**2 + ez.imag**2
        q = np.imag(alpha) * I_e * k_env * incident_power * released_power_scaling
        D_T = np.sum( q / dist_probe )
        if kappa_subst != kappa_env:
            D_T += np.sum( q / dist_probe_mirror ) * (kappa_subst - kappa_env)/(kappa_subst + kappa_env)
        D_T /= kappa_env
        
#        if np.isnan(D_T):
#            print D_T, np.sum(I_e), np.imag(alpha[0])
        
        return D_T
    
    
    ## --- SINGLE POSITION:
    if len(np.shape(r_probe)) == 1 or len(r_probe) == 1:
        D_T = calc_heat_single_position(sim, r_probe)
    ## --- MULTIPLE POSITIONS:
    else:
        D_T = []
        if np.shape(r_probe)[1] != 3:
            r_probe = np.transpose(r_probe)
        for i in r_probe:
            D_T.append([i[0], i[1], i[2], 
                        calc_heat_single_position(sim, i)])
        D_T = np.array(D_T)
    
    return D_T





def decay_eval(sim, SBB, mx,my,mz, verbose=False):
    """evaluate decay rate of electric or magnetic dipole transition
    
    Evaluate the decay rate modification of a dipole with complex amplitude
    (mx,my,mz) using pre-calculated tensors (:func:`.core.decay_rate`).
    
    Parameters
    ----------
      sim : :class:`.core.simulation`
        simulation description
      
      SBB : int or list of lists
          index of wavelength in `sim` or tensor-list as returned by 
          :func:`.core.decay_rate`
      
      mx,my,mz : float 
          x/y/z amplitude of dipole transition vector
      
      verbose : bool default=False
          print some runtime info
    
    Returns
    -------
      gamma_map: list of coordinates: [[x1,y1,z1, gamma1], [x2,..] ...]
          Each element consists of:
           - x,y,z: coordinates of evaluation position
           - gamma: normalized decay-rate (real) gamma / gamma_0 at each map-position
      
    Notes
    -----
     - For detailed information about the underlying formalism, see:
       Wiecha, P. R., Girard, C., Cuche, A., Paillard, V. & Arbouet, A. 
       **Decay Rate of Magnetic Dipoles near Non-magnetic Nanostructures.** 
       Phys. Rev. B 97, 085411 (2018).
    
     - Requires scipy v0.17.0 or later to work also inside the volume of a nanostructure
      
    """
    from . import fields
    if sim.efield.field_generator == fields.dipole_electric:
        dp_type = 'electric'
    elif sim.efield.field_generator == fields.dipole_magnetic:
        dp_type = 'magnetic'
    else:
        raise ValueError("Wrong incident field: `decay_rate` requires the " + 
                         "incident field to be " + 
                         "either an electric or a magnetic dipole emitter. " +
                         "Please use `fields.dipole_electric` or " +
                         "`fields.dipole_magnetic`.")
    
    ## --- evaluation coordinates of dipole
    xyz = [[pos['x0'], pos['y0'], pos['z0']] for pos in sim.efield.kwargs_permutations]
    MAP = np.array(xyz)
    
    if type(SBB) == int:
        if sim.S_P is None: 
            raise ValueError("Error: Decay tensors not yet evaluated. Run `core.decay_rate` first.")
        SBB = sim.S_P[SBB]
        
    wavelength = SBB[0]
    SBB = SBB[1]
        
    ak0 = 2.*np.pi / (wavelength)
    
    if verbose:
        print("decay-rate evaluated using:")
        print("  - wavelength: {}nm".format(wavelength))
        print("  - dipole type: {}".format(dp_type))
        print("  - dipole vector: ({}, {}, {})".format(mx,my,mz))
    
    
    ## --- normalization
    ## normalize by chi/step^3 (divide by step**3 because cell-volume is already 
    ##                          taken into account in GDM polarizabilities)
    sim.struct.getNormalization(wavelength)
    Chi = sim.struct.getPolarizability(wavelength)
    if len(np.unique(Chi)) != 1:
        raise ValueError("Anisotropic structures not supported by the decay " +
                         "rate calculation at the moment! Please use a " +
                         "constant material dispersion for the whole structure.")
    Chi = Chi[0]
    normSBB = Chi / sim.struct.step**3  
    
    
    ## calc. unitary dipole orientation: divide by dipole length
    meg = np.sqrt(mx**2 + my**2 + mz**2)
    mu = np.array([mx, my, mz]) / meg
    
    ## vacuum value for decay
    gamma_0 = 1
    
    ## get propagator SBB
    try:
        from scipy.linalg import norm
        ## -- check if scipy.linalg.norm supports 'axis' kwarg
        import scipy
        if int(scipy.__version__.split('.')[0]) == 0 and int(scipy.__version__.split('.')[1]) < 17:
            no_norm_axis = True
        else:
            no_norm_axis = False
    except ImportError:
        no_norm_axis = True
        
        
    gamma_map = np.zeros( (len(MAP),4) )
    for i,R in enumerate(MAP):
        gamma_map[i][0] = R[0]
        gamma_map[i][1] = R[1]
        gamma_map[i][2] = R[2]
        gamma_map[i][3] = (gamma_0 + (3./2.) * (1./ak0**3) * gamma_0 * 
                           np.dot(np.dot(mu, (normSBB * SBB[i]).imag), mu))
        
        ## --- outside structure
        if no_norm_axis:
            warnings.warn("No scipy or scipy with version < 0.17.0 installed! LDOS will NOT be evaluated correctly inside the structure !!!")
            gamma_map[i][3] = (gamma_0 + (3./2.) * (1./ak0**3) * gamma_0 * 
                               np.dot(np.dot(mu, (normSBB * SBB[i]).imag), mu))
        elif abs(sorted(norm(sim.struct.geometry - R, axis=1))[0]) >= sim.struct.step:
            gamma_map[i][3] = (gamma_0 + (3./2.) * (1./ak0**3) * gamma_0 * 
                               np.dot(np.dot(mu, (normSBB * SBB[i]).imag), mu))
        ## --- inside structure
        else:
            if dp_type == 'magnetic':
                raise Exception("In the current pyGDM version, the magnetic LDOS cannot be calculated `inside` the particle.")
            gamma_map[i][3] = (gamma_0 + (3./2.) * (1./ak0**3) * gamma_0 * 
                               np.dot(np.dot(mu, (1./sim.struct.step**3   * SBB[i]).imag), mu))
    
    return gamma_map

