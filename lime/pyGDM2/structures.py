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
#
"""
Collection of structure generators and geometry related tools

"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

import numpy as np
import copy

from .core import simulation
from . import tools







#==============================================================================
# Structure container class
#==============================================================================
class struct(object):
    """structure container
    
    Defines a nanostructure to perform a GDM simulation on. This includes the
    geometry, the material specific dielectric function(s), as well as the 
    environment in which the structure located.
    
    
    Parameters
    ----------
    step : float
        stepsize used for discretization
    
    geometry : list of tuples
        list of coordinates of the meshpoints of the geometry. 
        Several structure generators are provided in the module `structures`
    
    material : instance of `materials` class or list of such
        use material classes provided by the `materials` module. 
        If a single instance is provided, the according material will be 
        used for the entire structure
    
    n1, n2 : complex
        complex refractive index of environmental layers 
        "1" (substrate), "2" (environment containing structure)
    
    normalization : float
        can be obtained using :func:`.structures.get_normalization`. 
        =1 for cubic mesh, =sqrt(2) for hexagonal compact mesh.
    
    n3 : complex, default: value of n2
        complex refractive index of (optional) top cladding layer (layer "3"). 
        By default, use ref.index of structure environment (layer "2", ref.index n2)
    
    spacing : float, default = 5000
        thickness of layer "2" in nm. Layers "1" and "2" are infinitely thick 
        ("1" from -inf to 0; "3" from `spacing` to +inf)
        
    with_radiation_correction : bool, default: True
        Adds an optional radiative correction to the self-term, like described
        by *B. Draine, in: Astrophys. J. 333, 848–872 (1988)*.
        Using the correction usually leads to better convergence for very large 
        discretization stepsizes (>=20nm), and has only a weak influence on the 
        results from fine meshes.
    

    """
    def __init__(self, step, geometry, material, n1,n2, normalization, 
                 n3=None, spacing=5000.0, with_radiation_correction=True,
                 auto_shift_structure=True):
        """Initialization"""
        self.step = step
        if type(geometry) == list:
            geometry = np.array(geometry)
            
        ## structure consistenct check:
        self.geometry, _wrong_dipoles, index_correct = tools.test_geometry(
                            geometry, step, return_index=True, plotting=False)
        if len(_wrong_dipoles) != 0:
            warnings.warn("using only consistent part of structure. " +
                          "Please verify the geometry.")
        
        self.normalization = normalization
        self.with_radiation_correction = with_radiation_correction
        self.n_dipoles = len(self.geometry)
        
        self.n1 = complex(n1)
        self.n2 = complex(n2)
        self.n3 = n3 or self.n2
        self.n3 = complex(self.n3)
        
        self.spacing = spacing
        
        ## set precision to single by default
        self.setDtype(np.float32, np.complex64)
        
        ## Geometry check: Lowest dipole MUST NOT lie on Z=0. If < 0, elevate to Z_min = step/2
        if len(self.geometry) > 0:
            if self.geometry.T[2].min() - self.step/2.0 < 0:
                if auto_shift_structure:
                    warnings.warn("Minimum structure Z-value lies below substrate level!" +
                                  " Shifting structure bottom to Z=step/2.")
                    self.geometry.T[2] += (self.geometry.T[2].min() + self.step/2.0)
                else:
                    warnings.warn("Minimum structure Z-value lies below substrate level!" +
                                  " Must be shifted above zero if a substrate is used!")
                
            
            ## --- material at each meshpoint
            if isinstance(material, (list, tuple, np.ndarray )):
                if len(material) != len(geometry):  # cf wirh original geometry here!
                    raise ValueError("Error in structure initialization: " +
                                     "Number of material definitions must " +
                                     "equal number of dipoles.")
                self.material = np.array(material)[index_correct]  # maintain order
            else:
                self.material = [material for i in range(len(self.geometry))]
        
        else:
            warnings.warn("Emtpy structure geometry.")
            self.material = material
        
    
    def setDtype(self, dtypef, dtypec):
        """set dtype of arrays"""
        self.dtypef = dtypef
        self.dtypec = dtypec
        self.geometry = np.asfortranarray(self.geometry, dtype=dtypef)


    def getPolarizability(self, wavelength):
        """Returns the scalar polarizabilities of each meshpoint at "wavelength"
        
        Calculate the "polarizabilites" for each meshpoint using its volume 
        (defined by step) and its dielectric function ("material") as well as 
        the environment refractive index ("n2").
        
        For the moment, only isotropic materials (scalar espilon) are supported.
        (anistropic tensor support is already implemented in fortran routines 
        but not exposed to python)
        
        Calling this function will set the internal variables 
        `struct.wavelength`, `struct.alpha`, `struct.areal` (real part of alpha) 
        and `struct.aimag` (imag part of alpha)
        
        
        Parameters
        ----------
        
        wavelength : float
            wavelength (in nm) at which to calculate polarizabilites
        
        Returns
        -------
        
        alpha : list of complex
            complex "polarizabilities" for each meshpoint
        
        Notes
        -----
        
        These are not actual polarizabilities and in particular not 
        identical to the ones used e.g. in the coupled dipole approximation 
        software *DDSCAT*. 
        For details, see e.g.: Girard, C. **Near fields in nanostructures.** 
        Reports on Progress in Physics 68, 1883-1933 (2005).
        
        """
        if self.dtypef is None:
            raise ValueError("Error in structure evaluation: 'dtype' not " +
                             "set yet. Please call 'setDtype' first.")
        
        self.wavelength = wavelength
        
        ## environment index at each meshpoint position
        epsilon_environment = np.ones(len(self.geometry)) * self.n2**2.0
        epsilon_environment[self.geometry.T[2]>self.spacing] = self.n3**2.0
        epsilon_environment[self.geometry.T[2]<0] = self.n1**2.0
        
        vcell_norm = self.step**3 / float(self.normalization)
        
        epsilon = np.array([mat.epsilon(wavelength) for mat in self.material])
        ## case of list of wavelengths:
        if len(epsilon.shape) == 2:
            epsilon_environment = epsilon_environment[:, None] 
            
        ## --- isotropic polarizability
        alpha = np.asfortranarray((epsilon - epsilon_environment) * 
                                      vcell_norm / (4.0 * np.pi), dtype=self.dtypec)
        areal = np.asfortranarray(alpha.real, dtype=self.dtypef)
        aimag = np.asfortranarray(alpha.imag, dtype=self.dtypef)
        
        self.alpha = alpha
        self.areal = areal
        self.aimag = aimag
        
        return alpha
    
    
    def getPolarizabilityTensor(self, wavelength):
        """Returns the (3x3) polarizability tensor of each meshpoint at "wavelength"
        
        requires material class which supports an `epsilonTensor` method.
        Otherwise, isotropic polarizabilities will be returned.
        
        """
        epsilon = []
        for mat in self.material:
            if hasattr(mat, "epsilonTensor"):
                epsilon.append(mat.epsilonTensor(wavelength))
            else:
                ## scalar permittivity: use diagonal identity tensor
                epsilon.append(mat.epsilon(wavelength)*np.identity(3, dtype=self.dtypec))
        epsilon = np.array(epsilon)
        
        self.wavelength = wavelength
        
        ## --- environment epsilon (tensor) at each meshpoint position
        epsilon_environment = np.ones(len(self.geometry)) * self.n2**2.0
        epsilon_environment[self.geometry.T[2]>self.spacing] = self.n3**2.0
        epsilon_environment[self.geometry.T[2]<0] = self.n1**2.0
        
        epsilon_env_tensor = np.zeros(epsilon.shape, dtype=self.dtypec)
        epsilon_env_tensor[:,0,0]=epsilon_environment
        epsilon_env_tensor[:,1,1]=epsilon_environment
        epsilon_env_tensor[:,2,2]=epsilon_environment
        
        vcell_norm = self.step**3 / float(self.normalization)
        
        ## --- 3x3 polarizability tensor at each mesh cell
        alphatensor = np.asfortranarray((epsilon - epsilon_env_tensor) * 
                                      vcell_norm / (4.0 * np.pi), dtype=self.dtypec)
        
        return alphatensor


    def getNormalization(self, wavelength):
        """Returns the normalization term at "wavelength"
        
        Calling this function will set the internal variables `struct.wavelength`
        and `struct.cnorm`, `self.norm_nonrad` and `self.norm_rad`
        
        
        Parameters
        ----------
        
        wavelength : float
            wavelength (in nm) at which to calculate polarizabilites
        
        
        Returns
        -------
        
        cnorm : complex
            normalization term for used mesh and wavelength
        
        
        Notes
        -----
        
        For details on the normalization derivation, see e.g.
        Girard, C., Dujardin, E., Baffou, G. & Quidant, R. **Shaping and 
        manipulation of light fields with bottom-up plasmonic structures.** 
        New J. Phys. 10, 105016 (2008).
        """
        self.wavelength = wavelength
        k0 = 2.0*np.pi / float(wavelength)
        epsilon_environment = self.n2 ** 2.0
        
        norm_nonrad = -4.0 * np.pi * self.normalization / (3.0 * (self.step**3) * epsilon_environment)
        norm_rad = 1j * 2.0 * (k0**3)/3.0
        
        if self.with_radiation_correction:
            self.cnorm = norm_nonrad + norm_rad
        else:
            self.cnorm = norm_nonrad
        self.norm_nonrad = norm_nonrad
        self.norm_rad = norm_rad
        
        return self.cnorm












#==============================================================================
# STRUCTURE GENERATORS
#==============================================================================
def rect_wire(step, L,H,W, mesh='cube', ORIENTATION=1):
    """
    Generate rectangular wire with long axis along X on X/Y plane
    
    Parameters
    ----------
    step : float
        stepsize in nw
    L, H, W : int
        Length, Height and Width as Nr. of steps
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    def conditionRect(xi,yi,zi):
        return (-L/2.<xi<=L/2. and -W/2.<yi<=W/2. and 0<=zi<H - 1/np.sqrt(2))
    
    if mesh == 'cube':
        sp = _meshCubic( (-int(L), int(L)), (-int(W), int(W)+1), (-1, int(H)+1), conditionRect)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact( (-(int(L)), int(L)+1), (-int(W), int(W)+1), (-1, 2*int(H)+1), 
                                    conditionRect, ORIENTATION)
    
    sp = np.array(sp, dtype=np.float).T
#    sp[0] -= L/2; sp[1] -= W/2
    sp *= step
    dipoles = sp.T
    
    
    
    return dipoles

def rect_dimer(step, L,H,W,G, mesh='cube', ORIENTATION=1):
    """
    Generate rectangular wire dimer with long axis along X on X/Y plane
    
    Using rectwire model for generation of the two dimer particles.
    
    Note : Using an even number gap will result in non-symmetric position with respect to (X=0,Y=0)
    
    
    Parameters
    ----------
    step : float
        stepsize in nw
    L, H, W : int
        Length, Height and Width as Nr. of steps
    G : int (zero or positive)
        Gap between dimers, in Nr. of steps
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    G = float(G)
    if G < 0:
        raise ValueError("Gap must be >= 0!")
    
    ## wire 1
    STRUCT0 = rect_wire(step, L=L,H=H,W=W, mesh=mesh, ORIENTATION=ORIENTATION).T
    ## wire 2
    STRUCT1 = rect_wire(step, L=L,H=H,W=W, mesh=mesh, ORIENTATION=ORIENTATION).T
    
    
    ## shift and concatenate wires
    STRUCT0[0] -= (L/2. + G/2.)*step
    STRUCT0 = STRUCT0.T
    
    STRUCT1[0] += (L/2. + G/2.)*step
    STRUCT1 = STRUCT1.T
    
    dipoles = np.concatenate( [STRUCT0, STRUCT1] )
    
    return dipoles


def prism(step, NSIDE, H, truncate_ratio=0.0, mesh='cube', ORIENTATION=1):
    """
    Generate prism on X/Y plane
    
    Parameters
    ----------
    step : float
        stepsize in nw
    NSIDE : int
        sidelength of regular prism (in nrs of dipoles)
    H : int
        height of prism (in nrs of dipoles)
    truncate_ratio : float
        truncate edges. give the ratio of total sidelength to remove.
        value between 0 [no truncation] and 1 [truncate at sidelength center]
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    
    NL = NSIDE/2.
    Hs = truncate_ratio*NL*np.sqrt(3)/2.
    Habs = (np.sqrt(3)/2.)*NL
    
    def conditionPrism(xi,yi,zi):
        inPrism = ((abs(xi) <= (Habs-(yi+0.5))/np.sqrt(3))
                    and (abs(xi) <= (((3*Habs - np.sqrt(3)*NL*truncate_ratio) + (yi+0.5))/np.sqrt(3)))
                    and (yi <= Habs-Hs) 
                    and (yi >= -Habs) 
                    and (0 <= zi < H - 1/np.sqrt(2)))
        return inPrism
    
    if mesh == 'cube':
        sp = _meshCubic( (-1*int(NL)-10, int(NL)+10), (-1*int(np.ceil(NL))-10, int(np.ceil(NL))+10), 
                         (-1*int(np.ceil(H))-10, int(np.ceil(H))+10),  conditionPrism)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact( (-1*int(NL)-10, int(NL)+10), (-1*int(np.ceil(NL))-10, int(np.ceil(NL))+10), 
                                    (-1*int(np.ceil(H))-10, int(np.ceil(H))+10), 
                                    conditionPrism, ORIENTATION)
    
    sp = np.array(sp, dtype=np.float).T*step
    sp[2] -= np.min(sp[2])  # shift Z so that bottom at z=0
    dipoles = sp.T
    
    return dipoles



def rhombus(step, L, H, alpha=90.0, mesh='cube', ORIENTATION=1):
    """
    Generate planar rhombus structure on X/Y plane
    
    Parameters
    ----------
    step : float
        stepsize in nw
        
    L : int
        sidelength of regular prism (in nrs of dipoles)
        
    H : int
        height of prism (in nrs of dipoles)
        
    alpha : float, default: 90
        top/bottom angle of rhombus (in degrees)
        
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
        
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    
    NL = L/2.
    alpha = np.pi * alpha / 180.
    alpha05 = alpha / 2.
    
    def conditionRhombus(xi,yi,zi):
        inRhombus = ((abs(xi) <= ((L*np.cos(alpha05)-abs(yi-0.5))*np.tan(alpha05)) )
                    and (abs(yi) <= (L*np.cos(alpha05)+0.5) )
                    and (0 <= zi < H - 1/np.sqrt(2)))
        return inRhombus
    
    if mesh == 'cube':
        sp = _meshCubic( (-1*int(L)-5, int(L)+5), (-1*int(L)-5, int(L)+5), 
                         (-1, int(H)+1), conditionRhombus)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact( (-1*int(L)-1, int(L)+1), (-1*int(L)-5, int(L)+5), 
                                    (-1, int(H)+1), conditionRhombus, ORIENTATION)
    
    sp = np.array(sp, dtype=np.float).T*step
    sp[2] -= np.min(sp[2])  # shift Z so that bottom at z=0
    dipoles = sp.T
    
    return dipoles


def hexagon(step, NSIDE, H, rotate=0.0, mesh='cube', ORIENTATION=1):
    """
    Generate regular hexagon on X/Y plane
    
    Parameters
    ----------
    step : float
        stepsize in nw
    NSIDE : int
        sidelength of regular hexagon (in nrs of dipoles)
    H : int
        height of hexagon (in nrs of dipoles)
    rotate : float, default: 0
        rotation angle (in degrees)
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    NL = NSIDE
    A = NSIDE
    if mesh =='hex': A -= 0.3
    
    def conditionHexagon(xi,yi,zi):
        Habs = (np.sqrt(3)/2.)*A
        
        inHex = ((abs(yi)-A/2 < (Habs-(abs(xi)+0.5))/np.sqrt(3))
                    and (abs(xi) < Habs)
                    and (0 < zi <= H))
                    
        return inHex
    
    if mesh == 'cube':
        sp = _meshCubic( (-1*int(NL)-10, int(NL)+10), (-1*int(np.ceil(NL))-10, int(np.ceil(NL))+10), 
                         (-1*int(np.ceil(H))-10, int(np.ceil(H))+10),  conditionHexagon)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact( (-1*int(NL)-10, int(NL)+10), (-1*int(np.ceil(NL))-10, int(np.ceil(NL))+10), 
                                    (-1*int(np.ceil(H))-10, int(np.ceil(H))+10), 
                                    conditionHexagon, ORIENTATION)
    
    sp = np.array(sp, dtype=np.float).T*step
    sp[2] -= np.min(sp[2])  # shift Z so that bottom at z=0
    
    rotate *= np.pi/180.
    rotX = sp[0]*np.cos(rotate) - sp[1]*np.sin(rotate)
    rotY = sp[0]*np.sin(rotate) + sp[1]*np.cos(rotate)
    sp[0]=rotX
    sp[1]=rotY
    dipoles = sp.T
    
    return dipoles


def double_hexagon(step, NSIDE1, NSIDE2, DX, DY, H, rotate=0.0, mesh='cube', ORIENTATION=1):
    """
    Generate regular hexagon on X/Y plane
    
    Parameters
    ----------
    step : float
        stepsize in nw
    NSIDE1 : int
        sidelength of hexagon1 (in nrs of dipoles)
    NSIDE2 : int
        sidelength of hexagon2 (in nrs of dipoles)
    DX : int
         X-offset of hexagon2 (in nrs of dipoles)
    DY : int
         Y-offset of hexagon2 (in nrs of dipoles)
    H : int
        height of hexagon (in nrs of dipoles)
    rotate : float, default: 0
        rotation angle (in degrees)
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    NL1 = NSIDE1
    A1 = NSIDE1
    Habs1 = (np.sqrt(3)/2.)*A1
    NL2 = NSIDE2
    A2 = NSIDE2
    Habs2 = (np.sqrt(3)/2.)*A2
    if mesh =='hex': 
        A1 -= 0.3
        A2 -= 0.3
    
    def conditionHexagon(xi,yi,zi):
        
        inHex1 = ( ( (abs(yi)-A1/2. < (Habs1-(abs(xi)+1.0))/np.sqrt(3) and yi>0)
                      or (abs(yi)-A1/2. < (Habs1-(abs(xi)-1.0))/np.sqrt(3) and yi<=0))
                   and (abs(xi) <= Habs1)
                   and (0 <= zi < H))
        xi2 = xi - DX
        inHex2 = ( ( (abs(yi)-A2/2. < (Habs2-(abs(xi2)+1.0))/np.sqrt(3) and yi>0)
                     or (abs(yi)-A2/2. < (Habs2-(abs(xi2)-1.0))/np.sqrt(3) and yi<=0))
                    and (Habs2 >= xi2 > -1*(Habs2))
                    and (0 <= zi < H))
                    
        return (inHex1 or inHex2)
    
    NL = max([NL1, NL2])
    if mesh == 'cube':
        sp = _meshCubic( (-1*int(NL) + min([-2,DX-5]), int(NL) + max([2,DX+5])), 
                         (-1*int(np.ceil(NL)) + min([-2,DY-5]), int(np.ceil(NL)) + max([2,DY+5])), 
                         (-1*int(np.ceil(H)) - 2, int(np.ceil(H)) + 2),  conditionHexagon)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact( (-1*int(NL) + min([-2,DX-5]), int(NL) + max([2,DX+5])), 
                                    (-1*int(np.ceil(NL)) + min([-2,DY-5]), int(np.ceil(NL)) + max([2,DY+5])), 
                                    (-1*int(np.ceil(H)) - 2, int(np.ceil(H)) + 2), 
                                    conditionHexagon, ORIENTATION)
    
    sp = np.array(sp, dtype=np.float).T*step
    sp[2] -= np.min(sp[2])  # shift Z so that bottom at z=0
    
    rotate *= np.pi/180.
    rotX = sp[0]*np.cos(rotate) - sp[1]*np.sin(rotate)
    rotY = sp[0]*np.sin(rotate) + sp[1]*np.cos(rotate)
    sp[0]=rotX
    sp[1]=rotY
    dipoles = sp.T
    
    return dipoles


def diabolo(step, nside_upper, nside_lower, length, width, H, dir_prisms='in', 
            mesh='cube', ORIENTATION=1):
    """generate diabolo structure on X/Y plane
    
    Parameters
    ----------
    step : float
        stepsize in nw
        
    nside_upper : int
        sidelength of upper regular prism (in nrs of dipoles)
        
    nside_lower : int
        sidelength of upper regular prism (in nrs of dipoles)
    
    length : int
        length of bridge between prisms (in nrs of dipoles)
    
    width : int
        width of bridge between prisms (in nrs of dipoles)
    
    H : int
        height of prism (in nrs of dipoles)
    
    dir_prisms : str, default="in"
        "in" or "out". Direction of prism tips, inwards or outwards.
    
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    
    nu = nside_upper/2.
    nl = nside_lower/2.
    L = length
    W = width
    
    def conditionPrism(xi,yi,zi):
        Habs1 = (np.sqrt(3)/2.)*nu
        Habs2 = (np.sqrt(3)/2.)*nl
        
        if dir_prisms.lower() == 'in':
            in_up_Prism = ((abs(xi) <= (Habs1+(yi - L/2 + 0.5))/np.sqrt(3)) 
                           and ( abs(yi - L/2) <= Habs1) 
                           and (0 <= zi < H - 1/np.sqrt(2)))
            
            in_low_Prism = ((abs(-xi) <= (Habs2+(-yi - L/2 + 0.5))/np.sqrt(3)) 
                           and ( abs(-yi - L/2) <= Habs2) 
                           and (0 <= zi < H - 1/np.sqrt(2)))
        else:
            in_up_Prism = ((abs(xi) <= (Habs1-(yi - L/2 + 0.5))/np.sqrt(3)) 
                           and ( abs(yi - L/2) <= Habs1) 
                           and (0 <= zi < H - 1/np.sqrt(2)))
            
            in_low_Prism = ((abs(xi) <= (Habs2-(-yi - L/2 + 0.5))/np.sqrt(3)) 
                           and ( abs(-yi - L/2) <= Habs2) 
                           and (0 <= zi < H - 1/np.sqrt(2)))
        
        in_bridge = ((abs(xi) <= W/2) 
                       and ( abs(yi) <= L/2) 
                       and (0 <= zi < H - 1/np.sqrt(2)))
        
        inDiabolo = in_up_Prism or in_low_Prism or in_bridge
        
        return inDiabolo
    
    NL = nu+nl+L
    if mesh == 'cube':
        sp = _meshCubic( (-1*int(NL)-1, int(NL)+1), (-1*int(np.ceil(NL))-1, int(np.ceil(NL))+1), 
                         (-1*int(np.ceil(H))-1, int(np.ceil(H))+1),  conditionPrism)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact( (-1*int(NL)-1, int(NL)+1), (-1*int(np.ceil(NL))-1, int(np.ceil(NL))+1), 
                                    (-1*int(np.ceil(H))-1, int(np.ceil(H))+1), 
                                    conditionPrism, ORIENTATION)
    
    sp = np.array(sp, dtype=np.float).T*step
    sp[2] -= np.min(sp[2])  # shift Z so that bottom at z=0
    dipoles = sp.T
    
    return dipoles


def polygon(step, config_dict, H, mesh='cube', ORIENTATION=1, plot_testing=False, verbose=False):
    """Generate polygonal, planar structure on X/Y plane
    
    Can take a list of polygons, they can be either added or removed. 
    The order of adding / removing polygons is important!
    
    Requires module `shapely`.
    
    Parameters
    ----------
    step : float
        stepsize in nm
        
    config_dict : dict or list of dict
        every dictionary describes one polygon and contains the keywords:
            - `N` (mandatory): number of eges
            - `S` (mandatory): side length in units of `step` 
            - `offset`: optional offset. tuple (X, Y). default: (0,0)
            - `aspect`: optional aspect ratio (X/Y). default: 1
            - `alpha`: optional rotation angle (in degrees). default: 0
            - `mode`: optional. either 'add' or 'remove'. Will either add the 
                    polygon to the collection or remove it. default: 'add'
    
    H : int
        height of planar structure in units of steps
    
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
        
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
        
    plot_testing : bool, default: False
        plot polygon outlines using `matplotlib`'s pyplot
    
    verbose : bool, default: False
        print timing info
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    import time
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    
    if type(config_dict) == dict:
        config_dict = [config_dict]
    
    ## --- generate polygons
    polygon_list = []
    allX = []
    allY = []
    for di in config_dict:
        if 'N' not in di.keys() or 'S' not in di.keys():
            raise ValueError("polygon dictionary must contain at least keys 'N' and 'S'.")
        if 'aspect' not in di.keys():
            di['aspect'] = 1
        if 'alpha' not in di.keys():
            di['alpha'] = 0
        if 'offset' not in di.keys():
            di['offset'] = [0,0]
        if 'mode' not in di.keys():
            di['mode'] = "add"
        else:
            if di['mode'].lower() not in ["add", "remove"]:
                raise ValueError("polygon dictionary entry 'mode' must be one of ['add', 'remove']")
        
        k_pts = np.arange(di['N'])
        
        X, Y = 1.01 * di['S'] * (np.array([np.cos(2*np.pi* k_pts/di['N']), np.sin(2 *np.pi * k_pts/di['N'])]))
        X = np.concatenate([X, [X[0]]]) * di['aspect']
        Y = np.concatenate([Y, [Y[0]]])
        
        alpha = -1*di['alpha']*np.pi / 180.0
        Xr = X*np.cos(alpha) - Y*np.sin(alpha) + di['offset'][0]
        Yr = X*np.sin(alpha) + Y*np.cos(alpha) + di['offset'][1]
        allX.append(Xr)
        allY.append(Yr)
        
        if plot_testing:
            import matplotlib.pyplot as plt
            plt.plot(Xr*step, Yr*step)
    
        poly_pts = np.transpose([Xr[:-1], Yr[:-1]])
        polygon_list.append(Polygon(poly_pts))
    
    
    def cond_poly(xi, yi, zi):
        if zi >= H: 
            return False
        else:    
            testpt = [xi, yi]
            testpoint = Point(testpt[0], testpt[1])
            poly_condition = None
            for i_poly, polygon in enumerate(polygon_list):
                mode = config_dict[i_poly]['mode'].lower()
                if mode == 'remove':
                    if poly_condition is None:
                        poly_condition = not polygon.contains(testpoint)
                    else:
                        poly_condition = not polygon.contains(testpoint) and poly_condition
                elif mode == 'add':
                    if poly_condition is None:
                        poly_condition = polygon.contains(testpoint)
                    else:
                        poly_condition = polygon.contains(testpoint) or poly_condition
            return poly_condition
        
    t0 = time.time()
    limits_X = [int(np.concatenate(allX).min()-4), int(np.concatenate(allX).max()+4)]
    limits_Y = [int(np.concatenate(allY).min()-4), int(np.concatenate(allY).max()+4)]
    limits_Z = [0, H]
    if mesh == 'cube':
        sp = _meshCubic(limits_X, limits_Y, limits_Z, cond_poly)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact(limits_X, limits_Y, limits_Z, 
                                   cond_poly, ORIENTATION)
    
    if verbose:
        print("polygon structure - elapsed time: {:.4f}s".format(time.time()-t0))
    
    sp = np.array(sp, dtype=np.float).T
    sp *= step
    dipoles = sp.T
    
    return dipoles


def sphere(step, R, mesh='cube', ORIENTATION=1):
    """
    Generate sphere with radius R on X/Y plane
    
    Parameters
    ----------
    step : float
        stepsize in nw
    R : float
        radius of sphere (in nrs of dipoles). Not limited to integer numbers.
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    
    def conditionSphere(xi,yi,zi):
        return (xi**2+yi**2+zi**2 <= (R+0.2)**2)
    
    if mesh == 'cube':
        sp = _meshCubic( (-2*int(R), 2*int(R)+1), (-2*int(R), 2*int(R)+1), 
                         (-2*int(R), 2*int(R)+1),  conditionSphere)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact( (-2*int(R), 2*int(R)+1), (-2*int(R), 2*int(R)+1), 
                                    (-2*int(R), 2*int(R)+1), 
                                    conditionSphere, ORIENTATION)
    
    sp = np.array(sp, dtype=np.float).T*step
    sp[2] -= np.min(sp[2])  # shift so that bottom at z=0
    dipoles = sp.T
    
    return dipoles


def spheroid(step, R1, R2, R3, mesh='cube', ORIENTATION=1):
    """Generate spheroid with radii R1, R2, R3 along X, Y, Z axis
    
      - if R3>R2=R1 --> prolate spheroid
      - if R3<R2=R1 --> oblate spheroid   
      - if R3=R2=R1 --> sphere
     
    contributed by C. Majorel.
    
    Parameters
    ----------
    step : float
        stepsize in nm
        
    R1 : float 
        radius along X axis (in nrs of steps). Not limited to integer numbers.
        
    R2 : float 
        radius along Y axis (in nrs of steps). Not limited to integer numbers. 
        
    R3 : float 
        radiu along Z axis  (in nrs of steps). Not limited to integer numbers.
    
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    
    def conditionSpheroid(xi,yi,zi):
        return ((xi/(R1+0.2))**2+(yi/(R2+0.2))**2+(zi/(R3+0.2))**2 <= 1)
    
    if mesh == 'cube':
        sp = _meshCubic( (-2*int(R1), 2*int(R1)+1), (-2*int(R2), 2*int(R2)+1), 
                         (-2*int(R3), 2*int(R3)+1),  conditionSpheroid)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact( (-2*int(R1), 2*int(R1)+1), (-2*int(R2), 2*int(R2)+1), 
                                    (-2*int(R3), 2*int(R3)+1), 
                                    conditionSpheroid, ORIENTATION)
    
    sp = np.array(sp, dtype=np.float).T*step
    sp[2] -= np.min(sp[2])  # shift so that bottom at z=0
    dipoles = sp.T
    
    return dipoles





def nanodisc(step, R, H, ELONGATED=0, mesh='cube', ORIENTATION=1):
    """
    Generate round nanodisc in X/Y plane. Height H, Radius R.
    
    Parameters
    ----------
    step : float
        stepsize in nw
    R : float
        radius of circular crosssection (in nrs of dipoles). Not limited to integer numbers.
    H : int
       Height of Structure (in nrs of dipoles).
    ELONGATED : int, default: 0
       add optional elongation "bridge" between half-circles (in nrs of steps)
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    L = ELONGATED
    
    def conditionRod(xi,yi,zi):
#        InRod = ((abs(zi) < H) and (np.sqrt(xi**2+yi**2)<(R+0.2)))
        InRod = ((abs(zi) < H) 
                    and 
                     ((np.sqrt((xi - int(L/2))**2 + yi**2) < (R+0.2)) or
                      (np.sqrt((xi + np.ceil(L/2.))**2 + yi**2) < (R+0.2)))
                    or
                     ((np.sqrt(yi**2) < (R+0.2)) and (abs(xi) < L/2))
                )
        return InRod
    
    if mesh == 'cube':
        sp = _meshCubic( (-1*int(R)-L/2-2, 1*int(R)+L/2+2), (-1*int(R)-2, 1*int(R)+2), 
                         (0, int(np.ceil(H))),  conditionRod)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact(  (-1*int(R)-5-L/2, 1*int(R)+5+L/2), 
                                     (-1*int(R)-5, 1*int(R)+5), (0, int(np.ceil(H))),
                                    conditionRod, ORIENTATION)
    
    sp = np.array(sp, dtype=np.float).T*step
    sp[2] -= np.min(sp[2])  # shift so that bottom at z=0
    dipoles = sp.T
    
    return dipoles


def nanorod(step, L,R, caps='flat', mesh='cube', ORIENTATION=1):
    """
    Generate round nanorod with axis along X on X/Y plane
    
    Parameters
    ----------
    step : float
        stepsize in nw
    L : int
       length of rod (in nrs of dipoles).
    R : float
        radius of circular crosssection (in nrs of dipoles). Not limited to integer numbers.
    caps : str, default: 'flat'
        'flat': flat caps, 'round': semispherical caps
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    
    def conditionRod(xi,yi,zi):
        InRod = ((abs(xi) <= L/2.) and (np.sqrt(yi**2+zi**2) <= (R+0.2)))
        
        if caps in ['round', 'true', True]:
            InCaps = (np.sqrt( (abs(xi)-L/2.)**2 + yi**2 + zi**2) <= (R+0.2))
        else:
            InCaps = False
        return (InRod or InCaps)
    
    if mesh == 'cube':
        sp = _meshCubic( (-1*int(L), 1*int(L)+1), (-2*int(R), 2*int(R)+1), 
                         (-2*int(R), 2*int(R)+1),  conditionRod)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact( (-1*int(L), 1*int(L)+1), (-2*int(R), 2*int(R)+1), 
                                    (-2*int(R), 2*int(R)+1), 
                                    conditionRod, ORIENTATION)
    
    sp = np.array(sp, dtype=np.float).T*step
    sp[2] -= np.min(sp[2])  # shift so that bottom at z=0
    dipoles = sp.T
    
    return dipoles



def nanocone(step, R, H, cutoff=1.0, mesh='cube', ORIENTATION=1):
    """
    Generate a right circular cone on X/Y plane. Height H, Radius R.
    
    Parameters
    ----------
    step : float
        stepsize in nw
    
    R : float
        radius of circular base (in nrs of dipoles). Not limited to integer numbers.
    
    H : int
        Height of cone (in nrs of dipoles).
    
    cutoff : float, deafult: 1.0
        percentage of cone to construct (1.0=100%, 0.5=stop at 50% of height)
    
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    H = int(H)
    
    def conditionCone(xi,yi,zi):
        InRod = (
                 (abs(zi) < H*cutoff) and 
                 (np.sqrt(xi**2+yi**2) < (H-zi)/(float(H))*(R+0.2))
                )
        return InRod
    
    if mesh == 'cube':
        sp = _meshCubic( (-1*int(R)-2, 1*int(R)+2), (-1*int(R)-2, 1*int(R)+2), 
                         (0, H+1),  conditionCone)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact(  (-1*int(R)-5, 1*int(R)+5), 
                                     (-1*int(R)-5, 1*int(R)+5), (0, H+3),
                                    conditionCone, ORIENTATION)
    
    sp = np.array(sp, dtype=np.float).T*step
    sp[2] -= np.min(sp[2])  # shift so that bottom at z=0
    dipoles = sp.T
    
    return dipoles


def lshape_rect(step, L,W,H, DELTA, mesh='cube', ORIENTATION=1):
    """
    Generate symmetric L-shaped nanoantenna in X/Y plane formed by
    two rectangular elements.
    
    Parameters
    ----------
    step : float
        stepsize in nw
    L,W,H : int
        length,width,height of each rod (in nrs of dipoles)
    DELTA : int (odd number)
        Gap width - only odd numbers work for symmetry reasons! (in nr of dipoles)
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    _DELTA = DELTA/2
    def conditionRod(xi,yi,zi):
        InRod1 = ( (_DELTA < xi <= _DELTA+L) and 
                   (-1*_DELTA > yi >= -1*_DELTA-W) and 
                   (0 <= zi < H - 1/np.sqrt(2)))
        InRod2 = ( (-1*_DELTA > xi >= -1*_DELTA-W) and 
                   (_DELTA < yi <= _DELTA+L) and 
                   (0 <= zi < H - 1/np.sqrt(2)))
        
        return (InRod1 or InRod2)
    
    if mesh == 'cube':
        sp = _meshCubic( (-1*abs(DELTA)-W-1, L+abs(DELTA)+2), (-1*abs(DELTA)-W-1, L+abs(DELTA)+2), (0, H+2), conditionRod)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact( (-1*DELTA-W-1, L+DELTA+2), (-1*DELTA-W-1, L+DELTA+2), (0, H+2), 
                                    conditionRod, ORIENTATION)
    
    Rect1 = np.array(sp, dtype=np.float).T*step
    dipoles = Rect1.T
    
    return dipoles


def lshape_rect_nonsym(step, L1,W1, L2,W2, H, DELTA, mesh='cube', ORIENTATION=1):
    """
    Generate symmetric L-shaped nanoantenna in X/Y plane formed by
    two rectangular elements.
    
    Parameters
    ----------
    step : float
        stepsize in nw
    L,W,H : int
        length,width,height of each rod (in nrs of dipoles)
    DELTA : int (odd number)
        Gap width - only odd numbers work for symmetry reasons! (in nr of dipoles)
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    _DELTA = int(DELTA/2)
    L1 = int(L1)
    L2 = int(L2)
    W1 = int(W1)
    W2 = int(W2)
    H = int(H)
    def conditionRod(xi,yi,zi):
        InRod1 = ( (_DELTA < xi <= _DELTA+L1) and 
                   (-1*_DELTA > yi >= -1*_DELTA-W1) and 
                   (0 <= zi < H - 1/np.sqrt(2)))
        InRod2 = ( (-1*_DELTA > xi >= -1*_DELTA-W2) and 
                   (_DELTA < yi <= _DELTA+L2) and 
                   (0 <= zi < H - 1/np.sqrt(2)))
        
        return (InRod1 or InRod2)
    W=W1+W2
    L=L1+L2
    if mesh == 'cube':
        sp = _meshCubic( (-1*abs(DELTA)-W-1, L+abs(DELTA)+2), (-1*abs(DELTA)-W-1, L+abs(DELTA)+2), (0, H+2), conditionRod)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact( (-1*DELTA-W-1, L+DELTA+2), (-1*DELTA-W-1, L+DELTA+2), (0, H+2), 
                                    conditionRod, ORIENTATION)
    
    Rect1 = np.array(sp, dtype=np.float).T*step
    dipoles = Rect1.T
    
    return dipoles



def lshape_round(step, L,W,H, DELTA, RAD, mesh='cube', ORIENTATION=1):
    """
    Generate symmetric L-shaped nanoantenna in X/Y plane formed by
    two rectangular elements.
    
    Parameters
    ----------
    step : float
        stepsize in nm
    L,W,H : int
        length,width,height of each rod (in nrs of dipoles)
    DELTA : int (odd number)
        Gap width - only odd numbers work for symmetry reasons! (in nr of dipoles)
    RAD : float
        radius of curvature in steps (nr of dipoles)
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    D = DELTA/2
    RAD += 0.2
    def conditionRod(xi,yi,zi):
        InRod1 = ( (D < xi <= D+L) and 
                   (-1*D > yi >= -1*D-W) and 
                   (0 <= zi < H - 1/np.sqrt(2)) )
        ## Roundings Rod 1
        R1R1 =   ( ( not ( (xi<int(D+RAD)) and (yi>-1.*int(D+RAD)) )) or 
                          ((xi-int(RAD+D+1))**2 + (yi+int(RAD+D+1))**2  <= RAD**2 ) )
        R1R2 =   ( ( not ( (xi<int(D+RAD)) and (yi<-1.*int(D+W-RAD)) )) or 
                          ((xi-int(RAD+D+1))**2 + (yi+int(D+W-RAD+1))**2  <= RAD**2 ) )
        R1R3 =   ( ( not ( (xi>int(D+L-RAD)) and (yi>-1.*int(D+RAD)) )) or 
                          ((xi-int(D+L-RAD+1))**2 + (yi+int(RAD+D+1))**2  <= RAD**2 ) )
        R1R4 =   ( ( not ( (xi>int(D+L-RAD)) and (yi<-1.*int(D+W-RAD)) )) or 
                          ((xi-int(D+L-RAD+1))**2 + (yi+int(D+W-RAD+1))**2  <= RAD**2 ) )
        Round1 = R1R1 and R1R2 and R1R3 and R1R4
        
        
        InRod2 = ( (-1*D > xi >= -1*D-W) and 
                   (D < yi <= D+L) and 
                   (0 <= zi < H - 1/np.sqrt(2)) )
        ## Roundings Rod 2
        R2R1 =   ( ( not ( (yi<int(D+RAD)) and (xi>-1.*int(D+RAD)) )) or 
                          ((yi-int(RAD+D+1))**2 + (xi+int(RAD+D+1))**2  <= RAD**2 ) )
        R2R2 =   ( ( not ( (yi<int(D+RAD)) and (xi<-1.*int(D+W-RAD)) )) or 
                          ((yi-int(RAD+D+1))**2 + (xi+int(D+W-RAD+1))**2  <= RAD**2 ) )
        R2R3 =   ( ( not ( (yi>int(D+L-RAD)) and (xi>-1.*int(D+RAD)) )) or 
                          ((yi-int(D+L-RAD+1))**2 + (xi+int(RAD+D+1))**2  <= RAD**2 ) )
        R2R4 =   ( ( not ( (yi>int(D+L-RAD)) and (xi<-1.*int(D+W-RAD)) )) or 
                          ((yi-int(D+L-RAD+1))**2 + (xi+int(D+W-RAD+1))**2  <= RAD**2 ) )
        Round2 = R2R1 and R2R2 and R2R3 and R2R4
        
        
        
        return (InRod1 and Round1) or (InRod2 and Round2)
    
    if mesh == 'cube':
        sp = _meshCubic( (-1*abs(DELTA)-W-1, L+abs(DELTA)+2), (-1*abs(DELTA)-W-1, L+abs(DELTA)+2), (0, H+2), conditionRod)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact( (-1*DELTA-W-10, L+DELTA+10), (-1*DELTA-W-10, L+DELTA+10), (0, H+2), 
                                    conditionRod, ORIENTATION)
    
    Rect1 = np.array(sp, dtype=np.float).T*step
    dipoles = Rect1.T
    
    return dipoles



def split_ring(step, R, W, H, G=-1, alphaG=0.0, mesh='cube', ORIENTATION=1):
    """
    Generate splitring structure on X/Y plane
    
    
    If G (gap) is set: close the resonator up to "gap" (G=)
    
    Parameters
    ----------
    step : float
        stepsize in nw
    R,W,H : int,int,int
        Dimensions: Radius, linewidth and height of structure (in nr. of dipoles)
    G : (optional) int, default: -1
        Gap width in numbers of steps. 
        If G == -1, alphaG is taken instead
    alphaG : (optional) float, default: 0
        Gap width as angle in degrees. 
        Allowed angles: Between 0 and 180.
        '0': entirely closed
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    if G != -1:
        alphaG = 2.*np.tan(G/(2.*R))*180. / np.pi

    def conditionSR(xi,yi,zi):
        A1 = R**2 >= (xi**2 + yi**2) >= (R-W)**2
        if yi > 0 and alphaG != 0:
            A2 = not ( np.arctan2(abs(xi),abs(yi))*180/np.pi <= alphaG/2. )
        else:
            A2 =True
        
        CZ = (0 <= zi < H - 1/np.sqrt(2))
        
        return ((A1 and A2) and CZ)
    
    if mesh == 'cube':
        sp = _meshCubic( (-1*int(R)-5, int(R)+5), (-1*int(R)-5, int(R)+5), (-1, int(H)+1), conditionSR)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact( (-1*int(R)-5, int(R)+5), (-1*int(R)-5, int(R)+5), (-1, int(H)+1), 
                                    conditionSR, ORIENTATION)
    
    sp = np.array(sp, dtype=np.float).T
#    sp[0] -= L1/2; sp[1] -= L2/2
    sp *= step
    dipoles = sp.T
    
    
    
    return dipoles



def rect_split_ring(step, L1,L2,H,W, G=False, mesh='cube', ORIENTATION=1):
    """
    Generate rectangular splitring structure on X/Y plane
    
    
    If G (gap) is set: close the resonator up to "gap" (G=)
    
    Parameters
    ----------
    step : float
        stepsize in nw
    L1,L2, H, W : int
        Length (L1:X,L2:Y), Height and Width in Nr. of steps
    G : (optional) int, default: False
        Gap of a closed splitring. 
            If False: totally open splitring, 
            If '0': entirely closed
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    def conditionSR(xi,yi,zi):
        A1 = (0<=xi<L1 and 0<=yi<W and (0 <= zi < H - 1/np.sqrt(2)))
        A2 = (0<=xi<W and 0<=yi<L2 and (0 <= zi < H - 1/np.sqrt(2)))
        A3 = (L1-W<=xi<L1 and 0<=yi<L2 and (0 <= zi < H - 1/np.sqrt(2)))
        
        if G != False:
            A4 = ((0<=xi<(L1-G)/2 or (L1+G)/2<=xi<L1) and L2-W<=yi<L2 and (0 <= zi < H - 1/np.sqrt(2))) 
        else:
            A4 = False
        
        return (A1 or A2 or A3 or A4)
    
    if mesh == 'cube':
        sp = _meshCubic( (-1, int(L1+W)+1), (-1, int(L2+W)+1), (-1, int(H)+1), conditionSR)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact( (-1, int(L1+W)+1), (-1, int(L2+W)+1), (-1, int(H)+1), 
                                    conditionSR, ORIENTATION)
    
    sp = np.array(sp, dtype=np.float).T
    sp[0] -= L1/2; sp[1] -= L2/2
    sp *= step
    dipoles = sp.T
    
    
    
    return dipoles

#==============================================================================
# MESHERS
#==============================================================================
def _meshCubic(XRANGE, YRANGE, ZRANGE, condition):
    """mesh on a cubic grid
    
    Mesh a 3d structure on a cubic grid within given spacial limits
    using a boolean selection function
    
    Parameters
    ----------
    XRANGE : tuple
        tuple of 2  ints (x0, x1): Indices of x-coordinates on mesh
    YRANGE : tuple
        tuple of 2  ints (y0, y1): Indices of y-coordinates on mesh
    ZRANGE : tuple
        tuple of 2  ints (z0, z1): Indices of z-coordinates on mesh
    condition : function
        function of mesh-coord.-indices, whether to place a meshpoint or not:
        'func(xi,yi,zi)', must returns a Boolean
        
    
    Returns
    -------
    list of tuples
         list of 3d coordinate-index tuples
    """
    sp=[]
    for xi in range(int(XRANGE[0]), int(XRANGE[1])):
        for yi in range(int(YRANGE[0]), int(YRANGE[1])):
            for zi in range(int(ZRANGE[0]), int(ZRANGE[1])):
                
                if condition(xi, yi, zi):
                    sp.append([xi,yi,zi])
    
    return sp


def _meshHexagonalCompact(XRANGE, YRANGE, ZRANGE, condition, ORIENTATION=1):
    """mesh on a hexagonal compact grid
    
    Parameters
    ----------
    XRANGE : tuple
        tuple of 2  ints (x0, x1): Indices of x-coordinates on mesh
    YRANGE : tuple
        tuple of 2  ints (y0, y1): Indices of y-coordinates on mesh
    ZRANGE : tuple
        tuple of 2  ints (z0, z1): Indices of z-coordinates on mesh
    condition : function
        function of mesh-coord.-indices, whether to place a meshpoint or not:
        'func(xi,yi,zi)', must returns a Boolean
    ORIENTATION : int, default: 1
        For hex. meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    list of tuples
         list of 3d coordinate-index tuples
    """
    sp = []
    if ORIENTATION==1:
        for zi in range(int(ZRANGE[0]), int(ZRANGE[1])):
            Z=zi*np.sqrt(2./3.)
            for yi in range(int(YRANGE[0]), int(YRANGE[1])):
                Y=yi*np.sqrt(3.)/2.+2.*abs(zi/2.-int(zi/2.))/np.sqrt(3.)
                for xi in range(int(XRANGE[0]), int(XRANGE[1])):
                    X=xi+abs(yi/2.-int(yi/2.))
                    
                    if condition(X,Y,Z):
                        sp.append([X,Y,Z])
    
    elif ORIENTATION==2:
        for xi in range(int(XRANGE[0]), int(XRANGE[1])):
            X=xi*np.sqrt(2./3.)
            for yi in range(int(YRANGE[0]), int(YRANGE[1])):
                Y=yi*np.sqrt(3.)/2.+2.*abs(xi/2.-int(xi/2.))/np.sqrt(3.)
                for zi in range(int(ZRANGE[0]), int(ZRANGE[1])):
                    Z=zi+abs(yi/2.-int(yi/2.))
                    
                    if condition(X,Y,Z):
                        sp.append([X,Y,Z])
    
    return sp





#==============================================================================
# Other Functions
#==============================================================================
def get_normalization(mesh):
    """Provide normalization factor for mesh type
    
    Parameters
    ----------
    mesh : string
        mesh used for structure generator
    
    Returns
    -------
    normalization : float
        normalization factor corresponding to `mesh`
    """
    if mesh == 'cube':
        normalization = 1.0
    elif mesh == 'hex':
        normalization = np.sqrt(2.0)
    else:
        raise ValueError('Meshing definition: "mesh" must be either "cube" or "hex"!')
    
    return normalization



def image_to_struct(img_name, nm_per_pixel, stepsize, H, threshold=100, 
                    useDarkPixel=True, returnImage=False, mesh='cube', ORIENTATION=1):
    """Convert an image to a planar structure
    
    Might be useful with SEM images
    
    Parameters
    ----------
    img_name : string
        path to image-file
    
    nm_per_pixel : float
        number of nanometers corresponding to one pixel in image
    
    stepsize : float
        target stepsize
    
    H : int
        Height of final structure in numbers of stepsize
    
    threshold : float, (default: 100)
        threshold value between [0, 255] to declare a pixel as *structure*
        all brighter pixels will be used
    
    useDarkPixel : bool, default: True
        if False, use bright pixels as structure (below threshold), instead 
        of the dark ones
    
    returnImage : bool, default: False
        if True, returns a 2D numpy array corresponding to the image
        AND the structure
    
    mesh : string, default: 'cube'
        meshing type. 'cube' or 'hex'
    
    ORIENTATION : int, default: 1
        For hexagonal meshing only. '1' or '2'. Orientation of Hexagons.
    
    Returns
    -------
    struct : list of tuples
         list of 3d coordinates in tuples (x,y,z)
    
    if returnImage==True:
        returns tuple: (img_array, struct)
    
    """
    from PIL import Image
    
    ## --- load image
    img = Image.open(img_name)
    img.load()   # required for png.split()
    
    ## --- remove alpha channel, if exists
    if len(img.split())==4:
        ## RGBA data --> remove alpha channel
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
    else:
        ## RGB data --> no alpha channel
        background = img
    
    ## original image to return if required
    img0 = background.convert('L')
    data0 = np.asarray(img0.getdata()).reshape(img0.size[::-1])
    
    
    ## --- rescale image to 1 stepsize per pixel
    X0 = background.size[0]
    basewidth = int(round(X0 * nm_per_pixel / float(stepsize)))
    
    hsize = int((float(background.size[1]) * float((basewidth / float(background.size[0])))))
    img = background.resize((basewidth, hsize), Image.ANTIALIAS)
    img = img.convert('L')
    
    data = np.asarray(img.getdata()).reshape(img.size[::-1])
    data = np.rot90(data, 3)  # rotate by 270 degrees
    
    ## --- generate structure using all pixel above threshold 
    if useDarkPixel:
        data[ data <= threshold ] = threshold
        data[ data > threshold ] = 0
    else:
        data[ data >= threshold ] = threshold
        data[ data < threshold ] = 0
    
    
    ## --- mesh the structure
    def condition(xi,yi,zi):
        return data[int(xi)][int(yi)] != 0
    
    if mesh == 'cube':
        sp = _meshCubic([0, data.shape[0]], [0, data.shape[1]], [0, H], condition)
    elif mesh == 'hex':
        sp = _meshHexagonalCompact([0, data.shape[0]], [0, data.shape[1]], [0, H], condition, ORIENTATION)
    sp = np.array(sp, dtype=np.float).T*stepsize
    sp[2] -= np.min(sp[2])  # shift Z so that bottom at z=0
    struct = sp.T
    
    struct = center_struct(struct)
    

    if returnImage:
        return data0, struct
    else:
        return struct



def struct_to_image(STRUCT, RGB=True, minimum_size=-1, projection='XY'):
    """generate image-compatible array from structure geometry
    
    Parameters
    ----------
    struct : list of tuple *or* :class:`.core.simulation`
        list of coordinate tuples or instance of :class:`.core.simulation`
    
    RGB : bool, default: True
        If True, generates an RGB array. If False, generate a 2D array with 
        simply 0 and 1 as entries (0: no material, 1: material)
    
    minimum_size : int, default: -1
        minimum canvas size in pixels. -1: use structure extensions. If >0:
        enlarge canvas and center structure if structure extensions smaller
        than image.
    
    projection : str, default: "XY"
        2D plane for projection. One of ['XY', 'XZ', 'YZ']
    
    Returns
    -------
    np.array for image conisting of rgb tuples for each pixel.
    Plot e.g. using `matplotlib`'s `imshow`. 
    Save to bitmap or png file e.g. using `scipy.misc.imsave`.
    
    """
    step = tools.get_step_from_geometry(STRUCT)
    _X,_Y,_Z = tools.get_geometry_2d_projection(STRUCT, projection).T
    if projection.lower() == 'xy':
        X,Y = _X,_Y
    elif projection.lower() == 'yz':
        X,Y = _Y,_Z
    elif projection.lower() == 'xz':
        X,Y = _X,_Z
        
    X /= float(step)
    Y /= float(step)
    X -= X.min()
    Y -= Y.min()
    
    ## -- optionally enlarge canvas and center struct
    if minimum_size <= X.max():
        DX = int(X.max()+1)
    else:
        X += int((minimum_size - X.max())/2.)
        DX = minimum_size
        
    if minimum_size <= Y.max():
        DY = int(Y.max()+1)
    else:
        Y += int((minimum_size - Y.max())/2.)
        DY = minimum_size
    
    struct=np.transpose([X,Y])
    
    ## -- generate 2D image array
    if RGB:
        arr = np.full( (DX,DY,3), 255 , dtype=np.uint8 )
        for i in struct:
            arr[int(i[0])][int(i[1])] = np.array([0,0,0])
    else:
        arr = np.full( (DX,DY), 1 , dtype=np.uint8 )
        for i in struct:
            arr[int(i[0])][int(i[1])] = 0
    
    return np.rot90(arr)


def rotate_XY(STRUCT, ALPHA):
    """Rotate a structure around the z axis
    
    see :func:`.rotate` for full documentation
    """
    return rotate(STRUCT, ALPHA, axis='z')

    
def rotate(geo, alpha, axis='z'):
    """Rotate a structure around the specified axis
    
    Parameters
    ----------
    geo : list of tuples
        geometry as list of (x,y,z) coordinate tuples.
        As generated by the functions of the structure module
    
    alpha : float
        rotation angle in degrees
        
    axis : str, default: 'z'
        rotation axis. one of ["x", "y", "z"]
    
    Returns
    -------
    list of tuples
         list of 3d coordinates in tuples (x,y,z)
    """
    A = alpha * np.pi / 180.
    
    if axis.lower() == 'x':
        yNew = np.cos(A)*geo.T[1] - np.sin(A)*geo.T[2]
        zNew = np.sin(A)*geo.T[1] + np.cos(A)*geo.T[2]
        geo.T[1:3] = yNew, zNew
    elif axis.lower() == 'y':
        xNew = np.cos(A)*geo.T[0] + np.sin(A)*geo.T[2]
        zNew = -np.sin(A)*geo.T[0] + np.cos(A)*geo.T[2]
        geo.T[0] = xNew
        geo.T[2] = zNew
    elif axis.lower() == 'z':
        xNew = np.cos(A)*geo.T[0] - np.sin(A)*geo.T[1]
        yNew = np.sin(A)*geo.T[0] + np.cos(A)*geo.T[1]
        geo.T[0:2] = xNew, yNew
    else:
        raise Exception("rotation axis '{}': Unknown axis description.".format(axis))
    
    return geo


def center_struct(STRUCT, returnOffsets=False):
    """
    Center a structure on the x/y plane around (0,0)
    
    Parameters
    ----------
    STRUCT : list of tuples, or `struct`-object
        list of (x,y,z) coordinate tuples, as generated by the functions 
        of the structure module; or `struct` or `simulation` object (which 
        includes the geometry information)
    
    returnOffsets : bool, default: False
        if True, return tuple of applied (X,Y)-offset together with structure
    
    Returns
    -------
         - list of 3d coordinates as tuples (x,y,z); or `struct` or 
           `simulation` object with adapted geometry.
    
         - optionally: Offset-tuple as second return value
         
    """
    if type(STRUCT) in [list, np.ndarray]:
        returnType = "list"
        X,Y,Z = np.transpose(STRUCT)
    elif type(STRUCT) == struct:
        returnType = "struct"
        try:
            X,Y,Z = STRUCT.geometry.T
        except:
            raise Exception("Not a valid structure object")
    elif type(STRUCT) == simulation:
        returnType = "simulation"
        try:
            X,Y,Z = STRUCT.struct.geometry.T
        except:
            raise Exception("Not a valid structure object")
    else:
        raise Exception("Got no valid structure data.")
    
    DX = (np.max(X) + np.min(X)) / 2.
    DY = (np.max(Y) + np.min(Y)) / 2.
    X -= DX
    Y -= DY
    
    
    if returnType == 'list':
        STRUCT = np.transpose([X,Y,Z])
        if returnOffsets:
            return STRUCT, (DX,DY)
        else:
            return STRUCT
    elif returnType == 'struct':
        STRUCT.geometry.T[0] = X
        STRUCT.geometry.T[1] = Y
        if returnOffsets:
            return STRUCT, (DX,DY)
        else:
            return STRUCT
    elif returnType == 'simulation':
        STRUCT.struct.geometry.T[0] = X
        STRUCT.struct.geometry.T[1] = Y
        if returnOffsets:
            return STRUCT, (DX,DY)
        else:
            return STRUCT











## -- list of all available structure generators
STRUCT_LIST = [sphere, spheroid, rect_wire, nanorod, nanocone, 
               nanodisc, prism, rhombus, hexagon, double_hexagon, diabolo, 
               lshape_rect, lshape_round, lshape_rect_nonsym,
               split_ring, rect_split_ring]