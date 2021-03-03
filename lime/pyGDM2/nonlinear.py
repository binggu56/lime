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
nonlinear optical effects

"""
from __future__ import print_function
from __future__ import absolute_import

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
# incoherent nonlinear effects
#==============================================================================
def tpl_ldos(sim, field_index, nonlin_order=2.0, beta=1.0E5, verbose=False):
    """calculate the TPL for nano-object under given illumination condition
    
    Calculate the two-photon photolouminescence (TPL). Higher order nonlinear 
    photoluminescence can also be calculated by changing the parameter 
    `nonlin_order`.
    
    - Can be used to simulate TPL signals using `nonlin_order`=2 (default)
      
    - Using `nonlin_order`=1 together with an unphysically tight 
      focused beam --> calculate the LDOS. 
      WARNING: The focus waist must not be smaller than some times the stepsize!
    
    - Might be used for Raman intensities using `nonlin_order`=1.
    
    
    Parameters
    ----------
    sim : :class:`.core.simulation`
        simulation description
    
    field_index : int
        index of evaluated self-consistent field to use for calculation. Can be
        obtained for specific parameter-set using :func:`.tools.get_closest_field_index`
    
    nonlin_order : float, default: 2.0
        order of nonlinear response. (I_npl ~ | |E|^2 | ^ nonlin_order).
    
    beta : float, default: 1E5
        incident laser power scaling. Use 1E5 for correct LDOS values.
    
    verbose : bool, default: False
        Enable some info printing
    
    
    Returns
    -------
    I_tpl : float
        TPL intensity in farfield from nano-object under given illumination
    
    
    Notes
    -----
    For details on TPL/LDOS calculation via focused beam rasterscan 
    simulations, see:
    Viarbitskaya, S. et al. **Tailoring and imaging the plasmonic local 
    density of states in crystalline nanoprisms**, Nat. Mater. 12, 426–432 (2013)

    """
    if verbose:
        if nonlin_order == 1:
            print("nonlinear order = 1 --> Using this is adapted for LDOS.")
        if nonlin_order == 2:
            print("nonlinear order = 2 --> Using this is adapted for TPL signal.")
        if nonlin_order > 2:
            print("nonlinear order > 2 --> Higher order incoherent nonlinear luminescence?")
        
    if sim.E is None: 
        raise ValueError("Error: Scattering field inside the structure not yet evaluated. Run `core.scatter` simulation first.")
    
    field_params = sim.E[field_index][0]
    wavelength   = field_params['wavelength']
    k0 = 2*np.pi / wavelength
    
    ## --- environment
    k1 = k0 * np.real(sim.struct.n2)
    
    ## --- constant factors
    pre_factor = (16.*(k1**4))/3.
    
    ## --- meshpoint electric fields
    E = sim.E[field_index][1]
    ex, ey, ez = E.T
    
    ## --- total TPL intensity
    I_e = np.power((np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2), nonlin_order)
    I_tpl = beta * pre_factor * np.sum(I_e)
    
    return I_tpl




#==============================================================================
# coherent nonlinear effects
#==============================================================================


## --- NONE SO FAR.




