# encoding: utf-8
"""
Collection of problems for the EO submodule of pyGDM2

    Copyright (C) 2017, P. R. Wiecha

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    

"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

import numpy as np
from PyGMO.problem import base

from .. import core
from .. import linear
from .. import tools



class BaseProblem(base):
    """Base pyGDM2 problem, inherits from PyGMO.problem.base
    
    Parameters:
    ---------------
    N_dim : int
        passing of the problem dimension as first argument is required by PyGMO
    
    model : instance inheriting from `pyGDM2.EO.models.BaseModel`
        Structure model for optimization
    
    field_index : int, default: 0
        field_index to use from pyGDM simulation.
    
    nthreads : int, default: 1
        number of threads for multithreading. Default=1. We recommend to use
        a single thread for the simulations and several 'islands' for multiple 
        parallel evolutions.
    
    """
    def __init__(self, N_dim=0, model=None, field_index=0, N_objectives=1, nthreads=1):
        ##
        ## `PyGMO.problem.base` constructor takes: (N_dim, N_int, N_obj)
        ##      N_dim: Dimensionality of problem
        ##      N_int: Number of integer parameter among `N_dim`
        ##      N_obj: Number of objective functions
        ##
        ## Pure float parameters:   base.__init__(N_dim, 0, 1) 
        ## Pure integer parameters: base.__init__(N_dim, N_dim, 1) 
        ##
        self.N_objectives = N_objectives
        
        if model is not None:
            ## --- main simulation definitions
            self.model = model
            self.sim = model.sim
            self.field_index = field_index
            self.nthreads = nthreads
            
            ## --- problem dimension, assuming `float` parameters:
            if N_dim != self.model.get_dim():
                raise ValueError("Dimension of problem doesn't match with structure model!")
            super(BaseProblem, self).__init__(N_dim, 0, N_objectives) 
            
            ## -- problem boundaries
            lbnds, ubnds = self.model.get_bounds()
            self.set_bounds(lbnds, ubnds)
        else:
            ## --- call base-constructore with prob.dimension, required by PyGMO
            super(BaseProblem, self).__init__(N_dim, 0, N_objectives) 
    
    
#==============================================================================
# Mandatory reinmplementations for specific problems:
#==============================================================================
    def objective_function(self, params):
        """Evaluates the objective function"""
        raise NotImplementedError("'problems.BaseProblem.objective_function' not re-implemented!")


#==============================================================================
# Optionally reinmplementations
#==============================================================================
    def _objfun_impl(self, x):
        """Virtual objective function, single objective (PyGMO requirement)"""
        ## --- implement like this:
        target_value = self.objective_function(x) # target function evaluation
        if hasattr(target_value, '__iter__'):
            ## --- multi objective
            return tuple([np.float64(i) for i in target_value])
        else:
            ## --- single objective must neverthless return tuple (with 1 element)
            return (np.float64(target_value), ) 
        

    def _compare_fitness_impl(self, f1, f2):
        """compare fitnesses: maximize a single objective (PyGMO requirement)"""
        ## f1 > f2 --> maximize
        pairwise_comparison = [f1[i] > f2[i] for i in range(self.N_objectives)]
        return all(map(lambda x: x==True, pairwise_comparison))==True


    def human_readable_extra(self):
        """Return problem description (PyGMO requirement)"""
        warnings.warn("'problems.BaseProblem.human_readable_extra' not re-implemented.")
        return "\n\t`BaseProblem` for maximization of a single target value"



    
    
    
        
        


        
class ProblemDirectivity(BaseProblem):
    """Problem to optimize directionality of scattering from nanostructure
    
    Use `EO.tools.calculate_solid_angle_by_dir_index` to define 'dir_index' 
    for the desired target solid angle
    
    """
    def __init__(self, N_dim=9999, model=None, field_index=0,
                       dir_index=[5], which_field='e_sc',
                       kwargs_farfield=dict(Nteta=3, Nphi=5, tetamin=0, tetamax=np.pi/2.),
                       consider_dS=False, averaging=True, absoluteI=False, 
                       kwargs_scatter={}, nthreads=-1):
        """constructor
        
        Parameters
        ----------
        
        N_dim : int
            number of free parameters (defined by used model class)
        
        model : instance of class implementing `pyGDM2.EO.models.BaseModel`
            Definition of structure model and pyGDM-simulation
        
        field_index : list of int, default: 0
            "field_index" in case of multiple pyGDM-simulation configurations
        
        dir_index : list of int; list of list of int, default: [5]
            which solid-angle elements to consider for optimization.
            If list of lists of int (e.g. [[4], [5], [1,2]]), will run 
            multi-objective optimization using each of the sub-lists as target
            solid angle
        
        which_field : str, default: 'e_tot'
            optimize using one of ['e0', 'e_sc', 'e_tot']. 
            'e0': fundamental field (rather for testing), 
            'e_sc' scattered field, 'e_tot' total field
        
        kwargs_farfield : dict, default: dict(Nteta=3, Nphi=5, tetamin=0, tetamax=np.pi/2.)
            kwargs, passed to `pyGDM2.linear.farfield`, defining the farfield 
            scattering calculation. Required arguments are: Nteta, Nphi, tetamin, tetamax
        
        consider_dS (False), averaging (True), absoluteI (False)
            Additional flags (defaults in paranthesis): 
                 - consider_dS: correct (True) solid angle integration or simple sum (False)
                 - averaging: avgerage (True) or integrate (False) solid angle intensity
                 - absoluteI: maximize total intensity through selected solid angle instead of ratio
        
        kwargs_scatter : dict, default: {}
            additional kwargs, passed to `pyGDM2.core.sactter`
        
        nthreads : int, default: -1
            number of parallel threads for pyGDM (default: All CPUs, if possible)
            
        """
        ## --- directionality problem setup
        if type(dir_index) not in (list, tuple, np.ndarray):
            self.dir_index = [dir_index]
        else:
            self.dir_index = dir_index
        
        if type(field_index) not in (list, tuple, np.ndarray):
            field_index = [field_index]
        else:
            field_index = field_index
        
        ## --- number of objectives
        if self.list_depth(self.dir_index) == 2:
            N_obj = len(self.dir_index)
            if len(field_index) == 1:
                field_index = [field_index[0]]*N_obj
            if len(field_index) != N_obj:
                raise ValueError("Number of 'field_index' entries must exactly match number of objectives!")
        elif self.list_depth(self.dir_index) == 1:
            N_obj = 1
            field_index = [field_index]
            self.dir_index = [self.dir_index]
        else:
            raise ValueError("Wrong shape of `dirindex` input parameter. Must be of 'depth' 1 or 2.")
        
        ## --- init base class
        super(self.__class__, self).__init__(N_dim, model, field_index, 
                                          N_objectives=N_obj, nthreads=nthreads)
                
        
        self.which_field = which_field 
        
        self.consider_dS = consider_dS
        self.averaging = averaging
        
        self.absoluteI = absoluteI
        
        self.kwargs_scatter = kwargs_scatter
        self.kwargs_farfield = kwargs_farfield
        
        self.dteta = ((kwargs_farfield['tetamax']-kwargs_farfield['tetamin']) / 
                      float(kwargs_farfield['Nteta']-1))
        self.dphi  = 2.*np.pi/float(kwargs_farfield['Nphi']-1)


    def list_depth(self, L):
        """'depth' of a list"""
        depth = lambda L: isinstance(L, list) and max(map(depth, L))+1
        return depth(L)
        
        
    
    def objective_function(self, params):
        """evaluate directionality ratio
        """
        self.model.generate_structure(params)
        
        ## --- main GDM simulation
        core.scatter(self.model.sim, nthreads=self.nthreads, verbose=0, **self.kwargs_scatter)
        
        ## --- iterate objective functions
        Iratios = []
        for i_obj, dir_idx in enumerate(self.dir_index):
            ## --- linear scattering to farfield, incoherent sum of all field_indices
            I_sc, I_tot, I0 = 0, 0, 0
            for di in self.field_index[i_obj]:
                tetalist, philist, _I_sc, _I_tot, _I0 = linear.farfield(
                                             self.model.sim, di, 
                                             nthreads=self.nthreads, 
                                             **self.kwargs_farfield)
                I_sc += _I_sc
                I_tot += _I_tot
                I0 += _I0
            
            if self.which_field.lower() == 'e0':
                I = I0
            elif self.which_field.lower() == 'e_sc':
                I = I_sc
            elif self.which_field.lower() == 'e_tot':
                I = I_tot
            else:
                raise ValueError("'which_field' must be one of ['e0', 'e_sc', 'e_tot']!")
            
            tetalist = tetalist.flatten()
            philist = philist.flatten()
            I = I.flatten()
            
            
            ## --- Processing (weighting, ratio-calc.)
            if self.consider_dS:
                ## --- weight intensities by solid angle of surface element
                dS = self.dteta*self.dphi*np.sin(tetalist)
                dS = dS + dS.max()*0.1  # slight cheating: increase all weights by 10% of overall max to not neglect 180degree backscattering 
            else:
                dS = np.ones(tetalist.shape)
            
            
            ## --- compute either absolute intensity or directionality ratio
            if self.absoluteI:
                if self.averaging:
                    Iratio = np.average(I[dir_idx]*dS[dir_idx])
                else:
                    Iratio = np.sum(I[dir_idx]*dS[dir_idx])
            else:
                if self.averaging:
                    non_idx = np.delete(range(len(I)), dir_idx)
                    Iratio = (np.average(I[dir_idx]*dS[dir_idx]) / 
                                             np.average(I[non_idx]*dS[non_idx]) )
                else:
                    Iratio = (np.sum( (I[dir_idx]*dS[dir_idx])) / 
                                 (np.sum(I*dS) - np.sum(I[dir_idx]*dS[dir_idx])) )
            
            Iratios.append(Iratio)
            
        return Iratios


    ## --- Problem description
    def human_readable_extra(self):
        return "\n\tMaximization of directional scattering."+\
               " Number of objectives: {}".format(len(self.dir_index))





class ProblemMaxScat(BaseProblem):
    """Problem to maximize scat./ext./abs. cross-section or efficiency of nano-structure"""
    
    def __init__(self, N_dim=9999, model=None, nthreads=1, opt_target='Qscat'):
        """constructor
        
        opt_target is one of ['Qscat', 'Qext', 'Qabs', 'CSscat', 'CSext', 'CSabs']
        CS --> cross secion
        Q  --> efficiency (CS divided by geometrical cross-section)
        """
        super(self.__class__, self).__init__(N_dim, model, nthreads=nthreads)
        
        self.opt_target = opt_target
    
    
    def objective_function(self, params):
        """evaluate directionality ratio
        """
        self.model.generate_structure(params)
        
        ## --- GDM simulation and cross-section calculation
        core.scatter(self.model.sim, nthreads=self.nthreads, verbose=0)
        ext_cs, sca_cs, abs_cs = linear.extinct(self.model.sim, field_index=0)
        
        if self.opt_target == "Qscat":
            geo_sect = tools.get_geometric_cross_section(self.model.sim)
            val = float(sca_cs) / geo_sect
        elif self.opt_target == "CSscat":
            val = float(sca_cs)
        elif self.opt_target == "Qext":
            geo_sect = tools.get_geometric_cross_section(self.model.sim)
            val = float(ext_cs) / geo_sect
        elif self.opt_target == "CSext":
            val = float(ext_cs)
        elif self.opt_target == "Qabs":
            geo_sect = tools.get_geometric_cross_section(self.model.sim)
            val = float(abs_cs) / geo_sect
        elif self.opt_target == "CSabs":
            val = float(abs_cs)
        
        return val


    def human_readable_extra(self):
        return "\n\tMaximization of scattering cross section"






class ProblemMaxNearfield(BaseProblem):
    """Problem to maximize near-field enhancement (|E|^2) in vicinity of nano-structure"""
    
    def __init__(self, N_dim=9999, model=None, nthreads=1, 
                 r_probe=(0,0,0), opt_target='E'):
        """constructor
        
        r_probe defines the (x,y,z) position where field enhancemnt is to be
        maximized
        
        opt_target defines wether the e-field ('E') or the b-field ('B') 
        shall be maximized
        
        """
        super(self.__class__, self).__init__(N_dim, model, nthreads=nthreads)
        
        self.r_probe = np.transpose( [r_probe] )
        self.opt_target = opt_target.lower()
        
        if self.opt_target not in ['e', 'b', 'h']:
            raise ValueError("'opt_target' must be one of ['e', 'b', 'h'].")
    
    
    def objective_function(self, params):
        """evaluate directionality ratio
        """
        self.model.generate_structure(params)
        
        ## --- GDM simulation and cross-section calculation
        core.scatter(self.model.sim, nthreads=self.nthreads, verbose=0)
        Es, Etot, Bs, Btot = linear.nearfield(self.model.sim, field_index=0,
                                                          r_probe=self.r_probe)
        
        if self.opt_target == "e":
            a = Es
        elif self.opt_target in ["b", "h"]:
            a = Bs
        
        I_NF = np.abs(a[0][3]**2 + a[0][4]**2 + a[0][5]**2)
        
        return I_NF


    def human_readable_extra(self):
        return "\n\tMaximization of scattering cross section"








