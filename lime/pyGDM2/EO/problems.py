# encoding: utf-8
"""
Collection of problems for the EO submodule of pyGDM2

    Copyright (C) 2017-2020, P. R. Wiecha

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

import numpy as np

import warnings


from .. import core
from .. import linear
from .. import tools




class BaseProblem(object):
    """Base pyGDM2 problem
    
    Problem classes must call the `BaseProblem` constructor with at least 
    an instance of a model-class (see :class:`.models.BaseModel`)
    
    Parameters
    ----------
    model : instance of class implementing :class:`.models.BaseModel`
            Definition of structure model and pyGDM-simulation
    
    field_index : int, default: 0
        field_index to use from pyGDM simulation.
    
    maximize : bool, default: True
        whether to maximize (True) or minimize (False) the fitness function
    
    """
    __name__ = 'BaseProblem'
    
    def __init__(self, model=None, field_index=0, maximize=True):
        
        if model is not None:
            ## --- main simulation definitions
            self.model = model
            self.sim = model.sim
            self.field_index = field_index
            self.maximize = maximize
            
            ## -- problem boundaries
            lbnds, ubnds = self.model.get_bounds()
            self.set_bounds(lbnds, ubnds)
        else:
            raise ValueError("No valid model provided. Please init the Problem with a valid geometry model instance.")
        
        
    
#==============================================================================
# Mandatory reinmplementations for specific problems:
#==============================================================================
    def objective_function(self, params):
        """To be reimplemented! Evaluates the objective function"""
        raise NotImplementedError("'problems.BaseProblem.objective_function' not re-implemented!")


#==============================================================================
# Optionally reinmplementations
#==============================================================================
    def equality_constraints(self, params):
        """Optional (nonlinear) equality constraints (as 1D array)
        
        Equality constraints are regarded satisfied if == 0
        """
        return []
        
    def inequality_constraints(self, params):
        """Optional (nonlinear) inequality constraints (as 1D array)
        
        Inequality constraints are regarded satisfied if <= 0
        """
        return []
        
    def get_name(self):
        """Return problem object name"""
        return self.__name__

    def get_extra_info(self):
        """Return extra info, e.g. a more detailed problem description"""
        return ""

    def get_nobj(self):
        """number of objectives"""
        return len(self.fitness(self.lbnds))
    
#==============================================================================
# Internally used
#==============================================================================    
    def set_bounds(self, lbnds, ubnds):
        self.lbnds = lbnds
        self.ubnds = ubnds
    
    def get_bounds(self):
        return (self.lbnds, self.ubnds)

    def fitness(self, dv):
        """Virtual objective function, single objective (PyGMO requirement)"""
        
        ## --- target function evaluation
        fitness = self.objective_function(dv)
        if not hasattr(fitness, '__iter__'):
            fitness = [fitness, ]
        if self.maximize:
            fitness = [-1*f for f in fitness]
        
        ## --- constraint functions
        equality_constraints = self.equality_constraints(dv)  # ==0 --> satisfied
        if not hasattr(equality_constraints, '__iter__'):
            equality_constraints = [equality_constraints, ]
        
        inequality_constraints = self.inequality_constraints(dv)  # <=0 --> satisfied
        if not hasattr(inequality_constraints, '__iter__'):
            inequality_constraints = [inequality_constraints, ]
        
        return_vector = np.concatenate( [ fitness, equality_constraints, inequality_constraints ] )
        
        return return_vector

    
        
    
    
    
    
    

# =============================================================================
# Pre-defined Problems
# =============================================================================
class ProblemScat(BaseProblem):
    """optimize for scat./ext./abs. cross-section or efficiency
    
    Parameters
    ----------
    model : instance of class implementing :class:`.models.BaseModel`
        Definition of structure model and pyGDM-simulation
    
    field_index : int, default: 0
        field_index to use from pyGDM simulation.
    
    opt_target : str, default: 'Qscat'
        Optimization target. 
        One of ['Qscat', 'Qext', 'Qabs', 'CSscat', 'CSext', 'CSabs']
         - CS: cross-section (absolute value)
         - Q: efficiency (CS divided by geometrical CS)
    
    maximize : bool, default: True
        whether to maximize (True) or minimize (False) the fitness function
    
    """
    
    def __init__(self, model, field_index=0, opt_target='Qscat', 
                 maximize=True):
        """constructor"""
        super(self.__class__, self).__init__(model, field_index, 
                                         maximize=maximize)
        self.opt_target = opt_target
    
    
    def objective_function(self, params):
        """evaluate directionality ratio
        """
        self.model.generate_structure(params)
        
        ## --- GDM simulation and cross-section calculation
        core.scatter(self.model.sim, verbose=0)
        ext_cs, sca_cs, abs_cs = linear.extinct(self.model.sim, field_index=self.field_index)
        
        ## --- scattering
        if self.opt_target == "Qscat":
            geo_sect = tools.get_geometric_cross_section(self.model.sim)
            val = float(sca_cs) / geo_sect
        elif self.opt_target == "CSscat":
            val = float(sca_cs)
        
        ## --- extinction
        elif self.opt_target == "Qext":
            geo_sect = tools.get_geometric_cross_section(self.model.sim)
            val = float(ext_cs) / geo_sect
        elif self.opt_target == "CSext":
            val = float(ext_cs)
        
        ## --- absorption
        elif self.opt_target == "Qabs":
            geo_sect = tools.get_geometric_cross_section(self.model.sim)
            val = float(abs_cs) / geo_sect
        elif self.opt_target == "CSabs":
            val = float(abs_cs)
        
        return val


    def get_extra_info(self):
        return "\n\tMaximization of scattering cross section"






class ProblemNearfield(BaseProblem):
    """optimize near-field enhancement ($|E|^2$ or $|B|^2$)

    opt_target defines wether the E-field ('E') or the B-field ('B') 
    shall be maximized (or optionally minimized)
    
    Parameters
    ----------
    model : instance of class implementing :class:`.models.BaseModel`
        Definition of structure model and pyGDM-simulation
    
    field_index : int, default: 0
        field_index to use from pyGDM simulation.
    
    r_probe : 3-tuple, default: (0,0,0)
        defines the (x,y,z) position where field enhancemnt is to be
        maximized / minimized
    
    opt_target : str, default: 'E'
        Optimization target. Electric or magnetic field intensity. 
        One of ['E', 'B'].
    
    maximize : bool, default: True
        whether to maximize (True) or minimize (False) the fitness function
    
    """
    
    def __init__(self, model, field_index=0,
                 r_probe=(0,0,0), opt_target='E',
                 maximize=True):
        """constructor"""
        super(self.__class__, self).__init__(model, field_index, 
                                         maximize=maximize)
        
        self.r_probe = np.transpose( [r_probe] )
        self.opt_target = opt_target.lower()
        
        if self.opt_target not in ['e', 'b', 'h']:
            raise ValueError("'opt_target' must be one of ['e', 'b', 'h'].")
    
    
    def objective_function(self, params):
        """evaluate field intensity"""
        self.model.generate_structure(params)
        
        ## --- GDM simulation and cross-section calculation
        core.scatter(self.model.sim, verbose=0)
        Es, Etot, Bs, Btot = linear.nearfield(self.model.sim, 
                                              field_index=self.field_index,
                                              r_probe=self.r_probe)
        
        if self.opt_target.lower() == "e":
            a = Es
        elif self.opt_target.lower() in ["b", "h"]:
            a = Bs
        
        I_NF = np.abs(a[0][3]**2 + a[0][4]**2 + a[0][5]**2)
        
        return I_NF


    def get_extra_info(self):
        return "\n\tMaximization of near-field intensity"






        
class ProblemDirectivity(BaseProblem):
    """Problem to optimize directionality of scattering from nanostructure
    
    Use `EO.tools.calculate_solid_angle_by_dir_index` to define 'dir_index' 
    for the desired target solid angle
    
    
    Parameters
    ----------
    
    model : instance of class implementing :class:`.models.BaseModel`
        Definition of structure model and pyGDM-simulation
    
    field_index : list of int, default: 0
        "field_index" in case of multiple pyGDM-simulation configurations
    
    dir_index : list of int; list of list of int, default: [5]
        which solid-angle elements to consider for optimization.
        If list of lists of int (e.g. [[4], [5], [1,2]]), will run 
        multi-objective optimization using each of the sub-lists as target
        solid angle
    
    which_field : str, default: 'e_sc'
        optimize using one of ['e_sc', 'e_tot', 'e0']. 
         - 'e_sc': scattered field
         - 'e_tot': total field (= e0 + e_sc)
         - 'e0': fundamental field (rather for testing) 
    
    kwargs_farfield : dict, default: dict(Nteta=3, Nphi=5, tetamin=0, tetamax=np.pi/2.)
        kwargs, passed to :func:`pyGDM2.linear.farfield`, defining the farfield 
        scattering calculation. Required arguments are: Nteta, Nphi, tetamin, tetamax
    
    consider_dS (False), averaging (True), absoluteI (False)
        Additional flags (defaults in paranthesis): 
             - consider_dS: correct (True) solid angle integration or simple sum (False)
             - averaging: avgerage (True) or integrate (False) solid angle intensity
             - absoluteI: maximize total intensity through selected solid angle instead of ratio
    
    kwargs_scatter : dict, default: {}
        additional kwargs, passed to :func:`pyGDM2.core.sactter`
    
    maximize : bool, default: True
        whether to maximize (True) or minimize (False) the fitness function
    
    """
    def __init__(self, model, field_index=0,
                       dir_index=[5], which_field='e_sc',
                       kwargs_farfield=dict(Nteta=3, Nphi=5, tetamin=0, tetamax=np.pi/2.),
                       consider_dS=False, averaging=True, absoluteI=False, 
                       kwargs_scatter={}, maximize=True):
        """constructor"""
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
        super(self.__class__, self).__init__(model, field_index, 
                                         maximize=maximize)
        
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
        """evaluate directionality ratio"""
        self.model.generate_structure(params)
        
        ## --- main GDM simulation
        core.scatter(self.model.sim, verbose=0, **self.kwargs_scatter)
        
        ## --- iterate objective functions
        Iratios = []
        for i_obj, dir_idx in enumerate(self.dir_index):
            ## --- linear scattering to farfield, incoherent sum of all field_indices
            I_sc, I_tot, I0 = 0, 0, 0
            for di in self.field_index[i_obj]:
                tetalist, philist, _I_sc, _I_tot, _I0 = linear.farfield(
                                             self.model.sim, di, 
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
    def get_extra_info(self):
        return "\n\tOptimize directionality of scattering."+\
               " Number of objectives: {}".format(len(self.dir_index))








