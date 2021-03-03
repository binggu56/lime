# encoding: utf-8
"""
Collection of structure models for the EO submodule of pyGDM2

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

import numpy as np
import time
import locale 


import copy
import sys
import warnings


from ..tools import print_sim_info
from .. import structures 


## del all duplicates with same x/y coordinates
def unique_rows_3D(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


#    a = np.ascontiguousarray(a)
#    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
#    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))



#==============================================================================
# Base class 
#==============================================================================
class BaseModel(object):
    """Base class describing a structure model for evolutionary optimization
    
    Model classes must take at least a pyGDM2 simulation object as input parameter
    
    Mandatory functions for re-implementation :
    ---------------
    `get_dim(self)` : no argmuents
        returns dimensionality of problem (integer)
    
    `get_bounds(self)` : no argmuents
        returns dimensionality of problem (integer)
        
    `generate_structure(self, params)` : input `params`
        `params`: list of length corresponding to the problem's dimensionality.
        
        resets `self.sim.struct` with the generated structre
        
    """
    def __init__(self, sim):
        self.sim = sim
        self.step = sim.struct.step


    def set_step(self, step):
        """set the stepsize of the geometry
        
        may also be used as optimization-parameter (in `generate_structure`)
        """
        self.step = step
        self.sim.struct.step = step
        
        
#==============================================================================
#     Mandatory for reimplementation
#==============================================================================
    def get_dim(self):
        """return dimension of problem"""
        raise NotImplementedError("'models.BaseModel.get_dim' not re-implemented!")
        
    
    def get_bounds(self):
        """return lower and upper bounds as required by pyGMO models"""
        raise NotImplementedError("'models.BaseModel.get_bounds' not re-implemented!")
        
        
    def generate_structure(self, params):
        """the main structure generator"""
        raise NotImplementedError("'models.BaseModel.generate_structure' not re-implemented!")
    
    
#==============================================================================
#     Optional for reimplementation: Information output
#==============================================================================
## --- Randomize structure
    def random_structure(self):
        """generate a randomized structure"""
        
        N_dim = self.get_dim()
        lbnds, ubnds = self.get_bounds()
        
        if not hasattr(lbnds, '__iter__'):
            lbnds = [lbnds] * N_dim
            ubnds = [ubnds] * N_dim
        
        random_parameters = ( (np.array(ubnds) - np.array(lbnds)) * 
                               np.random.random(N_dim) + np.array(lbnds) )
        
        self.generate_structure(random_parameters)
        
        
## --- Replace structure in `core.simulation` object
    def set_structure(self, geometry):
        """replace structure-geometry in `core.simulation` object"""
        if hasattr(self.sim.struct.material, '__iter__'):
            material = self.sim.struct.material[0]
        else:
            material = self.sim.struct.material
        
        struct = structures.struct(self.sim.struct.step, 
                                   geometry, material, # <-- the updated values
                                   self.sim.struct.n1, self.sim.struct.n2,
                                   self.sim.struct.normalization, 
                                   n3=self.sim.struct.n3, 
                                   spacing=self.sim.struct.spacing)
        self.sim.struct = struct
        self.sim.struct.setDtype(self.sim.dtypef, self.sim.dtypec)
        

## --- Information output
    def print_info(self):
        """print information about structure to `stdout`"""
        warnings.warn("'models.BaseModel.print_info' not re-implemented! Only printing simulation info.")
        
        print_sim_info(self.sim)
        
    
    def plot_structure(self, interactive=True, show=True, **kwargs):
        """plot the structure (2D, matplotlib)
        
        **kwargs are passed to `pyGMD2.visu.structure`
        """
        warnings.warn("'models.BaseModel.plot_structure' not re-implemented! Using `pyGDM2.visu.structure`.")
        
        import matplotlib.pyplot as plt
        from ..visu import structure as plt_struct
        
        if interactive: 
            plt.ion()
        
        plt_struct(self.sim, show=show, **kwargs)





#==============================================================================
# Simple structures
#==============================================================================
class RectangularAntenna(BaseModel):
    """optimization-model for simple geometry, consisting of one rectangular antenna
    
    free parameters: width W, length L, x_offset X, y_offset Y
     - W,L : units of `stepsize`
     - X,Y : units of nm
    
    Note : 
        The purpose of this model is merely for demonstration. 
    
    """
    def __init__(self, sim, limits_W, limits_L, limits_pos, height):
        """
        """
        ## --- init basemodel with simulation instance
        super(self.__class__, self).__init__(sim)
        
        print ("Rectangular Antenna optimziation model: Note that this simple model is rather intended for testing and demonstration purposes.")
        
        ## --- width and length limits for rectrangular antenna
        self.limits_W   = limits_W
        self.limits_L   = limits_L
        ## --- position boundaries (same for x and y)
        self.limits_pos = limits_pos
        
        ## --- fixed parameters
        self.height = height    # height of rectangular antenna
        
        ## --- init with random values. `set_step`and `random_structure` are 
        ##     defined in `BaseModel`
        self.set_step(self.sim.struct.step)
        self.random_structure()
    
    def get_dim(self):
        """four free parameters: width / length / posX / posY"""
        return 4
    
    def get_bounds(self):
        """Return lower and upper boundaries for parameters as required by pyGMO models"""
        self.lbnds = [self.limits_W[0], self.limits_L[0], self.limits_pos[0], self.limits_pos[0]]
        self.ubnds = [self.limits_W[1], self.limits_L[1], self.limits_pos[1], self.limits_pos[1]]
        
        return self.lbnds, self.ubnds
    
    def generate_structure(self, params):
        """generate the structure"""
        ## --- order of `params` must correspond to boundary order defined by `get_bounds`
        W, L, pos_x, pos_y = params
        
        from .. import structures
        geometry = structures.rect_wire(self.sim.struct.step, L, self.height, W)
        geometry.T[0] += pos_x
        geometry.T[1] += pos_y
        
        ## -- set new structure-geometry (`set_structure` is a `BaseModel` function)
        self.set_structure(geometry)





class CrossAntenna(BaseModel):
    """optimization-model for simple geometry, consisting of a cross-shaped structure
    
    free parameters: 2 widths W1, W2; 2 lengths L1, L2, x_offset X, y_offset Y
     - Wi,Li : units of `stepsize`
     - X,Y   : units of nm
    
    Note : 
        The purpose of this model is merely for demonstration. 
    
    """
    def __init__(self, sim, limits_W, limits_L, limits_pos, height):
        """
        """
        ## --- init basemodel with simulation instance
        super(self.__class__, self).__init__(sim)
        
        ## --- width and length limits for rectrangular antenna
        self.limits_W   = limits_W
        self.limits_L   = limits_L
        ## --- position boundaries (same for x and y)
        self.limits_pos = limits_pos
        
        ## --- fixed parameters
        self.height = height    # height of rectangular antenna
        
        ## --- init with random values. `set_step`and `random_structure` are 
        ##     defined in `BaseModel`
        self.set_step(self.sim.struct.step)
        self.random_structure()
        
    
    def get_dim(self):
        """four free parameters: 2 x width / 2 x length / posX / posY"""
        return 6
    
    def get_bounds(self):
        """Return lower and upper boundaries for parameters as required by pyGMO models"""
        self.lbnds = [self.limits_W[0], self.limits_W[0], self.limits_L[0], self.limits_L[0], 
                      self.limits_pos[0], self.limits_pos[0]]
        self.ubnds = [self.limits_W[1], self.limits_W[1], self.limits_L[1], self.limits_L[1], 
                      self.limits_pos[1], self.limits_pos[1]]
        
        return self.lbnds, self.ubnds
    
    
    def generate_structure(self, params):
        """generate the structure"""
        ## --- order of `params` must correspond to boundary order defined by `get_bounds`
        W1, W2, L1, L2, pos_x, pos_y = params
        
        from .. import structures
        wire1 = structures.rect_wire(self.sim.struct.step, L=L1, H=self.height, W=W1)
        wire2 = structures.rect_wire(self.sim.struct.step, L=L2, H=self.height, W=W2)
        geometry = np.concatenate([wire1, wire2])
        geometry.T[0] += pos_x
        geometry.T[1] += pos_y
        
        ## --- delete duplicates
        geometry = unique_rows_3D(geometry)
        
        ## -- set new structure-geometry (`set_structure` is a `BaseModel` function)
        self.set_structure(geometry)










#==============================================================================
# More complex structure models
#==============================================================================
class MultiRectAntenna(BaseModel):
    """optimization-model for geometry consisting of multiple rectangular pads
    
    free parameters: N times (width; length; x_pos; y_pos)
        - N: number of rectangles
        - all parameters in units of `stepsize`
     
    """
    def __init__(self, sim, N_antennas, limits_W, limits_L, limits_pos_x, 
                 limits_pos_y, height):
        """
        """
        ## --- init basemodel with simulation instance
        super(self.__class__, self).__init__(sim)
        
        ## --- width and length limits for each rectrangular sub-antenna
        self.limits_W   = limits_W
        self.limits_L   = limits_L
        ## --- position boundaries
        self.limits_pos_x = limits_pos_x
        self.limits_pos_y = limits_pos_y
        
        ## --- fixed parameters
        self.N = int(N_antennas)    # Number of rectangular sub-antenna
        self.height = height        # height of rectangular antenna
        
        ## --- init with random values
        self.set_step(self.sim.struct.step)
        self.random_structure()
        
    
    def get_dim(self):
        """four free parameters: 2 x width / 2 x length / posX / posY"""
        return self.N * 4
    
    def get_bounds(self):
        """Return lower and upper boundaries for parameters as required by pyGMO models"""
        self.lbnds = [self.limits_W[0], self.limits_L[0], self.limits_pos_x[0], 
                      self.limits_pos_y[0]] * self.N
        self.ubnds = [self.limits_W[1], self.limits_L[1], self.limits_pos_x[1], 
                      self.limits_pos_y[1]] * self.N
        
        return self.lbnds, self.ubnds
    
    
    def generate_structure(self, params):
        """generate the structure"""
        ## --- get 4-tuples describing each rectangular sub-antenna
        params_rects = [params[i:i + 4] for i in xrange(0, len(params), 4)]
        
        from .. import structures
        
        for i, p in enumerate(params_rects):
            W, L = int(round(p[0])), int(round(p[1]))
            rect = structures.rect_wire(self.sim.struct.step, L=L, H=self.height, W=W)
            rect.T[0] += int(round(p[2])) * self.sim.struct.step
            rect.T[1] += int(round(p[3])) * self.sim.struct.step
            if i == 0:
                geometry = rect
            else:
                geometry = np.concatenate([geometry, rect])
        
        ## --- delete duplicates
        geometry = unique_rows_3D(geometry)
        
        ## -- set new structure-geometry (`set_structure` is a `BaseModel` function)
        self.set_structure(geometry)





class BlockModel(BaseModel):
    """structure consisting of `N` blocks of variable positions
    """
    def __init__(self, sim, 
                 N, block_nx,block_ny,block_nz, area_limits, 
                 forbidden=[], 
                 symmetric=False, fit_offset=False):
        """
        """
        super(self.__class__, self).__init__(sim)
        
        ## config structure
        self.N_blocks = N
        
        self.block_nx = block_nx
        self.block_ny = block_ny
        self.block_nz = block_nz
        
        
        ## --- list of forbidden meshpoint-positions [[x1,y1], [x2,y2] ... ] 
        ##     (index coordinates [=divided by 'step'])
        self.forbidden = forbidden
        
        
        ## --- symmtery-axis: X? If True: Mirror structure at lower y-limit
        self.symmetric = symmetric
        self.fit_offset = fit_offset
        
        
        self.area_min, self.area_max = area_limits[0], area_limits[1]
        self.x_offset = 0
        self.y_offset = 0
        
        
        ## --- init with random positions      
        self.set_step(self.step)
        self.random_structure()
        
    
    def set_step(self, step):
        """set the stepsize"""
        self.step = step
        self.sim.struct.step = step
        
        self.step_grid_x = self.step*self.block_nx
        self.step_grid_y = self.step*self.block_ny
        
        self.generate_zero_block()
    
    
    def get_dim(self):
        if self.fit_offset:
            return 2 * self.N_blocks + 2            
        else:
            return 2 * self.N_blocks
    
    
    def get_bounds(self):
        """Return lower and upper bounds as required by pyGMO models"""
        self.lbnds = self.area_min
        self.ubnds = self.area_max
        
        return self.lbnds, self.ubnds
    

## --------- structure generation
    def generate_zero_block(self):
        """generate a list describing the fundamental block meshpoints"""
        self.zeroblock = []
        for xi in np.arange(self.block_nx):
            for yi in np.arange(self.block_ny):
                for zi in np.arange(self.block_nz):
                    self.zeroblock.append([xi, yi, zi+0.5])
        
        self.zeroblock = np.array(self.zeroblock)*self.step


    def get_position_tuples(self, params):
        """Generate position tuples (x,y) from the `params` list
        
        splitting up the `params` list: 
        (x0,y0) = [params[0], params[1]], (x1,y1) = [params[2], params[3]], ...
        
        positions are given as (x,y) index-tuples (=divided by `step`)
        """
        POS=[]
        for i, val in enumerate(params):
            if i%2==0:
                POS.append([])
            ## convention for float-->int parameter conversion: int(round(value))
            val = int(round(val))
            POS[-1].append(val)
        
        return np.array(POS, dtype=np.float)    
    
        
    def generate_structure(self, params):
        """generate the structure"""
        
        ## --- collect parameters, exception handling
        if self.fit_offset: 
            positions = self.get_position_tuples(params[:-2])
            self.x_offset, self.y_offset = params[-2:]
        else:
            positions = self.get_position_tuples(params)
            self.x_offset, self.y_offset = 0, 0
        
        if len(positions) != self.N_blocks:
            raise ValueError("Number of Positions must equal number of Block!")
        
        
        ## --- assemble blocks
        self.t0 = time.time()
        dipoles = []
        for xi,yi in positions:
            block = np.copy(self.zeroblock).T
            block[0] += round(xi)*self.step_grid_x
            block[1] += round(yi)*self.step_grid_y
            
            for dp in block.T:
                dipoles.append(tuple(dp))
        else:
            pass


        ## --- If symmetric: shift to positive Y and Duplicate sutructure
        dipoles = np.array(dipoles)
        if self.symmetric:
            if dipoles.T[1].min() < 0:
                dipoles.T[1] -= dipoles.T[1].min() #+ np.ceil(self.block_ny/2.-1)*self.step
            
            dipoles_sym = copy.deepcopy(dipoles)
            dipoles_sym.T[1] *= -1
            
            ## --- shift down by number of y-meshpoints per block
            dipoles_sym.T[1] += (self.block_ny-1)*self.step
            dipoles = np.concatenate([dipoles, dipoles_sym])
            ## --- center
            dipoles.T[1] -= np.floor((self.block_ny + 1-self.block_ny%2)/2.) * self.step
        
        
        ## --- delete forbidden positions
        delete_list = []
        for idp, (dpx,dpy,dpz) in enumerate(dipoles):
            if [int(round(dpx/self.step_grid_x)), int(round(dpy/self.step_grid_y))] \
                                                             in self.forbidden:
                delete_list.append(idp)
        dipoles = np.delete(dipoles, delete_list, 0)
        
        
        ## --- apply structure offset
        if self.fit_offset:
            dipoles.T[1] += round(self.y_offset) * self.step_grid_y
            dipoles.T[0] += round(self.x_offset) * self.step_grid_x
                
        
        ## --- delete duplicates
        self.len_original = len(dipoles)
        dipoles = unique_rows_3D(dipoles)
        
        
        ## --- replace structure in `self.sim`
        self.set_structure(dipoles)
        


    def print_info(self):
        """print model info"""
        print (" ----------- N BLOCKS MODEL -----------")
        print ('{} dipoles created.'.format(self.len_original))
        print ('{} duplicate dipoles deleted.'.format(self.len_original - self.sim.struct.n_dipoles))
        print ()
        print ("MATERIAL                 =", self.sim.struct.material[0].__name__)
        print ("STEPSIZE                 =", self.step)
        print ("BLOCK-SIZE               =", self.block_nx*self.block_ny*self.block_nz, "(meshpoints)")
        print ("NUMBER OF BLOCKS         =", self.N_blocks, end='')
        if self.symmetric: 
            print ("x2 (symmetric)")
        else:
            print ("")
        print ("AREA LIMITS (blocks)     =", self.area_min,'-->', self.area_max)
        print ("MAX. POSSIBLE MESHPOINTS =", self.N_blocks*self.block_nx*self.block_ny*self.block_nz, end='')
        if self.symmetric: 
            print ("x2 (symmetric)")
        else:
            print ("")
        print ()
        print ("TOTAL MESHPOINTS         =", self.sim.struct.n_dipoles)
        print (" --------------------------------------")
        print ()
        print ()

























#### --------------------------------------------------------------------------
### --           N variable size Antennas Model
#### --------------------------------------------------------------------------
#class AntennaModel():
#    def __init__(self, step, nblocks, h, limg, lims):
#        self.tinit = time.time()
##==============================================================================
##         config structure
##==============================================================================
#        self.setStep(step)
#        self.NBLOCKS = nblocks
#        
#        self.H = h
#        
#        self.LIM_GRID = np.array(limg)
#        self.LIM_SIZE = np.array(lims)
#        
#        
#        ## --- init at random positions        
#        self.get_bounds()
#        self.random_positions()
#        
#        self.t_config = time.time()
#    
#    
#    
#    def setStep(self, step):
#        self.STEP = step
#    
#    
#    def get_dim(self):
#        return self.NBLOCKS*4
#    
#    def get_bounds(self):
#        """Return lower and upper bounds as required by pyGMO models"""
#        limgrid = self.LIM_GRID
#        limsize = self.LIM_SIZE
#        self.lbnds = self.NBLOCKS * [float(limgrid[0]), float(limgrid[0]), 
#                                     float(limsize[0]), float(limsize[0])]
#        self.ubnds = self.NBLOCKS * [float(limgrid[1]), float(limgrid[1]), 
#                                     float(limsize[1]), float(limsize[1])]
#        
#        return self.lbnds, self.ubnds
#    
#    def random_positions(self):
#        ## chose random start positions on defined area and generate random start structure
#        PARAMS = []
#        for i in range(self.NBLOCKS):
#            POS = np.round(np.random.random(2)*(self.LIM_GRID[1]-self.LIM_GRID[0]) + 
#                           self.LIM_GRID[0])
#            DIM = np.round(np.random.random(2)*(self.LIM_SIZE[1]-self.LIM_SIZE[0]) + 
#                           self.LIM_SIZE[0])
#            PARAMS.append(np.array([POS, DIM]).flatten())
#
#        self.PARAMS = np.array(PARAMS).flatten()
#        self.generate_structure(self.PARAMS)
#        
#        
#    def generate_block(self, X,Y,L,D,H):
#        ## ------- Fundamental Brick
#        BLOCK = [[],[],[]]
#        for xi in np.arange(int(round(L))):
#            for yi in np.arange(int(round(D))):
#                for zi in np.arange(H):
#                    BLOCK[0].append(xi+int(round(X)))
#                    BLOCK[1].append(yi+int(round(Y)))
#                    BLOCK[2].append(zi+0.5)
#        
#        return BLOCK
#
#
#
#    def generate_structure(self, PARAMS):
#        ########################################################################
#        ## GENERATE STRUCTURE 
#        ########################################################################
#        P = np.array(PARAMS).reshape((len(PARAMS)/4, 4))
#        #print P
#        if len(P) != self.NBLOCKS:
#            raise ValueError("Number of parameters must equal 4*(number of blocks)!")
#        
#        self.t0 = time.time()
#        ## ------- Generate Structure
#        xm = np.array([])
#        ym = np.array([])
#        zm = np.array([])
#        for p in P:
#            block = self.generate_block(p[0],p[1],p[2],p[3],self.H)
#            xm = np.concatenate([xm, block[0]])
#            ym = np.concatenate([ym, block[1]])
#            zm = np.concatenate([zm, block[2]])
#        xm = xm.flatten()
#        ym = ym.flatten()
#        zm = zm.flatten()
#        
#        self.STRUCT = np.transpose([xm,ym,zm])*self.STEP
#        self.t_build = time.time()
#        
#        
#        ## del all duplicates with same x/y coordinates
#        def unique_rows_3D(a):
#            a = np.ascontiguousarray(a)
#            unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
#            return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
#        
#        self.len_original = len(self.STRUCT)
#        self.STRUCT = unique_rows_3D(self.STRUCT)
#        len1 = len(self.STRUCT)
#        self.t_duplicates = time.time()
#        
#        self.NDipPlane = int(len1/self.H)
#
#    
#    def get_individual_blocks(self, PARAMS):
#        """generate list of the individual blocks constituting the whole structure
#        
#        For analysis purpose (post-optimization)
#        """
#        ## --- check parameters
#        P = np.array(PARAMS).reshape((len(PARAMS)/4, 4))
#        if len(P) != self.NBLOCKS:
#            raise ValueError("Number of parameters must equal 4*(number of blocks)!")
#        
#        ## --- Constituents of structure
#        blocks = []
#        for p in P:
#            block = self.generateBlock(p[0], p[1], p[2], p[3], self.H)
#            blocks.append(np.transpose(block))
#        
#        return np.array(blocks)*self.STEP
#
#
#    def print_info(self):
#        ########################################################################
#        ## Print information
#        ########################################################################
#        print "   ----- N-ANTENNA MODEL -----"
#        print '{} dipoles created.'.format(self.len_original)
#        print '{} duplicate dipoles deleted.'.format(self.len_original-len(self.STRUCT))
#        print
#        print
#        print 'Last structure update (ms):', round(1000*(self.t_duplicates - self.t0),1)
#        print '        - initial config Time (ms) :', round(1000*(self.t_config - self.tinit),1)
#        print '        - last build Time (ms)     :', round(1000*(self.t_build - self.t0),1)
#        print '        - last duplicate Time (ms) :', round(1000*(self.t_duplicates - self.t_build),1)
#        print
#        print
#        print "STEPSIZE                 =", self.STEP
#        print "NUMBER OF BLOCKS         =", self.NBLOCKS
#        print "AREA LIMITS (X; Y)       =", self.LIM_GRID
#        print "SIZE LIMITS (X; Y)       =", self.LIM_SIZE
#        print "TOTAL MESHPOINTS         =", len(self.STRUCT)
#        print
#        print
#
#
#
#    def plot_structure(self, interactive=True, show=True):
#        ########################################################################
#        ## plot structure
#        ########################################################################
#        import matplotlib.pyplot as plt
#        ## VERY IMPORTANT! IF LOCALE NOT SET HERE, FORTRAN FORMATTING IS BROKEN!
#        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
#        
#        if show:
#            plt.clf()
#            if interactive:
#                plt.ion()
#        
#        XM,YM,ZM = self.STRUCT.T
#        plt.scatter(XM, YM, s=50)
#        
#        plt.xlim((self.LIM_GRID[0]-1)*self.STEP , self.STEP*(self.LIM_GRID[1]+self.LIM_SIZE[1]+1)); plt.xlabel("X (nm)")
#        plt.ylim((self.LIM_GRID[0]-1)*self.STEP , self.STEP*(self.LIM_GRID[1]+self.LIM_SIZE[1]+1)); plt.ylabel("Y (nm)")
#        
#        
#        if show:
#            if interactive:
#                plt.show()
#                plt.draw()
#            else:
#                plt.show()
#        if interactive:
#            plt.ion()
#        
#        
#
#
#
#
#
#### --------------------------------------------------------------------------
### --           Binary Array of Blocks Model
#### --------------------------------------------------------------------------
#class BinaryArrayModel():
#    def __init__(self, step, N, nx,ny,nz):
#        self.tinit = time.time()
#        ########################################################################
#        ## config structure
#        ########################################################################
#        """
#        binary array of (N x N) blocks of size (nx x ny x nz).
#        area from (0,0) to (N*step, N*step)
#        """
#        self.N = N
#        
#        self.NX_BLOCK = nx
#        self.NY_BLOCK = ny
#        self.NZ_BLOCK = nz
#        
#        self.set_step(step)
#        
#        
#        ## init at random positions        
#        self.get_bounds()
#        self.random_positions()
#        
#        self.t_config = time.time()
#    
#    def get_dim(self):
#        return self.N**2
#    
#    def get_bounds(self):
#        """Return lower and upper bounds as required by pyGMO models"""
#        self.lbnds = self.get_dim()*[0]
#        self.ubnds = self.get_dim()*[1]
#        
#        return self.lbnds, self.ubnds
#        
#    def random_positions(self):
#        ## chose random start positions on defined area and generate random start structure
#        POSITIONS = np.random.randint(0, 2, self.get_dim())
#        self.generate_structure(POSITIONS)
#    
#    def set_step(self, step):
#        self.STEP = step
#        self.generate_zero_block()
#        
#        
#    def generate_zero_block(self):
#        ## ------- FUNDAMENTAL BLOCK
#        self.ZEROBLOCK = []
#        for xi in np.arange(self.NX_BLOCK):
#            for yi in np.arange(self.NY_BLOCK):
#                for zi in np.arange(self.NZ_BLOCK):
#                    self.ZEROBLOCK.append([xi, yi, zi+0.5])
#        self.ZEROBLOCK = np.array(self.ZEROBLOCK)
#
#
#
#    def generate_structure(self, POSITIONS):
#        ########################################################################
#        ## GENERATE STRUCTURE 
#        ########################################################################
#        if len(POSITIONS) != self.get_dim():
#            raise ValueError("Number of Positions must equal size of array!")
#        self.POSITIONS = POSITIONS
#        
#        self.t0 = time.time()
#        ## ------- PUT BLOCKS AT EACH POSITION
#        dipoles = []
#        for i,BIN in enumerate(POSITIONS):
#            if BIN>=0.5:
#                BLOCK = np.copy(self.ZEROBLOCK).T
#                
#                BLOCK[0] += (i%self.N)*self.NX_BLOCK
#                BLOCK[1] += int(i/self.N)*self.NY_BLOCK
#                for dp in BLOCK.T:
#                    dipoles.append(tuple(dp))
#
#
#        self.t_build = time.time()
#
#        ## Fortran compatible arrays
#        self.dipoles = np.array(dipoles)*self.STEP
#        self.STRUCT  = self.dipoles
#        
#        self.NDipPlane = int(len(self.STRUCT)/self.NZ_BLOCK)
#
#
#
#    def print_info(self):
#        ########################################################################
#        ## Print information
#        ########################################################################
#        print "   ----- N BLOCKS MODEL -----"
#        print '{} dipoles created.'.format(len(self.STRUCT))
#        print
#        print
#        print 'Last structure update (ms):', round(1000*(self.t_build - self.t0),1)
#        print '        - initial config Time (ms) :', round(1000*(self.t_config - self.tinit),1)
#        print '        - last build Time (ms)     :', round(1000*(self.t_build - self.t0),1)
#        print
#        print
#        print "STEPSIZE                 =", self.STEP
#        print "BLOCK-SIZE               =", self.NX_BLOCK*self.NY_BLOCK*self.NZ_BLOCK, "(meshpoints)"
#        print "NUMBER OF BLOCKS         =", (self.N**2)
#        print "MAX. POSSIBLE MESHPOINTS =", (self.N**2)*self.NX_BLOCK*self.NY_BLOCK*self.NZ_BLOCK
#        print "TOTAL MESHPOINTS         =", len(self.dipoles)
#        print
#        print
#
#
#
#    def plot_structure(self, interactive=True, show=True):
#        ########################################################################
#        ## plot structure
#        ########################################################################
#        import matplotlib.pyplot as plt
#        
#        if show:
#            plt.clf()
#            if interactive:
#                plt.ion()
#        XM,YM,ZM = np.array(self.dipoles).T
#        plt.scatter(XM, YM, s=50)
#        
#        plt.xlim(-3*self.STEP, YM.max()+3*self.STEP); plt.xlabel("X (nm)")
#        plt.ylim(-3*self.STEP, XM.max()+3*self.STEP); plt.ylabel("Y (nm)")
#        
#        if show:
#            if interactive:
#                plt.show()
#                plt.draw()
#            else:
#                plt.show()
#        
        
        







if __name__=='__main__':
    
    testModel = BlockModel(step=10., nblocks=40, 
                            nx=5,ny=5,nz=5, 
                            blim=[-10,10])
    
#    testModel = AntennaModel(step=10., nblocks=3, 
#                            h=2,
#                            limg=[-10,10],lims=[2,6])
#    
#    testModel = BinaryArrayModel(step=10., N=10, 
#                            nx=2, ny=2, nz=2)

    #from mayavi import mlab
    #mlab.figure(size=(800,600), bgcolor=(1,1,1))
    
    #visu.plot2Dstruct(testModel.STRUCT)
    
    
    import matplotlib.pyplot as plt
    testModel.print_info()
    for i in range(2):
        testModel.random_positions()
        testModel.plot_structure(interactive=True)
        testModel.print_info()
        plt.pause(.1)
    
    plt.ioff()
    plt.show()











