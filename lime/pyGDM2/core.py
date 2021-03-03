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
pyGDM core routines

"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import gc
import warnings
import copy
import time



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
class CGnoConvergence(Exception):
    pass







#==============================================================================
# Simulation container class
#==============================================================================
class simulation(object):
    """
    Bases: object
    
    Main GDM simulation container class
    
    Defines a linear GDM simulation. Contains information on: 
        - *struct* : :class:`.structures.struct`:
            - the geometry of the nano structure 
            - its dielectric function
            - its environment
        - *efield* : :class:`.fields.efield`
            - the incident field and the wavelenghts 
            - possibly further incident field configuration 
    
    Parameters
    ----------
    
    struct : :class:`.structures.struct`
        structure object
        
    efield : :class:`.fields.efield`
        fundamental field
        
    dtype : (optional) `str`, default: 'f'
        precision of simulation
    
    
    """
    __name__ = "pygdm2 simulation"
    
    def __init__(self, struct, efield, dtype='f'):
        """Initialization"""
        
        ## --- struct includes: geometry, material(s), environment
        self.struct = struct
        
        ## --- efield includes: field_generator function, wavelengths, optional kwargs
        self.efield = efield
        
        ## --- precision
        if dtype in ['f', 'F']:
            self.dtypef = np.float32
            self.dtypec = np.complex64
        elif dtype in ['d', 'D']:
            self.dtypef = np.float64
            self.dtypec = np.complex128
        self.efield.setDtype(self.dtypef, self.dtypec)
        self.struct.setDtype(self.dtypef, self.dtypec)
        
        
        ## --- initialize unevaluated data to `None` (postprocessing data)
        self.E = None   # placeholder: scattered fields inside structure
        self.S_P = None # placeholder: decay-rate tensors
    
    
    def __repr__(self):
        from . import tools
        return tools.print_sim_info(self, prnt=False)
    
    
    def __eq__(self, other):
        ## identical geometries, identical field_kwargs keys, wavelengths, 
        ##     material, n1
        if (len(self.struct.geometry) == len(other.struct.geometry) and
               len(self.efield.wavelengths) == len(other.efield.wavelengths) and
               len(self.efield.kwargs.keys()) == len(other.efield.kwargs.keys())):
            if (np.sum(np.abs(self.struct.geometry - other.struct.geometry))==0 and 
                    np.sum(np.abs(self.efield.wavelengths - other.efield.wavelengths))==0 and 
                    self.efield.kwargs.keys() == other.efield.kwargs.keys() and 
                    self.struct.material[0].__name__ == other.struct.material[0].__name__ and 
                    self.struct.n1 == other.struct.n1 and 
                    self.struct.n2 == other.struct.n2 and 
                    self.struct.n3 == other.struct.n3 and 
                    self.struct.step == other.struct.step):
                return True
            else:
                return False
        else:
            return False
        
        
        
     


#==============================================================================
# scattered fields
#==============================================================================
def scatter(sim, method='lu', matrix_setup='fortran', 
            multithreaded=None, Nprocesses=None, verbose=True,
            callback=None):
    """Perform a linear scattering GDM simulation
    
    Calculate the electric field distribution inside a nano-structure.
    
    Parameters
    ----------
    sim : :class:`.simulation`
        simulation description
    
    method : string, default: "lu"
        inversion method. One of ["lu", "numpyinv", "scipyinv", "pinv2", "superlu", "cg", "pycg", "dyson"]
         - "lu" LU-decomposition (`scipy.linalg.lu_factor`)
         - "numpyinv" numpy inversion (`np.linalg.inv`, if numpy compiled accordingly: LAPACK's `dgesv`)
         - "scipyinv" scipy default inversion (`scipy.linalg.inv`)
         - "pinv2" is usually slower (req. `scipy.linalg.pinv2`)
         - "superlu" often has a <N^3 efficieny but is very expensive in memory (`scipy.sparse.linalg.splu`)
         - "dyson" uses a dyson sequence. Only solver that does not depend on external libraries
         - "skcuda" uses CUDA GPU via `pycuda` and `scikit-cuda`
         - "cupy" uses CUDA GPU via `cupy`
         - "cuda" (equivalent to "skcuda")
        
    matrix_setup : str, default: "fortran"
        Which routine to use for setup of the coupled dipole matrix.
        One of ["fortran", "numba", "cuda"]
        
    multithreaded, Nprocesses : int, bool, default: False
        deprecated parameters, no longer supported
        
    verbose : bool, default False
        Print timing info to stdout
        
    callback : func, default: None
        optional callback function, which is called after each wavelength.
        Passes a dict to `callback` containing current wavelength, 
        simulation and timing info.
    
      
    Returns
    -------
    E : list of lists
        list consists of incident field config and the electric field in 
        structure (total complex electric fields: E0 + E_scat).
        Each element of the list is composed as follows, where Exi is the x 
        component of the complex, total electric field at meshpoint i:
           
        elements : 
            [`dict`: kwargs of incident efield, 
             `list`: field as list of tuples [[Ex1,Ey1,Ez1], [Ex2,Ey2,Ez2], ...] ]
        
    Notes
    -----
    - The complex electric fields inside the structure are also copied into the 
      :class:`.simulation` instance as attribute `simulation.E`
    
    - For details on the concept of the generalized propagator, see e.g.:
      Martin, O. J. F. & Girard, C. & Dereux, A. **Generalized Field Propagator 
      for Electromagnetic Scattering and Light Confinement.**
      Phys. Rev. Lett. 74, 526–529 (1995).
    
    - The `scipy` solvers (like 'lu', 'ilu' and 'cg') run parallelized if BLAS
      is compiled with multithreading support. See:
      http://scipy-cookbook.readthedocs.io/items/ParallelProgramming.html#use-parallel-primitives
      (website called 05/2017)
      
    - To limit the number of threads for the multi-threaded parallelized parts, 
      you might do something like explained in: 
      https://stackoverflow.com/questions/29559338/set-max-number-of-threads-at-runtime-on-numpy-openblas
      (website called 06/2017)
    
    - multi-threaded incident field processing (generalized propagator * E0) requires
      "pathos" (https://github.com/uqfoundation/pathos)

        
    """
    if method.lower() == "cupy":
        import cupy
        if int(cupy.__version__[0])<7:
            raise ValueError("`cupy` version 7 or higher required. Found cupy version {}. Please upgrade and retry.".format(cupy.__version__))
    if method.lower() == "cuda":
        method = "skcuda"
        
    if method.lower() not in ["numpyinv", "scipyinv", "lu", "superlu", 
                                       "pinv2", "dyson", "skcuda", "cupy"]:
        raise ValueError('Error: Unknown solving method. Must be one of' +
                         ' ["lu", "numpyinv", "scipyinv", "pinv2", "superlu",'+
                         ' "dyson", "cuda", "skcuda", "cupy"].')
    
    if multithreaded is not None or Nprocesses is not None:
        warnings.warn("Deprecation warning: kwargs `multithreaded` and `Nprocesses` are no longer supported, have no effect and will be removed in future versions of pyGDM!")
    
    
    field_generator = sim.efield.field_generator
    wavelengths = sim.efield.wavelengths
    
    scattered_fields = []
    
    for i_wl, wavelength in enumerate(wavelengths):
        t0 = time.time()
        
        K = get_general_propagator(sim=sim, method=method, matrix_setup=matrix_setup,
                                   wavelength=wavelength, verbose=verbose)
        
        t1 = time.time()
        
        
#==============================================================================
#         Incident field evaluation
#==============================================================================
        ## --- At fixed wavelength: Use generalized propagator on all incident field parameters
        def generalized_propagator_operation(field_kwargs, K, sim, wavelength):
            E0 = field_generator(sim.struct, wavelength, **field_kwargs)
            E0_supervec = _fieldListToSuperVector(E0)
            
            ## --- generalized propagator times incident field:
            if method.lower() == 'superlu':
                E = K.solve(E0_supervec)
            elif method.lower() == 'lu':
                import scipy.linalg as la
                E = la.lu_solve(K, E0_supervec)
            elif method.lower() == 'skcuda':
                ## --- scikit-cuda GPU matrix vector multiplication
                import pycuda.gpuarray as gpuarray
                import skcuda.linalg as culinalg
                E0_supervec_gpu = gpuarray.to_gpu(np.ascontiguousarray(E0_supervec))
                E_gpu = culinalg.dot(K, E0_supervec_gpu)
                E = E_gpu.get()
            elif method.lower() == 'cupy':
                ## --- cupy GPU matrix vector multiplication
                import cupy as cp
                E0_supervec_gpu = cp.array(np.ascontiguousarray(E0_supervec))  # move vec to device memory
                E_gpu = cp.dot(K, E0_supervec_gpu)
                E = E_gpu.get()
            else:
                E = np.dot(K, E0_supervec)
            
            E = _superVectorToFieldList(E)
            kwargs_final = copy.deepcopy(field_kwargs)
            kwargs_final["wavelength"] = wavelength
            return [kwargs_final, E]
        
        ## --- loop over incident field configurations
        for field_kwargs in sim.efield.kwargs_permutations:
            scattered_fields.append(generalized_propagator_operation(
                                         field_kwargs, K, sim, wavelength))
                
        if verbose: print("repropa.: {:.1f}ms ({} field configs), tot: {:.1f} ms".format(
                                        1000.*(time.time()-t1), 
                                        len(sim.efield.kwargs_permutations), 
                                        1000.*(time.time()-t0)))
        if callback is not None:
            sim.E = scattered_fields
            cb_continue = callback(dict(
                i_wl=i_wl, wavelength=wavelength, sim=sim,
                t_inverse=1000.*(t1-t0), t_repropa=1000.*(time.time()-t1))
                                  )
            if not cb_continue:
                break           # quit if callback returns False
            
    gc.collect()
    
    sim.E = scattered_fields
    return scattered_fields



#==============================================================================
# scattered fields
#==============================================================================
def scatter_CG(sim, method='cg', cgKwargs={}, 
              cg_recycle_pc=True, pc_method='ilu', pc_recalc_thresh=0.5,
              matrix_setup='fortran', verbose=True):
    """linear scattering GDM simulation via conjugate gradients iterative solver
    
    Calculate the electric field distribution inside a nano-structure.
    For more details, see also :func:`.scatter`.
    
    Parameters
    ----------
    sim : :class:`.simulation`
        simulation description
    
    method : string, default: "cg"
        conjugate gradient iterative solver
        - "cg" (req. `scipy`)
        - "pycg" (req. `pyamg`)
        
    cgKwargs : dict, default: {}
        keywords passed to conjugate-gradient method, if used 
    
    cg_recycle_pc : bool, default: True
        Use same Preconditioner for all wavelengths (default: True)
        Remark: For large dispersion / large wavelength differences cg_recycle_pc=True 
        might slow down convergence significantly.
        Buy mostly this will speed up the Calcuation.
    
    pc_method : str, default 'ilu'
        Preconditioning method. 
        'ilu': scipy incomplete LU, 
        'lu': scipy complete LU (exact, but speed-imporvements for nearby wavelengths), 
        'amg' pyAMG amg, 
        'none': No preconditioning
    
    PCrecalcThresh : float, default 0.5
        Preconditioning recalculation time threshold.
        If speedup of PC-recycling falls below this threshold, recalculate 
        PC on next wavelength.
        
    matrix_setup : str, default: "fortran"
        Which routine to use for setup of the coupled dipole matrix.
        One of ["fortran", "numba", "cuda"]
        
    verbose : bool, default False
        Print timing info to stdout
    
      
    Returns
    -------
    E : list of lists
        list consists of incident field config and the electric field in 
        structure (total complex electric fields: E0 + E_scat).
        Each element of the list is composed as follows, where Exi is the x 
        component of the complex, total electric field at meshpoint i:
           
        elements : 
            [`dict`: kwargs of incident efield, 
             `list`: field as list of tuples [[Ex1,Ey1,Ez1], [Ex2,Ey2,Ez2], ...] ]
        
    """
    warnings.warn("Untested CG implementation. Use with caution!")
    
    
    if method.lower() not in ["cg", "pycg"]:
        raise ValueError('Error: Unknown CG solver. Must be one of' +
                                                             ' "cg", "pycg"].')
    field_generator = sim.efield.field_generator
    wavelengths = sim.efield.wavelengths
    
    scattered_fields = []
    PC = None; PCreset = True
    
    for i_wl, wavelength in enumerate(wavelengths):
        if verbose: t0=time.time()
        
        ## --- evaluate polarizabilities and normalization terms at "wavelength"
        sim.struct.getPolarizability(wavelength)
        sim.struct.getNormalization(wavelength)
        
        ## --- construct matrix
        as_csc = False
        CAM0 = get_side_by_side(sim, wavelength=wavelength, invertible=True, 
                            times_alpha=True, as_csc=as_csc)       
        
        if verbose: 
            print('timing for wl={:.2f}nm - setup: {:.1f} ms, '.format(
                                wavelength, 1000.*(time.time()-t0)), end='')
            t0b = time.time()
# =============================================================================
#       Iterative Conjugate Gradients solver 
# =============================================================================
        if len(sim.efield.kwargs_permutations) > 10:
            warnings.warn("Efficiency warning: Many field-configurations" +
                          " per wavelength (N_conf={})! Conjugate gradients" +
                          " will probably be very slow.".format(
                                      len(sim.efield.kwargs_permutations)))
        if cg_recycle_pc==False: PC = None
        tPCreset = time.time()
        
        for field_kwargs in sim.efield.kwargs_permutations:
            E0 = field_generator(sim.struct, wavelength, **field_kwargs)
            E0_supervec = _fieldListToSuperVector(E0)
            E, PC, STATUS = get_efield_by_cg(CAM0, E0_supervec, PC=PC, 
                                          method=method, pc_method=pc_method, 
                                          **cgKwargs)
            
            if STATUS != 0: # on cgs convergence fail --> exact LU-inversion
                warnings.warn("CG convergence warning: Conjugate gradients" +
                              " did not converge. Trying LU instead.")
                ## PC == K: Use as next preconditioner afterwards (avoid resetting PC).
                PC = get_general_propagator(CAM0, method="LU", sim=sim, 
                                          wavelength=wavelength, 
                                          retLinearOperator=1)
                E = PC*E0_supervec
                tPCreset = time.time()
                
            E = _superVectorToFieldList(E)
            kwargs_final = copy.deepcopy(field_kwargs)
            kwargs_final["wavelength"] = wavelength
            scattered_fields.append([kwargs_final, E])
        
        ## Check last CG duration: If > pc_recalc_thresh * Preconditionung-time: 
        ## recalculate PC in next step
        if cg_recycle_pc:
            if PCreset:
                PCreset = False
                tCG0 = time.time() - tPCreset
            elif time.time()-tPCreset > pc_recalc_thresh*tCG0:
                PCreset = True
                PC = None
                if verbose: print("   Slow convergence (limit {:.1f}ms) " + \
                                  "--> Recalc. PC on next iter.\n   --> ".format(
                                              pc_recalc_thresh*tCG0*1000), end='')
        
        if verbose: print("CG solver.: {:.1f}ms ({} field configs), tot: {:.1f} ms".format(
                                            1000.*(time.time()-t0b), len(sim.efield.kwargs_permutations), 
                                            1000.*(time.time()-t0)))
            
    
    gc.collect()
    
    sim.E = scattered_fields
    return scattered_fields



        
def scatter_mpi(sim, verbose=False, scatter_verbose=False, **kwargs):
    """MPI wrapper to :func:`.scatter` for embarrassingly parallel calculation of spectra
    
    *requires:* **mpi4py**
    
    run with "mpirun -n X python scriptname.py", where `X` is the number of 
    parallel processes (ideally an integer divisor of the number of wavelengths)
    
    
    Parameters
    ----------
    sim : :class:`.simulation` 
        simulation description
    
    verbose, scatter_verbose: bool, default: False
        turns on some mpi-routine info printing, respectively controls verbose setting for :func:`.scatter`
    
    **kwargs : 
        all kwargs are passed to :func:`.scatter`


    Notes 
    -----
    - On single machines it is usually easier to install `scipy` 
      compiled with parallel BLAS (parallel LU / CG routines). Usually the
      parallel BLAS is will be already installed automatically. Try to not use 
      both parallelisation techniques simultaneously unless properly
      configured for instance via a batch script (e.g. `SLURM`). 
      *Overloading the CPUs will usually result in decreased calculation speed.*
    
    - see :func:`.scatter` for main documentation
    
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    nprocs, rank = comm.Get_size(), comm.Get_rank()

    

    ## --- create list of jobs and split in equal parts depending on `nprocs`
    def split(jobs, nprocs):
        return [jobs[i::nprocs] for i in range(nprocs)]
    
    if comm.rank == 0:
        if nprocs == 1:
            warnings.warn("Executing only one MPI process! Should be run using" +
                          " e.g. 'mpirun -n X python scriptname.py', where X" +
                          " is the number of parallel processes.")
        if verbose: 
            print("")
            print("number of MPI processes:        ", nprocs)
            print("number of wavelengths:          ", len(sim.efield.wavelengths))
            print("number of wavelengths / process:", int(np.ceil(len(sim.efield.wavelengths) / float(nprocs))))
            print("")
        
        if verbose: print("Generating and splitting jobs... ", end='')
        jobs_all = []
        for i, wl in enumerate(sim.efield.wavelengths):
            ## --- generate simulation objects for each individual wavelength
            from . import fields
            _sim = copy.deepcopy(sim)
            _efield = fields.efield(_sim.efield.field_generator, [wl], _sim.efield.kwargs)
            _efield.setDtype(_sim.dtypef, _sim.dtypec)
#            _sim.efield.wavelengths = [wl]
            _sim.efield = copy.deepcopy(_efield)
            jobs_all.append(_sim)
        jobs = split(jobs_all, nprocs)
        if len(np.unique([len(i) for i in jobs])) > 1:
            warnings.warn("Efficiency warning: Number of wavelengths ({}) ".format(
                                                len(sim.efield.wavelengths)) + 
                          "not divisable by Nr of processes ({})!".format(nprocs))
        if verbose: print("Done.")
    else:
        jobs = None
    
    
    ## --- Scatter jobs across processes and perform GDM simulations for each wavelength
    jobs = comm.scatter(jobs, root=0)
    
    results = []
    for job in jobs:
        if verbose: print(" process #{}: Calculating wavelength".format(rank) + \
                               " {}nm".format(job.efield.wavelengths[0]))
        scattered_field = scatter(job, verbose=scatter_verbose, **kwargs)
        for _scatf in scattered_field:
            results.append(_scatf)
    
    
    ## --- Gather results on rank 0
    results = MPI.COMM_WORLD.gather(results, root=0)
    
    if comm.rank == 0:
        if verbose: print("All simulations done. Recombining... ", end='')
        ## --- recombine data and sort by wavelength
        results = [i for temp in results for i in temp]
        results = sorted(results, key=lambda k: k[0]['wavelength'])
        sim.E = results
        if verbose: print("Done.")
    
    return results
    




# =============================================================================
# Decay rate calculation
# =============================================================================
def decay_rate(sim, method='lu', matrix_setup='fortran', verbose=False):
    """Calculate the decay rate of a dipole emitter
    
    Calculate the propagator necessary to derive the change of decay rate of
    an electric or magnetic dipolar emitter in the vicinity of a photonic
    nanostructure.
    The result of this routine can be used to calculate the actual decay rate
    for a dipole transition of any orientation / amplitude using 
    :func:`.linear.decay_eval`.
    
    Parameters
    ----------
      sim : :class:`.simulation`
        simulation description
          
      method : string, default: "lu"
        inversion method. One of ["lu", "numpyinv", "scipyinv", "pinv2", "superlu", "cg", "pycg", "dyson"]
         - "lu" LU-decomposition (`scipy.linalg.lu_factor`)
         - "numpyinv" numpy inversion (`np.linalg.inv`, if numpy compiled accordingly: LAPACK's `dgesv`)
         - "scipyinv" scipy default inversion (`scipy.linalg.inv`)
         - "pinv2" is usually slower (req. `scipy.linalg.pinv2`)
         - "superlu" often has a <N^3 efficieny but is very expensive in memory (`scipy.sparse.linalg.splu`)
         - "dyson" uses a dyson sequence. Only solver that does not depend on external libraries
         - "skcuda" uses CUDA GPU via `pycuda` and `scikit-cuda`
         - "cupy" uses CUDA GPU via `cupy`
         - "cuda" (equivalent to "skcuda")
         
      matrix_setup : str, default: "fortran"
        Which routine to use for setup of the coupled dipole matrix.
        One of ["fortran", "numba", "cuda"]
      
      verbose : bool default=False
          print some info
    
    
    Returns
    -------
      list of lists:  each element of the main list contains [wavelength, S_P].
              `S_P` again is a list of the 3x3 tensors S_P^EE (or S_P^BB) for 
              each map-position (EE: electric dipole, BB: magnetic dipole)
      
    
    Notes
    -----
    `sim` must contain either an electric or magnetic dipole emitter as 
    fundamental field. Its orientation and amplitude will however be ignored.
    These need to be specified later, when the actual evaluation is done using
    :func:`.linear.decay_eval`
    
    For detailed information about the underlying formalism, see:
    Wiecha, P. R., Girard, C., Cuche, A., Paillard, V. & Arbouet, A. 
    **Decay Rate of Magnetic Dipoles near Non-magnetic Nanostructures.** 
    Phys. Rev. B 97(8), 085411 (2018)

    """
    from .pyGDMfor import propa_sp_decay as forDecay
    
    if method.lower() not in ["numpyinv", "scipyinv", "lu", "superlu", 
                              "pinv2", "dyson", "cupy", "skcuda", "cuda"]:
        raise ValueError('Error: Unknown solving method. Must be one of' +
                         ' ["lu", "numpyinv", "scipyinv", "pinv2",' +
                         ' "superlu", "dyson", "cuda", "cupy"].')
    
    field_generator = sim.efield.field_generator
    wavelengths = sim.efield.wavelengths
    xyz = [[pos['x0'], pos['y0'], pos['z0']] for pos in sim.efield.kwargs_permutations]
    xmap, ymap, zmap = np.transpose(xyz)
    
    dp_orient = [[pos['mx'], pos['my'], pos['mz']] for pos in sim.efield.kwargs_permutations]
    if len([x for x in dp_orient if dp_orient.count(x) >= 2]) > 1:
        warnings.warn("The dipole orientation is configured multiple times. " + 
                      "This setting will be ignored by `core.decay_rate`. " +
                      "Please use `linear.decay` to compute the decay-rates for " +
                      "several dipole orientations.")
        
    from . import fields
    if field_generator == fields.dipole_electric:
        magnetic = 0
        dipole_type = 'electric'
    elif field_generator == fields.dipole_magnetic:
        magnetic = 1
        dipole_type = 'magnetic'
        if sim.struct.n1 != sim.struct.n2 or sim.struct.n2 != sim.struct.n3:
            warnings.warn("Substrate or cladding layers are not supported in " + 
                                "magnetic dipole decay rate simulations until " +
                                "now. No surface term will be used.")
    else:
        raise ValueError("Wrong incident field: `decay_rate` requires the " + 
                         "incident field to be " + 
                         "either an electric or a magnetic dipole emitter. " +
                         "Please use `fields.dipole_electric` or " +
                         "`fields.dipole_magnetic`.")
    
    xm, ym, zm = sim.struct.geometry.T
    
    SBB_spectral = []
    for i_wl, wavelength in enumerate(wavelengths):
        if verbose: 
            t0=time.time()
            print("Wavelength: {}nm - {} dipole".format(wavelength, dipole_type))
        
        ## --- evaluate polarizabilities and normalization terms at "wavelength"
        sim.struct.getNormalization(wavelength)
        alpha = sim.struct.getPolarizability(wavelength)
        
        K = get_general_propagator(sim, method=method.lower(), 
                                   matrix_setup=matrix_setup, wavelength=wavelength)
            
        ## --- generalized propagator: explicit matrix
        if method.lower() == 'superlu':
            K = K.solve(np.identity(3*len(xm)))
        elif method.lower() == 'lu':
            import scipy.linalg as la
            K = la.lu_solve(K, np.identity(3*len(xm)))
        elif method.lower() in ["skcuda", "cupy", 'cuda']:
            K = K.get()
        

        ## --- decay-rate using multi-threaded fortran routine
        SP = forDecay(sim.struct.step, xm,ym,zm, 
                      sim.struct.n1, sim.struct.n2, sim.struct.n3,
                      wavelength, 
                      xmap, ymap, zmap,
                      magnetic, K, alpha)
        
        SBB = [SP[i].reshape( 3,3 ) for i in range(len(SP))]
        
        if verbose:
            print("  - decay repropagation:  {:.3f}s".format(time.time()-t0))
            t0=time.time()
            
        SBB_spectral.append([wavelength, SBB])
    
    sim.S_P = SBB_spectral
    
    return SBB_spectral
    
    
























#==============================================================================
# Matrix operations
#==============================================================================
def get_side_by_side(sim, wavelength, invertible=True, times_alpha=True, 
                         as_csc=False):
    """Build side-by-side matrix CA0 for the structure at given wavelength
    
    Parameters
    ----------
    sim : :class:`.simulation`
        simulation description
    
    wavelength: float
        Wavelength at which to calculate susceptibility matrix (in nm)
        
    invertible: bool, default: True
        return invertible matrix (I-CA0)
        
    times_alpha: bool, default: True
        multiply by alpha. If True: CA0 = alpha.G; else CA0 = G
        
    as_csc: bool, default: True
        return as csc sparse matrix format
    
    Returns
    -------
    CA0 : array-like
        return the side-by-side matrix CA0 ("S"). 
        Inverse of CAM0=(I-CA0) is generalized propagator
    
    
    Notes
    -----
    For the analytical expression of the asymptotic propagators with substrate, 
    see e.g.: 
    Girard, C. **Near fields in nanostructures.** 
    Reports on Progress in Physics 68, 1883–1933 (2005).
    """
    from .pyGDMfor import setupmatrix as forSetupmatrix
    
    xm, ym, zm = sim.struct.geometry.T

    CA0 = forSetupmatrix(alambda=wavelength, space=sim.struct.spacing, 
                          xm=xm,ym=ym,zm=zm,
                          cap=sim.struct.alpha,
                          cn1=sim.struct.n1, cn2=sim.struct.n2, cn3=sim.struct.n3,
                          cnorm=sim.struct.cnorm,
                          timesalpha=times_alpha,
                          propagator=1) 
    
    N = np.shape(CA0)[0]
    if not invertible:
        CA0 = np.asfortranarray(np.identity(N) - CA0, dtype=sim.dtypec)
    if as_csc:
        from scipy.sparse import csc_matrix
        CA0 = csc_matrix(CA0, shape=(N, N))
    
    return CA0





        
def get_general_propagator(sim=None, CAM0=None, wavelength=None, 
                           method='lu', matrix_setup='fortran', 
                           return_linear_operator=False, return_susceptibility=False,
                           verbose=False):
    """Invert Matrix CAM0
    
    Parameters
    ----------
    sim : :class:`.simulation`
        simulation description
    
    CAM0 : array-like
        alternative to `sim`: Matrix to invert (I-CA0). As returned by :func:`.get_side_by_side` 
        ("dyson" calculates CAM0 itself, may be set to `None` in that case)
           
    wavelength: float
        Wavelength at which to calculate susceptibility matrix (in nm)
    
    method : string, default: "lu"
        inversion method. One of ["lu", "numpyinv", "scipyinv", "pinv2", "superlu", "cg", "pycg", "dyson"]
         - "lu" LU-decomposition (`scipy.linalg.lu_factor`)
         - "numpyinv" numpy inversion (`np.linalg.inv`, if numpy compiled accordingly: LAPACK's `dgesv`)
         - "scipyinv" scipy default inversion (`scipy.linalg.inv`)
         - "pinv2" is usually slower (req. `scipy.linalg.pinv2`)
         - "superlu" often has a <N^3 efficieny but is very expensive in memory (`scipy.sparse.linalg.splu`)
         - "dyson" uses a dyson sequence. Only solver that does not depend on external libraries
         - "skcuda" uses CUDA GPU via `pycuda` and `scikit-cuda`
         - "cupy" uses CUDA GPU via `cupy`
         - "cuda" (equivalent to "skcuda")
        
    matrix_setup : str, default: "fortran"
        Which routine to use for setup of the coupled dipole matrix.
        One of ["fortran", "numba", "cuda"]
        
    return_linear_operator : bool, default: True
        return K as LinearOperator Class
    
    return_susceptibility : bool, default: False
        return the field susceptibility S inside the structure instead of the 
        generalized propagator (K = 1 + chi*S)
    
    
    Returns
    -------
      - K: Generalized Propagator
    
    
    Notes
    -----
    For details on the concept of the generalized propagator, see e.g.:
    Martin, O. J. F. & Girard, C. & Dereux, A. **Generalized Field Propagator 
    for Electromagnetic Scattering and Light Confinement.**
    Phys. Rev. Lett. 74, 526–529 (1995).
    """
    if return_susceptibility and method.lower() != 'dyson':
        raise ValueError("Field susceptibility can for the moment only be "+
                         "calculated using the 'dyson' solver. Please use 'dyson'.")
        
    if sim is None and CAM0 is None:
        raise Exception("One of the following kwargs must be given: `sim` or `CAM0`.")
    if sim is not None and CAM0 is not None:
        raise Exception("Only one of the following kwargs must be given, not both: `sim` or `CAM0`.")
    
# =============================================================================
#     setup coupling matrix
# =============================================================================
    t0 = time.time()
    if type(CAM0) != np.ndarray:
        ## --- evaluate polarizabilities and normalization terms at "wavelength"
        sim.struct.getPolarizability(wavelength)
        sim.struct.getNormalization(wavelength)
        
        ## --- construct matrix
        if method.lower() == "cuda":
            method = "skcuda"
        if method.lower() in ["lu", "numpyinv", "scipyinv", "superlu", 
                                                "pinv2", "skcuda", "cupy"]:
            as_csc = True if method.lower() == 'superlu' else False
            if matrix_setup.lower() == 'numba':
                CAM0 = get_SBS_numba(sim, wavelength=wavelength, invertible=True, 
                                     times_alpha=True, as_csc=as_csc)                
            elif matrix_setup.lower() == 'cuda':
                CAM0 = get_SBS_cuda(sim, wavelength=wavelength, invertible=True, 
                                    times_alpha=True, as_csc=as_csc)                
            else:
                if matrix_setup.lower() != 'fortran':
                    warnings.warn("Unknown `matrix_setup`. Must be one of ['fortran', 'numba', 'cuda']. Falling back to `fortran`.")
                CAM0 = get_side_by_side(sim, wavelength=wavelength, invertible=True, 
                                    times_alpha=True, as_csc=as_csc)       
        else:
            CAM0 = None
        
        if verbose: 
            print('timing for wl={:.2f}nm - setup: {:.1f} ms, '.format(
                                wavelength, 1000.*(time.time()-t0)), end='')
    else:
        warnings.warn("Deprecation warning: `get_general_propagator` will no longer support matrix-input in future versions of pyGDM. Please update your code and use the `sim` class as input to `get_general_propagator`.")
    t0b = time.time()
    
# =============================================================================
#     the actual inversion (via chosen method)
# =============================================================================
    ## --- superlu-LU decomposition (scipy.sparse)
    if method.lower() in ["superlu"]:
        import scipy.sparse.linalg as la
        K = la.splu(CAM0)

        ## --- superLU-object can be transformed to a scipy linearoperator by
        if return_linear_operator: 
            _dtype = CAM0.dtype
            N = CAM0.shape[0]
            K = la.LinearOperator(shape=(N, N), matvec=K.solve, dtype=_dtype)
            del CAM0; gc.collect()
    

    ## --- standard scipy.linalg inversion methods
    elif method.lower() in ["numpyinv", "scipyinv", "lu", "pinv2"]:
        if method.lower() in ["numpyinv"]:     # pyGDM2 default
            K = np.linalg.inv(CAM0)
        elif method.lower() in ["scipyinv"]:
            import scipy.linalg as la
            K = la.inv(CAM0, overwrite_a=True)
        elif method.lower() in ["lu"]:
            import scipy.linalg as la
            K = la.lu_factor(CAM0, overwrite_a=True)
        elif method.lower() in ["pinv2"]: 
            import scipy.linalg as la
            K = la.pinv2(CAM0)
        del CAM0; gc.collect()
    
    
    ## --- Dyson sequence (openMP - fortran routine)
    elif method.lower() == "dyson":
        from .pyGDMfor import dysonseq as forDysonSeq
        
        if sim is None:
            raise ValueError("Instance of core.simulation must be provided if using 'dyson' solver.")
        if return_susceptibility:
            return_susceptibility = 1
        else:
            return_susceptibility = 0
        xm, ym, zm = sim.struct.geometry.T
        CAM0 = forDysonSeq(wavelength, sim.struct.spacing, 
                              xm,ym,zm, sim.struct.alpha,
                              sim.struct.n1, sim.struct.n2, sim.struct.n3,
                              sim.struct.cnorm, returnsusc=return_susceptibility)
        
        if return_linear_operator:
            import scipy.sparse.linalg as la
            K = la.aslinearoperator(CAM0)
            del CAM0; gc.collect()
        else:
            K = CAM0
    
    ## --- CUDA based inversion on GPU viua `skcuda` or `cupy` - EXPERIMENTAL
    elif method.lower() in ["skcuda", "cuda"]:
        import pycuda.autoinit
        import pycuda.gpuarray as gpuarray
        import skcuda.linalg as culinalg
        culinalg.init()
        ## move array to cuda device, cuda inversion
        Agpu = gpuarray.to_gpu(np.ascontiguousarray(CAM0))
        Ainv_gpu = culinalg.inv(Agpu, overwrite=True)
        K = Ainv_gpu
    elif method.lower() == "cupy":
        import cupy as cp
        import cupy.linalg
        ## move array to cuda device, cuda inversion
        Agpu = cp.array(np.ascontiguousarray(CAM0))
        Ainv_gpu = cupy.linalg.inv(Agpu)
        K = Ainv_gpu
    else:
        raise ValueError('Invalid inversion method. Must be one of ["numpyinv", "scipyinv", "lu", "pinv2", "superlu", "dyson"].')
    
    if verbose: 
        print('inv.: {:.1f} ms, '.format(1000.*(time.time()-t0b)), end='')
    
    return K



def get_efield_by_cg(CAM0, E0, pcTol=2E-3, fill_factor=15, cgsTol=1E-2, 
                  maxiter=200, PC=None, method='cg', pc_method='ilu', 
                  verbose=False):
    """Invert Matrix CAM0
    
    Parameters
    ----------
    CAM0 : array-like
        Matrix to invert (I-CA0). As returned by get_side_by_side. 
    
    E0 : array-like
        Incident electric field, 3N super-vector 
        (you may use :func:`._fieldListToSuperVector`)
    
    pcTol, cgsTol : float, float, defaults: 2E-3 and 1E-2
        tolerances for preconditioning / conjugate gradients
        pcTol > 1: Don't use preconditioning
    
    maxiter : int, default: 500
        Max. iterations for CG
        
    fill_factor : int, default: 15
        Preconditioning Parameter
    
    PC : array-like, default: None
        Optional Preconditer Matrix. If not None, overrides preconditioning.
    
    method : string, default: 'cg'
        iterative solver to use. 
        'cg': scipy stab-cg; 'pycg' pyAMG implementation of stab-cg (threadsafe)
    
    pc_method : string, default: 'ilu'
        Preconditioning method. 
         - 'ilu': scipy incomplete LU, 
         - 'lu': scipy complete LU (exact, but speed-improvements for close 
           wavelengths when recycling), 
         - 'amg' pyAMG amg, 
         - 'none': No preconditioning
    
    Returns
    -------
    E : array-like
    M : array-like, STATUS : int
    
    - E :       Solution of CAM0*E = E0 (3N supervector)
    - M :       Used Preconditioner Matrix (for possible re-use)
    - STATUS :  Status of conj. gradient. (0: converged, !=0: not converged)
    """
    if method.lower() in ["pycg"]:
        from pyamg.krylov._bicgstab import bicgstab as cgsAMG
        cgSolver = cgsAMG
    elif method.lower() in ["cg"]:
        from scipy.sparse.linalg import bicgstab as scipycgs
        cgSolver = scipycgs
    else:
        raise ValueError("Error: Unknown solver for CG.")
        
    N = np.shape(CAM0)[0]
    DTYPE = CAM0.dtype
    
    ## preconditioning
    if PC==None and pcTol > 1. or (pc_method.lower() in ['none', 'false']):
        M = np.identity(N)
    elif PC==None and pcTol <= 1.:
        if verbose: print("Preconditioning...")
        if pc_method.lower() == 'ilu':
            import scipy.sparse.linalg as la
            PCILU = la.spilu(CAM0, drop_tol=pcTol, fill_factor=fill_factor, drop_rule='basic, area')
            M = la.LinearOperator(shape=(N, N), matvec=PCILU.solve, dtype=DTYPE)
        elif pc_method.lower() == 'lu':
            import scipy.sparse.linalg as la
            PCLU = la.splu(CAM0) ## EXACT!!
            M = la.LinearOperator(shape=(N, N), matvec=PCLU.solve)
        elif pc_method.lower() == 'amg':
            from pyamg import smoothed_aggregation_solver
            ml = smoothed_aggregation_solver(CAM0, coarse_solver="pinv2")
            M = ml.aspreconditioner(cycle='V')
    else:
        if verbose: print("Re-use PC...")
        M = PC
    
    if verbose: print("Iterate conjugate gradients...")
    E, STATUS = cgSolver(CAM0, E0, M=M, tol=cgsTol, maxiter=maxiter)
    del CAM0; gc.collect()
    return E, M, STATUS











#==============================================================================
# Internal Helper Functions
#==============================================================================
def _superVectorToFieldList(E):
    """convert complex 3N supervector E to list of N field tuples (Ex,Ey,Ez)"""
    return np.reshape(E, (int(len(E)/3), 3))


def _fieldListToSuperVector(E):
    """convert list of N field tuples (Ex,Ey,Ez) to complex 3N supervector E"""
    return np.reshape(E, (np.product(E.shape)))


def _complexToFortranFloatFieldList(efield, dtype='f'):
    """Convert complex to fortran-compatible float valued lists of Re/Im values
    
    Parameters
    ----------
    efield : np.array of field tuples, complex
        efield list as returned by "scatter" simulation
    
    Returns
    -------
    EX1,EX2, EY1,EY2, EZ1, EZ2 : lists
        1: real part, 2: imag part
    """
    EX, EY, EZ = efield.T
    
    EX1 = np.asfortranarray(EX.real, dtype=dtype)
    EX2 = np.asfortranarray(EX.imag, dtype=dtype)
    EY1 = np.asfortranarray(EY.real, dtype=dtype)
    EY2 = np.asfortranarray(EY.imag, dtype=dtype)
    EZ1 = np.asfortranarray(EZ.real, dtype=dtype)
    EZ2 = np.asfortranarray(EZ.imag, dtype=dtype)
    
    return EX1,EX2, EY1,EY2, EZ1,EZ2


def _sortByParameterConfig(sim, sortList):
    """Sort `sortList` by incident field kwargs configurations using definitions in `sim`
    
    Parameters
    ----------
    sim : :class:`.simulation`
        simulation describtion
    
    sortList : list
        list of fields sorted by incident-field-parameters such as returned e.g. 
        by `pyGDM2.core.scatter`
    
    Returns
    -------
    sortedResults : list of lists
        each list contains a full spectrum with otherwise identical 
        first element, the latter being the incident field's parameter-set
        
        
    """
    Npermutations = len(sortList) / len(sim.efield.wavelengths)
    sortedFields = [ [] for i in range(Npermutations) ]
    
    for i, e in enumerate(sortList):
        sortedFields[i%Npermutations].append(e)
    
    return sortedFields


def _insertInSpectralEfieldFortran(efield, E, ilambda):
    """insert E in spectral efield-array at position (ilambda)
    
    Use for passing field data to parallelized fortran routines.
    - Shape of 'efield': (3, Ndp, Nlambda), complex.
    - Shape of 'E': list of 3-tuples, complex. 
        Example: E = [[Ex1,Ey1,Ez1],  [Ex2,Ey2,Ez2],  ...].
    
    Parameters
    ----------
    efield : np.array, complex
        Spectral E-field array as required by some parallelized fortran
        routines (shape: (3, Ndp, Nlambda, Ntheta), complex)
    
    E : list of 3-tuples, complex
        E-Field, list fo complex 3-tuples: [[Ex1,Ey1,Ez1],  [Ex2,Ey2,Ez2],  ...]
    
    ilambda : int
        index of wavelength to insert in efield 
    
    Returns
    -------
    efield : np.array, complex (shape: (3, Ndp, Nlambda))
        Efield list where field at position 'ilambda' is replaced by E
    """
    S = np.shape(efield)
    if not (0 <= ilambda < S[2]):
        raise ValueError("Invalid Index for Wavelength! Must be Nlambda >= ilambda > 0.")
    
    Ex, Ey, Ez = E.T
    
    for i,(ex,ey,ez) in enumerate(zip(Ex,Ey,Ez)):
        efield[0][i][ilambda] = ex
        efield[1][i][ilambda] = ey
        efield[2][i][ilambda] = ez
        
    return np.asfortranarray(efield, dtype=ex.dtype)









# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # # NUMBA 
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================





#%%

# =============================================================================
# CPU - side-by-side matrix setup
# =============================================================================
def get_SBS_numba(sim, wavelength, invertible=True, times_alpha=True, 
                         as_csc=False):
    # =========================================================================
    # create CUDA propagator and kernel
    # =========================================================================
    from pyGDM2.propagators import t_sbs
    
    ##  --- simulation config
    geo = sim.struct.geometry
    spacing = sim.struct.spacing
    cnorm = sim.struct.getNormalization(wavelength)
    alpha = sim.struct.getPolarizabilityTensor(wavelength)
    eps1 = sim.struct.n1**2
    eps2 = sim.struct.n2**2
    eps3 = sim.struct.n3**2
    

    ## --- run numba routine
    M = np.zeros((len(geo)*3,len(geo)*3), dtype=np.complex64)
    t_sbs(geo, wavelength, cnorm, eps1, eps2, eps3, spacing, alpha, M)
    
    
    N = np.shape(M)[0]
    if not invertible:
        M = np.asfortranarray(np.identity(N) - M, dtype=sim.dtypec)
    if as_csc:
        from scipy.sparse import csc_matrix
        M = csc_matrix(M, shape=(N, N))
    
    return M






# =============================================================================
# GPU-CUDA - side-by-side matrix setup
# =============================================================================
def get_SBS_cuda(sim, wavelength, invertible=True, times_alpha=True, 
                 as_csc=False, threadsperblock=(16,16)):
    # =========================================================================
    # create CUDA propagator and kernel
    # =========================================================================
    import math
    from numba import cuda
    from pyGDM2.propagators import G0, Gs123


    ## --- cuda version of propagator
    G0_cuda = cuda.jit(device=True)(G0)
    Gs123_cuda = cuda.jit(device=True)(Gs123)
    
    ## --- cuda kernel
    @cuda.jit()
    def t_sbs_kernel_cuda(geo, lamda, cnorm, eps1, eps2, eps3, spacing, alpha, M):
        """
        Code for kernel.
        """
        i, j = cuda.grid(2)    # indices of current cuda thread
        if i < M.shape[0]/3 and j < M.shape[1]/3:
            R1 = geo[i]
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
                xx, yy, zz, xy, xz, yx, yz, zx, zy = G0_cuda(R1, R2, lamda, eps2)
            
            ## --- 1-2-3 surface dyad
            xxs,yys,zzs,xys,xzs,yxs,yzs,zxs,zys = Gs123_cuda(R1, R2, lamda, 
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


    
    ##  --- simulation config
    geo = sim.struct.geometry
    spacing = sim.struct.spacing
    cnorm = sim.struct.getNormalization(wavelength)
    alpha = sim.struct.getPolarizabilityTensor(wavelength)
    eps1 = sim.struct.n1**2
    eps2 = sim.struct.n2**2
    eps3 = sim.struct.n3**2
    
    ## --- cuda threads config
    M = np.zeros((len(geo)*3,len(geo)*3), dtype=np.complex64)
    blockspergrid_x = math.ceil((M.shape[0]/3) / threadsperblock[0])
    blockspergrid_y = math.ceil((M.shape[1]/3) / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    ## -- run cuda kernel
    cuda.select_device(0)
    t_sbs_kernel_cuda[blockspergrid, threadsperblock](geo, wavelength, 
                                     cnorm, eps1, eps2, eps3, spacing, alpha, M)
    cuda.close()
    
    
    N = np.shape(M)[0]
    if not invertible:
        M = np.asfortranarray(np.identity(N) - M, dtype=sim.dtypec)
    if as_csc:
        from scipy.sparse import csc_matrix
        M = csc_matrix(M, shape=(N, N))
        
    return M




#%%
if __name__ == "__main__":
    pass
    

