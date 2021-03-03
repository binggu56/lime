# encoding: utf-8
"""
Main routines of EO submodule of pyGDM2

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
    
    
Includes:
    
    Functions to run and continue evolutionary optimizations of 
    nano-structures for nano-optical problems

"""
from __future__ import print_function
from __future__ import absolute_import

import six; from six.moves import cPickle as pickle

import os
import sys
import time
import warnings

from PyGMO import migration, archipelago, island, topology, algorithm
from PyGMO import population as pygmo_population
import numpy as np




########################################################################
def _plot_print_structure(champ, problem, runtime_plotting):
    """Internal function for runtime plotting and printing of best opt. solution"""
    
    parameters = champ.x
    problem.model.generate_structure(parameters)
    
    problem.model.print_info()
    
    if runtime_plotting:
        import matplotlib.pyplot as plt
        problem.model.plot_structure(interactive=True)
        plt.pause(.1)
    
    
    
########################################################################
def continue_eo(results_folder, results_suffix,
                  max_time=3600, max_iter=200, max_nonsuccess=20, 
                  n_report=10,
                  runtime_plotting=False,
                  save_each_generation=False,
                  verbose=True):
    """Continue evolution, re-loading saved optimization
    
    Parameters:
    ---------------
    results_folder : str
        output folder of saved optimization
    
    results_suffix : str
        output file suffix
    
    other kwargs :
        all other arguments are explained in `pyGDM2.EO.core.do_eo`
    """
    
    do_eo(results_folder, results_suffix,
          problem=None,
          max_time=max_time, max_iter=max_iter, max_nonsuccess=max_nonsuccess, 
          n_report=n_report,
          topology=None, s_pol=None, r_pol=None, algo2=None, 
          ## --- flags
          runtime_plotting=runtime_plotting,
          save_each_generation=save_each_generation,
          continue_eo=True,
          verbose=verbose)



########################################################################
def do_eo(results_folder, results_suffix,
          problem, 
          ## --- default optimization algorithm configuration
          algo=algorithm.jde(gen=1, memory=True), 
          islands=-1, population=20,
          max_time=3600, max_iter=200, max_nonsuccess=20, 
          n_report=10,
          ## --- topology and migration policies, if doing multi-thread optimization. Optinal: alternative EO algorithm
          topology=topology.fully_connected(), 
          s_pol=migration.best_s_policy(0.3, migration.rate_type.fractional),
          r_pol=migration.fair_r_policy(0.3, migration.rate_type.fractional),
          algo2=None, 
          ## --- flags
          runtime_plotting=False,
          save_each_generation=False,
          continue_eo=False,
          verbose=True):
    """Run evolutionary optimization of nano-structure geometry for nano-optics
    
    -----    
    allows to continue former evolution.
    In case an evolution is continued, "prob", "structure", "algo", "ISLANDS", 
    "POPULATION" arguments are ignored and the formerly used objects 
    are loaded from disc.
    
    For this, "problem", "archipelago", "structure" and "halloffame" files are 
    required on the disc.
    
    if given, "algo2" will be used on every second island.
    
    Parameters:
    ---------------
    results_folder : str
        output folder to save results
    
    results_suffix : str
        output file suffix
        
    problem : class inheriting from `pyGDM2.EO.problems.BaseProblem`
        problem definition including the structure-model and pyGDM simulation setup
    
    algo : PyGMO.algorithm instance, default: `jde`
        EO algorithm
    
    islands : int, default: -1
        number of islands to evolve in parallel. '-1': N_cpu
    
    population : int, default: 20
    
    max_time : int, default: 3600
        stop criterion: maximum time in seconds
    
    max_iter : int, default: 200
        stop criterion: maximum number of iterations
    
    max_nonsuccess : int, default: 20
        stop criterion: maximum consecutive iterations without improvement
    
    n_report : int, default: 10
        report every n_report iterations (plot+print best structure)
        If 0: disable print best structure details
    
    topology : `PyGMO.topology` instance, default: topology.fully_connected()
        topology in case of multi-island optimization
    
    s_pol, r_pol : `PyGMO.migration` instance, default: migration.best_s/r_policy(0.3, migration.rate_type.fractional)
        migration policies in case of multi-island optimization
    
    algo2 : PyGMO.algorithm instance, default: None
        alternative evolution algorithm to use on each second island
    
    runtime_plotting : bool, default: False
        enable / disable plotting of optimum structure at runtime
    
    save_each_generation : bool, default: False
        wether or not to save the full population at each iteration. 
        If False, only the best candidate in each iteration will be saved
    
    continue_eo : bool, default: False
        wether to continue an existing optimization. Prefer use of `continue_eo`.
    
    verbose : bool, default: True
        print some more detailed runtime info
        
          
    Note:
    ---------------
    To reload and continue an optimization, "results_folder" and 
    "results_suffix" are used to identify the saved optimization.
    
    For details on the used 'generalized island model', see
    
    1. Biscani, F., Izzo, D. & Yam, C. H.: 
        "A Global Optimisation Toolbox for Massively Parallel Engineering Optimisation." 
        arXiv:1004.3824 [cs, math] (2010).
    
    2. Izzo, D., Ruciński, M. & Biscani, F.: 
        "The Generalized Island Model"
        in 'Parallel Architectures and Bioinspired Algorithms'
        (eds. Vega, F. F. de, Pérez, J. I. H. & Lanchares, J.) 
        pp. 151–169 (Springer Berlin Heidelberg, 2012).

    
    -----    
    """
    if runtime_plotting:
        import matplotlib.pyplot as plt
    
    if runtime_plotting:
        plt.ion()
        plot_archi = True
    
    if islands > 1 and problem.nthreads != 1:
        warnings.warn("Using multi-island optimization together with multi-threaded problem. Might result in PyGDM freezing due to non-threadsave parts of code.")
    

#==============================================================================
# Prepare optimization: initialize or reload
#==============================================================================
    if continue_eo:
        ## --- load and continue former optimization
        print ('')
        print ('----------------------------------------------' )
        print (' Reload and continue former optimization')
        print ('----------------------------------------------\n' )
        try: 
            problem    = pickle.load( open( os.path.join(results_folder, "EO_{}_final_problem.p".format(results_suffix)), "rb" ) )
            archi      = pickle.load( open( os.path.join(results_folder, "EO_{}_final_archipelago.p".format(results_suffix)), "rb" ) )
            halloffame = pickle.load( open( os.path.join(results_folder, "EO_{}_halloffame.p".format(results_suffix)), "rb" ) )
        except IOError:
            raise IOError("Required files not found! Please check file- and foldernames.")
        
        if save_each_generation:
            try: 
                allGenerations = pickle.load( open( os.path.join(results_folder, "EO_{}_allGenerations.p".format(results_suffix)), "rb" ) )
            except IOError:
                allGenerations = []
                
        print ("Done. \n\n --> Continuing evolution...\n\n")
        
    else:
        ## --- start new optimization
        if islands == -1:
            import multiprocessing
            islands = multiprocessing.cpu_count()
        elif islands > 1:
            import multiprocessing
            if islands > multiprocessing.cpu_count():
                warnings.warn("Innefficiency warning: Using more islands for evolution than avaiable processors.")
           
        print ('')
        print ('----------------------------------------------' )
        print (' Starting new optimization')
        print ('----------------------------------------------\n\n' )
        print ("Generate archipelago for populations...", end='')
        t0 = time.time()
        
        halloffame=[]
        if save_each_generation:
            allGenerations = []
        
        ## --- define evolution environment
        archi = archipelago(topology=topology)
        isls = []
        for i in xrange(islands):
            alg = algo
            if i%2==1 and algo2 is not None: 
                alg = algo2
            isls.append( island(alg, problem, population, s_policy=s_pol, r_policy=r_pol) )
        print ("Done in {:.1f}s.".format(time.time()-t0))
        
        print ("\nDistribute islands on archipelago...", end='')
        t0 = time.time()
        for isl in isls:
            archi.push_back(isl)
        print ("Done in {:.1f}s.".format(time.time()-t0))
        print ("\n", algo)
        if algo2:
            print ("\n --- alternate algorithm on each second island:", algo2)
        
        
        ## --- save problem definition and structure-model
        pickle.dump( problem, open( os.path.join(results_folder, "EO_{}_final_problem.p".format(results_suffix)), "wb" ) )
        
        print ("\nInitialization finished. \n\n --> starting evolution...\n\n")
    
     
    
#==============================================================================
# Run the optimization
#==============================================================================
    ## --- fittest candidate: 'champ'
    champ = sorted([[isl.population.champion.f, isl.population.champion] for isl in archi])[0][1]
    
    i_iter = 0
    i_progress = 0
    i_nonsuccess = 0
    t0 = time.time()
    
    ## --- Main evolution loop
    while int(time.time()-t0)<max_time and i_iter<max_iter and i_nonsuccess<max_nonsuccess:
        lastChamp = champ.f
        i_iter += 1
        i_nonsuccess += 1
        
        ## --- evolve entire population
        archi.evolve(1)
        champs = [[isl.population.champion.f, isl.population.champion] for isl in archi]
        champ = sorted(champs)[0][1]
        
        
        ## --- optinally: save whole generation
        if save_each_generation:
            populationCurrentGen = []
            if len(archi)>1:
                for isl in archi:
                    for ind in isl.population:
                        populationCurrentGen.append([ind.best_f, ind.best_x])
            else:
                populationCurrentGen = archi[0].population
            allGenerations.append(populationCurrentGen)
        
        ## --- print status
        f_string = ["{:.5g}".format(_f) for _f in champ.f]
        if verbose:
            print ('\r#i={}, t={}s, best fitness: {}       '.format( i_iter, int(time.time()-t0), f_string))
        
        ## --- if new champion or each 'n_report' iterations, save evolution
        if not (lastChamp == champ.f):
            i_nonsuccess = 0
            i_progress += 1
            print ("  --> New champion (#iter={}, #imprv={}, f={}), saving... ".format(i_iter, i_progress, f_string), end='')
            
            halloffame.append([i_iter, champ.f, champ.x])
            pickle.dump( archi, open( os.path.join(results_folder, "EO_{}_final_archipelago.p".format(results_suffix)), "wb" ) )
            pickle.dump( halloffame, open( os.path.join(results_folder, "EO_{}_halloffame.p".format(results_suffix)), "wb" ) )
            if save_each_generation:
                pickle.dump( allGenerations, open( os.path.join(results_folder, "EO_{}_allGenerations.p".format(results_suffix)), "wb" ) )
            print ('Done.')
        
        ## --- report best candidate
        if n_report != 0:
            if (i_iter%n_report==0):
                print ("\n")
                print ("-------------------------------------------------")
                print ("  --> {} iterations. Report on best solution".format(n_report))
                print ("-------------------------------------------------")
                _plot_print_structure(champ, problem, runtime_plotting)
                if runtime_plotting and len(archi)>1 and plot_archi:
                    try:
                        print ("\n Archipelago (the more red the better solutions on island)")
                        archi.draw()
                    except ImportError:
                        warnings.warn("Failed plotting archipelago network. This requires `networkx` python module.")
                        plot_archi = False
                print ("-------------------------------------------------")
                print ("\n")
                
        sys.stdout.flush()
    
    
#==============================================================================
# If stop-criterion met: finish and save all results
#==============================================================================
    ## --- Join parallel threads (if using archipelage)
    archi.join()
    
    ## --- print exit reason
    if int(time.time()-t0) >= max_time:
        print ("Timelimit reached.")
    elif i_iter >= max_iter:
        print ("Max. iterations limit reached.")
    elif i_nonsuccess >= max_nonsuccess:
        print ("Max. non-success limit reached.")
    
    
    ## --- generate final population and save results
    print ("Generate total population... ", end='')
    t0=time.time()
    if len(archi)>1:
        pop = pygmo_population(problem)
        for isl in archi:
            for ind in isl.population:
                pop.push_back(ind.best_x)
    else:
        pop = archi[0].population
    print ("Done in {:.1f}s.".format(time.time()-t0))
    
    
    print ("Saving final data (pickle python objects)...")
    pickle.dump( pop, open( os.path.join(results_folder, "EO_{}_final_population.p".format(results_suffix)), "wb" ) )
    pickle.dump( archi, open( os.path.join(results_folder, "EO_{}_final_archipelago.p".format(results_suffix)), "wb" ) )
    pickle.dump( halloffame, open( os.path.join(results_folder, "EO_{}_halloffame.p".format(results_suffix)), "wb" ) )
    if save_each_generation:
        pickle.dump( allGenerations, open( os.path.join(results_folder, "EO_{}_allGenerations.p".format(results_suffix)), "wb" ) )
    
    
    ## --- if Multiobjective: save and plot Pareto front
    if problem.f_dimension >= 2:
        F = np.array([[ind.cur_f, [int(round(XX)) for XX in ind.cur_x]]  for ind in pop])
        with open( os.path.join(results_folder, "EO_{}_MO_Opt_pareto.txt".format(results_suffix)), 'w') as f:
            f.write(str(F))
    
    if runtime_plotting and problem.f_dimension==2:
        plt.figure()
        pop.plot_pareto_fronts()
        plt.xlabel(r"$f^{(1)}$" +r"($\lambda$1)")
        plt.ylabel(r"$f^{(2)}$" +r"($\lambda$2)")
        
        plt.show()



