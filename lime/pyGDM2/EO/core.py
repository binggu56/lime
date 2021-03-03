# encoding: utf-8
"""
Main routines of EO submodule of pyGDM2

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

import numpy as np
import pygmo as pg



########################################################################
def get_mo_champion(pop):
    """return the champion of `pop`
    
    SO: return: champion_f, champion_x, 1, 1
    MO: return: ideal, nadir, N_pareto_fronts, length_first_front
    
    
    """
    f = pop.get_f()
    NObj = len(f[0])
    
    if NObj == 1: 
        champion_f = pop.champion_f
        champion_x = pop.champion_x
        N_paretofronts = 1
        l_paretofront = 1
    else:
        ndf, dl, dc, ndl = pg.fast_non_dominated_sorting(pop.get_f())
        N_paretofronts = len(ndf)
        l_paretofront = len(ndf[0])
        
        champion_f = pg.ideal(pop.get_f())
        champion_x = pg.nadir(pop.get_f())
        
    
    return champion_f, champion_x, N_paretofronts, l_paretofront
    
    
########################################################################
def continue_eo(filename,
                plot_interval=0, 
                generations=10,
                max_time=3600, max_iter=200, max_nonsuccess=20,
                save_all_generations=False,
                verbose=False):
    """Continue an evolutionary optimization
    
    Parameters
    ----------
    filename : str or dict
        path to file or dict containing the saved optimization
    
    Notes
    -----
    all other arguments are explained in :func:`.run_eo`
    
    """
    return run_eo(problem=None, 
                       plot_interval=plot_interval, 
                       generations=generations,
                       max_time=max_time, 
                       max_iter=max_iter, 
                       max_nonsuccess=max_nonsuccess,
                       filename=filename,
                       save_all_generations=save_all_generations,
                       continue_eo=True,
                       verbose=verbose)
    

########################################################################
def save_eo_dict(eo_dict, filename):
    try:
        pickle.dump(eo_dict, open(filename, "wb"))
    except TypeError:
        warnings.warn("Save EO exception: It seems that 'pop' and 'prob' cannot be pickled. Removing from dict and retry and replacing by x/f values.")
        eo_dict["pop_x"] = eo_dict['pop'].get_x()
        eo_dict["pop_f"] = eo_dict['pop'].get_f()
        eo_dict.pop('prob', None)
        eo_dict.pop('pop', None)
        pickle.dump(eo_dict, open(filename, "wb"))
    


########################################################################


def run_eo(problem=None, 
           population=20,
           algorithm=pg.sade, algorithm_kwargs=dict(),
           plot_interval=0, 
           generations=10,
           max_time=3600, max_iter=200, max_nonsuccess=20,
           filename='',
           save_interval=1,
           save_all_generations=False,
           continue_eo=False,
           verbose=False):
    """run nano-photonic geometry evolutionary optimization
    
    Parameters
    ----------
    problem : class inheriting from :class:`.problems.BaseProblem`
        problem definition including the structure-model and pyGDM simulation setup
    
    population : int, default: 20
        number of individuals in the population
    
    algorithm : pygmo.algorithm object, default: `pygmo.sade`
        the EO algorithm to use. 
    
    algorithm_kwargs : dict, default {}
        optional kwargs, passed to **algorithm**
    
    plot_interval : int, default: 0
        plot each *N* number of improvements. 0 = no plotting
    
    generations : int, default: 10
        number of generations to evolve per iteration (i.e. nr of generations 
        between status reports)
    
    max_time : int, default: 3600
        stop criterion: maximum time in seconds
    
    max_iter : int, default: 200
        stop criterion: maximum number of iterations
    
    max_nonsuccess : int, default: 20
        stop criterion: maximum consecutive iterations without improvement
    
    filename : str, default ''
        path to file. dictionary containing optimization status and results 
        will be saved for later re-use / analysis using pythons `pickle`. 
        Empty string means no run-time saving.
    
    save_interval : int, default: 1
        save results to file each N improvements
    
    save_each_generation : bool, default: False
        wether or not to save the full population at each iteration. 
        If False, only the best candidate in each iteration will be saved
    
    continue_eo : bool, default: False
        wether to continue an existing optimization. 
        We encourag to use the wrapper `continue_eo` instead.
    
    verbose : bool, default: False
        if True, print more detailed runtime info
        
        
    Returns
    -------
    dict : dictionary containing optimization data and log
          
    
    
    Notes
    -----
    To reload and continue an optimization (continue_eo==True), "filename" 
    is used to identify the stored optimization ("filename" can be either 
    a dict as returned by :func:`.run_eo`, or a path to a file containing 
    the pickeled dict).
    
    The *halloffame* in the eo-dictionary contains 
     - for single-objective: [#iter, best_f, best_x]
     - for multi-objective: [#iter, `ideal`, `nadir`]
    
    For details on the underlyign pagmo/pygmo library, see also
    
    [1] https://esa.github.io/pagmo2/
    
    [2] Biscani, F., Izzo, D. & Yam, C. H.: 
        **A Global Optimisation Toolbox for Massively Parallel Engineering Optimisation.** 
        arXiv:1004.3824 [cs, math] (2010).
    
    [3] Izzo, D., Ruciński, M. & Biscani, F.: 
        **The Generalized Island Model**
        in *Parallel Architectures and Bioinspired Algorithms*
        (eds. Vega, F. F. de, Pérez, J. I. H. & Lanchares, J.) 
        pp. 151–169 (Springer Berlin Heidelberg, 2012).

    """
# =============================================================================
# Reload or initialize optimization
# =============================================================================
    if continue_eo:
        print ('\n----------------------------------------------')
        print (' Reload and continue former optimization')
        print ('----------------------------------------------\n')
        
        from .tools import reload_eo
        eo_dict = reload_eo(filename)
        
        problem = eo_dict['problem']
        prob = eo_dict['prob']
        pop = eo_dict['pop']
        algo = eo_dict['algo']
        
        halloffame = eo_dict['halloffame']
        all_gen_best = eo_dict['all_gen_best']
        all_gen_all_f = eo_dict['all_gen_all_f']
        all_gen_pop = eo_dict['all_gen_pop']
        
        i_iter = eo_dict['i_iter']
        i_progress = eo_dict['i_progress']
        i_nonsuccess = 0
                
        print ("Done. \n\n --> Continuing evolution...\n\n")
        
    else:
        if problem is None:
            raise ValueError("Problem not defined!")
        
        print ('\n----------------------------------------------')
        print (' Starting new optimization')
        print ('----------------------------------------------\n\n')
        
        prob = pg.problem(problem)
        pop = pg.population(prob, size=population)
        algo = pg.algorithm(algorithm(gen=generations, **algorithm_kwargs))
        
        halloffame = []
        all_gen_best = []
        all_gen_all_f = []
        all_gen_pop = []
        
        i_iter = 0
        i_nonsuccess = 0
        i_progress = 0
    
    
# =============================================================================
# Main evolution
# =============================================================================
    new_nadir = last_nadir = 0
    new_Npareto = last_Npareto = 0
    new_lpareto = last_lpareto = 0
    t_start = time.time()
    plotted = False
    
    while 1:
        champion_f, champion_x, N_paretofronts, l_paretofront = get_mo_champion(pop)
        last_champ = [float("{:.5g}".format(_f)) for _f in champion_f]
        if problem.get_nobj() != 1:
            last_nadir = [float("{:.5g}".format(_f)) for _f in champion_x]
            last_Npareto = N_paretofronts
            last_lpareto = l_paretofront
        
        i_iter += 1
        i_nonsuccess += 1
        
        
# =============================================================================
# check stop criteria
# =============================================================================
        if time.time() - t_start > max_time:
            print ("\n -------- timelimit reached")
            break
        elif i_iter > max_iter:
            print ("\n -------- maximum interations reached")
            break
        elif i_nonsuccess > max_nonsuccess:
            print ("\n -------- maximum non-successful iterations reached")
            break
        
    
# =============================================================================
# evolve the population
# =============================================================================
        pop = algo.evolve(pop)
        
        ## --- check if improvement was made
        champion_f, champion_x, N_paretofronts, l_paretofront = get_mo_champion(pop)
        new_champ = [float("{:.5g}".format(_f)) for _f in champion_f]
        if problem.get_nobj() != 1:
            new_nadir = [float("{:.5g}".format(_f)) for _f in champion_x]
            new_Npareto = N_paretofronts
            new_lpareto = l_paretofront
        condition_new_champ = ((not (last_champ == new_champ)) or 
                               (problem.get_nobj() != 1 and 
                                     ((not last_nadir == new_nadir)  or
                                      (not last_Npareto == new_Npareto) or
                                      (not last_lpareto == new_lpareto))))
        
        if condition_new_champ or i_iter == 1:
            halloffame.append([i_iter, champion_f, champion_x])
            plotted = False
            i_nonsuccess = 0
            i_progress += 1
        
        if save_all_generations:
            all_gen_best.append([i_iter, champion_f, champion_x])
            all_gen_all_f.append([i_iter, [pop.get_f()[i] for i in range(len(pop))]])
            all_gen_pop.append([i_iter, pop])
        
        ## --- bundle information
        eo_dict = dict(problem=problem, prob=prob, pop=pop, algo=algo,
                       halloffame=halloffame, 
                       all_gen_best=all_gen_best,
                       all_gen_all_f=all_gen_all_f,
                       all_gen_pop=all_gen_pop,
                       i_iter=i_iter,
                       i_nonsuccess=i_nonsuccess,
                       i_progress=i_progress)
        
        
# =============================================================================
# status output / save optimization
# =============================================================================
        print ("iter #{:>3}, time: {:>6.1f}s, progress # {:>2}, f_evals: {}".format(i_iter, 
                        time.time()-t_start, i_progress, int(pop.problem.get_fevals())), end='')
        if i_nonsuccess >= 1:
            print ("(non-success: {})".format(i_nonsuccess))
        else:
            print ('')
        
        if plot_interval > 0 and ((i_progress%plot_interval == 0 and not plotted) or i_iter == 1):
            if problem.get_nobj() == 1:
                problem.model.generate_structure(champion_x)
                problem.model.plot_structure()
            else:
                pg.plot_non_dominated_fronts(pop.get_f()) 
            plotted = True
        
        if condition_new_champ or i_iter == 1:
            if problem.get_nobj() == 1:
                print ("        - champion fitness: {}\n".format( new_champ ))
            else:
                print ("        - ideal: {}  (= minimum objective values)".format( new_champ ))
                print ("        - nadir: {}  (= maximum objective values)".format( new_nadir ))
                print ("        - {} fronts, length best front: {}\n".format(N_paretofronts, l_paretofront))
            
            ## --- new champion, save EO info
            if filename:
                if (i_progress+1)%save_interval == 0:
                    save_eo_dict(eo_dict, filename)
                    
    
    ## --- iterations finished, save and return
    if filename:
        if (i_progress+1)%save_interval != 0 or not condition_new_champ:
            save_eo_dict(eo_dict, filename)
    return eo_dict

    

