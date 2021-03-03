# encoding: utf-8
"""
tools for EO submodule of pyGDM2

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

import six; from six.moves import cPickle as pickle

import os
import copy
import warnings

import numpy as np



def reload_eo(filename):
    """Reload saved evol. optimization from file
    
    Parameters
    ----------
    filename : str or dict
        either path to filename (pickled optimization data) or the optimization
        dictionary. In the latter case, the input will simply be returned 
        unaltered.
        
    Returns
    -------
    dict : evolutionary optimization data
    
    """
    if type(filename) == str:
        try:
            eo_dict = pickle.load( open(filename, "rb") )
        except IOError:
            raise IOError("Saved optimization file not found!")
    elif type(filename) == dict:
        eo_dict = filename
    else:
        raise ValueError("No valid optimization data. Must be either string (path to file) or dict.")
        
    return eo_dict
    


#==============================================================================
# generate simuation dictionary for best solution
#==============================================================================
def get_best_candidate(filename, iteration=-1, do_test=True, verbose=False):
    """generate simuation object for best solution
    
    Parameters
    ----------
    filename : str or dict
        either path to filename (pickled optimization data) or the optimization
        dictionary. In the latter case, the input will simply be returned 
        unaltered.
        
    iteration : int, default: -1
        iteration (Nr of improvements) to use. Best candidate of selected iteration
        will be returned. Python indexing, hence positive and negative numbers 
        possible.
    
    do_test : bool, default: True
        whether to test the stored fitness (redo the simulation) or not
    
    verbose : bool, default: False
        print some information
        
    Returns
    -------
    sim : :class:`pyGDM2.core.simulation` instance of best candidate
    """
    ## --- reload
    eo_dict = reload_eo(filename)
    
    ## --- champion of selected iteration
    halloffame = eo_dict['halloffame']
    champ = halloffame[iteration]
    f = champ[1]
    x = champ[2]
    
    ## --- generate geometry
    problem = eo_dict['problem']
    problem.model.generate_structure(x)
    
    if verbose:
        i_iter = champ[0]
        if iteration < 0:
            iteration = len(halloffame) + iteration
        print ("Best candidate after {} iterations (with {} improvements): fitness = {}".format(
                                    i_iter+1, iteration, ["{:.5g}".format(_f) for _f in f]))
    
    ## --- test
    if do_test:
        if verbose: print ("Testing: recalculating fitness...", end='')
        f_test = problem.fitness(x)
        differ_flag = False
        for f_test_1, f_test_2 in zip(f_test, f):
            f1 = round(f_test_1, 5)
            f2 = round(f_test_2, 5)
            if f1-f2 != 0:
                differ_flag = True
        
        if differ_flag:
            warnings.warn("Stored and recalculated fitness values seem to differ. " + 
                          "f_stored={}, f_recalc={}".format(
                                      ["{:.5g}".format(_f) for _f in f],
                                      ["{:.5g}".format(_f) for _f in f_test]))
        elif verbose: print ("Done. Everything OK.")
    
    ## --- copy and return simulation dict
    problem.model.generate_structure(x)
    sim = copy.deepcopy(problem.model.sim)
    return sim


def get_best_candidate_f_x(filename, iteration=-1):
    """get best fitness and parameter vector of a specific generation.
    
    Returns also the total number of generations with improvements accessible 
    by the `iteration` parameter.
    
    Parameters
    ----------
    filename : str or dict
        either path to filename (pickled optimization data) or the optimization
        dictionary. In the latter case, the input will simply be returned 
        unaltered.
        
    iteration : int, default: -1
        iteration (Nr of improvements) to use. Best candidate of selected iteration
        will be returned. Python indexing, hence positive and negative numbers 
        possible.
    
    Returns
    -------
    f : array like
        fitness(-vector) of best candidate
    
    x : array like
        parameter vector of best candidate
    
    N : int
        number of iterations with improvement
    
    """
    ## --- reload
    eo_dict = reload_eo(filename)
    
    ## --- champion of selected iteration
    halloffame = eo_dict['halloffame']
    champ = halloffame[iteration]
    f = champ[1]
    x = champ[2]
    
    return f, x, len(halloffame)
    

def get_model(filename):
    """get instance of model object used in optimization
    
    Parameters
    ----------
    filename : str or dict
        either path to filename (pickled optimization data) or the optimization
        dictionary. In the latter case, the input will simply be returned 
        unaltered.
    
    Returns
    -------
    model : Instance of class based on :class:`.model.BaseClass` 
        Model used in the loaded EO
    """
    eo_dict = reload_eo(filename)
    problem = eo_dict['problem']
    
    return problem.model
    

def get_problem(filename):
    """get instance of problem object used in optimization
    
    Parameters
    ----------
    filename : str or dict
        either path to filename (pickled optimization data) or the optimization
        dictionary. In the latter case, the input will simply be returned 
        unaltered.
    
    Returns
    -------
    model : Instance of class based on :class:`.problems.BaseProblem`
        Problem used in the loaded EO
    """
    eo_dict = reload_eo(filename)
    problem = eo_dict['problem']
    
    return problem
    

def get_population(filename):
    """get full final population
    
    Parameters
    ----------
    filename : str or dict
        either path to filename (pickled optimization data) or the optimization
        dictionary. In the latter case, the input will simply be returned 
        unaltered.
    
    Returns
    -------
    list of lists : list of tuples (f, x)
        Full population of final iteration. Each individual is stored as
        tuple (*f*, **x**) with *f* fitness the of parameter-vector **x**
    """
    eo_dict = reload_eo(filename)
    pop = eo_dict['pop']
    pop_list = [(pop.get_f()[i], pop.get_x()[i]) for i in range(len(pop))]
    
    return pop_list




def get_pareto_fronts(filename, iteration=-1, recalc=False, verbose=False):
    """get Pareto fronts and corresponding simulations of multi-objective optimization
    
    Parameters
    ----------
    filename : str or dict
        either path to filename (pickled optimization data) or the optimization
        dictionary. In the latter case, the input will simply be returned 
        unaltered.
    
    iteration : int, default: -1
        iteration number. Python indexing (pos. and neg. indices possible).
    
    Returns
    -------
    list of lists : list of tuples (f, x)
        Full population of final iteration. Each individual is stored as
        tuple (*f*, **x**) with *f* fitness the of parameter-vector **x**
    """
    import pygmo as pg
    
    eo_dict = reload_eo(filename)
    
    if iteration != -1 and eo_dict['all_gen_pop'] == []:
        raise ValueError("Population-evolution not stored. To obtain the " +
                         "Pareto-fronts for others than the final population, " +
                         "`save_all_generations` flag must be enabled " +
                         "during optimization. ")
    if iteration == -1:
        pop = eo_dict['pop']
    else:
        pop = eo_dict['all_gen_pop'][iteration][1]
    
    pf, dl, dc, ndl = pg.fast_non_dominated_sorting(pop.get_f())
    problem = eo_dict['problem']

    allpareto = []
    allsim = []
    allx = []
    for p in pf:
        allpareto.append([])
        allsim.append([])
        allx.append([])
        for i in p:
            best_x = pop.get_x()[i]
            best_f = pop.get_f()[i]
            problem.model.generate_structure(best_x)
            if recalc:
                problem._objfun_impl(best_x)
            
            allpareto[-1].append(best_f)
            allsim[-1].append( copy.deepcopy(problem.model.sim) )
            allx[-1].append( copy.deepcopy(best_x) )
            
        allpareto[-1] = np.transpose(allpareto[-1])
        allpareto[-1] *= -1
        allpareto[-1] = np.append(allpareto[-1],[np.arange(len(allpareto[-1][0]))], axis=0)
        allpareto[-1] = allpareto[-1].T
        
        ## sort pareto front and get sorting indices
        if np.shape(allpareto[-1])[0] != 1:    
            allpareto[-1] = np.sort(allpareto[-1].view('f8,f8,i8'), order=['f1'], axis=0).view(np.float).T
            sort_idx = [int(i) for i in allpareto[-1][2]]
        else:
            ## one single element in Pareto front
            sort_idx = [0]
        allpareto[-1] = -1 * allpareto[-1][:-1].T
        
        ## Sort simulations and parameter vectors according to pareto-front order
        allsim[-1] = [allsim[-1][s_idx] for s_idx in sort_idx]
        allx[-1] = [allx[-1][s_idx] for s_idx in sort_idx]

    return allpareto, allsim, allx
    


def plot_pareto_2d(pareto, print_indices=True, cutoff_indexprint=-1, 
                   marker='x', mew=1.5, ms=7, 
                   show=True, **kwargs):
    """plot the pareto-front, optionally print index numbers for identification
    
    Parameters
    ----------
    pareto : list of tuples
        Pareto-front as list of objective-value tuples (2 objectives)
    
    print_indices : bool, default: True
        print index of individuals next to pareto-front points
    
    cutoff_indexprint : float, default: -1
        optional: manual value for minimum distance between two individuals in
        Pareto plot for allowing the printing of a new index-value label
    
    marker, mew, ms: str, float, float. defaults: 'x', 1.5, 7
        marker type, marker linewith and marker size for step-plot
    
    show : bool, default: True
        whether or not to run `plt.show()`
    
    **kwargs: are passed to matplotlib's `plt.step`
    
    Returns
    -------
    return value of matplotlib's `plt.step`
    """
    import matplotlib.pyplot as plt
    
    ## plot
    if pareto.shape[1] == 0:
        warnings.warn("Empty Pareto-front. Ignoring.")
    else:
        step_plot = plt.step(pareto.T[0], pareto.T[1], where='post',
                                     marker=marker, mew=mew, ms=ms, **kwargs)
        
        ## labels with indices
        if print_indices:
            if cutoff_indexprint == -1:
                cutoffx = (pareto.T[0].max() - pareto.T[0].min())*0.05  # 5% of objective #1 value range
                cutoffy = (pareto.T[1].max() - pareto.T[1].min())*0.05  # 5% of objective #1 value range
            else:
                cutoffx = cutoff_indexprint
                cutoffy = cutoff_indexprint
            
            x_last, y_last = 0, 0
            for i, (x,y) in enumerate(zip(pareto.T[0], pareto.T[1])):
                ## check whether to print or not
                if not (abs(x - x_last)<cutoffx and abs(y - y_last)<cutoffy):
                    plt.text(x + (pareto.T[0].max()-pareto.T[0].min())*0.0, 
                             y + (pareto.T[1].max()-pareto.T[1].min())*0.04, 
                             i)
                    x_last, y_last = x, y
        
        plt.xlabel('fitness objective #1')
        plt.ylabel('fitness objective #2')
        
        if show:
            plt.show()
        return step_plot


def plot_all_pareto_fronts_2d(all_paretos, cmap=None, show=True, **kwargs):
    """Plot all Pareto-fronts
    
    Parameters
    ----------
    all_paretos : list of lists of tuples
        List of all non-dominated fronts as obtained by 
        :func:`.get_pareto_fronts`
    
    cmap : matplotlib colormap, default: None
        A colormap for the different pareto-fronts. Default "None" will use 
        matplotlib's `Blues`
    
    show : bool, default: True
        whether or not to run `plt.show()`
    
    **kwargs: are passed to :func:`.plot_pareto_2d`
    
    """
    import matplotlib.pyplot as plt
    if cmap is None:
        cmap = plt.cm.Blues
    
    colors = cmap(np.linspace(0.3, 1.0, len(all_paretos)))[::-1]
    for i, pareto in enumerate(all_paretos):
        plot_pareto_2d(pareto, show=False, color=colors[i], 
                                   label='front #{}'.format(i), **kwargs)
    plt.legend(loc='best')

    if show:
        plt.show()




#==============================================================================
# Get index-lists for directionality problem
#==============================================================================
verbose=True
plot=True
plot_type='mesh' #'scatter'

def calculate_solid_angle_by_dir_index(Nteta, Nphi, tetamin, tetamax, dir_index,
                                      plot=True, show=True,
                                      plot_type='scatter', verbose=True):
    """calculate target solid angle for `pyGDM2.EO.problems.ProblemDirectivity`
    
    plot_type : str, default: "scatter"
        either "mesh", "scatter" or "both". 
        Note: Since the field is evaluated at points, 'mesh' is only an 
        approximation and represents the true solid angle not entirely correctly!
        
    """
    def theta_phi_lists(Nteta, Nphi, tetamin, tetamax):
        dteta = (tetamax-tetamin)/(Nteta-1.)
        dphi = 2.*np.pi/(Nphi) 
        
        tetalist = np.zeros( (Nteta, Nphi) )
        philist  = np.zeros( (Nteta, Nphi) )
        for j in range(Nteta):
            for k in range(Nphi):
                tetalist[j][k] = tetamin + j*dteta
                philist[j][k] = k*dphi
        
        return tetalist.flatten(), philist.flatten(), dteta, dphi
    
    
    if type(dir_index) == list:
        dir_index = np.array(dir_index)
    
    theta, phi, dTheta,dPhi = theta_phi_lists(Nteta, Nphi, tetamin, tetamax)
    dS = np.sin(theta)*dTheta*dPhi
    dS += dS.max()*0.1
    
    
    
    #==============================================================================
    # list of selected/unselected points
    #==============================================================================
    teta_phi_sel = []
    teta_phi_unsel = []
    for idx, tp in enumerate(zip(theta, phi)):
        if dir_index is not None:
            if idx in dir_index:
                teta_phi_sel.append(tp)
            else:
                teta_phi_unsel.append(tp)
    teta_phi_sel = np.transpose(teta_phi_sel)
    teta_phi_unsel = np.transpose(teta_phi_unsel)
    
    #==============================================================================
    # selection colormesh (0 --> not selected, 1--> selected)
    #==============================================================================
#    Nteta2 = Nteta+2
    Nteta2 = Nteta+1
    theta2, phi2,dTheta2,dPhi2 = theta_phi_lists(Nteta2, Nphi, tetamin, tetamax)
#    dir_index2 = dir_index + Nphi
    dir_index2 = dir_index
    
    ## --- create colormesh arrays
    t_p_colors = np.zeros( theta2.shape )
    t_p_colors[dir_index2] = 1
    
    shape = (Nteta2, Nphi)
    t_p_colors = t_p_colors.reshape(shape)
    t_p_teta = theta2.reshape(shape)
    t_p_phi  = phi2.reshape(shape) - (dPhi2/2.)
    
    ### --- for plotting only: Add 360degrees (copy of 0 degrees)
    t_p_teta   = np.concatenate([t_p_teta.T, [t_p_teta.T[-1]]]).T
    t_p_phi    = np.concatenate([t_p_phi.T, [np.ones(t_p_phi.T[-1].shape) * t_p_phi.max()+dPhi2]]).T #- np.pi/float(Nphi)
    t_p_colors = np.concatenate([t_p_colors.T, [t_p_colors.T[-1]]]).T
    
    
    #==============================================================================
    # Print index-list
    #==============================================================================
    if verbose:
        print ('    idx, theta, phi' )
        print ('-----------------------' )
        for i, tp in enumerate(zip(theta, phi)):
            if dir_index is not None:
                if i in dir_index:
                    print ("XXX -- ", end='')
                else:
                    print (" o  -- ", end='')
                
            print ('{: >3}:   t={:.1f}, p={:.1f},     dS={:.3f}'.format(i, tp[0] * (180./np.pi), tp[1] * (180./np.pi), dS[i]))
    
    
    #==============================================================================
    # PLOT
    #==============================================================================
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('mycmap', [(0,  (1.0, 0.7, 0.7) ),
                                                            (1,  (0.4, 0.7, 0.4) )])
        
        ## --- scatter
        plt.subplot(polar=True)
        
        if plot_type in ['mesh', 'both']:
            plt.pcolormesh(t_p_phi, t_p_teta*180/np.pi, t_p_colors, cmap=cmap, alpha=1, edgecolors="none")
        
        if plot_type in ['scatter', 'both']:
            plt.scatter(teta_phi_unsel[1], teta_phi_unsel[0]*180/np.pi, marker='x', color='r', label='unsel.')
            plt.scatter(teta_phi_sel[1], teta_phi_sel[0]*180/np.pi, marker='x', color='g', label='sel.', lw=3, s=35)
            
        plt.ylim(0, 90)
        if show:
            plt.show()
                
    
    
if __name__=='__main__':
    ##--- TESTING

    
    
    #==============================================================================
    # Configuration
    #==============================================================================
    dir_index = None
    tetamin = 0
    tetamax = np.pi/2.
    
    
    #==============================================================================
    # DONUT
    #==============================================================================
    Nteta = 7
    Nphi = 36
    dir_index = np.arange(144, 180) # donut --> ring at teta=58deg
    
    
    
    #==============================================================================
    # FOCUSED
    #==============================================================================
    Nteta = 7
    Nphi = 36
    dir_index = np.array([109,142,143]) # tight focus 1 - 44 window x3
    
    
    dir_index = np.array([143+36]) # tight focus 2 - 58s
    toffset  = 1*Nphi
    dir_index = np.array([109+toffset,142+toffset,143+toffset]) # tight focus 58 w3
    dir_index = np.array([143+72]) # tight focus 3 - 73s
    dir_index = np.array([143+108]) # tight focus 4 - 90s
    toffset  = 3*Nphi
    dir_index = np.array([109+toffset,142+toffset,143+toffset]) # tight focus 90 w3
    dir_index = np.array([109+toffset+17,143+toffset-18]) # tight focus 90 w3
    
    #toffset  = 1*Nphi; toffset2 = 2*Nphi
    #dir_index = np.array([109+toffset,142+toffset,143+toffset, 109+toffset2,142+toffset2,143+toffset2]) # tight focus 5 - 6pts window
    #
    #dir_index = np.array([109-36,110-36,111-36,112-36,139-36,140-36,141-36,142-36,143-36,
    #                     109,110,111,112,139,140,141,142,143,
    #                     109+36,110+36,111+36,112+36,139+36,140+36,141+36,142+36,143+36]) # loose focus
    
    
    #==============================================================================
    # DOUBLE FOCUS
    #==============================================================================
    #Nteta = 7
    #Nphi = 37
    #toffset  = 0*Nphi; toffset2 = 0
    #dir_index = np.array([149+toffset,183+toffset,184+toffset, 165+toffset2,166+toffset2,167+toffset2]) # double focus at 0deg / 180deg
    
    
    #==============================================================================
    # SIMPLE
    #==============================================================================
    Nteta = 3
    Nphi=5
    tetamin=0
    tetamax=np.pi/2.
    dir_index = [10]
    
    
    calculate_solid_angle_by_dir_index(Nteta, Nphi, tetamin, tetamax, dir_index, plot_type='scatter')    
    calculate_solid_angle_by_dir_index(Nteta, Nphi, tetamin, tetamax, dir_index, plot_type='mesh')    
    calculate_solid_angle_by_dir_index(Nteta, Nphi, tetamin, tetamax, dir_index, plot_type='both')    
    
    
    
    
    
    
    
    
    
