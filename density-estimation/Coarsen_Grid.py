import sys, os
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
import pysgpp
import numpy as np

def coefficient_thresholding(grid, alpha, fac, *args):
        """
        Removes the less important factors from the factor graph
        arguments -- Grid, Co-efficients (alpha), factor graph and coefficient_threshold 
        returns -- updated factor graph
        """
        alpha_threshold = args[0]
        alpha_levels = {}
        max_len = 0
        for k in xrange(grid.getSize()):
            grid_index = grid.getStorage().get(k)
            levels = tuple()
            for d in xrange(fac.dim):
                """Fetch the interacting factors"""
                if grid_index.getLevel(d) != 1:
                    levels = levels + (d,)
            """Store the grid point index and corresponding tuple of interacting factors"""
            alpha_levels[k] = tuple(sorted(levels))
            if max_len < len(levels):
                max_len = len(levels)
            #print "Levels, alpha: ",alpha_levels
        """
        Delete all the higher order interacting factors in the factor_graph which are higher 
        than the maximum length of the <interacting factors> obtained in the previuos step
        """
        if fac.dim > max_len+1:
            for k in xrange(fac.dim-1,max_len,-1):
                fac.factors[k] = []

        level_alphas = {}
        for key, value in alpha_levels.iteritems():
            if value not in level_alphas:
                level_alphas[value] = [alpha[key]]
            else:
                level_alphas[value][len(level_alphas[value]):] = [alpha[key]]

        delete_list = []
        for key, value in level_alphas.iteritems():
            #print "ALpha avg: ",  sum(np.absolute(value))/float(len(value))

            """
            if the average absolute values of the alphas (co-efficients) corresponding 
            to a <interacting factor> tuple is less than some <alpha_threshold> then 
            add the <interacting factor> tuple to the delete list
            """
            if sum(np.absolute(value))/float(len(value)) < alpha_threshold:
                delete_list[len(delete_list):] = [key]

        fac.coarsen_factor_graph(delete_list)
 
        return fac

def coarsening_function(grid, alpha, factor_graph):
    """
    Updates the Grid based on the components in the factor graph
    arguments -- Grid, Co-efficients (alpha), factor graph 
    returns -- updated grid and coefficients
    """
    dim = factor_graph.dim
    storage = grid.getStorage()
    delete_counter = 0
    for i in xrange(grid.getSize()):
        grid_index = storage.get(i)
        levels = tuple()
        for d in xrange(dim):
            """Fetch the interacting factors"""
            if grid_index.getLevel(d) != 1:
                levels = levels + (d,)
    if len(levels) != 0:
        levels = tuple(sorted(levels))
        nInteract_factors = factor_graph.factors[len(levels)]
        """If the corresponding interaction is not in the factor graph then set the correspoding alpha to 0"""
        if levels not in nInteract_factors:
            alpha[i] = 0
            delete_counter += 1

    coarseningFunctor = pysgpp.SurplusCoarseningFunctor(alpha, delete_counter, 0.5)
    grid.createGridGenerator().coarsen(coarseningFunctor, alpha)
    #print "New grid size:", grid.getSize()
    return grid, alpha


def update_grid(grid, alpha, fac, coarsening_strategy=coefficient_thresholding, *args):
    """
    coarsens the grid
    arguments -- Grid, Co-efficients, factor graph, coarsening strategy (function), respective parameters
    returns -- Updated - Grid, Coefficients and factor graph
    """
    fac = coarsening_strategy(grid, alpha, fac, *args)
    gsize = grid.getSize()
    newgsize = 0
    while gsize != newgsize:
        gsize = newgsize
        grid, alpha = coarsening_function(grid, alpha, fac)
        newgsize = grid.getSize()

    return grid, alpha, fac
