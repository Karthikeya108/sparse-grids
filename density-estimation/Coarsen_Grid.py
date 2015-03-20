import sys, os
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
import pysgpp
import numpy as np

def coarsening_function(grid, alpha_mask, factor_graph):
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
        """If the corresponding interaction is not in the factor graph then set the correspoding alpha_mask to 0"""
        if levels not in nInteract_factors:
            alpha_mask[i] = 0
            delete_counter += 1

    coarseningFunctor = pysgpp.SurplusCoarseningFunctor(alpha_mask, delete_counter, 0.5)
    grid.createGridGenerator().coarsen(coarseningFunctor, alpha_mask)
    #print "New grid size:", grid.getSize()
    return grid, alpha_mask

