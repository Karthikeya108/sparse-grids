import sys, os
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
import pysgpp
import numpy as np

def coarseningFunction(grid, factor_graph):
	dim = factor_graph.dim
        storage = grid.getStorage()
        alpha = pysgpp.DataVector(grid.getSize())
        alpha.setAll(1.0)
        delete_counter = 0
        for i in xrange(grid.getSize()):
                grid_index = storage.get(i)
                levels = tuple()
        	for d in xrange(dim):
			#Fetch the interacting factors
                	if grid_index.getLevel(d) != 1:
                        	levels = levels + (d,)
		if len(levels) != 0:
			levels = tuple(sorted(levels))
			nInteract_factors = factor_graph.factors[len(levels)]
			#If the corresponding interaction is not in the factor graph then set the correspoding alpha to 0
			if levels not in nInteract_factors:
                		alpha[i] = 0
                		delete_counter += 1

        coarseningFunctor = pysgpp.SurplusCoarseningFunctor(alpha, delete_counter, 0.5)
        grid.createGridGenerator().coarsen(coarseningFunctor, alpha)
        print "New grid size:", grid.getSize()
        return grid, alpha

