import sys, os
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
import pymc as pm
import pysgpp
import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.integrate import *
import copy
from pylab import hist, plot, show, legend

def coarseningFunction(grid, dim):
	storage = grid.getStorage()
	alpha = pysgpp.DataVector(grid.getSize())
	alpha.setAll(1.0)
	delete_counter = 0
	for i in xrange(grid.getSize()):
    		grid_index = storage.get(i)
    		levels = []
    	for d in xrange(dim):
        	if grid_index.getLevel(d) != 1:
            		levels.append(d)
   	# this part is customary to my particular example. something more general should be programmed later
    	if len(levels) >= 3 \
    	or (len(levels) == 2 and ((levels[0] +1) != levels[1] and (levels[1] +1)%dim != levels[0])):
        	alpha[i] = 0
        	delete_counter += 1
	coarseningFunctor = pysgpp.SurplusCoarseningFunctor(alpha, delete_counter, 0.5)
	grid.createGridGenerator().coarsen(coarseningFunctor, alpha)
	print "New grid size:", grid.getSize()
	return grid, alpha

def computeMarginals(grid, alpha, dim, n=500):
    """Compute the expected value of the sparse grid functions and normalisation constant 
       using naive Monte-Carlo sampling.

       Parameters:
       grid -- Grid object
       alpha -- DataVector with surplusses
       dim -- number of dimensions
       n -- number of sampling points

       Return values:
       expected values
       normalisation constant
    """
    # sample data
    x = np.random.random(dim*n).reshape([n, dim])
    X = pysgpp.DataMatrix(x)
    
    # evaluate grids and exponents
    operationEvaluation = pysgpp.createOperationMultipleEval(grid, X)
    y = pysgpp.DataVector(n)
    operationEvaluation.mult(alpha, y)
    e = np.exp(y.array())
    
    # get normalization factor
    normFactor = np.mean(e)
    
    grid_size = grid.getSize()
    result = pysgpp.DataVector(grid_size)
    alpha_tmp = pysgpp.DataVector(n)
    alpha_tmp.setAll(1.0)
    operationEvaluation.multTranspose(alpha_tmp, result)
    
    return result.array()/n, normFactor

def make_model(grid, alpha, dim):
    """ Creates the variables for my graphical model. This function is not generic"""
    x = np.empty(dim, dtype=object)
    for i in xrange(dim):
        x[i] = pm.distributions.Uniform('x'+str(i), lower=0, upper=1.0)
        # Normal distribution for diagnostics
        #x[i] = pm.distributions.Normal('x'+str(i), mu= 0.5, tau=1)
    
    # univariate potentials
    def psi_i_logp(anova, i): 
        return anova[(i,)]
    
    psi_i = np.empty(dim, dtype=object)
    
    # bivariate potentials
    def psi_ij_logp(anova, i, j):
    	return anova[tuple(sorted([i,j]))]

    psi_ij = np.empty(dim, dtype=object)
     
    # anova components as deterministic variable
    def compute_ANOVA_components(x, alpha, grid):
        """Evaluates the sparse grid at the point x and returns 
           the individual ANOVA components of the result"""
        x = np.hstack(x).reshape(1, -1)
        dataPoint = pysgpp.DataMatrix(x)
        result = pysgpp.DataVector(grid.getSize())
        result.setAll(0.0)
        a = pysgpp.DataVector(1)
        a[0] = 1.0
        opMultipleEval = pysgpp.createOperationMultipleEval(grid, dataPoint)
        opMultipleEval.multTranspose(a, result)
        result = result.array()*alpha.array()
        
        anovaComponents = {(-1,):0}
        for i in xrange(dim):
            anovaComponents[(i,)] = 0
            anovaComponents[tuple(sorted((i, (i+1)%dim)))] = 0

        storage = grid.getStorage()
        for i in xrange(grid.getSize()):
            grid_index = storage.get(i)
            key = []
            for d in xrange(dim):
                levelDimension = grid_index.getLevel(d)
                if levelDimension != 1:
                    key.append(d)
            if key == []: anovaComponents[(-1,)] += result[i]
            else:
                key = tuple(key)
		if key in anovaComponents:
                	anovaComponents[key] += result[i]
        return anovaComponents # it is fine to return dictionary
            
            
    anova = pm.Deterministic(eval = lambda x: compute_ANOVA_components(x, alpha, grid),
                  name = 'anova',
                  parents = {'x': x},
                  doc = 'Individual anova component contributions',
                  trace = True,
                  verbose = 0,
                  plot=False,
                  cache_depth = 2)
    
    # constant factor ot the ANOVA component
    psi_0 = pm.Potential(logp = lambda anova: anova[(-1,)],
                            name = 'psi_0',
                            parents = {'anova': anova},
                            doc = 'Potential corresponding to the constant factor',
                            verbose = 0,
                            cache_depth = 1)
    
    
    for i in xrange(dim):  
        psi_i[i] = pm.Potential(logp = lambda anova, i=i: psi_i_logp(anova, i),
                            name = 'psi_i'+str(i),
                            parents = {'anova': anova},
                            doc = 'A univariate potential',
                            verbose = 0,
                            cache_depth = 1)
    
    
    for i in xrange(dim):  
        j = (i+1)%dim # this is just how I connect the variables in my graphical model. It's not generic
        psi_ij[i] = pm.Potential(logp = lambda anova, i=i, j=j: psi_ij_logp(anova, i, j),
                            name = 'psi_ij%d%d'%(i,j),
                            parents = {'anova': anova},
                            doc = 'A bivariate potential',
                            verbose = 0,
                            cache_depth = 2)
    
    # for some reason pm.Model cannot be created is the arrays are not converted to
    # ArrayContainers before (dict. name becomes an integer and Python complains)
    x = pm.ArrayContainer(x)
    psi_i = pm.ArrayContainer(psi_i)
    psi_ij = pm.ArrayContainer(psi_ij)
    return locals()
