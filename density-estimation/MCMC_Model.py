import sys, os
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
import pymc as pm
import pysgpp
import numpy as np
import copy

def make_model(grid, alpha, factor_graph):
    dim = factor_graph.dim
    x = np.empty(dim, dtype=object)
    for i in xrange(dim):
        x[i] = pm.distributions.Uniform('x'+str(i), lower=0, upper=1.0)
        # Normal distribution for diagnostics
        #x[i] = pm.distributions.Normal('x'+str(i), mu= 0.5, tau=1)
    
    # anova components as deterministic variable
    def compute_ANOVA_components(x, alpha, grid, factor_graph):
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
        for key, value in factor_graph.factors.iteritems():
            for v in value: 
                anovaComponents[v] = 0

        storage = grid.getStorage()
        for i in xrange(grid.getSize()):
            grid_index = storage.get(i)
            key = []
            for d in xrange(dim):
                levelDimension = grid_index.getLevel(d)
                if levelDimension != 1:
                    key.append(d)

            if key == []:
                anovaComponents[(-1,)] += result[i]
            else:
                key = tuple(key)
                if key in anovaComponents:
                    anovaComponents[key] += result[i]

        return anovaComponents # it is fine to return dictionary
            
            
    anova = pm.Deterministic(eval = lambda x: compute_ANOVA_components(x, alpha, grid, factor_graph),
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

    #Potentials
    def psi_logp(anova, v):
        return anova[v]

    psi = {}
    for key, value in factor_graph.factors.iteritems():
        psi[key] = np.empty(len(value), dtype=object)
        for v in xrange(len(value)):
            psi[key][v] = pm.Potential(logp = lambda anova, value=value, v=v: psi_logp(anova, value[v]),
                            name = 'psi_'+str(key)+str(value[v]),
                            parents = {'anova': anova},
                            doc = 'A '+str(key)+' Variate potential',
                            verbose = 0,
                            cache_depth = key)
        #<value> HAS to be DELETED, otherwise the pymc module complains. Looks like no unnecessary array of elements can be left in memory
        del value

    x = pm.ArrayContainer(x)

    for key in psi:
        psi[key] = pm.ArrayContainer(psi[key])
    
    psi = pm.DictContainer(psi) 

    return locals()

