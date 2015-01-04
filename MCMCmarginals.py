import sys, os
sys.path.append('<Path to pysgpp>')
import pymc as pm
import pysgpp
import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.integrate import *
import copy

dim = 4
level = 5
grid = pysgpp.Grid.createModLinearGrid(dim)
generator = grid.createGridGenerator()
generator.regular(level)
print "Grid size:", grid.getSize()

gsize = grid.getSize()
newGsize = 0

#Continue execute the block until the grid size stops changing
while gsize != newGsize:
	gsize = newGsize
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
	newGsize = grid.getSize()
	print "New grid size:", grid.getSize()

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

def make_model():
    """ Creates the variables for my graphical model. This function is not generic"""
    x = np.empty(4, dtype=object)
    for i in xrange(4):
        x[i] = pm.distributions.Uniform('x'+str(i), lower=0, upper=1.0)
        # Normal distribution for diagnostics
        #x[i] = pm.distributions.Normal('x'+str(i), mu= 0.5, tau=1)
    
    # univariate potentials
    def psi_i_logp(anova, i): 
        return anova[(i,)]
    
    psi_i = np.empty(4, dtype=object)
    
    # bivariate potentials
    def psi_ij_logp(anova, i, j):
    	return anova[tuple(sorted([i,j]))]

    psi_ij = np.empty(4, dtype=object)
     
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
        for i in xrange(4):
            anovaComponents[(i,)] = 0
            anovaComponents[tuple(sorted((i, (i+1)%4)))] = 0

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
    
    
    for i in xrange(4):  
        psi_i[i] = pm.Potential(logp = lambda anova, i=i: psi_i_logp(anova, i),
                            name = 'psi_i'+str(i),
                            parents = {'anova': anova},
                            doc = 'A univariate potential',
                            verbose = 0,
                            cache_depth = 1)
    
    
    for i in xrange(4):  
        j = (i+1)%4 # this is just how I connect the variables in my graphical model. It's not generic
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

#model = make_model()
#mcmc = pm.MCMC(model, name="MCMC")
#mcmc.sample(iter=10000, burn=100, thin=5)

#from pylab import hist, show
#hist(mcmc.trace('x0')[:])
#show()

#pm.Matplot.plot(mcmc)
#mcmc.stats()

#dot = pm.graph.graph(model)
#dot.write_png("graph.png")

mu = np.array([0.5]*dim) # distribution centered at [0.5, ..., 0.5]
sigma2 = 0.05**2
SigmaInv = np.eye(dim)/sigma2 # variables are independent in this example
Sigma = np.eye(dim) * sigma2 

# I can compute alpha by doing interpolation and hierarchisation
func = lambda x: -0.5*(x-mu).dot(SigmaInv).dot(x-mu) # computes the exponent of the pdf
storage = grid.getStorage()
pointDV = pysgpp.DataVector(dim)
alpha = pysgpp.DataVector(grid.getSize())

for j in xrange(grid.getSize()):
    grid_index = storage.get(j)
    grid_index.getCoords(pointDV)
    x = pointDV.array()
    alpha[j] = func(x)
opHierarchisation = pysgpp.createOperationHierarchisation(grid)
opHierarchisation.doHierarchisation(alpha)

model = pm.Model(input=make_model(), name="sg_normal_indep")
mcmc = pm.MCMC(model, name="MCMC")

mcmc.sample(iter=10000, burn=100, thin=5)

pm.Matplot.plot(mcmc)
mcmc.stats()

print model.psi_i[3].logp
print model.anova.get_value()

from pylab import hist, plot, show, legend

# Define the paraemters for the gaussian distribution function
mu1d = 0.0
sigma1d = 1.0
size = 10000 # sampling size
epsilon = 0.5

# we sample from a unit distribution
x_uniform = np.random.random(size)*200-100 # sample on the interval [-100, 100)
y_uniform = sp.stats.norm.pdf(x_uniform, loc=mu1d, scale=sigma1d)/sp.stats.uniform.pdf(x_uniform, loc=-100, scale=200)
points_uniform = np.array([[s, np.sum(y_uniform[:s])/s] for s in xrange(500, size, 500)])

# we sample from a similar gaussian distribution
x_gauss = np.random.randn(size)*(sigma1d + epsilon) + (mu1d + epsilon)
y_gauss= sp.stats.norm.pdf(x_gauss, loc=mu1d, scale=sigma1d)/sp.stats.norm.pdf(x_gauss, loc=(mu1d+epsilon), scale=(sigma1d+epsilon))
points_gauss = np.array([[s, np.sum(y_gauss[:s])/s] for s in xrange(500, size, 500)])

# plot the approximation for different number of sampling points
plot(points_uniform[:,0], points_uniform[:,1], 'r-', label="uniform")
plot(points_gauss[:,0], points_gauss[:,1], 'g-', label="gaussian")
plot(points_gauss[:,0], np.array([1.0]*points_gauss.shape[0]), 'k--', label='correct')
legend(loc=4)
show()


#Computing expecting values of basis function using MCMC
#First we evaluate the expected values for grid functions using the MCMC sampling (almost from the true distribution)
data = np.vstack([model.x[i].trace() for i in xrange(dim)]).T
X = pysgpp.DataMatrix(data)
    
# evaluate grids and exponents
operationEvaluation = pysgpp.createOperationMultipleEval(grid, X)
y = pysgpp.DataVector(grid.getSize())
a = pysgpp.DataVector(data.shape[0])
a.setAll(1.0)
operationEvaluation.multTranspose(a, y)

avgs = y.array()/data.shape[0]
for i in xrange(5):
    grid_index = storage.get(i)
    level = np.empty(dim)
    index = np.empty(dim)
    for d in xrange(dim):
        level[d] = grid_index.getLevel(d)
        index[d] = grid_index.getIndex(d)
    print level, index, avgs[i]

# functions for explicit computation of the modified linear functions
from itertools import izip 

def __phi(x):
    """ Evaluates 1-D hat function at x """
    return max([1 - abs(x), 0])
    
def eval_function_modlin(
    point,
    level_vector,
    index_vector
    ):
    """ Evaluate an individual function define by multilevel and multiindex
    at the given point """

    product = 1
    for (l, i, x) in izip(level_vector, index_vector, point):
        if l == 1 and i == 1:
            val = 1
        elif l > 1 and i == 1:
            if x >= 0 and x <= 2 ** (1 - l):
                val = 2 - 2 ** l * x
            else:
                val = 0
        elif l > 1 and i == 2 ** l - 1:
            if x >= 1 - 2 ** (1 - l) and x <= 1:
                val = 2 ** l * x + 1 - i
            else:
                val = 0
        else:
            val = __phi(x * 2 ** l - i)
        product *= val
        if product == 0:
            break
    return product

# compute the expected value of the grid point [3 1 1 1][3 1 1 1]
level_vector = np.array([1]*dim)
level_vector[0] = 3
index_vector = np.array([1]*dim)
index_vector[0] = 3
myfunc = lambda x, level_vector=level_vector, index_vector=index_vector: \
    eval_function_modlin(x, level_vector, index_vector)

rv = norm(loc=mu[0], scale=np.sqrt(sigma2))
prod = 1.0
for d in xrange(dim):
    myfunc = lambda x: eval_function_modlin([x], [level_vector[d]], [index_vector[d]])*rv.pdf(x)
    res = quad(myfunc, 0, 1, epsabs=1e-14, epsrel=1e-12)
    prod*= res[0]
print level_vector, index_vector, prod

# comapred to the naive monte carlo
data = np.random.random(len(data[:])*100).reshape([-1,dim])
X = pysgpp.DataMatrix(data)
    
# evaluate grids and exponents
operationEvaluation = pysgpp.createOperationMultipleEval(grid, X)
y = pysgpp.DataVector(grid.getSize())
a = pysgpp.DataVector(data.shape[0])
a.setAll(1.0)
operationEvaluation.multTranspose(a, y)

avgs = y.array()/data.shape[0]
for i in xrange(5):
    grid_index = storage.get(i)
    level = np.empty(dim)
    index = np.empty(dim)
    for d in xrange(dim):
        level[d] = grid_index.getLevel(d)
        index[d] = grid_index.getIndex(d)
    print level, index, avgs[i]

