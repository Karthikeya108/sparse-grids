import sys, os
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
import pymc as pm
import pysgpp
import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.integrate import *
from MCMC_GridUtils import *

if len(sys.argv) != 3:
        print "Incorrect number of arguments: "
        print "Usage: testMCMC_Grid.py <dim> <level>"
        exit(0)

dim, level = int(sys.argv[1]), int(sys.argv[2])
print dim, level

grid = pysgpp.Grid.createModLinearGrid(dim)
generator = grid.createGridGenerator()
generator.regular(level)
print "Grid size:", grid.getSize()

gsize = grid.getSize()
newGsize = 0

alpha = pysgpp.DataVector(grid.getSize())
alpha.setAll(1.0)

while gsize != newGsize:
        gsize = newGsize
        grid, alpha = coarseningFunction(grid, dim)
        newGsize = grid.getSize()

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

model = pm.Model(input=make_model(grid, alpha, dim), name="sg_normal_indep")

mcmc = pm.MCMC(model, name="MCMC")

mcmc.sample(iter=10000, burn=100, thin=5)

print model.psi_i[3].logp
print model.anova.get_value()

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

