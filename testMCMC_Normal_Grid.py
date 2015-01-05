import sys, os
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
import pymc as pm
import pysgpp
import numpy as np
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

pm.Matplot.plot(mcmc)
mcmc.stats()

print model.psi_i[3].logp
print model.anova.get_value()

