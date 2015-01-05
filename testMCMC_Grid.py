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

#dim = 4
#level = 5

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

model = make_model(grid, alpha, dim)
mcmc = pm.MCMC(model, name="MCMC")
mcmc.sample(iter=10000, burn=100, thin=5)

from pylab import hist, show
hist(mcmc.trace('x0')[:])
show()

pm.Matplot.plot(mcmc)
mcmc.stats()

