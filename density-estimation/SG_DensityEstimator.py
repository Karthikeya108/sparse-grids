import sys
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
from matplotlib.pylab import *
from tools import *
from pysgpp import *
from math import *
import random
from optparse import OptionParser
from array import array
from painlesscg import cg,sd,cg_new

import numpy as np
from MCMC_Model import *
from Coarsen_Grid import *
from factor_graph import *

#Required for 'visualizeResult' method
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Required for 'compareExpVal' method
from scipy.stats import norm
from scipy.integrate import *

from pylab import hist, show

#Reuired for 'eval_function_modlin' method
from itertools import izip

import scipy.sparse.linalg as la

class SG_DensityEstimator:
    def __init__(self, data, gridLevel):
        self.data = data
        self.gridLevel = gridLevel

        dim = data["data"].getNcols()

        training = buildTrainingVector(data)

        grid = Grid.createModLinearGrid(dim)
        generator = grid.createGridGenerator()
        generator.regular(level)

        alpha = DataVector(grid.getSize())
        alpha.setAll(1.0)

        fac = factor_graph(int(dim))
        fac.create_factor_graph(int(level))

    def __repr__(self):
        return ""+self

    def updateGrid(grid, alpha, fac):
        gsize = grid.getSize()
        newGsize = 0
        while gsize != newGsize:
            gsize = newGsize
            grid, alpha = coarseningFunction(grid, alpha, fac)
            newGsize = grid.getSize()

     # @param filename filename of the file
     # @return the data stored in the file as a set of arrays
     def openFile(filename):
        if "arff" in filename:
                return readData(filename)
        else:
                return readDataTrivial(filename)

    ## Builds the training data vector
    def buildTrainingVector(data):
        return data["data"]

    def calcq(grid, data):
        X = DataMatrix(data)

        # evaluate grids and exponents
        operationEvaluation = createOperationMultipleEval(grid, X)
        y = DataVector(grid.getSize())
        a = DataVector(data.getNrows())
        a.setAll(1.0)
        operationEvaluation.multTranspose(a, y)

        avgs = y.array()/data.getNrows()

        return avgs

    #Computing the Expected Value (\varphi(x)) using MCMC
    def computeNLterm(grid, alpha, fac):
        dim = fac.dim

        model = pm.Model(input=make_model(grid, alpha, fac), name="sg_normal_indep")
        db = pm.database.pickle.load('sg_mcmc.pickle')
        #print "x0 trace: ",len(db.trace('x0',chain=None)[:])
        #print "x1 trace: ",len(db.trace('x1',chain=None)[:])
        mcmc = pm.MCMC(model, name="MCMC", db=db)
        mcmc.sample(iter=10000, burn=10, thin=2)
        mcmc.db.close()

        #Picks up the samples from the last chain
        data = vstack([model.x[i].trace()[:] for i in xrange(dim)]).T
        X = pysgpp.DataMatrix(data)

        # evaluate grids and exponents
        operationEvaluation = pysgpp.createOperationMultipleEval(grid, X)
        y = pysgpp.DataVector(grid.getSize())
        a = pysgpp.DataVector(data.shape[0])
        a.setAll(1.0)
        operationEvaluation.multTranspose(a, y)
        #print "Size of Last chain of MCMC samples: ",data.shape[0]

        avgs = y.array()/data.shape[0]
        print "Psi values: ",avgs

        return avgs

    def updateFactorGraph(grid, alpha, fac):
        alpha_threshold = 0.5

        alpha_levels = {}
        max_len = 0
        for k in xrange(grid.getSize()):
                grid_index = grid.getStorage().get(k)
                levels = tuple()
                for d in xrange(fac.dim):
                        #Fetch the interacting factors
                        if grid_index.getLevel(d) != 1:
                                levels = levels + (d,)
                #Store the grid point index and corresponding tuple of interacting factors
                alpha_levels[k] = tuple(sorted(levels))
                if max_len < len(levels):
                        max_len = len(levels)
                #print "Levels, alpha: ",alpha_levels
        #Delete all the higher order interacting factors in the factor_graph which are higher 
        #than the maximum length of the <interacting factors> obtained in the previuos step
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
                print "ALpha avg: ",  sum(np.absolute(value))/float(len(value))

                #if the average absolute values of the alphas (co-efficients) corresponding 
                #to a <interacting factor> tuple is less than some <alpha_threshold> then 
                #add the <interacting factor> tuple to the delete list

                if sum(np.absolute(value))/float(len(value)) < alpha_threshold:
                        delete_list[len(delete_list):] = [key]

        fac.coarsen_factor_graph(delete_list)

        return fac

    def run(grid, alpha, fac, training):
        errors = None
        gridSize = grid.getStorage().size()

        #Parameters
        paramW = 0.01
        epsilon = 0.5
        imax = gridSize
        residual = 1
        i = 1

        q = calcq(grid, training)
        print "q value: ",q

        #Create OperationMatrix Object
        #opL = createOperationLaplace(grid)
        opL = createOperationIdentity(grid)

        lambdaVal = options.regparam

        #Form the LinearOperator
        def matvec_mult(v, opL, lambdaVal):
            result = DataVector(gridSize)
            opL.mult(DataVector(v), result)
            result.mult(lambdaVal)
            return result.array()

        matvec = lambda x: matvec_mult(x, opL, lambdaVal)

        A_lambda = la.LinearOperator((gridSize, gridSize), matvec=matvec, dtype='float64')

        print A_lambda

        #A = np.eye(gridSize)

        alpha_true = computeTrueCoeffs(grid, fac.dim)
        print "True ALpha: ",alpha_true
        #This is just to initialize the pickle db to store the MCMC state
        model = pm.Model(input=make_model(grid, alpha, fac), name="sg_normal_indep")
        mcmc = pm.MCMC(model, name="MCMC", db="pickle", dbname="sg_mcmc.pickle")
        mcmc.db
        mcmc.sample(iter=10, burn=1, thin=1)
        mcmc.db.close()

        mcmc_expVal = computeNLterm(grid, alpha, fac)
     print mcmc_expVal

    alpha_mask = DataVector(grid.getSize())
    alpha_mask.setAll(1.0)

    while residual > epsilon and i <= imax:

        b = q - mcmc_expVal

        alpha_old = DataVector(alpha)
        ## Conjugated Gradient method for sparse grids, solving A.alpha=b
        alpha, info = la.cg(A_lambda, b, alpha)
        #print "Conjugate Gradient output:"
        #print "cg residual: ",res
        print "CG Info: ",info
        print "old alpha: ",alpha_old

        if i > 1:
                val = alpha
                val = val * paramW
                alpha = alpha_old + val

        print "new alpha: ",alpha

        #Test
        #alpha = b_lambda

        A_alpha = DataVector(gridSize)

        A_lambda = la.aslinearoperator(A_lambda)
        A_alpha = A_lambda.matvec(alpha)

        alpha = DataVector(alpha)

        #if 'toy' not in options.data[0]:
        mcmc_expVal = computeNLterm(grid, alpha, fac)
        #else:
       #        mcmc_expVal = computeNLterm(grid, alpha_true, fac)





























