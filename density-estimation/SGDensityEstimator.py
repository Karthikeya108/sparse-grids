import sys
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
from matplotlib.pylab import *
from pysgpp import *
from math import *
import random
from array import array

import numpy as np
from MCMC_Model import *
from Sampling_Module import *
from Coarsen_Grid import *
from Factor_Graph import *

import scipy.sparse.linalg as la

class SG_DensityEstimator:
    def __init__(self, data, gridLevel, regparam, alpha_threshold):
        """
        Constructor
        arguments -- Input Data, Grid Level, Regularization Parameter (lambda), Alpha Threshold for Coarsening the Factor Graph
        returns -- SG_DensityEstimator Object
        """
        self.data = data
        self.gridLevel = gridLevel
        self.regparam = regparam
        self.alpha_threshold = alpha_threshold
        
        self.dim = data["data"].getNcols()
        
        """Create a ModLinear Grid"""
        self.grid = Grid.createModLinearGrid(self.dim)
        generator = self.grid.createGridGenerator()
        generator.regular(gridLevel)
        
        """Initialize the Co-effecients"""
        self.alpha = DataVector(self.grid.getSize())
        self.alpha.setAll(1.0)
        
        """Create a fully connected factor graph"""
        self.fac = Factor_Graph(int(self.dim))
        self.fac.create_factor_graph(int(gridLevel))
        
        """Initialize the Sampling Module"""
        self.model = pm.Model(input=make_model(self.grid, self.alpha, self.fac), name="sg_normal_indep")
        self.model = initialize_mcmc(self.model)

    def __repr__(self):
        return ""+self

    def calculate_prior_info(self, data):
        """
        priorInfo(q) = \frac{1}{n} \sum_{i=1}^{n} \varphi_k (x_i)

        arguments -- Input Data
        returns -- an array of means of the values obtained from the evaluation of the grid on the datapoints
        """
        X = DataMatrix(data)
    
        # evaluate grids and exponents
        operationEvaluation = createOperationMultipleEval(self.grid, X)
        y = DataVector(self.grid.getSize())
        a = DataVector(data.getNrows())
        a.setAll(1.0)
        operationEvaluation.multTranspose(a, y)
    
        avgs = y.array()/data.getNrows()

        return avgs

    def compute_nl_term(self):
        """
        Computing the Expected Value (\varphi(\alpha^i)) using MCMC
        arguments -- 
        returns -- \varphi(\alpha^i)
        """
        dim = self.fac.dim

        self.model = pm.Model(input=make_model(self.grid, self.alpha, self.fac), name="sg_normal_indep")
        self.model = sample_mcmc(self.model)
        
        """Picks up the samples from the last chain"""
        data = vstack([self.model.x[i].trace()[:] for i in xrange(dim)]).T
        X = pysgpp.DataMatrix(data)
        
        """Evaluate grids and exponents"""
        operationEvaluation = pysgpp.createOperationMultipleEval(self.grid, X)
        y = pysgpp.DataVector(self.grid.getSize())
        a = pysgpp.DataVector(data.shape[0])
        a.setAll(1.0)
        operationEvaluation.multTranspose(a, y)
        #print "Size of Last chain of MCMC samples: ",data.shape[0]

        avgs = y.array()/data.shape[0]
        print "Psi values: ",avgs

        return avgs
        
    def update_factor_graph(self):
        """
        Removes the less important factors from the factor graph
        arguments -- 
        returns -- 
        """
        alpha_levels = {}
        max_len = 0
        for k in xrange(self.grid.getSize()):
            grid_index = self.grid.getStorage().get(k)
            levels = tuple()
            for d in xrange(self.fac.dim):
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
        if self.fac.dim > max_len+1:
            for k in xrange(self.fac.dim-1,max_len,-1):
                self.fac.factors[k] = []

        level_alphas = {}
        for key, value in alpha_levels.iteritems():
            if value not in level_alphas:
                level_alphas[value] = [self.alpha[key]]
            else:
                level_alphas[value][len(level_alphas[value]):] = [self.alpha[key]]

        delete_list = []
        for key, value in level_alphas.iteritems():
            #print "ALpha avg: ",  sum(np.absolute(value))/float(len(value))
                    
            """
            if the average absolute values of the alphas (co-efficients) corresponding 
            to a <interacting factor> tuple is less than some <alpha_threshold> then 
            add the <interacting factor> tuple to the delete list
            """
            if sum(np.absolute(value))/float(len(value)) < self.alpha_threshold:
                delete_list[len(delete_list):] = [key]
        
        self.fac.coarsen_factor_graph(delete_list)

    def update_grid(self, alpha_mask):
        """
        coarsens the grid
        arguments -- alpha_mask
        returns -- updated alpha_mask
        """
        gsize = self.grid.getSize()
        newgsize = 0
        while gsize != newgsize:
            gsize = newgsize
            self.grid, alpha_mask = coarsening_function(self.grid, alpha_mask, self.fac)
            newgsize = self.grid.getSize()
            
        return alpha_mask
        
    def compute_coefficients(self):
        """
        arguments -- lambdaVal (Regularization parameter)
        returns -- alpha(co-efficients), updated grid
        """
        grid_size = self.grid.getSize()

        """Initialize Parameters"""
        learning_rate = 0.01
        epsilon = 0.5
        imax = grid_size
        residual = 1
        lambda_val = self.regparam
        i = 1

        prior_info = self.calculate_prior_info(self.data["data"])
        print "priorInfo value: ", prior_info

        """Regularization factor"""
        #opL = createOperationLaplace(self.grid)
        opL = createOperationIdentity(self.grid)

        def matvec_mult(v, opL, lambda_val):
            result = DataVector(grid_size)
            opL.mult(DataVector(v), result)
            result.mult(lambda_val)
            return result.array()
        
        matvec = lambda x: matvec_mult(x, opL, lambda_val)

        A = la.LinearOperator((grid_size, grid_size), matvec=matvec, dtype='float64')
        
        mcmc_expVal = self.compute_nl_term()

        alpha_mask = DataVector(self.grid.getSize())
        alpha_mask.setAll(1.0)

        while residual > epsilon and i <= imax:

            b = prior_info - mcmc_expVal

            alpha_old = DataVector(self.alpha)
            """Conjugated Gradient method for sparse grids, solving A.alpha=b"""
            self.alpha, info = la.cg(A, b, self.alpha)
            #print("Conjugate Gradient output:")
            #print("cg residual: ",res)
            print "CG Info: ",info
            print "old alpha: ",alpha_old

            """ \alpha^{i+1} = \alpha^{i} + \omega \tilde{\alpha} """
            if i > 1:
                val = self.alpha
                val = val * learning_rate
                self.alpha = alpha_old + val
            
            print "new alpha: ",self.alpha

            A_alpha = DataVector(grid_size)
            A = la.aslinearoperator(A)
            A_alpha = A.matvec(self.alpha)
            
            """residual = $\|\emph{A} \alpha^{i+1} - q + \Phi(\alpha^{i+1}) \|$ """
            self.alpha = DataVector(self.alpha)
            mcmc_expVal = self.compute_nl_term()
                
            q_val = mcmc_expVal - prior_info
            value = DataVector(grid_size)
            value = A_alpha + q_val
            residual = np.linalg.norm(value)

            i = i + 1
            print "*****************Residual***************  ", residual
            print "+++++++++++++++++i+++++++++++++++++++++   ",i-1

            self.update_factor_graph()

            print "------------------------------factors---", self.fac.factors

            alpha_mask = self.update_grid(alpha_mask)

        print "Alpha: ",self.alpha
        
        return self.grid, DataVector(self.alpha)
        
    def evaluate_density_function(self, dim, inputData):
        """
        arguments -- dim
        returns -- f(x) = \exp{(\sum_{i=1}^{n} \alpha_i \varphi_i(x))} -- A list of values corresponding to each data points
        """
        result = []
        q = DataVector(dim)
        for i in xrange(inputData.getNrows()):
            inputData.getRow(i,q)
            value = createOperationEval(self.grid).eval(self.alpha,q)
            result.append(value)

        result = np.exp(result)
     
        return result
