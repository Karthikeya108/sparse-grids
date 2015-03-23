import sys
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
from matplotlib.pylab import *
from pysgpp import *
from math import *
import random
from array import array

import numpy as np
from MCMC_Model import *
import Sampling_Module as sm
import Coarsen_Grid as cgrid
from Factor_Graph import *

import scipy.sparse.linalg as la

class SG_DensityEstimator:
    def __init__(self, data, gridLevel, regparam, regstr, *args):
        """
        Constructor
        arguments -- Input Data, Grid Level, Regularization Parameter (lambda), Alpha Threshold for Coarsening the Factor Graph
        returns -- SG_DensityEstimator Object
        """
        self.data = data
        self.gridLevel = gridLevel
        self.regparam = regparam
        self.regstr = regstr

        if len(args) > 0:
            self.alpha_threshold = args[0]
        
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
        self.model = sm.initialize_mcmc(self.model)

    def __repr__(self):
        return ""+self

    def calculate_prior_info(self, data):
        """
        priorInfo(q) = \frac{1}{n} \sum_{i=1}^{n} \varphi_k (x_i)

        arguments -- Input Data
        returns -- an array of means of the values obtained from the evaluation of the grid on the datapoints
        """
        X = DataMatrix(data)
    
        operationEvaluation = createOperationMultipleEval(self.grid, X)
        y = DataVector(self.grid.getSize())
        a = DataVector(data.getNrows())
        a.setAll(1.0)
        operationEvaluation.multTranspose(a, y)
    
        avgs = y.array()/data.getNrows()

        return avgs

    def compute_nl_term(self, sampling_size):
        """
        Computing the Expected Value (\varphi(\alpha^i)) using MCMC
        arguments -- 
        returns -- \varphi(\alpha^i)
        """
        dim = self.fac.dim

        self.model = pm.Model(input=make_model(self.grid, self.alpha, self.fac), name="sg_normal_indep")
        self.model = sm.sample_mcmc(self.model, sampling_size)
        
        """Picks up the samples from the last chain"""
        data = vstack([self.model.x[i].trace()[:] for i in xrange(dim)]).T
        X = pysgpp.DataMatrix(data)
        
        """Evaluate grid"""
        operationEvaluation = pysgpp.createOperationMultipleEval(self.grid, X)
        y = pysgpp.DataVector(self.grid.getSize())
        a = pysgpp.DataVector(data.shape[0])
        a.setAll(1.0)
        operationEvaluation.multTranspose(a, y)
        #print "Size of Last chain of MCMC samples: ",data.shape[0]

        avgs = y.array()/data.shape[0]
        #print "Psi values: ",avgs

        return avgs
        
    def compute_regfactor(self):
        """
        arguments -- 
        returns -- regularization factor
        """
        lambda_val = self.regparam
        grid_size = self.grid.getSize()
        """Regularization factor"""
        if self.regstr == 'laplace':
            opL = createOperationLaplace(self.grid)
        elif self.regstr == 'identity':
            opL = createOperationIdentity(self.grid)

        def matvec_mult(v, opL, lambda_val):
            result = DataVector(grid_size)
            opL.mult(DataVector(v), result)
            result.mult(lambda_val)
            return result.array()

        matvec = lambda x: matvec_mult(x, opL, lambda_val)

        A = la.LinearOperator((grid_size, grid_size), matvec=matvec, dtype='float64')

        return A

    def update_coefficients(self):
        """
        arguments -- 
        returns -- 
        """
        print "updating alpha"
        print self.alpha
        nonzero_index_array = np.nonzero(self.alpha)
        nonzero_index_list = nonzero_index_array[0]
        cnt = 0
        new_alpha = [None]*len(nonzero_index_list)
        for i in nonzero_index_list:
            new_alpha[cnt] = self.alpha[int(i)]
            cnt = cnt + 1
        print new_alpha

        self.alpha = DataVector(new_alpha)
        
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
        i = 1

        prior_info = self.calculate_prior_info(self.data["data"])
        #print "priorInfo value: ", prior_info

        A = self.compute_regfactor()
        
        mcmc_expVal = self.compute_nl_term(500)

        alpha_mask = DataVector(self.grid.getSize())
        alpha_mask.setAll(1.0)

        while residual > epsilon and i <= imax:

            b = prior_info - mcmc_expVal
  
            #print "b: ",b

            alpha_old = DataVector(self.alpha)
            """Conjugated Gradient method for sparse grids, solving A.alpha=b"""
            self.alpha, info = la.cg(A, b, self.alpha);
            #print("Conjugate Gradient output:")
            #print("cg residual: ",res)
            print "CG Info: ",info
            #print "old alpha: ",alpha_old

            """ \alpha^{i+1} = \alpha^{i} + \omega \tilde{\alpha} """
            if i > 1:
                val = self.alpha
                val = val * learning_rate
                self.alpha = alpha_old + val
            
            #print "new alpha: ",self.alpha

            A_alpha = DataVector(grid_size)
            A = la.aslinearoperator(A)
            A_alpha = A.matvec(self.alpha)
            
            """residual = $\|\emph{A} \alpha^{i+1} - q + \Phi(\alpha^{i+1}) \|$ """
            self.alpha = DataVector(self.alpha)
            if i == 1:
                mcmc_expVal = self.compute_nl_term(1000)
            else:
                mcmc_expVal = self.compute_nl_term(100)
                
            q_val = mcmc_expVal - prior_info
            value = DataVector(grid_size)
            value = A_alpha + q_val
            residual = np.linalg.norm(value)

            i = i + 1
            print "*****************Residual***************  ", residual
            print "+++++++++++++++++i+++++++++++++++++++++   ",i-1

            curr_grid_size = grid_size
            self.grid, self.alpha, self.fac = cgrid.update_grid(self.grid, self.alpha, self.fac, cgrid.coefficient_thresholding, self.alpha_threshold)

            if curr_grid_size != self.grid.getSize():
                grid_size = self.grid.getSize()
                self.update_coefficients()
                """Recompute regularozation factor and prior info and expected value"""
                A = self.compute_regfactor()
                prior_info = self.calculate_prior_info(self.data["data"])
                mcmc_expVal = self.compute_nl_term(100)
        
        print "Alpha: ",self.alpha
        
        return self.grid, DataVector(self.alpha)
        
    def evaluate_density_function(self, dim, inputData):
        """
        arguments -- dim, input data
        returns -- f(x) = \exp{(\sum_{i=1}^{n} \alpha_i \varphi_i(x))} -- A list of values (density) corresponding to each data point
        """
        operationEvaluation = createOperationMultipleEval(self.grid, inputData)
        y = DataVector(inputData.getNrows())
        operationEvaluation.mult(self.alpha, y)
        e = np.exp(y.array())
        
        return e
