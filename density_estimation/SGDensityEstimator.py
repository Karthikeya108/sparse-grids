import sys
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
sys.path.append('/home/perun/opt/eclipse/plugins/org.python.pydev_3.9.2.201502050007/pysrc')
from matplotlib.pylab import *
from pysgpp import *
from math import *
import random
from array import array

import numpy as np
from MCMC_Model import *
import Sampling_Module as sm
from Coarsen_Grid import GridCoarsener
from Factor_Graph import *

import scipy.sparse.linalg as la
import scipy as sp
import scipy.optimize
import scipy.stats
import pydevd


class SG_DensityEstimator:
    def __init__(self, data, gridLevel, regparam, regstr, **kwargs):
        """
        Constructor
        arguments -- Input Data, Grid Level, Regularization Parameter (lambda), Alpha Threshold for Coarsening the Factor Graph
        returns -- SG_DensityEstimator Object
        """
        self.data = data['data']
        self.gridLevel = gridLevel
        self.regparam = regparam
        self.regstr = regstr

        self.alpha_threshold = kwargs.get('alpha_threshold', 0)
        self.epsilon = kwargs.get('epsilon', 0.5)
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.sampling_size_init = kwargs.get('sampling_size_init', 500)
        self.sampling_size_first = kwargs.get('sampling_size_first', 1000)
        self.sampling_size_last = kwargs.get('sampling_size_last', 100)
        self.imax = kwargs.get('imax', 15)
        self.alpha_true = kwargs.get('alpha_true', None)
        #pydevd.settrace()

        self.dim = data["data"].getNcols()
        
        # Create a ModLinear Grid
        self.grid = Grid.createModLinearGrid(self.dim)
        generator = self.grid.createGridGenerator()
        generator.regular(gridLevel)
        
        # Initialize the Co-effecients
        #self.alpha = DataVector(self.grid.getSize())
        #self.alpha.setAll(1.0)
        # sample alpha from truncated normal distr. [-0.2, 0.2]
        self.alpha = sp.stats.truncnorm.rvs(-0.2, 0.2, scale=0.2, size = self.grid.getSize())
        
        # Create a fully connected factor graph
        self.factor_graph = Factor_Graph(int(self.dim))
        self.factor_graph.create_factor_graph(int(gridLevel))
        
        # Initialize the Sampling Module
        self.model = pm.Model(input=make_model(self.grid, self.alpha, self.factor_graph), 
                              name="sg_normal_indep", verbose=0)
        self.model = sm.initialize_mcmc(self.model)


    def __repr__(self):
        return ""+self


    def calc_empirical_expected_values(self, data):
        """
        Computes emirical zeroth moment of the sufficient statistics
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


    def calc_model_expected_values(self, iter_step):
        """
        Computing the model Expected Value (\varphi(\alpha^i)) of sufficient statistics
         using MCMC
        arguments -- 
        returns -- \varphi(\alpha^i)
        """
        
        if iter_step > 1:
            sampling_size = self.sampling_size_last
        else:
            sampling_size = self.sampling_size_first
            
        dim = self.factor_graph.dim

        self.model = pm.Model(input=make_model(self.grid, self.alpha, 
                    self.factor_graph), name="sg_normal_indep", verbose=0)
        self.model = sm.sample_mcmc(self.model, sampling_size)
        
        # Picks up the samples from the last chain
        data = vstack([self.model.x[i].trace()[:] for i in xrange(dim)]).T
        X = pysgpp.DataMatrix(data)
        
        # Evaluate grid
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
        Compute regularisation factor (Laplace or Identity)
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
        DEPRECATED
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
        
        
    def __get_cond_unnorm_prob(self, x, j, theta):
        """See p. 970 at the bottom
        x_j = x[j]
        x_not_j = x_1, ..., x_{j-1}, 0, x_{j+1}, ..., x_n
        return the log of the probability:
        \ln P(x_j | \vec x_{-j}) = \ln \tilde{P} (x_j , \vec x_{-j}) - \ln \sum_{x'_j} \tilde{P} (x'_j | \vec x_{-j})
        where \tilde{P} is unnormalised probability
        """
        stepsize = 2**-self.gridLevel
        nodes = np.arange(0, 1+stepsize, stepsize)
        
        all_coords_dm = DataMatrix(nodes.shape[0]+1, x.shape[0])
        x_dv = DataVector(x)
        
        # compile the dataset first for fast evaluation
        for i,node in enumerate(nodes):
            all_coords_dm.setRow(i, x_dv)
            all_coords_dm.set(i, j, node)
        
        # pack the original coordinate for nominator evaluation
        all_coords_dm.setRow(nodes.shape[0], x_dv)
            
        opEval = createOperationMultipleEval(self.grid, all_coords_dm)
        
        funcval_dv = DataVector(nodes.shape[0]+1)
        # the parameter is called alpha in sparse grid community and theta in statistics
        alpha_dv = DataVector(theta)
        
        # compute joint log probabilities
        opEval.mult(alpha_dv, funcval_dv)
        
        # compute integral, it's easy since exponent is linear between nodes:
        # \int_v^w exp(ax + b) dx = e^b * 1/a [exp(ax)]_v^w
        # w-v = stepsize so I can always assume I start at 0 and b=f(v), a=(f(w)-f(v))/stepsize
        int_val = 0
        for i in xrange(1,nodes.shape[0]):
            b = funcval_dv[i-1]
            slope = (funcval_dv[i] - funcval_dv[i-1])/stepsize
            if slope*stepsize > 700 or b > 700:
                int_val += 1e10 #np.finfo('d').max/100/nodes.shape[0]
            else:
                try:
                    int_val += 1.0/(slope + 1e-8) * (np.exp(slope*stepsize + b) - np.exp(b))
                except:
                    pydevd.settrace();
                    pass
        
        # log of fraction, hence difference of logs
        try:
            result = funcval_dv[nodes.shape[0]] - np.log(int_val)
        except:
            pydevd.settrace();
            pass
        return result
            
        
    def compute_pseudo_loglikelihood(self, theta):
        """ Computes pseudo-loglikelihood for current self.data, self.grid. 
        See Koller and Friedman  "Probabilistic Graphical Models"
        Ch. 20.6.1
        $\ell_{PL}(\theta; D) = 1/M \sum_{m=1}^M \sum_{j=1}^n \ln P(x_j[m] | \vec x_{-j} [m], \theta)$
        where D is dataset, M is the number of data points, n is the number of dimensions
        """
        #pydevd.settrace()
        M = self.data.getNrows()
        n = self.data.getNcols()
        pseudo_loglikelihood = 0
        for x in self.data.array():
            for j in xrange(n):
                pseudo_loglikelihood += self.__get_cond_unnorm_prob(x, j, theta)
        
        return pseudo_loglikelihood/M
    
    
    def __get_gradient_point_pseudelikelikhood(self, x, j, theta):
        grid_size = self.grid.getSize()
        stepsize = 2**-self.gridLevel
        nodes = np.arange(0, 1+stepsize, stepsize)
        
        all_coords_dm = DataMatrix(nodes.shape[0]+1, x.shape[0])
        x_dv = DataVector(x)
        
        # compile the dataset first for fast evaluation
        for node_idx,node in enumerate(nodes):
            all_coords_dm.setRow(node_idx, x_dv)
            all_coords_dm.set(node_idx, j, node)
        
        # pack the original coordinate for nominator evaluation
        all_coords_dm.setRow(nodes.shape[0], x_dv)
            
        opEval = createOperationMultipleEval(self.grid, all_coords_dm)
        
        funcval_dv = DataVector(nodes.shape[0]+1)
        # the parameter is called alpha in sparse grid community and theta in statistics
        alpha_dv = DataVector(theta)
        
        # compute joint log probabilities
        opEval.mult(alpha_dv, funcval_dv)
        
        # prepace computation of individual basis functions
        one_dv = DataVector(nodes.shape[0]+1)
        one_dv.setAll(0.)
        one_dv[0] = 1
        basis_evaluations_dv = DataVector(grid_size)
        opEval.multTranspose(one_dv, basis_evaluations_dv)
        basis_evaluations_old = basis_evaluations_dv.array()
        one_dv[0] = 0
        
        # compute integral, it's easy since exponent is linear between nodes:
        # \int_v^w(ax+b) exp(cx + d) dx = (a exp(d))/c^2 [exp(cx)(cx-1)]_v^w + 
        # b exp(d)/c [exp(cx)]_v^w
        # w-v = stepsize so I can always assume I start at 0
        
        int_val = np.zeros(grid_size)
        norm_constant = 0
        for node_idx in xrange(1,nodes.shape[0]):
            one_dv[node_idx] = 1
            opEval.multTranspose(one_dv, basis_evaluations_dv)
            one_dv[node_idx] = 0
            b = basis_evaluations_old
            a = (basis_evaluations_dv.array() - b)/stepsize
            d = funcval_dv[node_idx-1]
            c = (funcval_dv[node_idx] - d)/stepsize + 1e-8
            cx = c*stepsize

            int_val += a*np.exp(d)/(c**2)*(np.exp(cx)*(cx-1) + 1) + \
                       b*np.exp(d)/c*(np.exp(cx) - 1)
            norm_constant += np.exp(d)/c*(np.exp(cx) - 1)
            basis_evaluations_old = basis_evaluations_dv.array()
            
        int_val /= norm_constant
        
        # log of fraction, hence difference of logs
        return int_val

    
    
    def compute_gradient_pseudo_loglikelihood(self, theta):
        """Computes the gradient of the pseudo-likelihood function for current self.data
        and self.grid."""
        M = self.data.getNrows()
        n = self.data.getNcols()
        
        q = self.calc_empirical_expected_values(self.data)

        gradient_pseudo_loglikelihood = np.zeros(self.grid.getSize())
        for x in self.data.array():
            for j in xrange(n):
                gradient_pseudo_loglikelihood += \
                        self.__get_gradient_point_pseudelikelikhood(x, j, theta)
        
        return q*n - gradient_pseudo_loglikelihood/M

    def run(self):
        """
        Newton-Raphson procedure for penalised maximum likelihood
         
        arguments -- lambdaVal (Regularization parameter)
        returns -- alpha(co-efficients), updated grid
        """
        
        # TODO use scipy.optimize.newton_krylov instead

        # Initialize Parameters
        likelihood_grad_norm = np.inf
        iteration_step = 1
        grid_coarsener = GridCoarsener()
        f_val_at_alpha = None

        E_empir = self.calc_empirical_expected_values(self.data) # same as q
        print  "\nE_empir", E_empir
        E_model = self.calc_model_expected_values(self.sampling_size_init) # same as \Phi(\alpha^{i-1})
        print  "\nInitial E_model", E_model
        A = self.compute_regfactor()
        b = E_empir - E_model - A.dot(self.alpha) #NEGATIVE gradient of the likelihood
        
        if self.alpha_true != None:
            dist = np.linalg.norm(self.alpha_true - self.alpha)/np.linalg.norm(self.alpha_true)
            print "Initial relative Distance to the true alpha %f"%dist
        
        while likelihood_grad_norm > self.epsilon and iteration_step <= self.imax:
            #alpha_old = DataVector(self.alpha)
            # Conjugated Gradient method for sparse grids, solving A.alpha=b
            # TODO: what is the miningful x0 here?
            alpha_direction, info = la.cg(A, b); # direction of DESCENT since negatve gradient used
            print "\nCG Info: ",info
            
            p_k = self.compute_gradient_pseudo_loglikelihood(self.alpha)

            # \alpha^{iteration_step+1} = \alpha^{iteration_step} + \omega \tilde{\alpha} """
            # TODO: the call to self.compute_pseudo_loglikelihood(self.alpha) can be saved
            negate_loglike = lambda theta: -self.compute_pseudo_loglikelihood(theta)
            if f_val_at_alpha == None: f_val_at_alpha = negate_loglike(self.alpha)
            pydevd.settrace()
            learning_rate, f_count, f_val_at_alpha = sp.optimize.linesearch.line_search_armijo(negate_loglike, 
                        self.alpha, alpha_direction, -p_k, f_val_at_alpha, alpha0=.3)
            if learning_rate == None or learning_rate < 0:
                learning_rate = 1e-3
                f_val_at_alpha = None
            try:
                print "Number of armijo step:%d, learning rate:%f"%(f_count, learning_rate)
            except:
                pass
        
            self.alpha += learning_rate * alpha_direction
            
            if self.alpha_true != None:
                dist = np.linalg.norm(self.alpha_true - self.alpha)/np.linalg.norm(self.alpha_true)
                print "Relative Distance to the true alpha %f"%dist
            sys.stdout.flush()
            # reestimate model expected values for suff. statistics using new alphas
            E_model = self.calc_model_expected_values(iteration_step)
            print  "\nE_model:", E_model
            
            #likelihood_grad_norm = $\|\emph{A} \alpha^{iteration_step+1} - q + \Phi(\alpha^{iteration_step+1}) \|$ """
            b = E_empir - E_model - A.dot(self.alpha)
            likelihood_grad_norm = np.linalg.norm(b)

            print  "*****************Norm of Likelihood Gradient***************  ", likelihood_grad_norm
            print  "+++++++++++++++++iteration_step+++++++++++++++++++++   ",iteration_step
            
            # TODO: call grid compression 
#            self.grid, self.alpha, self.factor_graph = grid_coarsener.update_grid(self.grid, self.alpha,
#                 self.factor_graph, grid_coarsener.coefficient_thresholding, self.alpha_threshold)
             
            # Recompute regularozation factor and prior info and expected value
#             A = self.compute_regfactor() #superfast, 'cause implicit
#             E_empir = grid_coarsener.compress_array(E_empir) #self.calc_empirical_expected_values(self.data["data"])
#             E_model = grid_coarsener.compress_array(E_model) #self.calc_model_expected_values(100)
#             self.alpha = grid_coarsener.compress_array(self.alpha)
            
            iteration_step += 1
                
        print "Alpha: ",self.alpha
        
        return self.grid, DataVector(self.alpha)
        

    def evaluate_density_function(self, dim, inputData):
        """
        arguments -- dim, input data
        returns -- f(x) = \exp{(\sum_{i=1}^{n} \alpha_i \varphi_i(x))} -- A list of values (density) corresponding to each data point
        """
        alpha_dv = DataVector(self.alpha)
        operationEvaluation = createOperationMultipleEval(self.grid, inputData)
        y = DataVector(inputData.getNrows())
        operationEvaluation.mult(alpha_dv, y)
        e = np.exp(y.array())
        
        return e
