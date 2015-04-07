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
from Factor_Graph import *

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
#-------------------------------------------------------------------------------
## Outputs a deprecated warning for an option
# @param option Parameter set by the OptionParser
# @param opt Parameter set by the OptionParser
# @param value Parameter set by the OptionParser
# @param parser Parameter set by the OptionParser
def callback_deprecated(option, opt, value, parser):
    print "Warning: Option %s is deprecated." % (option)


#-------------------------------------------------------------------------------
## Formats a list of type mainlist as a string
# <pre>
#     main_list  = {plain_list | string}*
#     plain_list = string*
# </pre>
# @param l Mainlist
def format_optionlist(l):
    def join_inner_list(entry):
        if type(entry) is list:
            return "("+' OR '.join(entry)+")"
        else:
            return entry
    return ' '*4+' AND '.join(map(lambda entry: join_inner_list(entry), l))


#-------------------------------------------------------------------------------
## Checks whether a valid mode is specified,
# whether all required options for the mode are given and
# executes the corresponding action (function)
#
# @todo remove hack for level new when porting to classifier.new.py
#
# @param mode current mode
def exec_mode(mode):

    if mode=="help":
        print "The following modes are available:"
        for m in modes.keys():
            print "%10s: %s" % (m, modes[m]['help'])
        sys.exit(0)

    # check valid mode
    if not modes.has_key(mode):
        print("Wrong mode! Please refer to --mode help for further information.")
        sys.exit(1)

    # check if all required options are set
    a = True
    for attrib in modes[mode]['required_options']:
        # OR
        if type(attrib) is list:
            b = False
            for attrib2 in attrib:
                # hack for level 0
                if attrib2 == "level":
                    option = getattr(options, attrib2, None)
                    if option >= 0:
                        b = True
                else:
                    if getattr(options, attrib2, None):
                        b = True
            a = a and b
        else:
            if not getattr(options, attrib, None):
                a = False
    if not a:
        print ("Error!")
        print ("For --mode %s you have to specify the following options:\n" % (mode)
               + format_optionlist(modes[mode]['required_options']))
        print ("More on the usage of %s with --help" % (sys.argv[0]))
        sys.exit(1)

    # execute action
    modes[mode]['action']()


#-------------------------------------------------------------------------------
## Opens and read the data of an ARFF (or plain whitespace-separated data) file.
# Opens a file given by a filename.
# @param filename filename of the file
# @return the data stored in the file as a set of arrays
def openFile(filename):
    if "arff" in filename:
        return readData(filename)
    else:
        return readDataTrivial(filename)

#-------------------------------------------------------------------------------
## Builds the training data vector
#
def buildTrainingVector(data):
    return data["data"]

## Calculates the number of points, that should be refined
# @param options: options object
# @param grid: grid
# @return: number of points, that should be refined
def getNumOfPoints(options, grid):
    numOfPoints = 0
    if options.adapt_rate:
        numOfPoints = int(ceil( options.adapt_rate * grid.createGridGenerator().getNumberOfRefinablePoints()))
    else: numOfPoints = options.adapt_points
    return numOfPoints

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

# functions for explicit computation of the modified linear functions
def __phi(x):
    """ Evaluates 1-D hat function at x """
    return max([1 - abs(x), 0])
    
def eval_function_modlin(point, level_vector, index_vector):
    """ Evaluate an individual function define by multilevel and multiindex at the given point """

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

def computeTrueCoeffs(grid, dim):
   mu = np.array([0.5]*dim)
   sigma2 = 0.1**2
   SigmaInv = np.eye(dim)/sigma2 # variables are independent in this example
   Sigma = np.eye(dim) * sigma2

   #Computing alpha by doing interpolation and hierarchisation
   func = lambda x: -0.5*(x-mu).dot(SigmaInv).dot(x-mu) # computes the exponent of the pdf
   storage = grid.getStorage()
   pointDV = pysgpp.DataVector(dim)
   alpha = pysgpp.DataVector(grid.getSize())
   for j in xrange(grid.getSize()):
        grid_index = storage.get(j)
        grid_index.getCoords(pointDV)
        x = pointDV.array()
    	print "Debug x: ", x
        alpha[j] = func(x)
    	print "Debug alpha[j]: ", alpha[j]
   opHierarchisation = pysgpp.createOperationHierarchisation(grid)
   opHierarchisation.doHierarchisation(alpha)
   
   return alpha

#Custom code, not generic - Corresponds to toy dataset
def compareExpVal(grid, mcmc_expVal, dim):
    storage = grid.getStorage()

    mu = np.array([0.5]*dim)
    sigma2 = 0.1**2
    rv = norm(loc=mu[0], scale=np.sqrt(sigma2))

    for i in xrange(grid.getSize()):
        grid_index = storage.get(i)
        level = np.empty(dim)
        index = np.empty(dim)

        prod = 1.0
        for d in xrange(dim):
            level[d] = grid_index.getLevel(d)
            index[d] = grid_index.getIndex(d)
    
            myfunc = lambda x: eval_function_modlin([x], [level[d]], [index[d]])*rv.pdf(x)
            res = quad(myfunc, 0, 1, epsabs=1e-14, epsrel=1e-12)
            prod*= res[0]

    print "MCMC_ExpVal",level, index, mcmc_expVal[i]
    print "----------------------------------------------------------"

    print "Ture ExpVal",level, index, prod
    print ".........................................................."
    

#-------------------------------------------------------------------------------
def run(grid, alpha, fac, training):
    errors = None
    gridSize = grid.getStorage().size()

    #Parameters
    learningRate = 0.01
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

    A = la.LinearOperator((gridSize, gridSize), matvec=matvec, dtype='float64')
    
    print A
    
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
    
    if 'toy' in options.data[0]:
        compareExpVal(grid, mcmc_expVal, fac.dim)

    alpha_mask = DataVector(grid.getSize())
    alpha_mask.setAll(1.0)

    while residual > epsilon and i <= imax:

        b = q - mcmc_expVal

        alpha_old = DataVector(alpha)
        ## Conjugated Gradient method for sparse grids, solving A.alpha=b

        alpha, info = la.cg(A, b, alpha)
        #print "Conjugate Gradient output:"
        #print "cg residual: ",res
        print "CG Info: ",info
        print "old alpha: ",alpha_old

        if i > 1:
            val = alpha
            val = val * learningRate
            alpha = alpha_old + val
        
        print "new alpha: ",alpha

        A_alpha = DataVector(gridSize)
        A = la.aslinearoperator(A)
        A_alpha = A.matvec(alpha)
        
        alpha = DataVector(alpha)
        
        mcmc_expVal = computeNLterm(grid, alpha, fac)

        if 'toy' in options.data[0]:
            compareExpVal(grid, mcmc_expVal, fac.dim)
            
        q_val = mcmc_expVal - q

        value = DataVector(gridSize)
        
        value = A_alpha + q_val
        '''
        summ = 0
        for k in range(gridSize):
            summ = summ + value[k]*value[k]
            
        residual = np.sqrt(summ)
        '''
        residual = np.linalg.norm(value)

        i = i + 1
        print "*****************Residual***************  ", residual
        print "+++++++++++++++++i+++++++++++++++++++++   ",i-1

        fac = updateFactorGraph(grid, alpha, fac)

        print "------------------------------factors---", fac.factors

        grid, alpha_mask = coarsening_function(grid, alpha_mask, fac) 

    print "True Alpha: ", alpha_true
    print "Alpha: ",alpha
    
    return grid, DataVector(alpha)

#-------------------------------------------------------------------------------

def evaluateDensityFunction(grid, alpha, dim, data):

    result = []
    q = DataVector(dim)
    for i in xrange(data.getNrows()):
        data.getRow(i,q)
        value = createOperationEval(grid).eval(alpha,q)
        result.append(value)

    result = np.exp(result)
 
    return result

def visualizeResult(training, result, dim):

    #x_axis = np.arange(0, 500, 0.1)

    #plt.plot(x_axis, result, 'r--', x_axis, kde_result, 'b-')
    #plt.show()

    #hist(result)
    #show()

    x = DataVector(training.getNrows())
    y = DataVector(training.getNrows()) 
    
    training.getColumn(0,x)
    training.getColumn(1,y)

    if dim == 2:
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        p = ax.scatter(x, y, c=result)
        fig.colorbar(p)
    elif dim == 3:
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        z = DataVector(training.getNrows())
        training.getColumn(2,z)
        p = ax.scatter(x, y, z, c=result)
        fig.colorbar(p)
    elif dim == 1:
        pdf_true = norm(0.5, 0.1).pdf(x)
        fig, ax = plt.subplots()
        ax.plot(x, result, color='blue', alpha=0.5, lw=3)
        ax.fill(x, pdf_true, ec='gray', fc='gray', alpha=0.4)
    
    plt.show()

## Density Estimation
def doDensityEstimation():
    # read data
    data = openFile(options.data[0])
    dim = data["data"].getNcols()
    numData = data["data"].getNrows()

    level = options.level

    #if options.verbose:
    print "Dimension is:", dim
    print "Size of datasets is:", numData
    print "Level is: ", options.level

    training = buildTrainingVector(data)

    #grid = constructGrid(dim)

    grid = Grid.createModLinearGrid(dim)
    generator = grid.createGridGenerator()
    generator.regular(level)

    gsize = grid.getSize()
    newGsize = 0

    print "Gridsize is:", gsize

    alpha = DataVector(grid.getSize())
    alpha.setAll(1.0)

    fac = Factor_Graph(int(dim))
    fac.create_factor_graph(int(level))

    #Code to remove all the interactions among the factors in the factor graph
    #for k in xrange(2,level+1):
    #   fac.factors[k] = []    

    while gsize != newGsize:
        gsize = newGsize
        grid, alpha = coarsening_function(grid, alpha, fac)
        newGsize = grid.getSize()
    
    grid, alpha = run(grid, alpha, fac, training)

    result = evaluateDensityFunction(grid, alpha, dim, training)

    print "Mean of the result: ",np.mean(result)
   
    if dim < 4:
        visualizeResult(training, result, dim)
 
    if options.outfile:
        writeAlphaARFF(options.outfile, alpha)
    if options.gridfile:
        writeGrid(options.gridfile, grid)

    return alpha

#-------------------------------------------------------------------------------

if __name__=='__main__':
        # Initialize OptionParser, set Options
        parser = OptionParser()
        parser.add_option("-l", "--level", action="store", type="int", dest="level", help="Gridlevel")
        parser.add_option("-D", "--dim", action="callback", type="int",dest="dim", help="Griddimension", callback=callback_deprecated)
        parser.add_option("-m", "--mode", action="store", type="string", default="apply", dest="mode", help="Specifies the action to do. Get help for the mode please type --mode help.")
        parser.add_option("-C", "--CMode", action="store", type="string", default="laplace", dest="CMode", help="Specifies the action to do.")
        parser.add_option("-L", "--lambda", action="store", type="float",default=0.01, metavar="LAMBDA", dest="regparam", help="Lambda")
        parser.add_option("-i", "--imax", action="store", type="int",default=500, metavar="MAX", dest="imax", help="Max number of iterations")
        parser.add_option("-d", "--data", action="append", type="string", dest="data", help="Filename for the Datafile.")
        parser.add_option("-t", "--test", action="store", type="string", dest="test", help="File containing the testdata")
        parser.add_option("-A", "--alpha", action="store", type="string", dest="alpha", help="Filename for a file containing an alpha-Vector")
        parser.add_option("-o", "--outfile", action="store", type="string", dest="outfile", help="Filename where the calculated alphas are stored")
        parser.add_option("--gridfile", action="store", type="string", dest="gridfile", help="Filename where the resulting grid is stored")
        parser.add_option("-v", "--verbose", action="store_true", default=False, dest="verbose", help="Provides extra output")
        parser.add_option("--grid", action="store", type="string", dest="grid", help="Filename for Grid-resume. For fold? and test. Full filename.")
        parser.add_option("--mse_limit", action="store", type="float", default="0.0", dest="mse_limit", help="If MSE of test data fall below this limit, refinement will stop.")
        parser.add_option("--grid_limit", action="store", type="int", default="0", dest="grid_limit", help="If the number of points on grid exceed grid_limit, refinement will stop.")
        parser.add_option("--Hk", action="store", type="float", default="1.0", dest="Hk", help="Parameter k for regularization with H^k norm. For certain CModes.")

        parser.add_option("--function_type", action="store", type="choice", default=None, dest="function_type", choices=['modWavelet'],
                      help="Choose type for non-standard basis functions")
    # parse options
        (options,args)=parser.parse_args()

    # specifiy the modes:
        # modes is an array containing all modes, the options needed by the mode and the action
        # that is to be executed
        modes = {
                 'density'   : {'help': "learn a dataset",
                      'required_options': ['data', 'level'],
                      'action': doDensityEstimation}
                }

    
        exec_mode(options.mode.lower())
