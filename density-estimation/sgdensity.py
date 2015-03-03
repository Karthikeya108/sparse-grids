import sys, os
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
from matplotlib.pylab import *
from tools import *
from pysgpp import *
from math import *
import random
from optparse import OptionParser
from array import array
from painlesscg import cg,sd,cg_new

from skmonaco import mcquad #TODO: Remove
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
## Constructs a new grid.
# If options.grid is set, then read in a stored grid. If not, construct a new
# grid dependent on the dimension dim, on options.level and options.polynom.
# Sets the use of boundary functions according to options.border.
# @param dim the grid dimension
# @return a grid
# @todo Integrate into all modes
def constructGrid(dim):
    if options.grid == None:
        # grid points on boundary
        if options.trapezoidboundary == True or options.completeboundary == True:
            if options.polynom > 1:
                print "Error. Not implemented yet."
                sys.exit(1)
            if options.trapezoidboundary == True:
                if options.verbose:
                    print "LinearTrapezoidBoundaryGrid, l=%s" % (options.level)
                grid = Grid.createLinearTrapezoidBoundaryGrid(dim)
            if options.completeboundary == True:
                if options.verbose:
                    print "LinearBoundaryGrid, l=%s" % (options.level)
                grid = Grid.createLinearBoundaryGrid(dim)
        elif options.function_type == "modWavelet":
            if options.verbose:
                print "ModWaveletGrid, l=%s" % (options.level)
            grid = Grid.createModWaveletGrid(dim)
        else:
            # modified boundary functions?
            if options.border:
                if options.polynom > 1:
                    if options.verbose:
                        print "ModPolyGrid, p=%d, l=%d" %(options.polynom, options.level)
                    grid = Grid.createModPolyGrid(dim, options.polynom)
                else:
                    if options.verbose:
                        print "ModLinearGrid, l=%s" % (options.level)
                    grid = Grid.createModLinearGrid(dim)
            # grid points on boundary?
            elif options.boundary == 1:
                if options.polynom > 1:
                    print "Error. Not implemented yet."
                    sys.exit(1)
                else:
                    if options.verbose:
                        print "LinearTrapezoidBoundaryGrid, l=%s" % (options.level)
                    grid = Grid.createLinearTrapezoidBoundaryGrid(dim)
            # more grid points on boundary?
            elif options.boundary == 2:
                if options.polynom > 1:
                    print "Error. Not implemented yet."
                    sys.exit(1)
                else:
                    if options.verbose:
                        print "LinearBoundaryGrid, l=%s" % (options.level)
                    grid = Grid.createLinearBoundaryGrid(dim)
            else: #no border points
                if options.polynom > 1:
                    if options.verbose:
                        print "PolyGrid, p=%d, l=%d" %(options.polynom, options.level)
                    grid = Grid.createPolyGrid(dim, options.polynom)
                else:
                    if options.verbose:
                        print "LinearGrid, l=%s" % (options.level)
                    grid = Grid.createLinearGrid(dim)

	        generator = grid.createGridGenerator()
        generator.regular(options.level)
    else: #read grid from file
        if options.verbose:
            print "reading grid from %s" % (options.grid)
        grid = readGrid(options.grid)

    return grid


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

#Not used anymore: TODO - Remove
def basisFunction(gpoint, gdim, val):
	prod = 1
	for d in range(gdim):
		levInd = gpoint.get(d)
		level = levInd[0]
		index = levInd[1]
		basisMap = (2**level)*val[d] - index
		prod = prod * np.maximum(1 - np.absolute(basisMap), 0)
        return prod

def calcq(grid, data):
	X = pysgpp.DataMatrix(data)
    
	# evaluate grids and exponents
	operationEvaluation = pysgpp.createOperationMultipleEval(grid, X)
	y = pysgpp.DataVector(grid.getSize())
	a = pysgpp.DataVector(data.getNrows())
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
    	mcmc.sample(iter=10000, burn=100, thin=5)
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
	alpha_threshold = 0.1
	
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
		#print "ALpha avg: ",  sum(np.absolute(value))/float(len(value))
                
		#if the average absolute values of the alphas (co-efficients) corresponding 
                #to a <interacting factor> tuple is less than some <alpha_threshold> then 
                #add the <interacting factor> tuple to the delete list

                if sum(np.absolute(value))/float(len(value)) < alpha_threshold:
                        delete_list[len(delete_list):] = [key]

        fac.coarsen_factor_graph(delete_list)

	return fac
	

def conjugateGradient(b, alpha, imax, epsilon, A, reuse = False, verbose=True, max_threshold=None):
    epsilon2 = epsilon*epsilon

    i = 0
    temp = DataVector(len(alpha))
    q = DataVector(len(alpha))
    delta_0 = 0.0

    # calculate residuum
    if reuse:
        q.setAll(0)
        A.mult(q,temp)
        r = DataVector(b)
        r.sub(temp)
        delta_0 = r.dotProduct(r)*epsilon2
    else:
        alpha.setAll(0)

    A.mult(alpha, temp)
    r = DataVector(b)
    r.sub(temp)

    # delta
    d = DataVector(r)

    delta_old = 0.0
    delta_new = r.dotProduct(r)

    if not reuse:
        delta_0 = delta_new*epsilon2

    if verbose:
        print "Starting norm of residuum: %g" % (delta_0/epsilon2)
        print "Target norm:               %g" % (delta_0)

    while (i < imax) and (delta_new > delta_0) and (max_threshold == None or delta_new > max_threshold):
        # q = A*d
        A.mult(d, q)
        # a = d_new / d.q
        a = delta_new/d.dotProduct(q)

        # x = x + a*d
        alpha.axpy(a, d)

        if i % 50 == 0:
        # r = b - A*x
	    A.mult(alpha, temp)
            r.copyFrom(b)
	    r.sub(temp)
        else:
            # r = r - a*q
            r.axpy(-a, q)

        delta_old = delta_new
        delta_new = r.dotProduct(r)
        beta = delta_new/delta_old

        if verbose:
            print "delta: %g" % delta_new

        d.mult(beta)
        d.add(r)
	
        i += 1

    if verbose:
        print "Number of iterations: %d (max. %d)" % (i, imax)
        print "Final norm of residuum: %g" % delta_new
	
    return (i,delta_new)

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

#Custom code, not generic - Corresponds to data/toy2.txt
def compareExpVal(grid, mcmc_expVal, dim):
    storage = grid.getStorage()

    mu = np.array([0.5]*dim)
    sigma2 = 0.01**2
    rv = norm(loc=mu[0], scale=np.sqrt(sigma2))

    for i in xrange(grid.getSize()):
    	grid_index = storage.get(i)
    	level = np.empty(dim)
    	index = np.empty(dim)
    	for d in xrange(dim):
        	level[d] = grid_index.getLevel(d)
        	index[d] = grid_index.getIndex(d)
    
	print "MCMC_ExpVal",level, index, mcmc_expVal[i]
	print "----------------------------------------------------------"
    	myfunc = lambda x, level=level, index=index: eval_function_modlin(x, level, index)
    	prod = 1.0
    	for d in xrange(dim):
    		myfunc = lambda x: eval_function_modlin([x], [level[d]], [index[d]])*rv.pdf(x)
    		res = quad(myfunc, 0, 1, epsabs=1e-14, epsrel=1e-12)
    		prod*= res[0]

	print "Ture ExpVal",level, index, prod
	print ".........................................................."
    

#-------------------------------------------------------------------------------
def run(grid, alpha, fac, training):
    errors = None
    gridSize = grid.getStorage().size()
    #print grid.getSize()

    #Parameters
    paramW = 0.01
    epsilon = 0.1
    imax = gridSize
    u_0 = 0
    
    residual = 1
    i = 1

    q = calcq(grid, training)
    #print "q value: ",q
    A = createOperationLaplace(grid)

    #This is just to initialize the pickle db to store the MCMC state
    model = pm.Model(input=make_model(grid, alpha, fac), name="sg_normal_indep")
    mcmc = pm.MCMC(model, name="MCMC", db="pickle", dbname="sg_mcmc.pickle")
    mcmc.db
    mcmc.sample(iter=100, burn=10, thin=1)
    mcmc.db.close()

    mcmc_expVal = computeNLterm(grid, alpha, fac)
    #print mcmc_expVal
    if 'toy' in options.data[0]:
    	compareExpVal(grid, mcmc_expVal, fac.dim)

    alpha_mask = DataVector(grid.getSize())
    alpha_mask.setAll(1.0)

    while residual > epsilon and i <= imax:
        
        desMat = createOperationMultipleEval(grid, training)
	b = q - mcmc_expVal
	b = DataVector(b)
	#print "b value: ", b
	lambdaVal = 1
	b_lambda = DataVector(gridSize)
	for k in range(gridSize):
		b_lambda[k] = float(b[k] / lambdaVal)

	b_lambda = DataVector(b_lambda)
	#print "b_lambda: ",b_lambda

	alpha_old = DataVector(alpha)
	## Conjugated Gradient method for sparse grids, solving A.alpha=b
	#print alpha
        res = conjugateGradient(b_lambda, alpha, imax, options.r, A, False, options.verbose, max_threshold=options.max_r)
        #print "Conjugate Gradient output:"
        #print "cg residual: ",res
   	print "cg alpha: ",alpha
    	print "old alpha: ",alpha_old

	if i == 1:
		paramW = 1	
	
   	val = DataVector(alpha)
   	for k in range(grid.getStorage().size()):
   		val[k] = val[k] * paramW
   	
   	for k in range(grid.getStorage().size()):
    		alpha[k] = alpha_old[k] + val[k]
	#print "new alpha: ",alpha
   	A_alpha = DataVector(grid.getStorage().size())
   	
	A.mult(alpha, A_alpha)
    	mcmc_expVal = computeNLterm(grid, alpha, fac)
    	
    	if 'toy' in options.data[0]:
    		compareExpVal(grid, mcmc_expVal, fac.dim)

   	q_val = q + mcmc_expVal

   	value = DataVector(grid.getStorage().size())
   	for k in range(grid.getStorage().size()):
   		value[k] = A_alpha[k] - q_val[k]
   	
    	summ = 0
	for k in range(grid.getStorage().size()):
		summ = summ + value[k]*value[k]

    	residual = np.sqrt(summ)

	i = i + 1
    	print "*****************Residual***************  ", residual
    	print "+++++++++++++++++i+++++++++++++++++++++   ",i-1

    	fac = updateFactorGraph(grid, alpha, fac)

	print "------------------------------factors---", fac.factors

	grid, alpha_mask = coarseningFunction(grid, alpha_mask, fac) 

    print "Alpha: ",alpha
    #print grid.getSize()
    return grid, alpha

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

    hist(result)
    show()

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    x = DataVector(training.getNrows())
    y = DataVector(training.getNrows()) 
    
    training.getColumn(0,x)
    training.getColumn(1,y)

    if dim == 2:
    	ax.scatter(x, y, c=result)
    elif dim == 3:
    	z = DataVector(training.getNrows())
    	training.getColumn(2,z)
    	ax.scatter(x, y, z, c=result)
    
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

    fac = factor_graph(int(dim))
    fac.create_factor_graph(int(level))

    while gsize != newGsize:
    	gsize = newGsize
    	grid, alpha = coarseningFunction(grid, alpha, fac)
    	newGsize = grid.getSize()
    
    grid, alpha = run(grid, alpha, fac, training)

    result = evaluateDensityFunction(grid,alpha,dim,training)

    print "Mean of the result: ",np.mean(result)
   
    if dim < 4:
    	visualizeResult(training, result, dim)
 
    if options.outfile:
        writeAlphaARFF(options.outfile, alpha)
    if options.gridfile:
        writeGrid(options.gridfile, grid)

    if(options.gnuplot != None):
        if(dim != 2):
            print("Wrong dimension for gnuplot-Output!")
        else:
            writeGnuplot(options.gnuplot, grid, alpha, options.res)

    return alpha

#-------------------------------------------------------------------------------
def buildYVector(data):
    return data["classes"]

#TODO - Update
def testValues(grid,alpha,result,classes):
    correct = 0
    for i in xrange(len(result)):
	val = 1
	if (result[i] >= 0.7):
		val = 2
	else:
		val = 1
        if val == classes[i]:
            correct = correct + 1

    print "Accuracy: ", float(correct)/len(result)

## Density Estimation with Test
#TODO - Update
def doDETest():
    # read data
    data = openFile(options.data[0])
    dim = data["data"].getNcols()
    numData = data["data"].getNrows()

    if options.verbose:
        print "Dimension is:", dim
        print "Size of datasets is:", numData
        print "Gridsize is:", grid.getSize()

    training = buildTrainingVector(data)

    test = openFile(options.test)
    test_data = buildTrainingVector(test)
    test_values = buildYVector(test)

    grid = constructGrid(dim)

    alpha = run(grid, training)
    
    testValues(grid, alpha, test_data, test_values)

    if options.outfile:
        writeAlphaARFF(options.outfile, alpha)
    if options.gridfile:
        writeGrid(options.gridfile, grid)

    if(options.gnuplot != None):
        if(dim != 2):
            print("Wrong dimension for gnuplot-Output!")
        else:
            writeGnuplot(options.gnuplot, grid, alpha, options.res)

    return alpha

#-------------------------------------------------------------------------------


if __name__=='__main__':
    	# Initialize OptionParser, set Options
    	parser = OptionParser()
    	parser.add_option("-l", "--level", action="store", type="int", dest="level", help="Gridlevel")
    	parser.add_option("-D", "--dim", action="callback", type="int",dest="dim", help="Griddimension", callback=callback_deprecated)
    	parser.add_option("-a", "--adaptive", action="store", type="int", default=0, dest="adaptive", metavar="NUM", help="Using an adaptive Grid with NUM of refines")
    	parser.add_option("--adapt_points", action="store", type="int", default=1, dest="adapt_points", metavar="NUM", help="Number of points in one refinement iteration")
    	parser.add_option("--adapt_rate", action="store", type="float", dest="adapt_rate", metavar="NUM", help="Percentage of points from all refinable points in one refinement iteration")
    	parser.add_option("--adapt_start", action="store", type="int", default=0, dest="adapt_start", metavar="NUM", help="The index of adapt step to begin with")
    	parser.add_option("--adapt_threshold", action="store", type="float", default=0.0, dest="adapt_threshold", metavar="NUM", help="The threshold, an error or alpha has to be greater than in order to be reined.")
    	parser.add_option("-m", "--mode", action="store", type="string", default="apply", dest="mode", help="Specifies the action to do. Get help for the mode please type --mode help.")
    	parser.add_option("-C", "--CMode", action="store", type="string", default="laplace", dest="CMode", help="Specifies the action to do.")
    	parser.add_option("-f", "--foldlevel", action="store", type="int",default=10, metavar="LEVEL", dest="f_level", help="If a fold mode is selected, this specifies the number of sets generated")
    	parser.add_option("--onlyfoldnum", action="store", type="int", default=-1, metavar="I", dest="onlyfoldnum", help="Run only fold I in n-fold cross-validation. Default: run all")
    	parser.add_option("-L", "--lambda", action="store", type="float",default=0.000001, metavar="LAMBDA", dest="regparam", help="Lambda")
    	parser.add_option("-i", "--imax", action="store", type="int",default=500, metavar="MAX", dest="imax", help="Max number of iterations")
    	parser.add_option("-r", "--accuracy", action="store", type="float",default=0.0001, metavar="ACCURACY", dest="r", help="Specifies the accuracy of the CG-Iteration")
    	parser.add_option("--max_accuracy", action="store", type="float", default=None, metavar="ACCURACY", dest="max_r", help="If the norm of the residuum falls below ACCURACY, stop the CG iterations")
    	parser.add_option("-d", "--data", action="append", type="string", dest="data", help="Filename for the Datafile.")
    	parser.add_option("-t", "--test", action="store", type="string", dest="test", help="File containing the testdata")
    	parser.add_option("--val_proportion", action="store", type="string", dest="val_proportion", metavar="p", default=None,
                      help="Proportion (0<=p<=1) of training data to take as validation data (if applicable)")
    	parser.add_option("-A", "--alpha", action="store", type="string", dest="alpha", help="Filename for a file containing an alpha-Vector")
    	parser.add_option("-o", "--outfile", action="store", type="string", dest="outfile", help="Filename where the calculated alphas are stored")
    	parser.add_option("--gridfile", action="store", type="string", dest="gridfile", help="Filename where the resulting grid is stored")
    	parser.add_option("-g", "--gnuplot", action="store", type="string", dest="gnuplot", help="In 2D case, the generated can be stored in a gnuplot readable format.")
    	parser.add_option("--gnuplotdata", action="store_true", dest="gnuplotdata", default=False, help="In 2D case, the generated can be stored in a gnuplot readable format.")
    	parser.add_option("-R", "--resolution", action="store", type="int",default=50, metavar="RESOLUTION", dest="res", help="Specifies the resolution of the gnuplotfile")
    	parser.add_option("-s", "--stats", action="store", type="string", dest="stats", help="In this file the statistics from the test are stored")
    	parser.add_option("-p", "--polynom", action="store", type="int", default=0, dest="polynom", help="Sets the maximum degree for high order basis functions. Set to 2 or larger to activate. Works only with 'identity' and 'fold'-modes.")
    	parser.add_option("-b", "--border", action="store_true", default=False, dest="border", help="Enables special border base functions")
    	parser.add_option("--boundary", action="store", type="int", default=False, dest="boundary", help="Use basis functions on boundary (trapezoid boundary==1, boundary==2)")
    	parser.add_option("--trapezoid-boundary", action="store_true", default=False, dest="trapezoidboundary", help="Enables boundary functions that have a point on the boundary for every inner point (Trapezoid)")
    	parser.add_option("--complete-boundary", action="store_true", default=False, dest="completeboundary", help="Enables boundary functions that have more points on the boundary than inner points")
    	parser.add_option("-v", "--verbose", action="store_true", default=False, dest="verbose", help="Provides extra output")
    	parser.add_option("--normfile", action="store", type="string", dest="normfile", metavar="FILE", help="For all modes that read data via stdin. Normalizes data according to boundaries in FILE")
    	parser.add_option("--reuse", action="store_true", default=False, dest="reuse", help="Reuse alpha-values for CG")
    	parser.add_option("--seed", action="store", type="float", dest="seed", help="Random seed used for initializing")
    	parser.add_option("--regression", action="store_true", default=False, dest="regression", help="Use regression approach.")
    	parser.add_option("--checkpoint", action="store", type="string", dest="checkpoint", help="Filename for checkpointing. For fold? and test. No file extension.")
    	parser.add_option("--grid", action="store", type="string", dest="grid", help="Filename for Grid-resume. For fold? and test. Full filename.")
    	parser.add_option("--epochs_limit", action="store", type="int", default="0", dest="epochs_limit", help="Number of refinement iterations (epochs), MSE of test data have to increase, before refinement will stop.")
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
                      'required_options': ['data', ['level', 'grid']],
                      'action': doDensityEstimation},
                 'test'     : {'help': "learn a dataset with a test dataset",
                      'required_options': ['data', 'test', ['level', 'grid']],
                      'action': doDETest}

            	}

	
	exec_mode(options.mode.lower())
