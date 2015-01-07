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

from skmonaco import mcquad
import numpy as np
from MCMC_GridUtils import *
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
    return readData(filename)


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

def basisFunction(gpoint, gdim, val):
	prod = 1
	for d in range(gdim):
		levInd = gpoint.get(d)
		level = levInd[0]
		index = levInd[1]
		basisMap = (2**level)*val[d] - index
		prod = prod * np.maximum(1 - np.absolute(basisMap), 0)
        return prod

def calcq(grid,values):
        print "Calculating q"
	q = DataVector(grid.getStorage().size())
	noRows = values.getNrows()
	noCols = values.getNcols()
	for g in range(grid.getStorage().size()):
		gp = grid.getStorage().get(g)
		summ = 0
		for i in range(noRows):
                	prodd = 1
			val = DataVector(noCols)
			values.getRow(i,val)
			for j in range(noCols):
				levInd = gp.get(j)
				level = levInd[0]
				index = levInd[1]
				x = val[j]
				if not math.isnan(x):
					basisMap = (2**level)*x - index
					prodd = prodd * np.maximum(1 - np.absolute(basisMap), 0)
                	summ = summ + prodd
		q[g] = summ/noRows

        return q

def denFunction(grid, alpha, x):
	summ = 0
	gdim = grid.getStorage().dim()
	gsize = grid.getStorage().size()
	x = DataVector(gsize)
	for i in range(gsize):
		gpoint = grid.getStorage().get(i)
		summ = summ + alpha[i] * basisFunction(gpoint, gdim, x)
	return np.exp(summ)
	
def calcDenominator(grid, alpha):
	result, error = mcquad(lambda x: denFunction(grid, alpha, x), xl=[0.], xu=[1.], npoints=10000)
	print "den NL Result: ",result
	print "den NL Error: ",error
	return result

def wholeFunction(grid, alpha, g, x):
	summ = 0
	gdim = grid.getStorage().dim()
	gsize = grid.getStorage().size()
	for i in range(gsize):
		gpoint = grid.getStorage().get(i)
		summ = summ + alpha[i] * basisFunction(gpoint, gdim, x)
        numerator = np.exp(summ)
	
	ggpoint = grid.getStorage().get(g)
	return basisFunction(ggpoint, gdim, x) * numerator

def make_model(grid, alpha, dim):
    """ Creates the variables for my graphical model. This function is not generic"""
    x = np.empty(dim, dtype=object)
    for i in xrange(dim):
        #x[i] = pm.distributions.Uniform('x'+str(i), lower=0, upper=1.0)
        # Normal distribution for diagnostics
        x[i] = pm.distributions.Normal('x'+str(i), mu= 0.5, tau=1)
    
    # univariate potentials
    def psi_i_logp(anova, i): 
        return anova[(i,)]
    
    psi_i = np.empty(dim, dtype=object)
    
    # bivariate potentials
    def psi_ij_logp(anova, i, j):
    	return anova[tuple(sorted([i,j]))]

    psi_ij = np.empty(dim, dtype=object)
     
    # anova components as deterministic variable
    def compute_ANOVA_components(x, alpha, grid):
        """Evaluates the sparse grid at the point x and returns 
           the individual ANOVA components of the result"""
        x = np.hstack(x).reshape(1, -1)
        dataPoint = DataMatrix(x)
        result = DataVector(grid.getSize())
        result.setAll(0.0)
        a = DataVector(1)
        a[0] = 1.0
        opMultipleEval = createOperationMultipleEval(grid, dataPoint)
        opMultipleEval.multTranspose(a, result)
        result = result.array()*alpha.array()
        
        anovaComponents = {(-1,):0}
        for i in xrange(dim):
            anovaComponents[(i,)] = 0
            anovaComponents[tuple(sorted((i, (i+1)%dim)))] = 0

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
    
    
    for i in xrange(dim):  
        psi_i[i] = pm.Potential(logp = lambda anova, i=i: psi_i_logp(anova, i),
                            name = 'psi_i'+str(i),
                            parents = {'anova': anova},
                            doc = 'A univariate potential',
                            verbose = 0,
                            cache_depth = 1)
    
    
    for i in xrange(dim):  
        j = (i+1)%dim # this is just how I connect the variables in my graphical model. It's not generic
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

def calcNLterm(grid, alpha):
        print "Calculating the Non-linear term"
	denominator = calcDenominator(grid, alpha)
	nlTerm = DataVector(grid.getStorage().size())
	nlError = DataVector(grid.getStorage().size())
	gdim = grid.getStorage().dim()

	for g in range(grid.getStorage().size()):
        	func = lambda x: wholeFunction(grid, alpha, g, x)
		
		pointDV = pysgpp.DataVector(gdim)
	
		storage = grid.getStorage()	
		for j in xrange(grid.getSize()):
    			grid_index = storage.get(j)
    			grid_index.getCoords(pointDV)
    			x = pointDV.array()
    			alpha[j] = func(x)
		
		opHierarchisation = pysgpp.createOperationHierarchisation(grid)
		opHierarchisation.doHierarchisation(alpha)

		model = pm.Model(input=make_model(grid, alpha, gdim), name="sg_normal_indep")
		mcmc = pm.MCMC(model, name="MCMC")

		mcmc.sample(iter=10000, burn=100, thin=5)

		anovaComponents = mcmc.trace('anova')[:]
		result = [len(anovaComponents)]
		for i in anovaComponents:
			summ = 0
			for j in i.values():
				summ = summ + j
			result.append(summ/len(i.values()))

		print result

	return result

def calcA(grid, training):
        print "Calculating A"
	return np.matrix(np.identity(grid.getSize()), copy=False)

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

#-------------------------------------------------------------------------------
def run(grid, alpha, training):
    errors = None
    gridSize = grid.getStorage().size()
    print grid.getSize()

    #Parameters
    paramW = 0.01
    epsilon = 0.1
    imax = gridSize
    u_0 = 0
    
    residual = 1
    i = 1

    q = calcq(grid, training)
    print "q value: ",q
    A = createOperationLaplace(grid)

    #alpha = DataVector(grid.getStorage().size())
    #alpha.setAll(0.0)
    
    nlterm = calcNLterm(grid, alpha)
    
    while residual > epsilon and i <= imax:
        
        desMat = createOperationMultipleEval(grid, training)
	b = q - nlterm
	b = DataVector(b)
	print "b value: ", b
	lambdaVal = 1
	b_lambda = DataVector(gridSize)
	for k in range(gridSize):
		b_lambda[k] = float(b[k] / lambdaVal)

	b_lambda = DataVector(b_lambda)
	print "b_lambda: ",b_lambda

	alpha_old = DataVector(alpha)
	## Conjugated Gradient method for sparse grids, solving A.alpha=b
	print alpha
        res = conjugateGradient(b_lambda, alpha, imax, options.r, A, False, options.verbose, max_threshold=options.max_r)
        #print "Conjugate Gradient output:"
        print "cg residual: ",res
   	print "cg alpha: ",alpha
    	print "old alpha: ",alpha_old

	if i == 1:
		paramW = 1	
	
   	val = DataVector(alpha)
   	for k in range(grid.getStorage().size()):
   		val[k] = val[k] * paramW
   	
   	for k in range(grid.getStorage().size()):
    		alpha[k] = alpha_old[k] + val[k]
	print "new alpha: ",alpha
   	A_alpha = DataVector(grid.getStorage().size())
   	
	A.mult(alpha, A_alpha)
    	nlterm = calcNLterm(grid, alpha)
   	q_val = q + nlterm

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

    print "Alpha: ",alpha
    print grid.getSize()
    return alpha

#-------------------------------------------------------------------------------
## Density Estimation
def doDensityEstimation():
    # read data
    data = openFile(options.data[0])
    dim = data["data"].getNcols()
    numData = data["data"].getNrows()

    level = options.level

    if options.verbose:
        print "Dimension is:", dim
        print "Size of datasets is:", numData
        print "Gridsize is:", grid.getSize()

    training = buildTrainingVector(data)

    #grid = constructGrid(dim)

    grid = Grid.createLinearGrid(dim)
    generator = grid.createGridGenerator()
    generator.regular(level)
    print "Grid size:", grid.getSize()

    gsize = grid.getSize()
    newGsize = 0

    alpha = DataVector(grid.getSize())
    alpha.setAll(1.0)

    while gsize != newGsize:
    	gsize = newGsize
    	grid, alpha = coarseningFunction(grid, dim)
    	newGsize = grid.getSize()
    
    alpha = run(grid, alpha, training)   
 
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

def testValues(grid,alpha,test,classes):
    p = DataVector(test.getNcols())
    correct = 0
    for i in xrange(test.getNrows()):
        test.getRow(i,p)
        val = createOperationEval(grid).eval(alpha,p)
        if val == classes[i]:
            correct = correct + 1

    print "Accuracy: ", float(correct)/test.getNrows()

## Density Estimation with Test
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
