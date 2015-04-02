from optparse import OptionParser
from tools import *
from SGDensityEstimator  import *
import sys
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
#Required for 'visualizeResult' method
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

def exec_mode(mode):
    """
    Checks whether a valid mode is specified,
    whether all required options for the mode are given and
    executes the corresponding action (function)
    
    # @param mode current mode
    """
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
    
def open_file(filename):
    """
    arguments -- filename
    returns -- the data stored in the file as a set of arrays
    """
    if "arff" in filename:
        return readData(filename)
    else:
        return readDataTrivial(filename)
        
def visualize_result(training, result, dim):

    result = np.exp(result)
   
    for i in xrange(len(result)):
        if result[i] == 1:
            print "Debug result: ",result[i], " ",x[i]

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
        #result = result * 4
        pdf_true = norm(0.5, 0.1).pdf(x)
        fig, ax = plt.subplots()
        ax.scatter(x, result, color='blue')
        ax.scatter(x, pdf_true, color='gray')
    
    plt.show()

def classify(result, labels):
    result_labels = []
    for i in xrange(len(labels)):
        if result[i] <= 0:
            result_labels.append(-1.0)
        else:
            result_labels.append(1.0)
    
    correct = 0
    for k in xrange(len(result_labels)):
        if result_labels[k] == labels[k]:
            correct = correct + 1

    print "Accuracy: ",float(correct)/len(labels)

    print result_labels
        
def do_density_estimation():
    data = open_file(options.data[0])
    dim = data["data"].getNcols()
    num_data = data["data"].getNrows()
    
    sgde = SG_DensityEstimator(data, options.level, options.regparam, options.regstr, options.alpha_threshold)
    
    grid, alpha = sgde.compute_coefficients()

    result = sgde.evaluate_density_function(dim, data["data"])

    print "Mean of the result: ",np.mean(np.exp(result))

    if options.classify:
        classify(result, data["classes"])
   
    if dim < 4:
        visualize_result(data["data"], result, dim)
        
if __name__=='__main__':
    """Initialize OptionParser, set Options"""
    parser = OptionParser()
    parser.add_option("-l", "--level", action="store", type="int", dest="level", help="Gridlevel")
    parser.add_option("-m", "--mode", action="store", type="string", default="apply", dest="mode", help="Specifies the action to do. Get help for the mode please type --mode help.")
    parser.add_option("-L", "--lambda", action="store", type="float",default=0.01, metavar="LAMBDA", dest="regparam", help="Lambda")
    parser.add_option("-R", "--regstr", action="store", type="string",default='laplace', metavar="REGSTR", dest="regstr", help="RegStrategy")
    parser.add_option("-a", "--alphath", action="store", type="float",default=0.25, metavar="AlphaThreshold", dest="alpha_threshold", help="AlphaThreshold")
    parser.add_option("-i", "--imax", action="store", type="int",default=500, metavar="MAX", dest="imax", help="Max number of iterations")
    parser.add_option("-d", "--data", action="append", type="string", dest="data", help="Filename for the Datafile.")
    parser.add_option("-v", "--verbose", action="store_true", default=False, dest="verbose", help="Provides extra output")
    parser.add_option("--classify", action="store_true", default=False, dest="classify", help="Classify the data")
    
    (options,args)=parser.parse_args()
    
    modes = {
        'density'   : {'help': "learn a dataset",
        'required_options': ['data', 'level'],
        'action': do_density_estimation}
            }
   
    exec_mode(options.mode.lower())
