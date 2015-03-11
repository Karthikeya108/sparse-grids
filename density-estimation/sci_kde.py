import numpy as np
from sklearn.neighbors.kde import KernelDensity
from pylab import hist, show
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats.distributions import norm

def kde_scipy(data, dim):
	kde = stats.gaussian_kde(data)
	kde.set_bandwidth(bw_method='silverman')
	density = kde(data)

	if dim < 4:
		plotDD(data, density, dim)

	print "Scipy Mean: ", np.mean(density)
	return density

def kde_scikit(data, dim):
	data = data.T
	kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(data)
	density = kde.score_samples(data)

	if dim < 4:
		plotDD(data.T, np.exp(density), dim)

	print "Scikit Mean: ", np.mean(np.exp(density))

        return density

def plotDD(data, density, dim):
        if dim == 3:
		fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
                x, y, z = data
                p = ax.scatter(x, y, z, c=density)
        elif dim == 2:
		fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
                x, y = data
                p = ax.scatter(x, y, c=density)
	else:
		x = data.T
		pdf_true = norm(0.5, 0.1).pdf(x)
		fig, ax = plt.subplots()
		ax.plot(x, density, color='blue', alpha=0.5, lw=3)
		ax.fill(x, pdf_true, ec='gray', fc='gray', alpha=0.4)
		
        #fig.colorbar(p)
	plt.show()

def kde_eval(filename, dim):
	f = open(filename,'r')

	X = {}
        for i in xrange(dim):
		X[i] = []

	cls = []

        for line in f:
                row = line.split(' ')
		for k in xrange(dim):
			X[k].append(float(row[k]))
			cls.append(row[k+1])

	data = []

	for i in xrange(dim):
		data.append(X[i])

	values = np.asarray(data)

	scipy_result = kde_scipy(values, dim)
	scikit_result = kde_scikit(values, dim)

	return scipy_result, scikit_result


#val1, val2 = kde_eval('data/ripleyGarcke.train', 2)
#val1, val2 = kde_eval('data/3DOption/X_normalized.txt', 2)
val1, val2 = kde_eval('data/toy1.txt', 1)

#x_axis = np.arange(0, 500, 0.1)

#plt.plot(x_axis, val1, 'r--', x_axis, val2, 'b-')
#plt.show()

    
#hist(val1)
#show()

#hist(val2)
#show()

