import numpy as np
from sklearn.neighbors.kde import KernelDensity
from pylab import hist, show
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def kde_scipy(data):
	kde = stats.gaussian_kde(data)
	density = kde(data)

	fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
	x, y = data
	ax.scatter(x, y, c=density)
	plt.show()

	print "Scipy Mean: ", np.mean(density)
	return density


def kde_scikit(data):
	data = data.T
	kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)
	density = kde.score_samples(data)

	fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        x, y = data.T
        ax.scatter(x, y, c=np.exp(density))
        plt.show()

	print "Scikit Mean: ", np.mean(np.exp(density))

        return density

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

	scipy_result = kde_scipy(values)
	scikit_result = kde_scikit(values)

	return scipy_result, scikit_result


#val1, val2 = kde_eval('data/ripleyGarcke.train', 2)
val1, val2 = kde_eval('data/3DOption/X_normalized.txt', 2)

#x_axis = np.arange(0, 500, 0.1)

#plt.plot(x_axis, val1, 'r--', x_axis, val2, 'b-')
#plt.show()

    
hist(val1)
show()

hist(val2)
show()

