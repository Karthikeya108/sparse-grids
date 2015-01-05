import pymc as pm
import numpy as np
import scipy as sp
from pylab import hist, plot, show, legend

# Define the paraemters for the gaussian distribution function
mu1d = 0.0
sigma1d = 1.0
size = 10000 # sampling size
epsilon = 0.5

# we sample from a unit distribution
x_uniform = np.random.random(size)*200-100 # sample on the interval [-100, 100)
y_uniform = sp.stats.norm.pdf(x_uniform, loc=mu1d, scale=sigma1d)/sp.stats.uniform.pdf(x_uniform, loc=-100, scale=200)
points_uniform = np.array([[s, np.sum(y_uniform[:s])/s] for s in xrange(500, size, 500)])

# we sample from a similar gaussian distribution
x_gauss = np.random.randn(size)*(sigma1d + epsilon) + (mu1d + epsilon)
y_gauss= sp.stats.norm.pdf(x_gauss, loc=mu1d, scale=sigma1d)/sp.stats.norm.pdf(x_gauss, loc=(mu1d+epsilon), scale=(sigma1d+epsilon))
points_gauss = np.array([[s, np.sum(y_gauss[:s])/s] for s in xrange(500, size, 500)])

# plot the approximation for different number of sampling points
plot(points_uniform[:,0], points_uniform[:,1], 'r-', label="uniform")
plot(points_gauss[:,0], points_gauss[:,1], 'g-', label="gaussian")
plot(points_gauss[:,0], np.array([1.0]*points_gauss.shape[0]), 'k--', label='correct')
legend(loc=4)
show()


