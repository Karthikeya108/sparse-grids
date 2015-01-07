from numpy import *
import math
from skmonaco import mcquad

# covariance matrix
sigma = matrix([[0.1, 0, 0, 0],
           [0, 0.1, 0, 0],
           [0, 0, 0.1, 0],
           [0, 0, 0, 0.1]
          ])
# mean vector
mu = array([0,0,0,0])

# input
#x = array([2.1,3.5,8, 9.5])

def norm_pdf_multivariate(x, mu, sigma):

        #norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = matrix(x - mu)
        inv = sigma.I        
        result = exp(-0.5 * (x_mu * inv * x_mu.T))
        return result


det = linalg.det(sigma)
if det == 0:
	raise NameError("The covariance matrix can't be singular")

norm_const =  math.pow((2*pi),float(len(mu))/2) * math.pow(det,1.0/2)

print "norm_const: ",norm_const

result, error = mcquad(lambda x: norm_pdf_multivariate(x, mu, sigma), xl=[-1.], xu=[1.], npoints=100000)

print "MCMC Integration result: ",result
print "Error: ",error
