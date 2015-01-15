from numpy import *
import math
from skmonaco import mcquad

# covariance matrix
sigma = matrix([[0.5, 0, 0, 0],
           [0, 0.5, 0, 0],
           [0, 0, 0.5, 0],
           [0, 0, 0, 0.5]
          ])
# mean vector
mu = array([0,0,0,0])

def norm_pdf_multivariate(x, mu, sigma):

        x_mu = matrix(x - mu)
        inv = sigma.I        
        result = exp(-0.5 * (x_mu * inv * x_mu.T))
        return result


det = linalg.det(sigma)
if det == 0:
	raise NameError("The covariance matrix can't be singular")

norm_const =  math.pow((2*pi),float(len(mu))/2) * math.pow(det,1.0/2)

print "norm_const: ",norm_const

#Proper selection of the xl and xu is very important to get the right result
result, error = mcquad(lambda x: norm_pdf_multivariate(x, mu, sigma), xl=[-5.], xu=[5.], npoints=100000)

print "MCMC Integration result: ",result
print "Error: ",error
