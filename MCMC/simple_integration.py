from numpy import *
from skmonaco import mcquad

#Exponent part of the Normal Distribution -> exp(- (x - mean)^2 / (2 * sigma^2))
def f(x):
    return exp(- (x - 0)**2 / (2 * (1)**2 ))

#skmonaco
result, error = mcquad(lambda x: f(x), xl=[-5.], xu=[5.], npoints=100000)

#Constant part of the Normal Distribution -> sqrt(2 * pi * sigma^2)
const = sqrt(2 * pi * (1)**2)

print "Normalization factor: ",const

print "MCMC Integration result: ",result
print "Error: ",error

