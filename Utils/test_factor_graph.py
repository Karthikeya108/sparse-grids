from factor_graph import *

fac = factor_graph(4)
fac.create_factor_graph(3)

print "Node List: ",fac.nodes
print "Complete factors: ",fac.factors
fac.coarsen_factor_graph([(1,),(1, 2, 4), (1, 3, 4), (2, 3, 4)])

print "Factors after coarsening: ",fac.factors

