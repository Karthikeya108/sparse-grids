import itertools
'''
This class allows to create a factor graph and stores it. It also allows to coarsen the factor graph by deleting listed factors.

Class Variables:  
dim - Dimension of the dataset (Number of nodes in the graph)
nodes - list of nodes (in integer format)
factors - A dictionary with <key> indicating the number of interacting nodes and each <value> is a <list> of <set> objects indicating the factors

Class Methods:
__init__(dim) - Constructor with dimension as the parameter
create_factor_graph(max_factors) - Create the factor graph based on the specified <max_factors> 
coarsen_factor_graph(delete_list) - Coarsens the factor graph by deleting listed factors passed as parameter
'''
class factor_graph:
	def __init__(self, dim):
		self.dim = dim
		self.nodes = []
		for n in xrange(dim):
                        self.nodes[len(self.nodes):] = [n]
		
		self.factors = {}

        def create_factor_graph(self, max_factors):
		#if self.dim < max_factors:
		#	print "<max_factors> cannot be greater than number of nodes in the graph"
		#	exit(0)
		for f in xrange(1,max_factors+1):
			combination = itertools.combinations(self.nodes, f)
			self.factors[f] = list(combination)
		print self.factors

	def coarsen_factor_graph(self, delete_list):
		delete_list.sort(key = len, reverse=True)
		for i in delete_list:
			if len(i)+1 in self.factors:
				if len(self.factors[len(i)+1]) > 0:
					print "Cannot delete the following elements since the high order interactions are still present: ",delete_list
					return
			self.factors[len(i)].remove(tuple(sorted(i)))
			delete_list.remove(i)
		
	
