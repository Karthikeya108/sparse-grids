import itertools
from sets import Set

class Factor_Graph:
    '''
This class allows to create a factor graph and stores it. It also allows to coarsen 
the factor graph by deleting listed factors.

Class Variables:  
dim - Dimension of the dataset (Number of nodes in the graph)
nodes - list of nodes (in integer format)
factors - A dictionary with <key> indicating the number of interacting nodes and
each <value> is a <list> of <set> objects indicating the factors

Class Methods:
__init__(dim) - Constructor with dimension as the parameter
create_factor_graph(max_factors) - Create the factor graph based on the specified 
<max_factors> 
coarsen_factor_graph(delete_list) - Coarsens the factor graph by deleting listed 
factors passed as parameter
'''
    
    def __init__(self, dim):
        self.dim = dim
        self.nodes = []
        for n in xrange(dim):
                        self.nodes[len(self.nodes):] = [n]
        
        self.factors = {}


    #def __repr__(self):
    #    return ""+self


    def create_factor_graph(self, max_factors):
        #if self.dim < max_factors:
        #   print "<max_factors> cannot be greater than number of nodes in the graph"
        #   exit(0)
        for f in xrange(1,max_factors+1):
            combination = itertools.combinations(self.nodes, f)
            self.factors[f] = list(combination)
        print self.factors


    def coarsen_factor_graph(self, delete_list):
        if len(delete_list) > 0:
            delete_list.sort(key = len, reverse=True)
            for i in delete_list:
                d = i
                ## FIXME: why is it the double loop over j???
                for j in xrange(len(d)+1,self.dim+1):
                    if j in self.factors and len(d)>0:
                        print(len(d))
                        for k in self.factors[j]:
                            if Set(d).issubset(Set(k)) and len(d)>0: 
                                print("Cannot delete the following elements since the high order interactions are still present ")
                                print("Following element ignored from the delete_list: ")
                                print(d)
                                d = tuple()
                #Individual factors should not be deleted
                if len(d) > 1:
                    print(d)
                    print "Deleting factors: ",d
                    self.factors[len(d)].remove(tuple(sorted(d)))
                    print(self.factors)
                    
                    
    def get_grid_point_factor(self, grid, grid_point_index):
        grid_index = grid.getStorage().get(grid_point_index)
        factor = tuple()
        for d in xrange(self.dim):
            """Fetch the interacting factors"""
            if grid_index.getLevel(d) != 1:
                factor = factor + (d,)
        return tuple(sorted(factor))
    
    
    def max_interaction_thresholding(self, max_interact):
        """
           Delete all the higher order interacting factors in the factor_graph which are higher 
           than the maximum length of the <interacting factors> obtained in the previuos step
        """
        if self.dim > max_interact+1:
            for k in xrange(self.dim-1,max_interact,-1):
                self.factors[k] = []
                
    def contains(self, factor):
        """checks if the factor graph contains given factor"""
        return factor in self.factors[len(factor)]
