import sys, os
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
import pysgpp
import numpy as np
from collections import defaultdict

class GridCoarsener(object):
    
    def set_mask(self, mask):
        """ set new current mask: 1s will be REMOVED! """
        self.mask = mask 
        
    
    def compress_array(self, array):
        print self.mask
        masked_array = np.ma.array(array, mask=self.mask)
        return masked_array.compressed()
    

    def coefficient_thresholding(self, grid, alpha, factor_graph, *args):
           """
           Removes the less important factors from the factor graph
           arguments -- Grid, Co-efficients (alpha), factor graph and coefficient_threshold 
           returns -- updated factor graph
           """
           alpha_threshold = args[0]
           factor_alpha_collection = defaultdict(list)
           max_interaction = 0
           for grid_point_index in xrange(grid.getSize()):
               """Store the grid point index and corresponding tuple of interacting factors"""
               factor = factor_graph.get_grid_point_factor(grid, grid_point_index)
               factor_alpha_collection[factor].append(alpha[grid_point_index])
                      
               # TODO: why is it here?          
               if max_interaction < len(factor):
                   max_interaction = len(factor)
          
           # FIXME this part has nothing to do with coefficient thersholding
           # it should be moved somewhere else
           factor_graph.max_interaction_thresholding(max_interaction)
           
           factor_alpha_collection = dict(factor_alpha_collection)

           # FIXME: actually, the strategy-specific part is only in these last lines
           delete_list = []
           for factor, alphas_in_factor in factor_alpha_collection.iteritems():
               """
               if the average absolute values of the alphas (co-efficients) corresponding 
               to a <interacting factor> tuple is less than some <alpha_threshold> then 
               add the <interacting factor> tuple to the delete list
               """
               print "alpha_threshold: ",alpha_threshold
               print "mean of all alphas: ",np.mean(np.absolute(alpha))
               if np.mean(np.absolute(alphas_in_factor)) < alpha_threshold:
                   delete_list[len(delete_list):] = [factor]
           factor_graph.coarsen_factor_graph(delete_list)
    
           return factor_graph


    def coarsen_grid(self, grid, alpha, factor_graph):
        """
        Updates the Grid based on the components in the factor graph
        arguments -- Grid, Co-efficients (alpha), factor graph 
        returns -- updated grid and coefficients
        """
        dim = factor_graph.dim
        storage = grid.getStorage()
        delete_counter = 0
        #mask = pysgpp.DataVector(grid.getSize())
        #mask.setAll(1)
        for grid_point_index in xrange(grid.getSize()):
            factor = factor_graph.get_grid_point_factor(grid, grid_point_index)
            if len(factor) != 0:
                # If the corresponding interaction is not in the factor graph 
                # then set the correspoding alpha to 0"""
                if not factor_graph.contains(factor):
                    alpha[grid_point_index] = 0
                    delete_counter += 1
                    
            # to coarsen all needed points at once, assume all points are leafs
            # the leaf property is recalculated automatically after coarsening
            storage.get(grid_point_index).setLeaf(True)

        alpha = pysgpp.DataVector(alpha)
        # SG++ coarsening routine
        coarseningFunctor = pysgpp.SurplusCoarseningFunctor(alpha, delete_counter, 0.5)
        grid.createGridGenerator().coarsen(coarseningFunctor, alpha)
        grid.getStorage().recalcLeafProperty()
        return grid, alpha.array()
    
    
    def update_grid(self, grid, alpha, factor_graph, coarsening_strategy=coefficient_thresholding, *args):
        """
        coarsens the grid
        arguments -- Grid, Co-efficients, factor graph, coarsening strategy (function), respective parameters
        returns -- Updated - Grid, Coefficients and factor graph
        """
        # TODO: if the coarsening works without the loop, this function is really not needed
        factor_graph = coarsening_strategy(grid, alpha, factor_graph, *args)
        gsize = grid.getSize()
        newgsize = 0
        i = 0

        while gsize != newgsize:
            gsize = newgsize
            grid, alpha = self.coarsen_grid(grid, alpha, factor_graph)
            #self.mask = mask
            newgsize = grid.getSize()
            i += 1
            
        print "Numbe or recoarsenings needed", i
        print "Alpha :", alpha
    
        return grid, alpha, factor_graph
    


            

