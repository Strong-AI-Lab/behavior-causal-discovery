
import pandas as pd
import numpy as np

class CausalGraphFormatter():


    def __init__(self, graph : np.array, val_matrix : np.array, var_names : list = None):
        self.graph = graph
        self.val_matrix = val_matrix
        self.var_names = var_names

    
    def get_graph(self):
        return self.graph
    
    def get_val_matrix(self):
        return self.val_matrix
    
    def get_var_names(self):
        return self.var_names

    def get_results(self):
        return {**{"graph": self.graph, "val_matrix": self.val_matrix}, **({"var_names": self.var_names} if self.var_names is not None else {})}

    
    def low_filter(self, abs_min = 0.5):
        graph = self.graph.copy()
        val_matrix = self.val_matrix.copy()

        graph[np.abs(val_matrix) < abs_min] = ""
        val_matrix[np.abs(val_matrix) < abs_min] = 0

        return CausalGraphFormatter(graph, val_matrix)

    
    def high_filter(self, abs_max = 0.5):
        graph = self.graph.copy()
        val_matrix = self.val_matrix.copy()

        graph[np.abs(val_matrix) > abs_max] = ""
        val_matrix[np.abs(val_matrix) > abs_max] = 0

        return CausalGraphFormatter(graph, val_matrix)

    def var_filter(self, cause_vars_to_remove : list = None, effect_vars_to_remove : list = None, remove_bidirectional : bool = False):
        graph = self.graph.copy()
        val_matrix = self.val_matrix.copy()

        if cause_vars_to_remove is None:
            cause_vars_to_remove = []
        if effect_vars_to_remove is None:
            effect_vars_to_remove = []

        # If the link is bidirectional, do not remove
        for i in cause_vars_to_remove:
            graph[i,:,:][np.where(graph[i,:,:] != "o-o")] = ""
            val_matrix[i,:,:][np.where(graph[i,:,:] != "o-o")] = 0
        for j in effect_vars_to_remove:
            graph[:,j,:][np.where(graph[:,j,:] != "o-o")] = ""
            val_matrix[:,j,:][np.where(graph[:,j,:] != "o-o")] = 0

        # If the link is bidirectional and both ends belong to one of the sets to remove, remove
        if remove_bidirectional:
            bidirectional_var_to_remove = cause_vars_to_remove + effect_vars_to_remove
            for i in bidirectional_var_to_remove:
                for j in bidirectional_var_to_remove:
                    graph[i,j,:][np.where(graph[i,j,:] == "o-o")] = ""
                    val_matrix[i,j,:][np.where(graph[i,j,:] == "o-o")] = 0

        return CausalGraphFormatter(graph, val_matrix)

    def corr_filter(self):
        graph = self.graph.copy()
        val_matrix = self.val_matrix.copy()
        
        graph[np.where(graph == "o-o")] = ""
        val_matrix[np.where(graph == "o-o")] = 0

        return CausalGraphFormatter(graph, val_matrix)
    
    def row_filter(self, var_names : list = None):
        graph = self.graph.copy()
        val_matrix = self.val_matrix.copy()

        del_idxs = (graph=="").all(axis=(0,2)) & (graph=="").all(axis=(1,2))
        
        graph = graph[~del_idxs,:,:]
        graph = graph[:,~del_idxs,:]
        val_matrix = val_matrix[~del_idxs,:,:]
        val_matrix = val_matrix[:,~del_idxs,:]

        if var_names is None and self.var_names is not None:
            var_names = self.var_names

        if var_names is not None:
            var_names = [v for i,v in enumerate(var_names) if not del_idxs[i]]

        return CausalGraphFormatter(graph, val_matrix, var_names=var_names)


