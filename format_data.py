
import pandas as pd
import re
import numpy as np


class PandasFormatter():

    @staticmethod
    def format_df(column : pd.Series):
        column = column.str.lower()
        column = column.str.replace(' ', '_')
        return column

    @staticmethod
    def format_column(column : str):
        column = column.lower()
        column = column.replace(' ', '_')
        return column


    def __init__(self, df : pd.DataFrame, 
                        behaviour_key : str = "Behaviour", 
                        individual_key : str = "ID", 
                        zone_key : str = "Zone", 
                        time_key : str = "Time", 
                        close_neighbour_key : str = "Close_neighbours", 
                        distant_neighbour_key : str = "Distant_neighbours"):
        self.df = df
        self.behaviour_key = behaviour_key
        self.individual_key = individual_key
        self.zone_key = zone_key
        self.time_key = time_key
        self.close_neighbour_key = close_neighbour_key
        self.distant_neighbour_key = distant_neighbour_key

        self.individuals = None
        self.behaviours = None
        self.zones = None

    def get_individuals(self):
        if self.individuals is not None:
            return self.individuals

        individuals = self.df[self.individual_key]
        individuals = individuals.unique()

        self.individuals = individuals
        return individuals

    def get_behaviours(self):
        if self.behaviours is not None:
            return self.behaviours

        behaviours = PandasFormatter.format_df(self.df[self.behaviour_key])
        behaviours = behaviours.unique()

        self.behaviours = behaviours
        return behaviours
    
    def get_zones(self):
        if self.zones is not None:
            return self.zones

        zones = PandasFormatter.format_df(self.df[self.zone_key])
        zones = zones.unique()

        self.zones = zones
        return zones
    
    def get_formatted_columns(self):
        behaviours = self.get_behaviours()
        zones = self.get_zones()

        prefixes = ['', 'close_neighbour_', 'distant_neighbour_']
        columns = [''.join([prefix, z, '_zone']) for prefix in prefixes for z in zones] + \
                [''.join([prefix,b]) for prefix in prefixes for b in behaviours]
        
        return columns


    def format(self, merge : bool = False, to_np : bool = True, event_driven : bool = False):
        individuals = self.get_individuals()
        sequences = {i : None for i in individuals}

        columns = self.get_formatted_columns()
        
        def get_values_from_timeid(time, id):
            df_i = self.df[(self.df[self.individual_key] == id) & (self.df[self.time_key] == time)]
            if df_i.shape[0] == 0:
                None, None
            else:
                row = df_i.iloc[0]
                return PandasFormatter.format_column(row[self.zone_key]) + '_zone', PandasFormatter.format_column(row[self.behaviour_key])

        for individual in individuals:
            ts = pd.DataFrame(columns=columns)

            df_i = self.df[self.df[self.individual_key] == individual]
            for _, row in df_i.iterrows():
                zone = PandasFormatter.format_column(row[self.zone_key]) + '_zone'
                behaviour = PandasFormatter.format_column(row[self.behaviour_key])
                time = row[self.time_key]

                vec = [0] * len(columns)
                vec[columns.index(zone)] = 1
                vec[columns.index(behaviour)] = 1

                for cn in re.findall(r'(\d+)[;\}]', row[self.close_neighbour_key]):
                    cn = int(cn)
                    zone_cn, behaviour_cn = get_values_from_timeid(time, cn)
                    vec[columns.index('close_neighbour_' + zone_cn)] += 1
                    vec[columns.index('close_neighbour_' + behaviour_cn)] += 1
                
                for dn in re.findall(r'(\d+)[;\}]', row[self.distant_neighbour_key]):
                    dn = int(dn)
                    zone_dn, behaviour_dn = get_values_from_timeid(time, dn)
                    vec[columns.index('distant_neighbour_' + zone_dn)] += 1
                    vec[columns.index('distant_neighbour_' + behaviour_dn)] += 1

                ts.loc[len(ts)] = vec
                sequences[individual] = ts

        if event_driven: # keep only rows where there is a change in the sequence, discard contiguous duplicates
            # constants = []
            # non_constant_min_row = 5
            for individual in individuals:
                ts = sequences[individual]
                duplicates = ts.duplicated(keep='first')
                
                last_non_duplicate = None
                idx = 0
                while idx < len(ts):
                    if duplicates.iloc[idx].all() and (ts.iloc[idx] == last_non_duplicate).all():
                        pass
                    elif duplicates.iloc[idx].all() and (ts.iloc[idx] != last_non_duplicate).all(): # keep only contiguous duplicates
                        print(f"Warning: non-contiguous duplicate found for individual {individual} at index {idx}. Discarding it from the duplicate sequence.")
                        duplicates.iloc[idx] = False
                    else:
                        last_non_duplicate = ts.iloc[idx]
                    idx += 1

                # old_ts_len = len(ts)
                ts = ts[-duplicates]
                # if len(ts) < non_constant_min_row:
                #     print(f"Warning: individual {individual} has less than {non_constant_min_row} rows after removing duplicates ({old_ts_len} --> {len(ts)}). Discarding it from the sequence.")
                #     constants.append(individual)
                sequences[individual] = ts
            # for individual in constants:
            #     del sequences[individual]

        if merge:
            res = pd.concat(sequences.values())
            if to_np:
                return res.to_numpy(dtype=np.float64)
            else:
                return res
        else:
            res = list(sequences.values())
            if to_np:
                return [ts.to_numpy(dtype=np.float64) for ts in res]
            else:
                return res



class ResultsFormatter():

    @staticmethod
    def from_pandas(results : pd.DataFrame, var_names : list, tau_max : int):
        graph = np.zeros((len(var_names), len(var_names), tau_max+1), dtype='<U3')
        graph[:] = ""

        val_matrix = np.zeros((len(var_names), len(var_names), tau_max + 1))

        for idx, i, j, tau, link_type, link_value in results.itertuples(): # TODO: fix, put links in BOTH directions
            i = var_names.index(i)
            j = var_names.index(j)
            link_value = float(link_value)

            graph[i,j,tau] = link_type
            val_matrix[i,j,tau] = link_value
            
        return ResultsFormatter(graph, val_matrix)


    def __init__(self, graph : np.array, val_matrix : np.array):
        self.graph = graph
        self.val_matrix = val_matrix

    
    def get_graph(self):
        return self.graph
    
    def get_val_matrix(self):
        return self.val_matrix

    def get_results(self):
        return {"graph": self.graph, "val_matrix": self.val_matrix}

    
    def low_filter(self, abs_min = 0.5):
        graph = self.graph.copy()
        val_matrix = self.val_matrix.copy()

        graph[np.abs(val_matrix) < abs_min] = ""
        val_matrix[np.abs(val_matrix) < abs_min] = 0

        return ResultsFormatter(graph, val_matrix)

    def var_filter(self, cause_vars_to_remove : list = [], effect_vars_to_remove : list = []):
        graph = self.graph.copy()
        val_matrix = self.val_matrix.copy()

        # If the link is bidirectional, do not remove
        for i in cause_vars_to_remove:
            graph[i,:,:][np.where(graph[i,:,:] != "o-o")] = ""
            val_matrix[i,:,:][np.where(graph[i,:,:] != "o-o")] = 0
        for j in effect_vars_to_remove:
            graph[:,j,:][np.where(graph[:,j,:] != "o-o")] = ""
            val_matrix[:,j,:][np.where(graph[:,j,:] != "o-o")] = 0

        return ResultsFormatter(graph, val_matrix)

    def corr_filter(self):
        graph = self.graph.copy()
        val_matrix = self.val_matrix.copy()
        
        graph[np.where(graph == "o-o")] = ""
        val_matrix[np.where(graph == "o-o")] = 0

        return ResultsFormatter(graph, val_matrix)




    


    
