
from typing import List, Union
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
                        distant_neighbour_key : str = "Distant_neighbours",
                        skip_faults : bool = True):
        self.df = df
        self.behaviour_key = behaviour_key
        self.individual_key = individual_key
        self.zone_key = zone_key
        self.time_key = time_key
        self.close_neighbour_key = close_neighbour_key
        self.distant_neighbour_key = distant_neighbour_key
        self.skip_faults = skip_faults

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
        behaviours = [b for b in behaviours if isinstance(b,str)]

        self.behaviours = sorted(behaviours)
        return behaviours
    
    def get_zones(self):
        if self.zones is not None:
            return self.zones

        zones = PandasFormatter.format_df(self.df[self.zone_key])
        zones = zones.unique()
        zones = [z for z in zones if isinstance(z,str)]

        self.zones = sorted(zones)
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
        neighbor_graphs = {i : [] for i in individuals}

        columns = self.get_formatted_columns()
        
        def get_values_from_timeid(time, id):
            df_i = self.df[(self.df[self.individual_key] == id) & (self.df[self.time_key] == time)]
            if df_i.shape[0] == 0:
                return None, None
            else:
                row = df_i.iloc[0]
                return PandasFormatter.format_column(row[self.zone_key]) + '_zone', PandasFormatter.format_column(row[self.behaviour_key])

        for individual in individuals:
            ts = pd.DataFrame(columns=columns)

            df_i = self.df[self.df[self.individual_key] == individual]
            error_count = 0
            for _, row in df_i.iterrows():
                try:
                    zone = PandasFormatter.format_column(row[self.zone_key]) + '_zone'
                    behaviour = PandasFormatter.format_column(row[self.behaviour_key])
                    time = row[self.time_key]
                except AttributeError as e:
                    if self.skip_faults:
                        error_count += 1
                        print(f"\rWarning: faulty row. Discarding it from the sequence. ({error_count} errors)",end='')
                        continue
                    else:
                        raise e

                vec = [0] * len(columns)
                vec[columns.index(zone)] = 1
                vec[columns.index(behaviour)] = 1

                close_neighbors = []
                for cn in re.findall(r'(\d+)[;\}]', row[self.close_neighbour_key]):
                    try:
                        cn = int(cn)
                        zone_cn, behaviour_cn = get_values_from_timeid(time, cn)
                        vec[columns.index('close_neighbour_' + zone_cn)] += 1
                        vec[columns.index('close_neighbour_' + behaviour_cn)] += 1

                        close_neighbors.append(dn)
                    except AttributeError as e:
                        if self.skip_faults:
                            error_count += 1
                            print(f"\rWarning: faulty close neighbour. Discarding it from the sequence (keeping row and other close neigbours). ({error_count} errors)",end='')
                            continue
                        else:
                            raise e
                
                distant_neighbors = []
                for dn in re.findall(r'(\d+)[;\}]', row[self.distant_neighbour_key]):
                    try:
                        dn = int(dn)
                        zone_dn, behaviour_dn = get_values_from_timeid(time, dn)
                        vec[columns.index('distant_neighbour_' + zone_dn)] += 1
                        vec[columns.index('distant_neighbour_' + behaviour_dn)] += 1

                        distant_neighbors.append(dn)
                    except AttributeError as e:
                        if self.skip_faults:
                            error_count += 1
                            print(f"\rWarning: faulty distant neighbour. Discarding it from the sequence (keeping row and other distant neigbours). ({error_count} errors)",end='')
                            continue
                        else:
                            raise e

                ts.loc[len(ts)] = vec
                neighbor_graphs[individual].append((time, close_neighbors, distant_neighbors))

            sequences[individual] = ts

            if error_count > 0:
                print(f"\x1b[1K\rIndividual {individual} done with {error_count} errors.")

        if event_driven: # keep only rows where there is a change in the sequence, discard contiguous duplicates
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

                ts = ts[-duplicates]
                sequences[individual] = ts

                # Update neighbor graphs
                neighbor_graphs[individual] = [neighbor_graphs[individual][i] for i in range(len(neighbor_graphs[individual])) if not duplicates.iloc[i].all()]

        if merge:
            res = pd.concat(sequences.values())
            if to_np:
                res = res.to_numpy(dtype=np.float64)
        else:
            res = list(sequences.values())
            if to_np:
                res = [ts.to_numpy(dtype=np.float64) for ts in res]
        
        return res, sequences, neighbor_graphs
            

class PandasFormatterEnsemble(PandasFormatter):

    @staticmethod
    def merge_dataframes(dfs : List[pd.DataFrame], individual_key : str, time_key : str, close_neighbour_key : str, distant_neighbour_key : str):
        time_offset = 0
        individual_offset = 0
        for i in range(len(dfs)):
            max_time = dfs[i][time_key].max()
            max_id = dfs[i][individual_key].max()
            dfs[i][time_key] += time_offset
            dfs[i][individual_key] += individual_offset

            dfs[i][close_neighbour_key] = dfs[i][close_neighbour_key].str.replace(r'(\d+)', lambda x: str(int(x.group(1)) + individual_offset), regex=True)
            dfs[i][distant_neighbour_key] = dfs[i][distant_neighbour_key].str.replace(r'(\d+)', lambda x: str(int(x.group(1)) + individual_offset), regex=True)

            time_offset += max_time + 1
            individual_offset += max_id + 1
        
        return pd.concat(dfs)
            

    def __init__(self, dfs : Union[pd.DataFrame,List[pd.DataFrame]], 
                        behaviour_key : str = "Behaviour", 
                        individual_key : str = "ID", 
                        zone_key : str = "Zone", 
                        time_key : str = "Time", 
                        close_neighbour_key : str = "Close_neighbours", 
                        distant_neighbour_key : str = "Distant_neighbours",
                        skip_faults : bool = True):
        if isinstance(dfs, pd.DataFrame):
            super().__init__(dfs, behaviour_key, individual_key, zone_key, time_key, close_neighbour_key, distant_neighbour_key, skip_faults)
        elif isinstance(dfs, list):
            df = PandasFormatterEnsemble.merge_dataframes(dfs, individual_key, time_key, close_neighbour_key, distant_neighbour_key)
            super().__init__(df, behaviour_key, individual_key, zone_key, time_key, close_neighbour_key, distant_neighbour_key, skip_faults)
        



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

    def var_filter(self, cause_vars_to_remove : list = None, effect_vars_to_remove : list = None):
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

        return ResultsFormatter(graph, val_matrix)

    def corr_filter(self):
        graph = self.graph.copy()
        val_matrix = self.val_matrix.copy()
        
        graph[np.where(graph == "o-o")] = ""
        val_matrix[np.where(graph == "o-o")] = 0

        return ResultsFormatter(graph, val_matrix)




    


    
