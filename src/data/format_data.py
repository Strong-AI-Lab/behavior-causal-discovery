
from dataclasses import dataclass
from typing import List, Union, Optional, Tuple
import pandas as pd
import re
import numpy as np
import math


@dataclass
class FormatOutput():
    sequence_data : Union[np.ndarray, List[np.ndarray], pd.DataFrame, List[pd.DataFrame]]
    sequences : dict
    neighbor_graphs : dict
    variables : list
    num_individuals : int
    num_behaviours : int
    num_zones : int
    movements : dict


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
    
    @staticmethod
    def format_coord(column : str, normalisation: Optional[str] = None, normalisation_parameters: Optional[Tuple[float, float]] = None):
        res = re.match(r'\[(-?\d+(?:\.\d+));(-?\d+(?:\.\d+));(-?\d+(?:\.\d+))\]', column)
        if res is not None and len(res.groups()) == 3:
            x, y, z = float(res.group(1)), float(res.group(2)), float(res.group(3))
            if normalisation is not None:
                if normalisation == 'minmax':
                    assert normalisation_parameters is not None and len(normalisation_parameters) == 6, "Normalisation parameters must be a tuple of 6 elements (x_min, x_max, y_min, y_max, z_min, z_max)."

                    x = (x - normalisation_parameters[0]) / (normalisation_parameters[1] - normalisation_parameters[0])
                    y = (y - normalisation_parameters[2]) / (normalisation_parameters[3] - normalisation_parameters[2])
                    z = (z - normalisation_parameters[4]) / (normalisation_parameters[5] - normalisation_parameters[4])
                elif normalisation == 'mean':
                    assert normalisation_parameters is not None and len(normalisation_parameters) == 6, "Normalisation parameters must be a tuple of 6 elements (x_mean, x_std, y_mean, y_std, z_mean, z_std)."

                    x = (x - normalisation_parameters[0]) / math.sqrt(normalisation_parameters[1]**2 + 1e-6)
                    y = (y - normalisation_parameters[2]) / math.sqrt(normalisation_parameters[3]**2 + 1e-6)
                    z = (z - normalisation_parameters[4]) / math.sqrt(normalisation_parameters[5]**2 + 1e-6)
                else:
                    raise NotImplementedError(f"Normalisation method {normalisation} not implemented.")
            return x, y, z
        else:
            return None




    def __init__(self, df : pd.DataFrame, 
                        behaviour_key : str = "Behaviour", 
                        individual_key : str = "ID", 
                        zone_key : str = "Zone", 
                        time_key : str = "Time", 
                        close_neighbour_key : str = "Close_neighbours", 
                        distant_neighbour_key : str = "Distant_neighbours",
                        coordinates_key : str = "Coordinates",
                        skip_faults : bool = True):
        self.df = df
        self.behaviour_key = behaviour_key
        self.individual_key = individual_key
        self.zone_key = zone_key
        self.time_key = time_key
        self.close_neighbour_key = close_neighbour_key
        self.distant_neighbour_key = distant_neighbour_key
        self.coordinates_key = coordinates_key
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
        return self.behaviours
    
    def get_zones(self):
        if self.zones is not None:
            return self.zones

        zones = PandasFormatter.format_df(self.df[self.zone_key])
        zones = zones.unique()
        zones = [z for z in zones if isinstance(z,str)]

        self.zones = sorted(zones)
        return self.zones
    
    def get_formatted_columns(self):
        behaviours = self.get_behaviours()
        zones = self.get_zones()

        prefixes = ['', 'close_neighbour_', 'distant_neighbour_']
        columns = [''.join([prefix, z, '_zone']) for prefix in prefixes for z in zones] + \
                [''.join([prefix,b]) for prefix in prefixes for b in behaviours]
        
        return columns


    def format(self, 
               merge : bool = False, 
               to_np : bool = True, 
               event_driven : bool = False,
               output_format : str = 'tuple'
               ):
        individuals = self.get_individuals()
        sequences = {i : None for i in individuals} # sequences of behaviour and zone for individual and neighbour
        neighbor_graphs = {i : [] for i in individuals} # list of (time, close_neighbours, distant_neighbours) for individual
        movements = {i : None for i in individuals} # list of (time, coordinates) for individual

        columns = self.get_formatted_columns()
        
        def get_values_from_timeid(time, id):
            df_i = self.df[(self.df[self.individual_key] == id) & (self.df[self.time_key] == time)]
            if df_i.shape[0] == 0:
                return None, None
            else:
                row = df_i.iloc[0]
                return PandasFormatter.format_column(row[self.zone_key]) + '_zone', PandasFormatter.format_column(row[self.behaviour_key])

        for individual in individuals:
            ts = pd.DataFrame(columns=columns) # Behaviour and zone columns for individual and neighbour
            movements_i = pd.DataFrame(columns=['x', 'y', 'z']) # Movement columns for individual

            df_i = self.df[self.df[self.individual_key] == individual]
            error_count = 0
            for _, row in df_i.iterrows():
                try:
                    zone = PandasFormatter.format_column(row[self.zone_key]) + '_zone'
                    behaviour = PandasFormatter.format_column(row[self.behaviour_key])
                    time = row[self.time_key]
                    x, y, z = PandasFormatter.format_coord(row[self.coordinates_key])
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

                movements_i.loc[len(movements_i)] = [x, y, z]

                close_neighbors = []
                for cn in re.findall(r'(\d+)[;\}]', row[self.close_neighbour_key]):
                    try:
                        cn = int(cn)
                        zone_cn, behaviour_cn = get_values_from_timeid(time, cn)
                        vec[columns.index('close_neighbour_' + zone_cn)] += 1
                        vec[columns.index('close_neighbour_' + behaviour_cn)] += 1

                        close_neighbors.append(cn)
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
            movements[individual] = movements_i

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
        
        if output_format == 'tuple':
            return res, sequences, neighbor_graphs, columns, len(individuals), len(self.get_behaviours()), len(self.get_zones()), movements
        elif output_format == 'dataclass':
            return FormatOutput(
                res, 
                sequences, 
                neighbor_graphs, 
                columns, 
                len(individuals), 
                len(self.get_behaviours()), 
                len(self.get_zones()),
                movements
            )
        else:
            raise NotImplementedError(f"Output format {output_format} not implemented.")


            

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
                        coordinates_key : str = "Coordinates",
                        skip_faults : bool = True):
        if isinstance(dfs, pd.DataFrame):
            super().__init__(dfs, behaviour_key, individual_key, zone_key, time_key, close_neighbour_key, distant_neighbour_key, coordinates_key, skip_faults)
        elif isinstance(dfs, list):
            df = PandasFormatterEnsemble.merge_dataframes(dfs, individual_key, time_key, close_neighbour_key, distant_neighbour_key)
            super().__init__(df, behaviour_key, individual_key, zone_key, time_key, close_neighbour_key, distant_neighbour_key, coordinates_key, skip_faults)
        



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

        return ResultsFormatter(graph, val_matrix, var_names=var_names)




    


    
