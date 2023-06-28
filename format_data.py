
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
        columns = [''.join([prefix, z]) for prefix in prefixes for z in zones] + \
                [''.join([prefix,b]) for prefix in prefixes for b in behaviours]
        
        return columns


    def format(self, merge : bool = False, to_np : bool = True):
        individuals = self.get_individuals()
        sequences = {i : None for i in individuals}

        columns = self.get_formatted_columns()
        
        def get_values_from_timeid(time, id):
            df_i = self.df[(self.df[self.individual_key] == id) & (self.df[self.time_key] == time)]
            if df_i.shape[0] == 0:
                None, None
            else:
                row = df_i.iloc[0]
                return PandasFormatter.format_column(row[self.zone_key]), PandasFormatter.format_column(row[self.behaviour_key])

        for individual in individuals:
            ts = pd.DataFrame(columns=columns)

            df_i = self.df[self.df[self.individual_key] == individual]
            for _, row in df_i.iterrows():
                zone = PandasFormatter.format_column(row[self.zone_key])
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