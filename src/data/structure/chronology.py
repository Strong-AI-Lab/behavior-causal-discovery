
from typing import List, Dict, Tuple, Union, Optional, ClassVar
import pandas as pd
import numpy as np
from dataclasses import dataclass

from data.parsers.baseparser import Parser


class Chronology:

    @dataclass
    class State:
        individual_id: int
        zone: str
        behaviour: str
        coordinates: Tuple[float, float, float]
        close_neighbours: List[int]
        distant_neighbours: List[int]
        past_state: Optional['Chronology.State']
        future_state: Optional['Chronology.State']
        coord_epsilon: ClassVar[float] = 100

        def time_eq(self, other: 'Chronology.State') -> bool: # /!\ past and future states are not compared!
            return self.individual_id == other.individual_id and \
                   self.zone == other.zone and \
                   self.behaviour == other.behaviour and \
                   self.coordinates_eq(other) and \
                   self.close_neighbours == other.close_neighbours and \
                   self.distant_neighbours == other.distant_neighbours
        
        def coordinates_eq(self, other: 'Chronology.State') -> bool:
            return np.allclose(self.coordinates, other.coordinates, atol=self.coord_epsilon)
        
        def deep_copy(self) -> 'Chronology.State': # Deep copy /!\ past and future states are not copied
            return Chronology.State(self.individual_id, self.zone, self.behaviour, self.coordinates, self.close_neighbours.copy(), self.distant_neighbours.copy(), self.past_state, self.future_state)


    @dataclass
    class Snapshot:
        time: int
        close_adjacency_list: Dict[int, List[int]]
        distant_adjacency_list: Dict[int, List[int]]
        states: Dict[int, 'Chronology.State']

        def time_eq(self, other: 'Chronology.Snapshot') -> bool:
            return self.close_adjacency_list == other.close_adjacency_list and \
                   self.distant_adjacency_list == other.distant_adjacency_list and \
                   self.states.keys() == other.states.keys() and \
                   all([self.states[ind].time_eq(other.states[ind]) for ind in self.states.keys()])
        
        def deep_copy(self) -> 'Chronology.Snapshot': # Deep copy /!\ past and future states are not copied
            states_copy = {ind : state.deep_copy() for ind, state in self.states.items()}
            close_adjacency_list_copy = {ind : state.close_neighbours for ind, state in states_copy.items()}
            distant_adjacency_list_copy = {ind : state.distant_neighbours for ind, state in states_copy.items()}
            return Chronology.Snapshot(self.time, close_adjacency_list_copy, distant_adjacency_list_copy, states_copy)


    @classmethod
    def create(cls, file_path : Union[str, List[str]]) -> 'Chronology':
        from data.parsers.pandasparser import PandasParser
        parser = PandasParser()

        if isinstance(file_path, str):
            data = pd.read_csv(file_path)
            chronology = cls(data, parser)
        else:
            chronologies = []
            for path in file_path:
                data = pd.read_csv(path)
                chronologies.append(cls(data, parser))
            chronology = cls.merge_no_interlace_n(chronologies)

        return chronology
    
    @staticmethod
    def _closest_index(time : int, times : List[int]) -> int:
        if time < times[0]:
            return 0

        return times.index(max(filter(lambda x : x <= time, times)))
    

    def __init__(self, 
                    data : pd.DataFrame, 
                    parser : Optional[Parser] = None,
                    start_time : int = 0,
                    end_time : int = 0,
                    stationary_times : List[int] = None,
                    empty_times : List[int] = None,
                    snapshots : List[Optional['Chronology.Snapshot']] = None,
                    individuals_ids : List[int] = None,
                    zone_labels : List[str] = None,
                    behaviour_labels : List[str] = None,
                    parse_data : bool = True
                    ):
        self.start_time = start_time
        self.end_time = end_time
        self.stationary_times = stationary_times if stationary_times is not None else []
        self.empty_times = empty_times if empty_times is not None else []
        self.snapshots = snapshots if snapshots is not None else []
        self.individuals_ids = sorted(individuals_ids) if individuals_ids is not None else []
        self.zone_labels = zone_labels if zone_labels is not None else []
        self.behaviour_labels = behaviour_labels if behaviour_labels is not None else []

        self.raw_data = data
        self.parser = parser

        if parser is not None and parse_data:
            self._parse_data(data, parser)


    def _parse_data(self, data : pd.DataFrame, parser : Parser) -> None:
        parser.parse(data, self)

    def __eq__(self, other : 'Chronology') -> bool:
        return ((self.raw_data is None and other.raw_data is None) or self.raw_data.equals(other.raw_data)) and \
               ((self.parser is None and other.parser is None) or self.parser == other.parser) and \
               self.start_time == other.start_time and \
               self.end_time == other.end_time and \
               self.stationary_times == other.stationary_times and \
               self.empty_times == other.empty_times and \
               self.individuals_ids == other.individuals_ids and \
               self.zone_labels == other.zone_labels and \
               self.behaviour_labels == other.behaviour_labels and \
               len(self.snapshots) == len(other.snapshots) and \
               all([(self.snapshots[i] is None and other.snapshots[i] is None) or (self.snapshots[i].time_eq(other.snapshots[i])) for i in range(len(self.snapshots))])


    def get_snapshot(self, time : int) -> 'Chronology.Snapshot':
        return self.snapshots[time - self.start_time]

    def deep_copy(self) -> 'Chronology':
        # Copy attributes
        raw_data_copy = self.raw_data.copy() if self.raw_data is not None else None
        parser_copy = self.parser.copy() if self.parser is not None else None
        stationary_times_copy = self.stationary_times.copy()
        empty_times_copy = self.empty_times.copy()
        individuals_ids_copy = self.individuals_ids.copy()
        zone_labels_copy = self.zone_labels.copy()
        behaviour_labels_copy = self.behaviour_labels.copy()
        snapshots_copy = [None if snapshot is None else snapshot.deep_copy() for snapshot in self.snapshots]

        # Update past and future states
        for i in range(1, len(snapshots_copy)):
            snapshot = snapshots_copy[i]
            if snapshot is not None:
                for state in snapshot.states.values():
                    if state.past_state is not None:
                        state.past_state = snapshots_copy[i - 1].states[state.individual_id]
                        snapshots_copy[i - 1].states[state.individual_id].future_state = state

        return Chronology(data=raw_data_copy, parser=parser_copy, parse_data=False,
                            start_time=self.start_time, end_time=self.end_time,
                            stationary_times=stationary_times_copy, empty_times=empty_times_copy,
                            snapshots=snapshots_copy, individuals_ids=individuals_ids_copy,
                            zone_labels=zone_labels_copy, behaviour_labels=behaviour_labels_copy)



    @classmethod
    def _split_inplace(cls, chronology : 'Chronology', split_time : int) -> Tuple[Optional['Chronology'], Optional['Chronology']]:
        if split_time > chronology.end_time:
            return chronology, None
        elif split_time < chronology.start_time:
            return None, chronology

        # Split snapshots
        snapshot_split_idx = split_time - chronology.start_time
        chr0_snapshots = chronology.snapshots[:snapshot_split_idx]
        chr1_snapshots = chronology.snapshots[snapshot_split_idx:]

        # Split stationary and empty times
        stationary_split_idx = Chronology._closest_index(split_time, chronology.stationary_times)
        chr0_stationary_times = chronology.stationary_times[:stationary_split_idx]
        chr1_stationary_times = chronology.stationary_times[stationary_split_idx:]

        empty_split_idx = Chronology._closest_index(split_time, chronology.empty_times)
        chr0_empty_times = chronology.empty_times[:empty_split_idx]
        chr1_empty_times = chronology.empty_times[empty_split_idx:]

        # Split individuals ids
        chr0_individuals_ids = set()
        for snapshot in chr0_snapshots:
            if snapshot is not None:
                chr0_individuals_ids.update(snapshot.states.keys())
        chr0_individuals_ids = sorted(list(chr0_individuals_ids))

        chr1_individuals_ids = set()
        for snapshot in chr1_snapshots:
            if snapshot is not None:
                chr1_individuals_ids.update(snapshot.states.keys())
        chr1_individuals_ids = sorted(list(chr1_individuals_ids))

        # Cut past and future states
        if chr0_snapshots[-1] is not None and chr1_snapshots[0] is not None:
            for state in chr0_snapshots[-1].states.values():
                state.future_state = None
            for state in chr1_snapshots[0].states.values():
                state.past_state = None

        # Create new chronologies (/!\ they will share the same object references as the original chronology, prefer using the self.split method (that calls deep_copy internally) or use deep_copy before calling this method)
        chronology0 = Chronology(data=None, parser=None, 
                                start_time=chronology.start_time, 
                                end_time=split_time - 1,
                                stationary_times=chr0_stationary_times, 
                                empty_times=chr0_empty_times,
                                snapshots=chr0_snapshots, 
                                individuals_ids=chr0_individuals_ids,
                                zone_labels=chronology.zone_labels, 
                                behaviour_labels=chronology.behaviour_labels)
        
        chronology1 = Chronology(data=None, parser=None,
                                start_time=split_time, 
                                end_time=chronology.end_time,
                                stationary_times=chr1_stationary_times, 
                                empty_times=chr1_empty_times,
                                snapshots=chr1_snapshots, 
                                individuals_ids=chr1_individuals_ids,
                                zone_labels=chronology.zone_labels, 
                                behaviour_labels=chronology.behaviour_labels)
        
        return chronology0, chronology1
    
    def split(self, split_time : int) -> Tuple[Optional['Chronology'], Optional['Chronology']]:
        chronology = self.deep_copy()
        return Chronology._split_inplace(chronology, split_time)
                                 


    @classmethod
    def merge_not_interlace_2(cls, chronology0 : 'Chronology', chronology1 : 'Chronology', keep_individual_ids : bool = False) -> 'Chronology':
        # Copy chronologies to avoid modifying them
        chronology0 = chronology0.deep_copy()
        chronology1 = chronology1.deep_copy()

        # Check if chronologies are compatible
        if chronology0.zone_labels != chronology1.zone_labels:
            raise ValueError("Chronologies have different zone labels!")
        if chronology0.behaviour_labels != chronology1.behaviour_labels:
            raise ValueError("Chronologies have different behaviour labels!")
        
        # Offset chronology1 times
        chr1_start_time_offset = chronology0.end_time + 1 - chronology1.start_time
        for snapshot in chronology1.snapshots:
            if snapshot is not None:
                snapshot.time += chr1_start_time_offset
        chronology1.stationary_times = [time + chr1_start_time_offset for time in chronology1.stationary_times]
        chronology1.empty_times = [time + chr1_start_time_offset for time in chronology1.empty_times]

        if not keep_individual_ids: # Offset individual ids
            chr1_individuals_offset = max(chronology0.individuals_ids) + 1 - min(chronology1.individuals_ids)
            chronology1.individuals_ids = [ind_id + chr1_individuals_offset for ind_id in chronology1.individuals_ids]
            for snapshot in chronology1.snapshots:
                if snapshot is not None:
                    snapshot.states = {ind_id + chr1_individuals_offset : state for ind_id, state in snapshot.states.items()}
                    snapshot.close_adjacency_list = {ind_id + chr1_individuals_offset : [adj + chr1_individuals_offset for adj in adj_list] for ind_id, adj_list in snapshot.close_adjacency_list.items()}
                    snapshot.distant_adjacency_list = {ind_id + chr1_individuals_offset : [adj + chr1_individuals_offset for adj in adj_list] for ind_id, adj_list in snapshot.distant_adjacency_list.items()}
                    for state in snapshot.states.values():
                        state.individual_id += chr1_individuals_offset
                        state.close_neighbours = [adj + chr1_individuals_offset for adj in state.close_neighbours]
                        state.distant_neighbours = [adj + chr1_individuals_offset for adj in state.distant_neighbours]
            
            merged_individuals_ids = chronology0.individuals_ids + chronology1.individuals_ids
        else: # Keep individual ids and link snapshots
            if chronology0.snapshots[-1] is not None and chronology1.snapshots[0] is not None:
                for ind_id in chronology0.snapshots[-1].states.keys():
                    if ind_id in chronology1.snapshots[0].states:
                        chronology0.snapshots[-1].states[ind_id].future_state = chronology1.snapshots[0].states[ind_id]
                        chronology1.snapshots[0].states[ind_id].past_state = chronology0.snapshots[-1].states[ind_id]
            
            merged_individuals_ids = sorted(list(set(chronology0.individuals_ids + chronology1.individuals_ids)))


        # Merge chronologies
        return Chronology(data=None, parser=None,
                            start_time=chronology0.start_time,
                            end_time=chronology1.end_time + chr1_start_time_offset,
                            stationary_times=chronology0.stationary_times + chronology1.stationary_times,
                            empty_times=chronology0.empty_times + chronology1.empty_times,
                            snapshots=chronology0.snapshots + chronology1.snapshots,
                            individuals_ids=merged_individuals_ids,
                            zone_labels=chronology0.zone_labels,
                            behaviour_labels=chronology0.behaviour_labels)



    @classmethod
    def merge_no_interlace_n(cls, chronologies : List['Chronology'], keep_individual_ids : bool = False) -> 'Chronology':
        chronology = cls.merge_not_interlace_2(chronologies[0], chronologies[1], keep_individual_ids)

        for i in range(2, len(chronologies)):
            chronology = cls.merge_not_interlace_2(chronology, chronologies[i], keep_individual_ids)

        return chronology

