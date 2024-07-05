
from typing import List, Dict, Tuple, Union, Optional, ClassVar, Set
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json

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
        snapshot: Optional['Chronology.Snapshot']
        past_state: Optional['Chronology.State']
        future_state: Optional['Chronology.State']
        coord_epsilon: ClassVar[float] = 100

        def __hash__(self) -> int:
            return hash(self.individual_id) + hash(self.zone) + hash(self.behaviour) + hash(self.coordinates) + hash(tuple(self.close_neighbours)) + hash(tuple(self.distant_neighbours)) + (hash(self.snapshot) if self.snapshot is not None else 0)

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}(individual_id={self.individual_id}, zone={self.zone}, behaviour={self.behaviour}, coordinates={self.coordinates}, close_neighbours={self.close_neighbours}, " + \
                    f"distant_neighbours={self.distant_neighbours}, snapshot={str(self.snapshot.__class__.__name__) + '(time=' + str(self.snapshot.time) + '...)' if self.snapshot is not None else None}, " + \
                    f"past_state={str(self.past_state.__class__.__name__) + '(individual_id=' + str(self.past_state.individual_id) + '...)' if self.past_state is not None else None}, " + \
                    f"future_state={str(self.future_state.__class__.__name__) + '(individual_id=' + str(self.future_state.individual_id) + '...)' if self.future_state is not None else None})"

        def time_eq(self, other: 'Chronology.State') -> bool: # /!\ snapshots, past and future states are not compared!
            return self.individual_id == other.individual_id and \
                   self.zone == other.zone and \
                   self.behaviour == other.behaviour and \
                   self.coordinates_eq(other) and \
                   self.close_neighbours == other.close_neighbours and \
                   self.distant_neighbours == other.distant_neighbours
        
        def coordinates_eq(self, other: 'Chronology.State') -> bool:
            return np.allclose(self.coordinates, other.coordinates, atol=self.coord_epsilon)
        
        def deep_copy(self) -> 'Chronology.State': # Deep copy /!\ past and future states are not copied
            return Chronology.State(self.individual_id, self.zone, self.behaviour, self.coordinates, self.close_neighbours.copy(), self.distant_neighbours.copy(), self.snapshot, self.past_state, self.future_state)
        
        def to_json(self) -> dict: # Serialisation does not save snapshot, past and future states. They must be managed at upper levels (Snapshot handles `snapshot` attribute, Chronology handles `past_state` and `future_state` attributes)
            return {
                "hash" : hash(self),
                "individual_id" : self.individual_id,
                "zone" : self.zone,
                "behaviour" : self.behaviour,
                "coordinates" : self.coordinates,
                "close_neighbours" : self.close_neighbours,
                "distant_neighbours" : self.distant_neighbours,
                "snapshot" : hash(self.snapshot) if self.snapshot is not None else None,
                "past_state" : hash(self.past_state) if self.past_state is not None else None,
                "future_state" : hash(self.future_state) if self.future_state is not None else None
            }
        
        @classmethod
        def from_json(cls, json_data : dict) -> 'Chronology.State': # Deserialisation does not load snapshot, past and future states. They must be managed at upper levels
            return cls(json_data["individual_id"], json_data["zone"], json_data["behaviour"], tuple(json_data["coordinates"]), json_data["close_neighbours"], json_data["distant_neighbours"], None, None, None)


    @dataclass
    class Snapshot:
        time: int
        close_adjacency_list: Dict[int, List[int]]
        distant_adjacency_list: Dict[int, List[int]]
        states: Dict[int, 'Chronology.State']

        def __post_init__(self):
            for state in self.states.values():
                state.snapshot = self

        def __hash__(self) -> int:
            return hash(self.time) + hash(tuple(self.close_adjacency_list.keys())) + hash(tuple(self.distant_adjacency_list.keys())) + hash(tuple(self.states.keys()))

        def time_eq(self, other: 'Chronology.Snapshot') -> bool:
            return self.close_adjacency_list == other.close_adjacency_list and \
                   self.distant_adjacency_list == other.distant_adjacency_list and \
                   self.states.keys() == other.states.keys() and \
                   all([self.states[ind].time_eq(other.states[ind]) for ind in self.states.keys()])
        
        def deep_copy(self) -> 'Chronology.Snapshot': # Deep copy /!\ past and future states are not copied
            states_copy = {ind : state.deep_copy() for ind, state in self.states.items()}
            close_adjacency_list_copy = {ind : state.close_neighbours for ind, state in states_copy.items()}
            distant_adjacency_list_copy = {ind : state.distant_neighbours for ind, state in states_copy.items()}

            copy = Chronology.Snapshot(self.time, close_adjacency_list_copy, distant_adjacency_list_copy, states_copy)
            for state in states_copy.values():
                state.snapshot = copy
            return copy
        
        def to_json(self) -> dict:
            return {
                "hash" : hash(self),
                "time" : self.time,
                "close_adjacency_list" : self.close_adjacency_list,
                "distant_adjacency_list" : self.distant_adjacency_list,
                "states" : {ind : state.to_json() for ind, state in self.states.items()}
            }
        
        @classmethod
        def from_json(cls, json_data : dict) -> 'Chronology.Snapshot':
            states = {int(ind) : Chronology.State.from_json(state_data) for ind, state_data in json_data["states"].items()}
            close_adjacency_list = {int(ind) : adj_list for ind, adj_list in json_data["close_adjacency_list"].items()}
            distant_adjacency_list = {int(ind) : adj_list for ind, adj_list in json_data["distant_adjacency_list"].items()}

            snapshot = cls(json_data["time"], close_adjacency_list, distant_adjacency_list, states)
            for state in states.values(): # Handling of state snapshot attributes
                state.snapshot = snapshot
            return snapshot


    @classmethod
    def create(cls, file_path : Union[str, List[str]], fix_errors : bool = False, filter_null_state_trajectories : bool = False) -> 'Chronology':
        from data.parsers.pandasparser import PandasParser
        parser = PandasParser(filter_null_state_trajectories=filter_null_state_trajectories)

        if isinstance(file_path, str):
            data = pd.read_csv(file_path)
            chronology = cls(data, parser, fix_errors=fix_errors)
        else:
            chronologies = []
            for path in file_path:
                data = pd.read_csv(path)
                chronologies.append(cls(data, parser, fix_errors=fix_errors))
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
                    end_time : int = 0, # /!\ end_time is inclusive
                    stationary_times : List[int] = None,
                    empty_times : List[int] = None,
                    snapshots : List[Optional['Chronology.Snapshot']] = None,
                    individuals_ids : List[int] = None,
                    first_occurence : Dict[int, int] = None,
                    all_occurences : Dict[int, List[int]] = None,
                    zone_labels : Set[str] = None,
                    behaviour_labels : Set[str] = None,
                    parse_data : bool = True,
                    fix_errors : bool = False, # Find missing data and update with last values. If set false, some functions (e.g. split, merge) may return an error
                    ):
        self.start_time = start_time
        self.end_time = end_time
        self.stationary_times = stationary_times if stationary_times is not None else []
        self.empty_times = empty_times if empty_times is not None else []
        self.snapshots = snapshots if snapshots is not None else []
        self.individuals_ids = sorted(individuals_ids) if individuals_ids is not None else []
        self.first_occurence = first_occurence if first_occurence is not None else {}
        self.all_occurences = all_occurences if all_occurences is not None else {}
        self.zone_labels = zone_labels if zone_labels is not None else set()
        self.behaviour_labels = behaviour_labels if behaviour_labels is not None else set()

        self.raw_data = data
        self.parser = parser

        if parser is not None and parse_data:
            self._parse_data(data, parser)

            if fix_errors:
                self._fix_errors(time_threshold=999)


    def _parse_data(self, data : pd.DataFrame, parser : Parser) -> None:
        parser.parse(data, self)

    def __eq__(self, other : 'Chronology') -> bool:
        return ((self.raw_data is None and other.raw_data is None) or (self.raw_data is not None and other.raw_data is not None and self.raw_data.equals(other.raw_data))) and \
               ((self.parser is None and other.parser is None) or (self.parser is not None and other.parser is not None and self.parser == other.parser)) and \
               self.start_time == other.start_time and \
               self.end_time == other.end_time and \
               self.stationary_times == other.stationary_times and \
               self.empty_times == other.empty_times and \
               self.individuals_ids == other.individuals_ids and \
               self.first_occurence == other.first_occurence and \
               self.all_occurences == other.all_occurences and \
               self.zone_labels == other.zone_labels and \
               self.behaviour_labels == other.behaviour_labels and \
               len(self.snapshots) == len(other.snapshots) and \
               all([(self.snapshots[i] is None and other.snapshots[i] is None) or (self.snapshots[i] is not None and other.snapshots[i] is not None and (self.snapshots[i].time_eq(other.snapshots[i]))) for i in range(len(self.snapshots))])


    def get_snapshot(self, time : int) -> 'Chronology.Snapshot':
        return self.snapshots[time - self.start_time]

    def deep_copy(self) -> 'Chronology':
        # Copy attributes
        raw_data_copy = self.raw_data.copy() if self.raw_data is not None else None
        parser_copy = self.parser.copy() if self.parser is not None else None
        stationary_times_copy = self.stationary_times.copy()
        empty_times_copy = self.empty_times.copy()
        individuals_ids_copy = self.individuals_ids.copy()
        first_occurence_copy = self.first_occurence.copy()
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
                            first_occurence=first_occurence_copy,
                            all_occurences={ind_id : self.all_occurences[ind_id].copy() for ind_id in self.all_occurences.keys()},
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
        chr0_first_occurence = {ind_id : chronology.first_occurence[ind_id] for ind_id in chr0_individuals_ids}
        chr0_all_occurences = {ind_id : [time for time in chronology.all_occurences[ind_id] if time < split_time] for ind_id in chr0_individuals_ids}

        chr1_individuals_ids = set()
        chr1_first_occurence = {}
        for snapshot in chr1_snapshots:
            if snapshot is not None:
                chr1_individuals_ids.update(snapshot.states.keys())
                for ind_id in snapshot.states.keys():
                    if ind_id not in chr1_first_occurence:
                        chr1_first_occurence[ind_id] = snapshot.time # if individual first appears in chr0, we need to find the first time it appears in chr1
        chr1_individuals_ids = sorted(list(chr1_individuals_ids))
        chr1_all_occurences = {ind_id : sorted(list({chr1_first_occurence[ind_id]} | {time for time in chronology.all_occurences[ind_id] if time >= split_time})) for ind_id in chr1_individuals_ids}

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
                                first_occurence=chr0_first_occurence,
                                all_occurences=chr0_all_occurences,
                                zone_labels=chronology.zone_labels, 
                                behaviour_labels=chronology.behaviour_labels)
        
        chronology1 = Chronology(data=None, parser=None,
                                start_time=split_time, 
                                end_time=chronology.end_time,
                                stationary_times=chr1_stationary_times, 
                                empty_times=chr1_empty_times,
                                snapshots=chr1_snapshots, 
                                individuals_ids=chr1_individuals_ids,
                                first_occurence=chr1_first_occurence,
                                all_occurences=chr1_all_occurences,
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

        # Merge chronologiy labels
        merged_zone_labels = chronology0.zone_labels | chronology1.zone_labels
        merged_behaviour_labels = chronology0.behaviour_labels | chronology1.behaviour_labels
        
        # Offset chronology1 times
        chr1_start_time_offset = chronology0.end_time + 1 - chronology1.start_time
        for snapshot in chronology1.snapshots:
            if snapshot is not None:
                snapshot.time += chr1_start_time_offset
        chronology1.stationary_times = [time + chr1_start_time_offset for time in chronology1.stationary_times]
        chronology1.empty_times = [time + chr1_start_time_offset for time in chronology1.empty_times]
        chronology1.first_occurence = {ind_id : time + chr1_start_time_offset for ind_id, time in chronology1.first_occurence.items()}
        chronology1.all_occurences = {ind_id : [time + chr1_start_time_offset for time in times] for ind_id, times in chronology1.all_occurences.items()}

        if not keep_individual_ids: # Offset individual ids
            chr1_individuals_offset = max(chronology0.individuals_ids) + 1 - min(chronology1.individuals_ids)
            chronology1.individuals_ids = [ind_id + chr1_individuals_offset for ind_id in chronology1.individuals_ids]
            chronology1.first_occurence = {ind_id + chr1_individuals_offset : time for ind_id, time in chronology1.first_occurence.items()}
            chronology1.all_occurences = {ind_id + chr1_individuals_offset : times for ind_id, times in chronology1.all_occurences.items()}
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
        
        # Merge first occurences of individuals
        merged_first_occurence = chronology1.first_occurence
        merged_first_occurence.update(chronology0.first_occurence)
        
        # Find and merge start of sequences
        merged_all_occurences = chronology0.all_occurences
        for ind_id, times in chronology1.all_occurences.items():
            if ind_id in merged_all_occurences:
                merged_all_occurences[ind_id] = sorted(merged_all_occurences[ind_id] + times)
            else:
                merged_all_occurences[ind_id] = times
            
            # Handle links between chronology 0 and 1
            if chronology1.start_time in times and chronology1.get_snapshot(chronology1.start_time).states[ind_id].past_state is not None:
                merged_all_occurences[ind_id].remove(chronology1.start_time)


        # Merge chronologies
        return Chronology(data=None, parser=None,
                            start_time=chronology0.start_time,
                            end_time=chronology1.end_time + chr1_start_time_offset,
                            stationary_times=chronology0.stationary_times + chronology1.stationary_times,
                            empty_times=chronology0.empty_times + chronology1.empty_times,
                            snapshots=chronology0.snapshots + chronology1.snapshots,
                            individuals_ids=merged_individuals_ids,
                            first_occurence=merged_first_occurence,
                            all_occurences=merged_all_occurences,
                            zone_labels=merged_zone_labels,
                            behaviour_labels=merged_behaviour_labels)



    @classmethod
    def merge_no_interlace_n(cls, chronologies : List['Chronology'], keep_individual_ids : bool = False) -> 'Chronology':
        chronology = cls.merge_not_interlace_2(chronologies[0], chronologies[1], keep_individual_ids)

        for i in range(2, len(chronologies)):
            chronology = cls.merge_not_interlace_2(chronology, chronologies[i], keep_individual_ids)

        return chronology
    

    def to_json(self) -> dict: # Serialisation does not save raw_data and solver
        return {
            "raw_data" : None,
            "parser" : None,
            "start_time" : self.start_time,
            "end_time" : self.end_time,
            "stationary_times" : self.stationary_times,
            "empty_times" : self.empty_times,
            "snapshots" : [snapshot.to_json() if snapshot is not None else None for snapshot in self.snapshots],
            "individuals_ids" : self.individuals_ids,
            "first_occurence" : self.first_occurence,
            "all_occurences" : self.all_occurences,
            "zone_labels" : list(self.zone_labels),
            "behaviour_labels" : list(self.behaviour_labels)
        }

    @classmethod
    def from_json(cls, json_data : dict) -> 'Chronology': # Deserialisation does not load raw_data and solver
        snapshots = [Chronology.Snapshot.from_json(snapshot_data) if snapshot_data is not None else None for snapshot_data in json_data["snapshots"]]
        first_occurence = {int(ind_id) : time for ind_id, time in json_data["first_occurence"].items()}
        all_occurences = {int(ind_id) : times for ind_id, times in json_data["all_occurences"].items()}
        zone_labels = set(json_data["zone_labels"])
        behaviour_labels = set(json_data["behaviour_labels"])

        chronology = cls(data=None, parser=None, 
                    start_time=json_data["start_time"], 
                    end_time=json_data["end_time"],
                    stationary_times=json_data["stationary_times"], 
                    empty_times=json_data["empty_times"],
                    snapshots=snapshots, 
                    individuals_ids=json_data["individuals_ids"],
                    first_occurence=first_occurence,
                    all_occurences=all_occurences,
                    zone_labels=zone_labels,
                    behaviour_labels=behaviour_labels,
                    parse_data=False)
        
        # If json data was dumped, dict keys are strings, we need to convert them back to integers
        to_format = lambda x : x
        for snapshot in json_data["snapshots"]:
            if snapshot is not None:
                if isinstance(list(snapshot["states"].keys())[0], str):
                    to_format = lambda x : str(x)
                break
        
        # Handling of past and future states
        state_dict = {}
        for i in range(len(snapshots)):
            if snapshots[i] is not None:
                for ind_id in snapshots[i].states.keys():
                    state_dict[json_data["snapshots"][i]["states"][to_format(ind_id)]["hash"]] = {
                        "state" : snapshots[i].states[ind_id],
                        "past_state" : json_data["snapshots"][i]["states"][to_format(ind_id)]["past_state"],
                        "future_state" : json_data["snapshots"][i]["states"][to_format(ind_id)]["future_state"]
                    }
        for state_settings in state_dict.values():
            state = state_settings["state"]
            if state_settings["past_state"] is not None:
                state.past_state = state_dict[state_settings["past_state"]]["state"]
            if state_settings["future_state"] is not None:
                state.future_state = state_dict[state_settings["future_state"]]["state"]

        return chronology



    def serialize(self, file_path : str) -> None:
        with open(file_path, "w") as f:
            json.dump(self.to_json(), f, indent=4)

    @classmethod
    def deserialize(cls, file_path : str) -> 'Chronology':
        with open(file_path, "r") as f:
            json_data = json.load(f)
        return cls.from_json(json_data)


    def _get_neighbour_state_distribution(self, ind_id : int, time : int, state : str, proximity : str) -> Dict[str, int]:
        snapshot = self.get_snapshot(time)

        if proximity == "close":
            adjacency_list = snapshot.close_adjacency_list[ind_id]
        elif proximity == "distant":
            adjacency_list = snapshot.distant_adjacency_list[ind_id]
        else:
            raise ValueError("Proximity must be either 'close' or 'distant'!")
        
        if state == "behaviour":
            state_distribution = {behaviour : 0 for behaviour in self.behaviour_labels}
            attribute = "behaviour"
        elif state == "zone":
            state_distribution = {zone : 0 for zone in self.zone_labels}
            attribute = "zone"
        
        nb_nonzero_neighbours = 0
        for neighbour_id in adjacency_list:
            state = snapshot.states[neighbour_id]
            if getattr(state, attribute) is not None:
                state_distribution[getattr(state, attribute)] += 1
                nb_nonzero_neighbours += 1

        if nb_nonzero_neighbours > 0:
            for key in state_distribution.keys():
                state_distribution[key] /= nb_nonzero_neighbours

        return state_distribution

    def _fix_state(self, state : 'Chronology.State', weight_past : float = 2.0, weight_future : float = 1.0, weight_close_neighbours : float = 1.0, weight_distant_neighbours : float = 0.5, fix_mode : str = "max") -> None:
        ind_id = state.individual_id
        time = state.snapshot.time

        for attribute, labels in [("behaviour", self.behaviour_labels), ("zone", self.zone_labels)]:
            if getattr(state, attribute) is None:
                distribution = {attr_val : 0 for attr_val in labels}

                if state.past_state is not None and getattr(state, attribute) is not None:
                    past_attribute = getattr(state.past_state, attribute)
                    distribution[past_attribute] += weight_past

                if state.future_state is not None and state.future_state.behaviour is not None:
                    future_attribute = getattr(state.future_state, attribute)
                    distribution[future_attribute] += weight_future

                close_neighbour_attributes = self._get_neighbour_state_distribution(ind_id, time, attribute, "close")
                for attr_val, weight in close_neighbour_attributes.items():
                    distribution[attr_val] += weight * weight_close_neighbours

                distant_neighbour_attributes = self._get_neighbour_state_distribution(ind_id, time, attribute, "distant")
                for attr_val, weight in distant_neighbour_attributes.items():
                    distribution[attr_val] += weight * weight_distant_neighbours
                
                if fix_mode == "max":
                    setattr(state, attribute, max(distribution, key=distribution.get))
                elif fix_mode == "random":
                    setattr(state, attribute, np.random.choice(list(distribution.keys()), p=list(distribution.values())))
                else:
                    raise ValueError("Fix mode must be either 'max' or 'random'!")

    def _fix_states(self, weight_past : float = 2.0, weight_future : float = 1.0, weight_close_neighbours : float = 1.0, weight_distant_neighbours : float = 0.5, fix_mode : str = "max") -> None: # Find missing behaviour and zone data and update with most likely value
        try:
            self.zone_labels.remove(None)
        except KeyError:
            pass
        try:
            self.behaviour_labels.remove(None)
        except KeyError:
            pass
        
        for ind_id, start_time in self.first_occurence.items():
            state = self.snapshots[start_time].states[ind_id]
            self._fix_state(state, weight_past, weight_future, weight_close_neighbours, weight_distant_neighbours, fix_mode)

            while state.future_state is not None:
                state = state.future_state
                self._fix_state(state, weight_past, weight_future, weight_close_neighbours, weight_distant_neighbours, fix_mode)


    def _interpolate_states(self, state0 : 'Chronology.State', state1 : 'Chronology.State', time_ratio : float) -> 'Chronology.State':
        # Interpolate zone and behaviour
        if time_ratio > 0.5:
            zone = state1.zone
            behaviour = state1.behaviour
        else:
            zone = state0.zone
            behaviour = state0.behaviour

        # Interpolate coordinates
        coord0 = state0.coordinates
        coord1 = state1.coordinates
        coordinates = tuple([coord0[i] + time_ratio * (coord1[i] - coord0[i]) for i in range(len(coord0))])

        return Chronology.State(state0.individual_id, zone, behaviour, coordinates, [], [], None, None, None)

    def _attach_states(self, state0 : 'Chronology.State', state1 : 'Chronology.State') -> None:
        start_time = state0.snapshot.time
        end_time = state1.snapshot.time

        prev_state = state0
        for time in range(start_time+1, end_time):
            # Create new state for the individual
            state = self._interpolate_states(state0, state1, (time - start_time) / (end_time - start_time))
            
            # Link state to snapshot and past state
            snapshot = self.get_snapshot(time)
            if snapshot is None:
                snapshot = Chronology.Snapshot(time, {}, {}, {})
                self.snapshots[time - self.start_time] = snapshot

            snapshot.states[state.individual_id] = state
            snapshot.close_adjacency_list[state.individual_id] = state0.close_neighbours
            snapshot.distant_adjacency_list[state.individual_id] = state0.distant_neighbours
            state.snapshot = snapshot
            prev_state.future_state = state
            state.past_state = prev_state

            # Update past state
            prev_state = state
        
        # Link last state
        prev_state.future_state = state1
        state1.past_state = prev_state

    def _fix_sequences(self, time_threshold : int) -> None: # Re-attach cut trajectories within time threshold
        for ind_id in self.all_occurences.keys():
            occurence_times = self.all_occurences[ind_id]
            updated_times = occurence_times.copy()
            
            for i in range(len(occurence_times)-1):
                state = self.snapshots[occurence_times[i]].states[ind_id]
                
                # Find end time of current sequence
                while state.future_state is not None:
                    state = state.future_state
                end_time = state.snapshot.time

                # Find if next sequence starts within threshold
                time = occurence_times[i + 1]
                if time <= (end_time + time_threshold):
                    snapshot = self.get_snapshot(time)

                    # Attach sequence
                    self._attach_states(state, snapshot.states[ind_id])
                    
                    # Update current state and list of occurences
                    updated_times.remove(time)
            
            self.all_occurences[ind_id] = updated_times


    def _fix_errors(self,
                    weight_past : float = 2.0,
                    weight_future : float = 1.0,
                    weight_close_neighbours : float = 1.0,
                    weight_distant_neighbours : float = 0.5,
                    fix_mode : str = "max",
                    time_threshold : int = 20) -> None:
        
        # Fix missing behaviour and zone data
        self._fix_states(weight_past, weight_future, weight_close_neighbours, weight_distant_neighbours, fix_mode)

        # Fix cut sequences
        self._fix_sequences(time_threshold)

