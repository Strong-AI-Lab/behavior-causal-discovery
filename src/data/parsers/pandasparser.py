
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple
import re

from data.parsers.baseparser import Parser
from data.structure.chronology import Chronology


@dataclass
class Namekeys:
    behaviour_key : str = "Behaviour"
    individual_key : str = "ID"
    zone_key : str = "Zone"
    time_key : str = "Time"
    close_neighbour_key : str = "Close_neighbours"
    distant_neighbour_key : str = "Distant_neighbours"
    coordinates_key : str = "Coordinates"


State = Chronology.State
Snapshot = Chronology.Snapshot


class PandasParser(Parser):

    @staticmethod
    def tolowercase(name : str) -> str:
        return name.lower().replace(' ','_')

    @staticmethod
    def listtolowercase(namelist : List[str]) -> List[str]:
        return [PandasParser.tolowercase(name) for name in namelist]
    
    @staticmethod
    def toneighbours(neighbours : str) -> List[int]:
        neighbours = re.findall(r'(\d+)[;\}]', neighbours)
        return [int(neighbour) for neighbour in neighbours]



    def __init__(self, namekeys : Optional[Namekeys] = None, coordinates_normalisation : Optional[str] = None):
        super().__init__()

        if namekeys is None:
            namekeys = Namekeys()
        self.namekeys = namekeys
        self.coordinates_normalisation = coordinates_normalisation


    def __eq__(self, other : 'PandasParser') -> bool:
        return self.namekeys == other.namekeys and self.coordinates_normalisation == other.coordinates_normalisation
    
    def copy(self) -> 'PandasParser':
        return PandasParser(self.namekeys, self.coordinates_normalisation)


    def parse(self, data: pd.DataFrame, structure : Chronology) -> None:
        time_ids = data[self.namekeys.time_key]

        # Set structure attributes
        structure.start_time = time_ids.min()
        structure.end_time = time_ids.max()
        structure.individuals_ids = sorted(data[self.namekeys.individual_key].unique().tolist())
        structure.zone_labels = PandasParser.listtolowercase(data[self.namekeys.zone_key].unique().tolist())
        structure.zone_labels = [zone + '_zone' for zone in structure.zone_labels]
        structure.behaviour_labels = PandasParser.listtolowercase(data[self.namekeys.behaviour_key].unique().tolist())

        # Set normalisation parameters
        if self.coordinates_normalisation is not None:
            self._get_normalisation_parameters(data)

        # Parse snapshots
        previous_snapshot = None
        for time in range(structure.start_time, structure.end_time + 1):
            snapshot_data = data[data[self.namekeys.time_key] == time]
            
            if snapshot_data.empty: # Detect empty times
                snapshot = None
                structure.snapshots.append(snapshot)
                structure.empty_times.append(time)
            else:
                snapshot = self._parse_snapshot(time, snapshot_data, previous_snapshot)
                structure.snapshots.append(snapshot)

                # Detect stationary times
                if previous_snapshot is not None and snapshot.time_eq(previous_snapshot):
                    structure.stationary_times.append(time)

            previous_snapshot = snapshot


    def _parse_snapshot(self, time : int, data: pd.DataFrame, previous_snapshot : Optional[Snapshot]) -> Snapshot:
        states = {}
        close_adjacency_list = {}
        distant_adjacency_list = {}
        individuals = data[self.namekeys.individual_key].unique().tolist()

        if len(individuals) != len(data):
            raise ValueError("Individuals must be unique for each snapshot: a same individual cannot have multiple values at the same time!")
        
        # Parse individual states
        for ind_id in individuals:
            individual_data = data[data[self.namekeys.individual_key] == ind_id]
            state = self._parse_state(ind_id, individual_data)
            close_adjacency_list[ind_id] = state.close_neighbours
            distant_adjacency_list[ind_id] = state.distant_neighbours
            states[ind_id] = state

            # Set past and future states
            if previous_snapshot is not None:
                if ind_id in previous_snapshot.states:
                    previous_state = previous_snapshot.states[ind_id]
                    previous_state.future_state = state
                    state.past_state = previous_state

        return Snapshot(time, close_adjacency_list, distant_adjacency_list, states)
    
    
    def _parse_state(self, individual_id : int, data: pd.DataFrame) -> State:
        behaviour = PandasParser.tolowercase(data[self.namekeys.behaviour_key].iloc[0])
        zone = PandasParser.tolowercase(data[self.namekeys.zone_key].iloc[0]) + '_zone'
        coordinates = self.tocoordinates(data[self.namekeys.coordinates_key].iloc[0])
        close_neighbours = PandasParser.toneighbours(data[self.namekeys.close_neighbour_key].iloc[0])
        distant_neighbours = PandasParser.toneighbours(data[self.namekeys.distant_neighbour_key].iloc[0])

        return State(individual_id, zone, behaviour, coordinates, close_neighbours, distant_neighbours, None, None)
    
    
    
    def tocoordinates(self, coordinates : str) -> Tuple[float, float, float]:
        res = re.match(r'\[(-?\d+(?:\.\d+));(-?\d+(?:\.\d+));(-?\d+(?:\.\d+))\]', coordinates)

        if res is not None and len(res.groups()) == 3:
            x, y, z = res.groups()
            x, y, z = float(x), float(y), float(z)

            if self.coordinates_normalisation is not None:
                x, y, z = self._normalise_coordinates((float(x), float(y), float(z)))

            return (float(x), float(y), float(z))
        else:
            return None

    def _get_normalisation_parameters(self, data: pd.DataFrame) -> None:
        raise NotImplementedError("Normalisation parameters computation not implemented yet.")
    
    def _normalise_coordinates(self, coordinates : Tuple[float, float, float]) -> Tuple[float, float, float]:
        raise NotImplementedError("Coordinates normalisation not implemented yet.")
