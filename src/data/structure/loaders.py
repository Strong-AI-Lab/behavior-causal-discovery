
import abc
from typing import Tuple, List, Optional, Dict

from data.structure.chronology import Chronology
from data.constants import VECTOR_COLUMNS
from dynamics.solver import DynamicsSolver

import torch


class Loader(metaclass=abc.ABCMeta):
    def __init__(self, lookback : int, target_offset_start : int = 1, target_offset_end : int = 1):
        self.lookback = lookback
        self.target_offset_start = target_offset_start
        self.target_offset_end = target_offset_end

    @abc.abstractmethod
    def load(self, structure : Chronology) -> Tuple:
        return
    


class SeriesLoader(Loader):

    def __init__(self, lookback : int, target_offset_start : int = 1, target_offset_end : int = 1, skip_stationary : bool = False):
        super().__init__(lookback, target_offset_start, target_offset_end)
        self.skip_stationary = skip_stationary

    def _struct_to_sequences(self, structure : Chronology) -> Dict[int, torch.Tensor]:
        sequences = {}
        for ind_id in structure.individuals_ids:
            seq_ind = []
            snapshot = structure.get_snapshot(structure.first_occurence[ind_id])
            state = snapshot.states[ind_id]
            seq_ind.append(self._state_to_vector(state, structure))
            skip_states = []

            if self.skip_stationary: # Skip stationary states
                for time in structure.stationary_times:
                    sationary_snapshot = structure.get_snapshot(time)
                    if ind_id in sationary_snapshot.states:
                        skip_states.append(sationary_snapshot.states[ind_id])
                
            while state.future_state: # Build sequence by looking at future states
                state = state.future_state
                if state not in skip_states:
                    seq_ind.append(self._state_to_vector(state, structure))
            
            sequences[ind_id] = torch.stack(seq_ind) # [sequence_length, dimensions]
        
        return sequences

    @abc.abstractmethod
    def _state_to_vector(self, state : Chronology.State, structure : Chronology) -> torch.Tensor:
        return

    @abc.abstractmethod
    def load(self, structure : Chronology) -> Tuple:
        return
    



class BehaviourSeriesLoader(SeriesLoader):

    def __init__(self, lookback : int, target_offset_start : int = 1, target_offset_end : int = 1, skip_stationary : bool = False, vector_columns : List[str] = None):
        super().__init__(lookback, target_offset_start, target_offset_end, skip_stationary)
        if vector_columns is None:
            vector_columns = VECTOR_COLUMNS
        self.vector_columns = vector_columns

    def _state_to_vector(self, state : Chronology.State, structure : Chronology) -> torch.Tensor:
        nb_zones = len(structure.zone_labels)
        nb_behaviours = len(structure.behaviour_labels)

        vector = torch.zeros((nb_zones + nb_behaviours) * 3) # zone and behaviour information for current individual, close neighbours and distant neighbours
        vector[self.vector_columns.index(state.zone)] = 1.0
        vector[self.vector_columns.index(state.behaviour)] = 1.0

        for cn in state.close_neighbours:
            cn_state = state.snapshot.states[cn]
            vector[self.vector_columns.index('close_neighbour_' + cn_state.zone)] += 1.0
            vector[self.vector_columns.index('close_neighbour_' + cn_state.behaviour)] += 1.0

        for dn in state.distant_neighbours:
            dn_state = state.snapshot.states[dn]
            vector[self.vector_columns.index('distant_neighbour_' + dn_state.zone)] += 1.0
            vector[self.vector_columns.index('distant_neighbour_' + dn_state.behaviour)] += 1.0

        return vector

    def load(self, structure : Chronology) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        # Build individual sequences
        sequences = self._struct_to_sequences(structure)

        # Create lookback blocks
        x, y, individual = [], [], []

        for key, sequence in sequences.items():
            for i in range(len(sequence)-self.lookback-self.target_offset_end+1):
                feature = sequence[i:i+self.lookback]
                target = sequence[i+self.target_offset_start:i+self.lookback+self.target_offset_end]
                x.append(feature)
                y.append(target)
                individual.append(key)

        return torch.stack(x), torch.stack(y), individual
    



class DynamicSeriesLoader(SeriesLoader):

    def __init__(self, lookback : int, target_offset_start : int = 1, target_offset_end : int = 1, skip_stationary : bool = False, solver : Optional[DynamicsSolver] = None):
        super().__init__(lookback, target_offset_start, target_offset_end, skip_stationary)
        if solver is None:
            solver = DynamicsSolver(mass=1, dimensions=3)
        self.solver = solver

    def _struct_to_sequences(self, structure : Chronology) -> Dict[int, torch.Tensor]:
        self.v0 = torch.zeros(1, 1, 3) # [batch_size, 1, dimensions]
        sequences = super()._struct_to_sequences(structure)
        sequences = {ind_id : seq[:-1] for ind_id, seq in sequences.items()} # remove last state as its applied force cannot be computed
        return sequences

    def _state_to_vector(self, state : Chronology.State, structure : Chronology) -> torch.Tensor:
        x = torch.tensor(state.coordinates).float() # [3] = [dimensions]

        if state.future_state:
            x_next = torch.tensor(state.future_state.coordinates).float()
        
            x_full = torch.stack([x, x_next]).unsqueeze(0) # [1, 2, 3] = [batch_size, lookback, dimensions]
            force, a, v = self.solver.compute_force(x_full, self.v0, return_velocity=True) # 3 * [1, 2, 3] = 3 * [batch_size, lookback, dimensions]
            y = torch.stack([x, v[0, 0, :], force[0, 0, :]]) # coordinates, velocity, force applied at current state, [3, dimensions]
            self.v0 = v[:, -1:, :] # velocity at future state [batch_size, 1, dimensions]
        
        else:
            y = torch.stack([x, self.v0[0, 0, :], torch.zeros(3)])
            self.v0 = torch.zeros(1, 1, 3) # reset v0 for next individual, [batch_size, 1, dimensions]
        
        return y


    def load(self, structure : Chronology) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        # Build individual sequences
        sequences = self._struct_to_sequences(structure)

        # Create lookback blocks
        x, v, a, individual = [], [], [], []

        for key, sequence in sequences.items():
            for i in range(len(sequence)-self.lookback-self.target_offset_end+1):
                feature = sequence[i:i+self.lookback]
                target = sequence[i+self.target_offset_start:i+self.lookback+self.target_offset_end]
                x.append(feature[:,0])
                v.append(feature[:,1])
                a.append(target[:,2])
                individual.append(key)

        return torch.stack(x), torch.stack(v), torch.stack(a), individual
    


class DynamicGraphSeriesLoader(DynamicSeriesLoader):

    def __init__(self, lookback : int, target_offset_start : int = 1, target_offset_end : int = 1, skip_stationary : bool = False, solver : Optional[DynamicsSolver] = None):
        super().__init__(lookback, target_offset_start, target_offset_end, skip_stationary, solver)

    def _snapshot_to_graph(self, snapshot : Chronology.Snapshot, structure : Chronology) -> torch.Tensor:
        adjacency = torch.zeros(len(structure.individuals_ids), len(structure.individuals_ids))

        for ind_id, close_adj in snapshot.close_adjacency_list.items():
            for ind_id2 in close_adj:
                adjacency[structure.individuals_ids.index(ind_id), structure.individuals_ids.index(ind_id2)] = 1.0
        
        for ind_id, distant_adj in snapshot.distant_adjacency_list.items():
            for ind_id2 in distant_adj:
                adjacency[structure.individuals_ids.index(ind_id), structure.individuals_ids.index(ind_id2)] = 0.5

        return adjacency

    def _snapshot_to_ids(self, snapshot : Chronology.Snapshot, structure : Chronology) -> torch.Tensor:
        inds = torch.zeros(len(structure.individuals_ids))
        for ind_id in snapshot.states.keys():
            inds[structure.individuals_ids.index(ind_id)] = 1.0
        return inds

    def _struct_to_graphs(self, structure : Chronology) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, v, a, adjacency, individual = [], [], [], [], []
        nb_individuals = len(structure.individuals_ids)
        
        current_speed = {ind_id : torch.zeros(1, 1, 3) for ind_id in structure.individuals_ids}
        for snapshot in structure.snapshots:
            if snapshot is not None and (not self.skip_stationary or snapshot.time not in structure.stationary_times):
                x_snapshot = torch.zeros(nb_individuals, 3) # [num_individuals, dimensions]
                v_snapshot = torch.zeros(nb_individuals, 3) # [num_individuals, dimensions]
                a_snapshot = torch.zeros(nb_individuals, 3) # [num_individuals, dimensions]
                
                for ind_id, state in snapshot.states.items():
                    self.v0 = current_speed[ind_id]
                    feature = self._state_to_vector(state, structure) # [3, dimensions]
                    current_speed[ind_id] = self.v0
                    x_snapshot[structure.individuals_ids.index(ind_id)] = feature[0]
                    v_snapshot[structure.individuals_ids.index(ind_id)] = feature[1]
                    a_snapshot[structure.individuals_ids.index(ind_id)] = feature[2]
                
                adjacency_snapshot = self._snapshot_to_graph(snapshot, structure) # [num_individuals, num_individuals]
                individuals_snapshot = self._snapshot_to_ids(snapshot, structure) # [num_individuals]
                
                x.append(x_snapshot)
                v.append(v_snapshot)
                a.append(a_snapshot)
                adjacency.append(adjacency_snapshot)
                individual.append(individuals_snapshot)
        
        return torch.stack(x), torch.stack(v), torch.stack(a), torch.stack(adjacency), torch.stack(individual)

    def load(self, structure : Chronology) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Build individual sequences
        x, v, a, adjacency, individual = self._struct_to_graphs(structure)

        # Create lookback blocks
        x_lookback, v_lookback, a_lookback, adjacency_lookback, individual_lookback = [], [], [], [], []

        for i in range(len(x)-self.lookback-self.target_offset_end+1):
            x_lookback.append(x[i:i+self.lookback])
            v_lookback.append(v[i:i+self.lookback])
            a_lookback.append(a[i+self.target_offset_start:i+self.lookback+self.target_offset_end])
            adjacency_lookback.append(adjacency[i:i+self.lookback])
            individual_lookback.append(individual[i:i+self.lookback])

        return torch.stack(x_lookback), torch.stack(v_lookback), torch.stack(a_lookback), torch.stack(adjacency_lookback), torch.stack(individual_lookback)
