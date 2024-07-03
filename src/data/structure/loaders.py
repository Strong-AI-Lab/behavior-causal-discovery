
import abc
from typing import Tuple, List, Optional, Dict, Any, Union

from data.structure.chronology import Chronology
from data.constants import VECTOR_COLUMNS, MASKED_VARIABLES
from dynamics.solver import DynamicsSolver

import numpy as np
import torch


class Loader(metaclass=abc.ABCMeta):
    def __init__(self, skip_stationary : bool = False):
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
    def _state_to_vector(self, state : Chronology.State, structure : Chronology) -> Any:
        return

    @abc.abstractmethod
    def load(self, structure : Chronology) -> Any:
        return
    



class BehaviourSimpleLoader(Loader):
    def __init__(self, skip_stationary : bool = False, vector_columns : List[str] = None):
        super().__init__(skip_stationary)
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

    def load(self, structure : Chronology) -> Dict[int, np.ndarray]:
        sequences = self._struct_to_sequences(structure)
        sequences = {key: seq.numpy() for key, seq in sequences.items()}
        return sequences




class SeriesLoader(Loader):

    def __init__(self, lookback : int, target_offset_start : int = 1, target_offset_end : int = 1, skip_stationary : bool = False):
        super().__init__(skip_stationary)
        self.lookback = lookback
        self.target_offset_start = target_offset_start
        self.target_offset_end = target_offset_end

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



class GeneratorLoader(BehaviourSeriesLoader):
    def __init__(self, lookback : int, target_offset_start : int = 1, target_offset_end : int = 1, skip_stationary : bool = False, vector_columns : List[str] = None, masked_variables : List[str] = None):
        super().__init__(lookback, target_offset_start, target_offset_end, skip_stationary, vector_columns)
        if masked_variables is None:
            masked_variables = MASKED_VARIABLES
        self.masked_variables = masked_variables

        masked_ids = [self.vector_columns.index(var) for var in self.masked_variables]
        self.selected_idx = torch.where(~torch.tensor([i in masked_ids for i in range(len(self.vector_columns))]))[0]

    def load(self, structure : Chronology, model : torch.nn.Module, build_series : bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor, List[int]], Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        # Build individual sequences
        sequences = self._struct_to_sequences(structure)

        # Create lookback blocks and make model predictions
        updated_sequences = {}
        x, y, individual = [], [], []
        for ind_id, sequence in sequences.items():
            updated_sequences[ind_id] = []
            feature = sequence[:self.lookback] # [lookback, dimensions]
            for i in range(len(sequence)-self.lookback-self.target_offset_end+1):
                # Make prediction
                target_pred = model(feature.unsqueeze(0).to(model.device)).squeeze(0).cpu()

                # Extract ground truth
                target = sequence[i+self.target_offset_start:i+self.lookback+self.target_offset_end]

                # Remove masked variables
                target_pred = target_pred[:,self.selected_idx]
                target = target[:,self.selected_idx]
                feature_reduced = feature[:,self.selected_idx]

                # Quantize prediction and save only last
                target_pred = torch.nn.functional.gumbel_softmax(target_pred.clamp(min=1e-8, max=1-1e-8).log(), hard=True)
                target_pred = torch.cat((feature_reduced[1:,:], target_pred[-1:,:]), dim=0)

                # Update feature with generated values
                feature = sequence[i+1:i+1+self.lookback]
                feature[:,self.selected_idx] = target_pred

                updated_sequences[ind_id].append((target_pred[-1], target[-1])) # ([selected_dimensions], [selected_dimensions])
                x.append(target_pred) # [lookback, selected_dimensions]
                y.append(target) # [lookback, selected_dimensions]
                individual.append(ind_id)

        if build_series:
            return updated_sequences
        else:
            return torch.stack(x), torch.stack(y), individual



class DiscriminatorLoader(GeneratorLoader):
    def load(self, structure : Chronology, model : torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate sequences
        x, y, ind = super().load(structure, model, build_series=False)
        
        # Create discrimination task
        x_discr = []
        y_discr = []        
        for i in range(len(x)):
            x_discr.append(x[i])
            x_discr.append(y[i])
            y_discr.append(1) # Generated
            y_discr.append(0) # Sampled from ground truth

        return torch.stack(x_discr), torch.tensor(y_discr)
    


class GeneratorCommunityLoader(GeneratorLoader):
    def __init__(self, lookback: int, target_offset_start: int = 1, target_offset_end: int = 1, skip_stationary: bool = False, vector_columns: List[str] = None, masked_variables: List[str] = None):
        super().__init__(lookback, target_offset_start, target_offset_end, skip_stationary, vector_columns, masked_variables)

        self.single_loader = GeneratorLoader(lookback, target_offset_start, target_offset_end, skip_stationary, vector_columns, masked_variables)

    def _state_to_vector(self, state : Chronology.State, structure : Chronology) -> torch.Tensor:
        time = state.snapshot.time

        max_neighbours = len(structure.individuals_ids) - 1
        close_neighbours = state.close_neighbours
        distant_neighbours = state.distant_neighbours

        vector = torch.full((1 + 2 * max_neighbours,), -1) # time information + close and distant neighbours

        vector[0] = time
        for i, cn in enumerate(close_neighbours):
            vector[1 + i] = cn
        for j, dn in enumerate(distant_neighbours):
            vector[1 + max_neighbours + j] = dn

        return vector
    
    def _get_time_vector(self, time : int, data : torch.Tensor, metadata : torch.Tensor) -> torch.Tensor:
        idx = torch.where(metadata[:,0] == time)[0].item()
        return data[idx]
    
    def _get_time_neighbours(self, time : int, metadata : torch.Tensor, max_neighbours : int) -> Tuple[List[int], List[int]]:
        idx = torch.where(metadata[:,0] == time)[0].item()
        close_neighbours = metadata[idx,1:1+max_neighbours][metadata[idx,1:1+max_neighbours] != -1].tolist()
        distant_neighbours = metadata[idx,1+max_neighbours:][metadata[idx,1+max_neighbours:] != -1].tolist()
        return close_neighbours, distant_neighbours
    
    def _has_lookback(self, time : int, individual : int, time_index : Dict[int, Dict[int, torch.Tensor]], lookback : int) -> bool:
        for t in range(time-lookback, time):
            if (t not in time_index) or (individual not in time_index[t]) or (time_index[t][individual] is None):
                return False
        return True
    
    def _add_neighbor_information(self, vector : torch.Tensor, close_neighbours : torch.Tensor, distant_neighbours : torch.Tensor, max_neighbours : int) -> torch.Tensor:
        for value in list(filter(lambda x: 'neighbour_' not in x, self.vector_columns)):
            vector[self.vector_columns.index('close_neighbour_' + value)] = close_neighbours[self.vector_columns.index(value)]
            vector[self.vector_columns.index('distant_neighbour_' + value)] = distant_neighbours[self.vector_columns.index(value)]

        return vector
    
    def _get_lookback_vectors(self, time : int, individual : int, time_index : Dict[int, Dict[int, torch.Tensor]], metadata_dict : Dict[int, torch.Tensor], lookback : int, max_neighbours : int) -> List[torch.Tensor]:
        vectors = []
        for t in range(time-lookback, time):
            # Get individual vector
            individual_vector = time_index[t][individual]

            # Get neighbour information
            close_neighbours, distant_neighbours = self._get_time_neighbours(t, metadata_dict[individual], max_neighbours)
            if len(close_neighbours) == 0:
                close_neighbour_vectors = torch.zeros_like(individual_vector)
            else:
                close_neighbour_vectors = torch.stack([time_index[t][neighbour] for neighbour in close_neighbours]).sum(dim=0)
            if len(distant_neighbours) == 0:
                distant_neighbour_vectors = torch.zeros_like(individual_vector)
            else:
                distant_neighbour_vectors = torch.stack([time_index[t][neighbour] for neighbour in distant_neighbours]).sum(dim=0)

            # Add neighbour information
            individual_vector = self._add_neighbor_information(individual_vector, close_neighbour_vectors, distant_neighbour_vectors, max_neighbours)
            vectors.append(individual_vector)

        return torch.stack(vectors)
    
    def load(self, structure : Chronology, model : torch.nn.Module, build_series : bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor, List[int]], Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        # Build individual sequences
        sequences = self.single_loader._struct_to_sequences(structure)

        # Add community information
        sequences_information = self._struct_to_sequences(structure)

        # Build reverse index (time: individuals)
        time_index = {}
        for ind_id, vector in sequences_information.items():
            times = vector[:,0].unique().tolist()
            for time in times:
                if time not in time_index:
                    time_index[time] = {}
                time_index[time][ind_id] = None

        # Create lookback blocks and make model predictions
        updated_sequences = {ind_id : [] for ind_id in sequences.keys()}
        initial_states = {ind_id : 0 for ind_id in sequences.keys()}
        for time in sorted(time_index.keys()):
            individuals = time_index[time].keys()

            for ind_id in individuals:
                if not self._has_lookback(time+1-self.target_offset_end, ind_id, time_index, self.lookback+1-self.target_offset_start):
                    # Get ground truth for initial state
                    time_vector = self._get_time_vector(time, sequences[ind_id], sequences_information[ind_id]) # [dimensions]
                    time_index[time][ind_id] = time_vector

                    # Update sequence with initial state
                    time_vector_select = time_vector[self.selected_idx] # [selected_dimensions]
                    updated_sequences[ind_id].append((time_vector_select, time_vector_select)) # ([selected_dimensions], [selected_dimensions])
                    initial_states[ind_id] += 1

                else:
                    # Get lookback vectors
                    lookback_vectors = self._get_lookback_vectors(time+1-self.target_offset_end, ind_id, time_index, sequences_information, self.lookback+1-self.target_offset_start, len(structure.individuals_ids)-1) # [lookback, dimensions]

                    # Make prediction
                    target_pred = model(lookback_vectors.unsqueeze(0).to(model.device)).squeeze(0).cpu() # [lookback, dimensions]

                    # Extract ground truth
                    target = self._get_time_vector(time, sequences[ind_id], sequences_information[ind_id]) # [dimensions]

                    # Remove masked variables
                    target_pred_select = target_pred[-1, self.selected_idx] # [selected_dimensions]
                    target_select = target[self.selected_idx] # [selected_dimensions]

                    # Quantize prediction
                    target_pred_select = torch.nn.functional.gumbel_softmax(target_pred_select.clamp(min=1e-8, max=1-1e-8).log(), hard=True)

                    # Update sequence with generated values
                    updated_sequences[ind_id].append((target_pred_select, target_select)) # ([selected_dimensions], [selected_dimensions])
                    
                    # Update time index for next iteration (full vector is needed for indexes to match)
                    features = target
                    features[self.selected_idx] = target_pred_select
                    time_index[time][ind_id] = features # [dimensions]

        if build_series:
            return {ind_id : seq[initial_states[ind_id]:] for ind_id, seq in updated_sequences.items()}
        else:
            # Create data sequences
            x, y, individual = [], [], []
            for ind_id, sequence in updated_sequences.items():
                for i in range(len(sequence)-self.lookback):
                    feature = torch.stack([seq[0] for seq in sequence[i:i+self.lookback]])
                    target = torch.stack([seq[1] for seq in sequence[i:i+self.lookback]])
                    x.append(feature)
                    y.append(target)
                    individual.append(ind_id)

            return torch.stack(x), torch.stack(y), individual



class DiscriminatorCommunityLoader(GeneratorCommunityLoader):
    def load(self, structure : Chronology, model : torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate sequences
        x, y, ind = super().load(structure, model, build_series=False)
        
        # Create discrimination task
        x_discr = []
        y_discr = []        
        for i in range(len(x)):
            x_discr.append(x[i])
            x_discr.append(y[i])
            y_discr.append(1) # Generated
            y_discr.append(0) # Sampled from ground truth

        return torch.stack(x_discr), torch.tensor(y_discr)