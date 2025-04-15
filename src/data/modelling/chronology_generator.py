
from typing import Union, List, Dict, Tuple
import tqdm

from data.structure.chronology import Chronology
from data.structure.loaders import DynamicSeriesLoader, DynamicGraphSeriesLoader, BehaviourSeriesLoader
from model.dynamics_model import DynamicalPredictor
from model.graph_dynamics_model import DynamicalGraphPredictor
from model.behaviour_model import TSPredictor

import pytorch_lightning as pl
import torch

State = Chronology.State
Snapshot = Chronology.Snapshot


GENERATOR_MODE_LOADER = { # TODO: implement `all` mode that computes both coordinates and behaviours
    "dynamic" : DynamicSeriesLoader,
    "graph" : DynamicGraphSeriesLoader,
    "behaviour" : BehaviourSeriesLoader
}

ALLOWED_MODELS = {
    "dynamic" : [DynamicalPredictor],
    "graph" : [DynamicalGraphPredictor],
    "behaviour" : [TSPredictor]
}

def filter_kwargs_per_mode(mode : str, kwargs : Dict) -> Dict:
    if mode == "dynamic":
        return {k: v for k, v in kwargs.items() if k in DynamicSeriesLoader.__init__.__code__.co_varnames}
    elif mode == "graph":
        return {k: v for k, v in kwargs.items() if k in DynamicGraphSeriesLoader.__init__.__code__.co_varnames}
    elif mode == "behaviour":
        return {k: v for k, v in kwargs.items() if k in BehaviourSeriesLoader.__init__.__code__.co_varnames}
    else:
        raise ValueError(f"Mode {mode} not supported.")

class ChronologyGenerator:
    def __init__(self, chronology : Chronology, models : Union[pl.LightningModule, List[pl.LightningModule]], lookback : int, skip_stationary : bool = True, modes : Union[str,List[str]] = "dynamic", device : str = "cuda:0", **kwargs):
        self.chronology = chronology
        
        self.lookback = lookback
        self.skip_stationary = skip_stationary
        self.device = device
        
        if isinstance(modes, str):
            modes = [modes]
        self.modes = modes # modes: dynamic, graph, behaviour, all

        for mode in modes:
            if mode not in GENERATOR_MODE_LOADER.keys():
                raise ValueError(f"Mode {mode} not supported. Options: {','.join(GENERATOR_MODE_LOADER.keys())}.")

        if "graph" in modes and "dynamic" in modes:
            raise ValueError("Modes `graph` and `dynamic` cannot be used together. Only one dynamical mode (i.e. that updates coordinates) can be used at a time.")

        if isinstance(models, pl.LightningModule):
            models = [models]
        self.models = {mode: model.to(self.device) for mode, model in zip(modes, models)}

        if len(modes) != len(models):
            raise ValueError("The number of modes and models must be the same.")
        
        for i in range(len(modes)):
            if not any([isinstance(models[i], model_type) for model_type in ALLOWED_MODELS[modes[i]]]):
                raise ValueError(f"Model {type(models[i])} not supported for mode {modes[i]}. Options: {','.join([str(model) for model in ALLOWED_MODELS[modes[i]]])}.")
            if modes[i] == "dynamic" and isinstance(models[i], DynamicGraphSeriesLoader): # /!\ Test needed because the previous one will not catch graph models loaded with dynamic as DynamicGraphPredictor is a subclass of DynamicalPredictor
                raise ValueError(f"Model {type(models[i])} should not be used with mode `dynamic`. Use `graph` instead.")

        self.loaders = {}
        for mode in modes:
            self.loaders[mode] = GENERATOR_MODE_LOADER[mode](lookback=lookback, skip_stationary=skip_stationary, **filter_kwargs_per_mode(mode, kwargs))

        
    def get_last_idx_individuals(self, data : List[int]) -> List[int]:
        individuals = self.chronology.snapshots[-1].states.keys()
        last_idx_individuals = []
        rev_data = data[::-1]

        for idx in individuals:
            last_idx_individuals.append(len(data) - rev_data.index(idx) - 1)
            
        return last_idx_individuals

    def load_last(self, mode : str):
        data = self.loaders[mode].load(self.chronology)
        if mode == "behaviour":
            last_idx_individuals = self.get_last_idx_individuals(data[-1])
            return {'x': data[0][last_idx_individuals].to(self.device)}
        elif mode == "dynamic":
            last_idx_individuals = self.get_last_idx_individuals(data[-1])
            return {'x': data[0][last_idx_individuals].to(self.device), 'velocity': data[1][last_idx_individuals].to(self.device)}
        elif mode == "graph":
            batch = data[-1]
            return {'coordinates': batch.x.to(self.device), 'velocity': batch.v.to(self.device), 'adjacency_index': batch.edge_index.to(self.device), "adjacency_attr": batch.edge_attr.to(self.device)}
        else:
            raise ValueError(f"Mode {mode} not supported.")

    def update_adjacencies(self, coordinates : torch.Tensor) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        close_adjacency_list = {idx: [] for idx in coordinates.keys()}
        distant_adjacency_list = {idx: [] for idx in coordinates.keys()}

        for i, (idx_i, coordinates_i) in enumerate(coordinates.items()):
            for idx_j, coordinates_j in list(coordinates.items())[i+1:]:
                dist = torch.norm(coordinates_i - coordinates_j)

                if dist < 0.5: # TODO: verify distances
                    close_adjacency_list[idx_i].append(idx_j)
                    close_adjacency_list[idx_j].append(idx_i)

                elif dist < 1.0: # TODO: verify distances
                    distant_adjacency_list[idx_i].append(idx_j)
                    distant_adjacency_list[idx_j].append(idx_i)

        return close_adjacency_list, distant_adjacency_list

    def update_chronology(self, generated_data : Dict[str, torch.Tensor]):
        last_snapshot = self.chronology.snapshots[-1]
        individuals = last_snapshot.states.keys()

        if ("dynamic" not in self.modes) and ("graph" not in self.modes):
            coordinates = {idx: state.coordinates for idx, state in last_snapshot.states.items()}
            close_adjacency_list = last_snapshot.close_adjacency_list
            distant_adjacency_list = last_snapshot.distant_adjacency_list
        else:
            if "dynamic" in self.modes:
                coordinates = dict(zip(individuals, generated_data["dynamic"][:,-1,:].unbind()))
            else:
                coordinates = dict(zip(individuals, generated_data["graph"][-len(individuals):,:].unbind()))

            close_adjacency_list, distant_adjacency_list = self.update_adjacencies(coordinates)
            coordinates = {idx: tuple(coord.tolist()) for idx, coord in coordinates.items()}

        if "behaviour" not in self.modes:
            behaviours = {idx: state.behaviour for idx, state in last_snapshot.states.items()}
        else:
            behaviours = dict(zip(individuals, generated_data["behaviour"].tolist()))
        
        # Create new states
        new_states = {}
        for idx, state in last_snapshot.states.items():
            new_states[idx] = State(individual_id=idx, 
                                    zone=state.zone, # /!\ cannot be modified for now
                                    type=state.type, # /!\ cannot be modified (individuals cannot change type)
                                    behaviour=behaviours[idx],
                                    coordinates=coordinates[idx], 
                                    close_neighbours=close_adjacency_list[idx], 
                                    distant_neighbours=distant_adjacency_list[idx],
                                    snapshot=None, # Snaphot has not been created yet
                                    past_state=state,
                                    future_state=None)

        # Create new snapshot
        time = self.chronology.end_time + 1
        new_snapshot = Snapshot(time=time,
                                close_adjacency_list=close_adjacency_list,
                                distant_adjacency_list=distant_adjacency_list,
                                states=new_states)
        for state in new_states.values():
            state.snapshot = new_snapshot

        # Update chronology
        self.chronology.snapshots.append(new_snapshot)
        self.chronology.end_time = time
        for idx in new_states.keys():
            self.chronology.all_occurences[idx].append(time)
        if last_snapshot.time_eq(new_snapshot):
            self.chronology.stationary_times.append(time)
            

    def step(self):
        # Build data
        data = {}
        for mode in self.loaders.keys():
            data[mode] = self.load_last(mode)

        # Generate next timestep
        generated_data = {}
        for mode in self.modes:
            gen_data = self.models[mode](**data[mode])
            generated_data[mode] = gen_data

        # Update chronology
        self.update_chronology(generated_data)

    def generate(self, n_steps : int):
        for _ in tqdm.tqdm(range(n_steps)):
            self.step()

    def save(self, path : str):
        self.chronology.serialize(path)
        

