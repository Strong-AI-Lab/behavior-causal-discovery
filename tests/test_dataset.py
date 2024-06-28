
import pytest
import torch

from data.dataset import SeriesDataset, DiscriminatorDataset
from data.structure.loaders import BehaviourSeriesLoader, DynamicSeriesLoader, DynamicGraphSeriesLoader
from data.structure.chronology import Chronology
from dynamics.solver import DynamicsSolver


class TestSeriesDataset:
    @pytest.fixture
    def chronology(self):
        states = {}
        for ind_id in range(3):
            states_ind = []
            prev_state = None
            for _ in range(4):
                state = Chronology.State(ind_id, "zone_1", "behaviour_1", (0,0,0), [], [], None, prev_state, None)
                states_ind.append(state)
                if prev_state:
                    prev_state.future_state = state
                prev_state = state
            states[ind_id] = states_ind
        
        states[0][0].zone = "zone_2"
        states[0][2].behaviour = "behaviour_2"
        states[2][3].zone = "zone_2"
        states[2][3].close_neighbours = [0]
        states[2][3].distant_neighbours = [1]
        print([s.zone for s in states[0]])

        return Chronology(
            data=None,
            parser=None,
            start_time=0,
            end_time=3,
            stationary_times=None,
            empty_times=None,
            snapshots=[
                Chronology.Snapshot(
                    time=0,
                    close_adjacency_list={0:[], 1:[], 2:[]},
                    distant_adjacency_list={0:[], 1:[], 2:[]},
                    states={0: states[0][0], 1: states[1][0], 2: states[2][0]}
                ),
                Chronology.Snapshot(
                    time=1,
                    close_adjacency_list={0:[], 1:[], 2:[]},
                    distant_adjacency_list={0:[], 1:[], 2:[]},
                    states={0: states[0][1], 1: states[1][1], 2: states[2][1]}
                ),
                Chronology.Snapshot(
                    time=2,
                    close_adjacency_list={0:[], 1:[], 2:[]},
                    distant_adjacency_list={0:[], 1:[], 2:[]},
                    states={0: states[0][2], 1: states[1][2], 2: states[2][2]}
                ),
                Chronology.Snapshot(
                    time=3,
                    close_adjacency_list={0:[], 1:[], 2:[0]},
                    distant_adjacency_list={0:[], 1:[], 2:[1]},
                    states={0: states[0][3], 1: states[1][3], 2: states[2][3]}
                )
            ],
            individuals_ids=[0,1,2],
            first_occurence={
                0: 0,
                1: 0,
                2: 0
            },
            zone_labels=["zone_1", "zone_2"],
            behaviour_labels=["behaviour_1", "behaviour_2"],
            parse_data=False)

    @pytest.fixture
    def struct_loader(self):
        return BehaviourSeriesLoader(
            lookback=2, 
            skip_stationary=False,
            vector_columns=[
                "zone_1", "zone_2", "close_neighbour_zone_1", "close_neighbour_zone_2", "distant_neighbour_zone_1", "distant_neighbour_zone_2",
                "behaviour_1", "behaviour_2", "close_neighbour_behaviour_1", "close_neighbour_behaviour_2", "distant_neighbour_behaviour_1", "distant_neighbour_behaviour_2"
            ])
    
    @pytest.fixture
    def dataset(self, chronology, struct_loader):
        return SeriesDataset(chronology, struct_loader)


    def test_len(self, dataset):
        assert len(dataset) == 6

    def test_getitem(self, dataset):
        x, y, i = dataset[0]
        assert x.shape == (2, 12)
        assert y.shape == (2, 12)
        assert i == 0

        true_x = torch.zeros(2, 12)
        true_x[0, 1] = 1
        true_x[1, 0] = 1
        true_x[:, 6] = 1
        assert (x == true_x).all()

        true_y = torch.zeros(2, 12)
        true_y[:, 0] = 1
        true_y[0, 6] = 1
        true_y[1, 7] = 1
        assert (y == true_y).all()



    def test_getitem_last(self, dataset):
        x, y, i = dataset[-1]
        assert x.shape == (2, 12)
        assert y.shape == (2, 12)
        assert i == 2

        true_x = torch.zeros(2, 12)
        true_x[:, 0] = 1
        true_x[:, 6] = 1
        assert (x == true_x).all()

        true_y = torch.zeros(2, 12)
        true_y[0, 0] = 1
        true_y[1, 1] = 1
        true_y[:, 6] = 1
        true_y[1, 2] = 1
        true_y[1, 4] = 1
        true_y[1, 8] = 1
        true_y[1, 10] = 1
        assert (y == true_y).all()



class TestDynamicSeriesDataset:
    @pytest.fixture
    def chronology(self):
        states = {}
        for ind_id in range(3):
            states_ind = []
            prev_state = None
            for t in range(4):
                state = Chronology.State(ind_id, "zone_1", "behaviour_1", (t,t,t), [], [], None, prev_state, None)
                states_ind.append(state)
                if prev_state:
                    prev_state.future_state = state
                prev_state = state
            states[ind_id] = states_ind

        return Chronology(
            data=None,
            parser=None,
            start_time=0,
            end_time=3,
            stationary_times=None,
            empty_times=None,
            snapshots=[
                Chronology.Snapshot(
                    time=0,
                    close_adjacency_list={0:[], 1:[], 2:[]},
                    distant_adjacency_list={0:[], 1:[], 2:[]},
                    states={0: states[0][0], 1: states[1][0], 2: states[2][0]}
                ),
                Chronology.Snapshot(
                    time=1,
                    close_adjacency_list={0:[], 1:[], 2:[]},
                    distant_adjacency_list={0:[], 1:[], 2:[]},
                    states={0: states[0][1], 1: states[1][1], 2: states[2][1]}
                ),
                Chronology.Snapshot(
                    time=2,
                    close_adjacency_list={0:[], 1:[], 2:[]},
                    distant_adjacency_list={0:[], 1:[], 2:[]},
                    states={0: states[0][2], 1: states[1][2], 2: states[2][2]}
                ),
                Chronology.Snapshot(
                    time=3,
                    close_adjacency_list={0:[], 1:[], 2:[]},
                    distant_adjacency_list={0:[], 1:[], 2:[]},
                    states={0: states[0][3], 1: states[1][3], 2: states[2][3]}
                )
            ],
            individuals_ids=[0,1,2],
            first_occurence={
                0: 0,
                1: 0,
                2: 0
            },
            zone_labels=["zone_1", "zone_2"],
            behaviour_labels=["behaviour_1", "behaviour_2"],
            parse_data=False)
    
    @pytest.fixture
    def solver(self):
        return DynamicsSolver(mass=1, dimensions=3)

    @pytest.fixture
    def struct_loader(self, solver):
        return DynamicSeriesLoader(
            lookback=2, 
            skip_stationary=False,
            solver=solver)
    
    @pytest.fixture
    def dataset(self, chronology, struct_loader):
        return SeriesDataset(chronology, struct_loader)
    

    def test_len(self, dataset):
        assert len(dataset) == 3

    def test_getitem(self, dataset):
        x, v, a, i = dataset[0]
        assert x.shape == (2, 3)
        assert v.shape == (2, 3)
        assert a.shape == (2, 3)
        assert i == 0

    def test_getitem_last(self, dataset):
        x, v, a, i = dataset[-1]
        assert x.shape == (2, 3)
        assert v.shape == (2, 3)
        assert a.shape == (2, 3)
        assert i == 2



class TestDynamicGraphSeriesDataset:
    @pytest.fixture
    def chronology(self):
        states = {}
        for ind_id in range(4):
            states_ind = []
            prev_state = None
            for t in range(4):
                state = Chronology.State(ind_id, "zone_1", "behaviour_1", (t,t,t), [], [], None, prev_state, None)
                states_ind.append(state)
                if prev_state:
                    prev_state.future_state = state
                prev_state = state
            states[ind_id] = states_ind
        
        states[0][3].close_neighbours = [1]
        states[1][3].distant_neighbours = [2]

        return Chronology(
            data=None,
            parser=None,
            start_time=0,
            end_time=3,
            stationary_times=None,
            empty_times=None,
            snapshots=[
                Chronology.Snapshot(
                    time=0,
                    close_adjacency_list={0:[], 1:[], 2:[]},
                    distant_adjacency_list={0:[], 1:[], 2:[]},
                    states={0: states[0][0], 1: states[1][0], 2: states[2][0], 3: states[3][0]}
                ),
                Chronology.Snapshot(
                    time=1,
                    close_adjacency_list={0:[], 1:[], 2:[]},
                    distant_adjacency_list={0:[], 1:[], 2:[]},
                    states={0: states[0][1], 1: states[1][1], 2: states[2][1], 3: states[3][1]}
                ),
                Chronology.Snapshot(
                    time=2,
                    close_adjacency_list={0:[1], 1:[], 2:[]},
                    distant_adjacency_list={0:[], 1:[2], 2:[]},
                    states={0: states[0][2], 1: states[1][2], 2: states[2][2], 3: states[3][2]}
                ),
                Chronology.Snapshot(
                    time=3,
                    close_adjacency_list={0:[], 1:[], 2:[]},
                    distant_adjacency_list={0:[], 1:[], 2:[]},
                    states={0: states[0][3], 1: states[1][3], 2: states[2][3], 3: states[3][3]}
                )
            ],
            individuals_ids=[0,1,2,3],
            first_occurence={
                0: 0,
                1: 0,
                2: 0,
                3: 0
            },
            zone_labels=["zone_1", "zone_2"],
            behaviour_labels=["behaviour_1", "behaviour_2"],
            parse_data=False)
    
    @pytest.fixture
    def solver(self):
        return DynamicsSolver(mass=1, dimensions=3)

    @pytest.fixture
    def struct_loader(self, solver):
        return DynamicGraphSeriesLoader(
            lookback=2, 
            skip_stationary=False,
            solver=solver)
    
    @pytest.fixture
    def dataset(self, chronology, struct_loader):
        return SeriesDataset(chronology, struct_loader)

    def test_len(self, dataset):
        assert len(dataset) == 2

    def test_getitem(self, dataset):
        x, v, a, adj, i = dataset[0]
        assert x.shape == (2, 4, 3)
        assert v.shape == (2, 4, 3)
        assert a.shape == (2, 4, 3)
        assert adj.shape == (2, 4, 4)
        assert i.shape == (2, 4)
        
        assert (adj == torch.zeros(2, 4, 4)).all()

    def test_getitem_last(self, dataset):
        x, v, a, adj, i = dataset[-1]
        assert x.shape == (2, 4, 3)
        assert v.shape == (2, 4, 3)
        assert a.shape == (2, 4, 3)
        assert adj.shape == (2, 4, 4)
        assert i.shape == (2, 4)

        true_adj = torch.zeros(2, 4, 4)
        true_adj[1, 0, 1] = 1
        true_adj[1, 1, 2] = 0.5
        assert (adj == true_adj).all()



