
import pytest

from data.structure.chronology import Chronology
from data.constants import VECTOR_COLUMNS


class TestChronology:
    
    @pytest.fixture
    def structure(self):
        return Chronology.create("data/test/22-10-20_C2_20.csv")
    
    def test_parse(self, structure):
        assert structure.start_time == 0
        assert structure.end_time == 706
        assert structure.stationary_times == [1, 6, 19, 20, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 43, 124, 172, 173, 174, 175, 176, 204, 205, 214, 377, 642]
        assert structure.empty_times == list(range(49,121))
        assert len(structure.snapshots) == 707
        assert structure.raw_data.shape == (3907, 8)
        assert structure.parser is not None
        assert structure.individuals_ids == [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
        assert structure.first_occurence == {0: 0, 1: 23, 2: 24, 3: 12, 4: 0, 5: 3, 6: 4, 7: 7, 8: 41, 10: 121, 11: 121, 12: 125, 13: 126, 14: 127, 15: 129, 16: 138, 17: 141, 18: 177, 19: 243, 20: 248, 21: 248, 22: 251, 23: 255, 24: 256, 25: 256, 26: 256, 27: 256, 28: 259, 29: 266, 30: 278, 31: 301, 32: 322, 33: 342, 34: 367, 35: 374, 36: 383, 37: 383, 38: 385, 39: 391, 40: 487, 41: 687}
        assert structure.zone_labels == ['foraging_zone', 'background_zone', 'waiting_area_zone', 'door_zone']
        assert structure.behaviour_labels == ['foraging', 'raised_guarding_(vigilant)', 'moving', 'low_sitting/standing_(stationary)', 'groom', 'high_sitting/standing_(vigilant)', 'sunbathe', 'human_interaction', 'interacting_with_foreign_object', 'playfight', 'dig_burrow']

    def test_delete(self, structure):
        del structure
        with pytest.raises(NameError):
            structure

    def test_delete2(self, structure):
        assert len(structure.snapshots) == 707
        del structure
        structure = Chronology.create("data/test/22-10-20_C2_20.csv")
        assert len(structure.snapshots) == 707


    def test_individuals(self, structure):
        snapshot_individuals = []
        for snapshot in structure.snapshots:
            if snapshot is not None:
                for key, state in snapshot.states.items():
                    assert key == state.individual_id
                    snapshot_individuals.append(key)
        
        assert set(snapshot_individuals) == set(structure.individuals_ids)

    def test_state_snapshot(self, structure):
        for snapshot in structure.snapshots:
            if snapshot is not None:
                for state in snapshot.states.values():
                    assert state.snapshot is snapshot

    def test_first_occurence(self, structure):
        first_occurence_in_snapshot = {}
        for snapshot in structure.snapshots:
            if snapshot is not None:
                for key in snapshot.states.keys():
                    if key not in first_occurence_in_snapshot:
                        first_occurence_in_snapshot[key] = snapshot.time

        assert first_occurence_in_snapshot == structure.first_occurence

    def test_snapshot(self, structure):
        snapshot = structure.snapshots[0]
        assert snapshot.time == 0
        assert len(snapshot.states) == 2
        assert list(snapshot.states.keys()) == [0, 4]
        assert all([state.snapshot is snapshot for state in snapshot.states.values()])
        assert len(snapshot.close_adjacency_list) == 2
        assert len(snapshot.close_adjacency_list[0]) == 0
        assert len(snapshot.close_adjacency_list[4]) == 0
        assert len(snapshot.distant_adjacency_list) == 2
        assert len(snapshot.distant_adjacency_list[0]) == 0
        assert len(snapshot.distant_adjacency_list[4]) == 0

    def test_state(self, structure):
        state = structure.snapshots[0].states[0]
        assert state.individual_id == 0
        assert state.zone == 'foraging_zone'
        assert state.behaviour == 'foraging'
        assert state.close_neighbours == []
        assert state.distant_neighbours == []
        assert state.snapshot is structure.snapshots[0]
        assert state.future_state is not None
        assert state.past_state is None

    def test_state_sequence(self, structure):
        ind_id = 0
        state = structure.snapshots[0].states[ind_id]
        sequence = [state]
        while state.future_state is not None:
            state = state.future_state
            sequence.append(state)

        assert len(sequence) == 24

        reverse_sequence = [state]
        while state.past_state is not None:
            state = state.past_state
            reverse_sequence.append(state)

        assert len(reverse_sequence) == 24

    def test_closest_index(self):
        assert Chronology._closest_index(3, [1, 2, 3, 4, 5]) == 2
        assert Chronology._closest_index(2.5, [1, 2, 3, 4, 5]) == 1
        assert Chronology._closest_index(2.6, [1, 2, 3, 4, 5]) == 1
        assert Chronology._closest_index(2.4, [1, 2, 3, 4, 5]) == 1
        assert Chronology._closest_index(1, [1, 2, 3, 4, 5]) == 0
        assert Chronology._closest_index(5, [1, 2, 3, 4, 5]) == 4
        assert Chronology._closest_index(6, [1, 2, 3, 4, 5]) == 4
        assert Chronology._closest_index(0, [1, 2, 3, 4, 5]) == 0
        assert Chronology._closest_index(-1, [1, 2, 3, 4, 5]) == 0

    def test_get_snapshot(self, structure):
        snapshot = structure.get_snapshot(0)
        assert snapshot.time == 0
        assert len(snapshot.states) == 2
        assert list(snapshot.states.keys()) == [0, 4]
        assert len(snapshot.close_adjacency_list) == 2
        assert len(snapshot.close_adjacency_list[0]) == 0
        assert len(snapshot.close_adjacency_list[4]) == 0
        assert len(snapshot.distant_adjacency_list) == 2
        assert len(snapshot.distant_adjacency_list[0]) == 0
        assert len(snapshot.distant_adjacency_list[4]) == 0

    def test_equal(self, structure):
        assert structure == structure

    def test_equal2(self, structure):
        struct2 = Chronology.create("data/test/22-10-20_C2_20.csv")

        assert ((structure.raw_data is None and struct2.raw_data is None) or structure.raw_data.equals(struct2.raw_data))
        assert ((structure.parser is None and struct2.parser is None) or structure.parser == struct2.parser)
        assert structure.start_time == struct2.start_time
        assert structure.end_time == struct2.end_time
        assert structure.stationary_times == struct2.stationary_times
        assert structure.empty_times == struct2.empty_times
        assert structure.individuals_ids == struct2.individuals_ids
        assert structure.first_occurence == struct2.first_occurence
        assert structure.zone_labels == struct2.zone_labels
        assert structure.behaviour_labels == struct2.behaviour_labels
        assert len(structure.snapshots) == len(struct2.snapshots)
        assert all([(structure.snapshots[i] is None and struct2.snapshots[i] is None) or (structure.snapshots[i].time_eq(struct2.snapshots[i])) for i in range(len(structure.snapshots))])
        assert all([(structure.snapshots[i] is None and struct2.snapshots[i] is None) or structure.snapshots is not struct2.snapshots for i in range(len(structure.snapshots))])
        assert all([all([structure.snapshots[i].states[ind_id].snapshot is not struct2.snapshots[i].states[ind_id].snapshot for ind_id in structure.snapshots[i].states.keys()]) 
                    for i in range(len(structure.snapshots)) if structure.snapshots[i] is not None and struct2.snapshots[i] is not None]) 

        assert structure == struct2

    def test_not_equal(self, structure):
        struct2 = Chronology(None)
        struct2.start_time = 1
        assert structure != struct2

    def test_not_equal2(self, structure):
        struct2 = Chronology.create("data/test/22-10-20_C2_20.csv")
        struct2.end_time = 1
        assert structure != struct2

    def test_deep_copy(self, structure):
        # Check values
        copy = structure.deep_copy()
        assert copy == structure
        assert copy.start_time == structure.start_time
        assert copy.end_time == structure.end_time
        assert copy.stationary_times == structure.stationary_times
        assert copy.empty_times == structure.empty_times
        assert len(copy.snapshots) == len(structure.snapshots)
        assert copy.raw_data.shape == structure.raw_data.shape
        assert copy.parser is not None
        assert copy.individuals_ids == structure.individuals_ids
        assert copy.first_occurence == structure.first_occurence
        assert structure.zone_labels == structure.zone_labels
        assert structure.behaviour_labels == structure.behaviour_labels

        snapshot = structure.snapshots[0]
        copy_snapshot = copy.snapshots[0]
        assert copy_snapshot.time == snapshot.time
        assert len(copy_snapshot.states) == len(snapshot.states)
        assert list(copy_snapshot.states.keys()) == list(snapshot.states.keys())
        assert copy_snapshot.close_adjacency_list == snapshot.close_adjacency_list
        assert copy_snapshot.distant_adjacency_list == snapshot.distant_adjacency_list

        state = snapshot.states[0]
        copy_state = copy_snapshot.states[0]
        assert copy_state.individual_id == state.individual_id
        assert copy_state.zone == state.zone
        assert copy_state.behaviour == state.behaviour
        assert copy_state.close_neighbours == state.close_neighbours
        assert copy_state.distant_neighbours == state.distant_neighbours
        assert copy_state.future_state is not None
        assert copy_state.past_state is None
        assert copy_state.snapshot.time_eq(state.snapshot)

        # Check references
        assert copy is not structure
        assert copy.parser is not structure.parser
        assert copy.stationary_times is not structure.stationary_times
        assert copy.empty_times is not structure.empty_times
        assert copy.snapshots is not structure.snapshots
        assert copy.raw_data is not structure.raw_data
        assert copy.individuals_ids is not structure.individuals_ids
        assert copy.first_occurence is not structure.first_occurence
        assert copy.zone_labels is not structure.zone_labels
        assert copy.behaviour_labels is not structure.behaviour_labels

        assert copy_snapshot is not snapshot
        # assert copy_snapshot.time is not snapshot.time # can be identical as Python optimizes memory, not a problem as types are immutable
        assert copy_snapshot.states is not snapshot.states
        assert copy_snapshot.close_adjacency_list is not snapshot.close_adjacency_list
        assert copy_snapshot.distant_adjacency_list is not snapshot.distant_adjacency_list

        assert copy_state is not state
        # assert copy_state.individual_id is not state.individual_id # same as above
        # assert copy_state.zone is not state.zone
        # assert copy_state.behaviour is not state.behaviour
        assert copy_state.close_neighbours is not state.close_neighbours
        assert copy_state.distant_neighbours is not state.distant_neighbours
        assert copy_state.future_state is not state.future_state
        assert copy_state.snapshot is not state.snapshot

    def test_deep_copy2(self, structure):
        copy = structure.deep_copy()
        copy.snapshots.append(None)
        copy.snapshots[0].time = -1
        copy.snapshots[1].states[-1] = None
        copy.snapshots[2].states[0].individual_id =-1

        assert len(structure.snapshots) == 707
        assert len(copy.snapshots) == len(structure.snapshots) +1
        assert copy.snapshots is not structure.snapshots
        assert copy.snapshots[0].time != structure.snapshots[0].time
        assert len(copy.snapshots[1].states) == len(structure.snapshots[1].states) + 1
        assert -1 not in structure.snapshots[1].states
        assert copy.snapshots[2].states[0].individual_id ==  -1
        assert structure.snapshots[2].states[0].individual_id == 0


    def test_split(self, structure):
        struct0, struct1 = structure.split(300)
        assert struct0.start_time == 0
        assert struct0.end_time == 299
        assert struct1.start_time == 300
        assert struct1.end_time == 706
        assert len(struct0.snapshots) == 300
        assert len(struct1.snapshots) == 407
        assert struct0.raw_data is None
        assert struct1.raw_data is None
        assert struct0.parser is None
        assert struct1.parser is None
        assert struct0.individuals_ids == [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        assert struct1.individuals_ids == [21, 23, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
        assert struct0.first_occurence == {0: 0, 4: 0, 5: 3, 6: 4, 7: 7, 3: 12, 1: 23, 2: 24, 8: 41, 10: 121, 11: 121, 12: 125, 13: 126, 14: 127, 15: 129, 16: 138, 17: 141, 18: 177, 19: 243, 20: 248, 21: 248, 22: 251, 23: 255, 25: 256, 24: 256, 26: 256, 27: 256, 28: 259, 29: 266, 30: 278}
        assert struct1.first_occurence == {21: 300, 23: 300, 25: 300, 26: 300, 30: 300, 27: 301, 29: 301, 31: 301, 32: 322, 33: 342, 34: 367, 35: 374, 36: 383, 37: 383, 38: 385, 39: 391, 40: 487, 41: 687}
        assert struct0.zone_labels == struct1.zone_labels
        assert struct0.behaviour_labels == struct1.behaviour_labels

        shared_inds = set(struct0.snapshots[-1].states.keys()).intersection(set(struct1.snapshots[0].states.keys()))
        assert len(shared_inds) > 0

        for ind in shared_inds:
            state0 = struct0.snapshots[-1].states[ind]
            assert state0.future_state is None
            assert state0.past_state is not None

            state1 = struct1.snapshots[0].states[ind]
            assert state1.past_state is None
            assert state1.future_state is not None

    @pytest.mark.parametrize("split_time", [-1, 707])
    def test_split_out_of_bounds(self, structure, split_time):
        struct0, struct1 = structure.split(split_time)

        if struct0 is None:
            nonestruct = struct0
            fullstruct = struct1
        else:
            nonestruct = struct1
            fullstruct = struct0

        assert nonestruct is None
        assert fullstruct == structure
        assert fullstruct is not structure

    def test_merge_no_interlace_2(self, structure):
        structure.raw_data = None
        structure.parser = None
        struct0, struct1 = structure.split(300)
        merged = Chronology.merge_not_interlace_2(struct0, struct1, keep_individual_ids=False)

        assert merged != structure

        assert merged.raw_data is None
        assert merged.parser is None
        assert structure.start_time == merged.start_time
        assert structure.end_time == merged.end_time
        assert structure.stationary_times == merged.stationary_times
        assert structure.empty_times == merged.empty_times
        assert merged.individuals_ids == [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
        assert merged.first_occurence == {0: 0, 4: 0, 5: 3, 6: 4, 7: 7, 3: 12, 1: 23, 2: 24, 8: 41, 10: 121, 11: 121, 12: 125, 13: 126, 14: 127, 15: 129, 16: 138, 17: 141, 18: 177, 19: 243, 20: 248, 21: 248, 22: 251, 23: 255, 25: 256, 24: 256, 26: 256, 27: 256, 28: 259, 29: 266, 30: 278, 31: 300, 33: 300, 35: 300, 36: 300, 40: 300, 37: 301, 39: 301, 41: 301, 42: 322, 43: 342, 44: 367, 45: 374, 46: 383, 47: 383, 48: 385, 49: 391, 50: 487, 51: 687}
        assert struct0.zone_labels == struct1.zone_labels
        assert structure.zone_labels == merged.zone_labels
        assert structure.behaviour_labels == merged.behaviour_labels
        assert len(structure.snapshots) == len(merged.snapshots)
        assert all([(structure.snapshots[i] is None and merged.snapshots[i] is None) or (structure.snapshots[i].time_eq(merged.snapshots[i])) for i in range(300)])
        assert not any([(structure.snapshots[i] is None and merged.snapshots[i] is None) or (structure.snapshots[i].time_eq(merged.snapshots[i])) for i in range(300,707)])
        assert all([state.past_state is None for state in merged.snapshots[300].states.values()])
        assert all([state.future_state is None for state in merged.snapshots[299].states.values()])

        # Reset offset
        offset = -10
        for snapshot in merged.snapshots[300:]:
            if snapshot is not None:
                snapshot.states = {ind_id + offset : state for ind_id, state in snapshot.states.items()}
                snapshot.close_adjacency_list = {ind_id + offset : [adj + offset for adj in adj_list] for ind_id, adj_list in snapshot.close_adjacency_list.items()}
                snapshot.distant_adjacency_list = {ind_id + offset : [adj + offset for adj in adj_list] for ind_id, adj_list in snapshot.distant_adjacency_list.items()}
                for state in snapshot.states.values():
                    state.individual_id += offset
                    state.close_neighbours = [adj + offset for adj in state.close_neighbours]
                    state.distant_neighbours = [adj + offset for adj in state.distant_neighbours]

        # Equality should hold
        assert all([(structure.snapshots[i] is None and merged.snapshots[i] is None) or (structure.snapshots[i].time_eq(merged.snapshots[i])) for i in range(len(structure.snapshots))])

    def test_merge_no_interlace_2_keep_ids(self, structure):
        structure.raw_data = None
        structure.parser = None
        struct0, struct1 = structure.split(300)
        merged = Chronology.merge_not_interlace_2(struct0, struct1, keep_individual_ids=True)
        
        assert merged == structure

        shared_inds = set(struct0.snapshots[-1].states.keys()).intersection(set(struct1.snapshots[0].states.keys()))
        assert len(shared_inds) > 0

        for ind in shared_inds:
            state0 = merged.snapshots[299].states[ind]
            state1 = merged.snapshots[300].states[ind]
            assert state0.future_state is state1
            assert state1.past_state is state0

    def test_merge_no_interlace_n_keep_ids(self, structure):
        structure.raw_data = None
        structure.parser = None
        struct0, struct1 = structure.split(100)
        struct1, struct2 = struct1.split(200)
        struct2, struct3 = struct2.split(300)
        struct3, struct4 = struct3.split(400)

        merged = Chronology.merge_no_interlace_n([struct0, struct1, struct2, struct3, struct4], keep_individual_ids=True)
        
        assert merged == structure
        
        



        


