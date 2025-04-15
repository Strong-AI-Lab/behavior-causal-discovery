
import pytest
import torch

from dynamics.solver import DynamicsSolver


class TestDynamicsSolver:

    @pytest.fixture
    def solver(self):
        return DynamicsSolver(mass=1.0, dimensions=3)
    
    def test_apply_force(self, solver):
        x = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
        v = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
        force = torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float32)
        dt = 1

        x, v = solver.apply_force(x, v, force, dt)
        
        gt_x = [[[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]]
        gt_v = [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]

        assert x.tolist() == gt_x
        assert v.tolist() == gt_v

    def test_apply_force_2(self, solver):
        x = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
        v = torch.tensor([[[1.0, -1.0, 0.0]]], dtype=torch.float32)
        force = torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float32)
        dt = 2

        x, v = solver.apply_force(x, v, force, dt)
        
        gt_x = [[[1.0, 2.0, 3.0], [2.5, 1.5, 3.5], [5.0, 2.0, 5.0]]]
        gt_v = [[[1.0, -1.0, 0.0], [2.0, 0.0, 1.0], [3.0, 1.0, 2.0]]]
        
        assert x.tolist() == gt_x
        assert v.tolist() == gt_v

    def test_compute_acceleration(self, solver):
        x = torch.tensor([[[1.0, 2.0, 3.0],
                           [1.0, 2.0, 2.0],
                           [2.0, 1.0, 3.0],
                           [2.0, 1.0, 5.0],
                           [1.0, 2.0, 5.0],
                           [2.0, 1.0, 5.0]]], dtype=torch.float32)

        acc, v = solver.compute_acceleration(x, return_velocity=True)
        
        gt_acc =  [[[0.0, 0.0, -2.0],
                    [2.0, -2.0, 6.0],
                    [-4.0, 4.0, -4.0],
                    [2.0, -2.0, 0.0],
                    [2.0, -2.0, 0.0],
                    [0.0, 0.0, 0.0]]]
        
        gt_v =  [[[0.0, 0.0, 0.0],
                  [0.0, 0.0, -2.0],
                  [2.0, -2.0, 4.0],
                  [-2.0, 2.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [2.0, -2.0, 0.0]]]

        assert acc.tolist() == gt_acc
        assert v.tolist() == gt_v

    def test_compute_force(self, solver):
        x = torch.tensor([[[1.0, 2.0, 3.0],
                           [1.0, 2.0, 2.0],
                           [2.0, 1.0, 3.0],
                           [2.0, 1.0, 5.0],
                           [1.0, 2.0, 5.0],
                           [2.0, 1.0, 5.0]]], dtype=torch.float32)

        force = solver.compute_force(x)
        
        gt_force =  [[[0.0, 0.0, -2.0],
                      [2.0, -2.0, 6.0],
                      [-4.0, 4.0, -4.0],
                      [2.0, -2.0, 0.0],
                      [2.0, -2.0, 0.0],
                      [0.0, 0.0, 0.0]]]

        assert force.tolist() == gt_force

    def test_compute_apply(self, solver):
        x = torch.tensor([[[1.0, 2.0, 3.0],
                           [1.0, 2.0, 2.0],
                           [2.0, 1.0, 3.0],
                           [2.0, 1.0, 5.0],
                           [1.0, 2.0, 5.0],
                           [2.0, 1.0, 5.0]]], dtype=torch.float32)

        force, _, v = solver.compute_force(x, return_velocity=True)

        new_x = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
        new_v = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
        for i in range(5):
            new_x, new_v = solver.apply_force(new_x, new_v, force[:,i,:].unsqueeze(1), 1)

        assert new_x.tolist() == x.tolist()
        assert new_v.tolist() == v.tolist()

