
from typing import Optional, Tuple

import torch


class DynamicsSolver():
    """
    A class that represents a dynamics solver for simulating physical systems.

    Parameters:
    - mass (float): The mass of the object.
    - dimensions (int): The number of dimensions in the system.

    Methods:
    - apply_force(x, v=None, force=None, dt=1): Applies a force to the system.
    - compute_velocity(x): Computes the velocity of the system.
    - compute_acceleration(x, v=None): Computes the acceleration of the system.
    """

    def __init__(self, mass: float = 1, dimensions: int = 3):
        self.mass = mass
        self.dimensions = dimensions

    def apply_force(self, x: torch.Tensor, v: Optional[torch.Tensor] = None, force: Optional[torch.Tensor] = None, dt: int = 1):
        """
        Applies a force to the system.

        Parameters:
        - x (torch.Tensor): The position of the system.
        - v (torch.Tensor, optional): The velocity of the system. If not provided, it is initialized as zero.
        - force (torch.Tensor, optional): The force to be applied. If not provided, it is initialized as zero.
        - dt (int): The number of timesteps to apply the force.

        Returns:
        - x (torch.Tensor): The updated position of the system.
        - v (torch.Tensor): The updated velocity of the system.
        """
        batch_size = x.size(0)  # [batch_size, lookback, dimensions]

        if v is None:
            v = torch.zeros(x.shape, dtype=torch.float32)

        if force is None:
            force = torch.zeros((batch_size, 1, self.dimensions), dtype=torch.float32)

        acc = (force / self.mass).repeat(1, dt, 1)  # apply force for dt timesteps
        time = torch.arange(1,dt+1, dtype=torch.float32).reshape((1, dt, 1)).repeat(batch_size, 1, self.dimensions)  # time account for dt timesteps

        new_v = v[:, -1, :].unsqueeze(1).repeat(1, dt, 1) + acc * time
        new_x = x[:, -1, :].unsqueeze(1).repeat(1, dt, 1) + new_v * time

        v = torch.cat((v, new_v), dim=1)
        x = torch.cat((x, new_x), dim=1)

        return x, v  # [batch_size, lookback+dt, dimensions]

    def compute_velocity(self, x: torch.Tensor):
        """
        Computes the velocity of the system.

        Parameters:
        - x (torch.Tensor): The position of the system.

        Returns:
        - v (torch.Tensor): The velocity of the system.
        """
        batch_size = x.size(0)
        lookback = x.size(1)

        v = torch.zeros((batch_size, lookback - 1, self.dimensions), dtype=torch.float32)
        for i in range(lookback - 1):
            v[:, i, :] = x[:, i + 1, :] - x[:, i, :]

        return v  # [batch_size, lookback-1, dimensions]

    def compute_acceleration(self, x: torch.Tensor, v: Optional[torch.Tensor] = None):
        """
        Computes the acceleration of the system.

        Parameters:
        - x (torch.Tensor): The position of the system.
        - v (torch.Tensor, optional): The velocity of the system. If not provided, it is computed using compute_velocity().

        Returns:
        - a (torch.Tensor): The acceleration of the system.
        """
        batch_size = x.size(0)
        lookback = x.size(1)

        if v is None:
            v = self.compute_velocity(x)

        a = torch.zeros((batch_size, lookback - 2, self.dimensions), dtype=torch.float32)
        for i in range(lookback - 2):
            a[:, i, :] = v[:, i + 1, :] - v[:, i, :]

        return a  # [batch_size, lookback-2, dimensions]